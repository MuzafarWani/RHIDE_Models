#!/usr/bin/env python3
# Train_fedrated_vgg16.py
from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.io_helpers import read_yaml, ensure_dir
from src.utils.seeding import set_global_seed

from tensorflow.keras.datasets import cifar10

from src.models.vgg16 import build_vgg16_model
from src.models.split_graph_vgg16 import create_vgg16_submodels

from src.federated.priority import PriorityRegistry
from src.federated.messaging.bus import MessagingBus
from src.federated.messaging.messages import ModelUpdate
from src.federated.device_client.uploader import Uploader
from src.federated.edge_server.aggregator import EdgeAggregator
from src.federated.cloud_server.global_aggregator import CloudAggregator
from src.federated.cloud_server.distributor import Distributor
from src.federated.device_client.trainer import DeviceTrainer

from src.solver.mappings_vgg16 import vgg16_betas
from src.solver.param_feeds import DeviceStatic, BetaCoeffs, build_round_coeffs
from src.solver.gurobi_solver import RoundMILPSolver, SolverParams
from src.solver.postprocess import summarize_solution

from src.training.round_scheduler import build_round_batches
from src.federated.cloud_server.rl_agent import CloudRLAggregationAgent, CloudRLConfig


def build_cifar10_dataset(batch_size: int):
    (x_tr, y_tr), _ = cifar10.load_data()
    y_tr = y_tr.squeeze().astype("int64")

    def preprocess(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.image.resize(x, (224, 224))
        return x, y

    ds = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
    ds = ds.shuffle(10000)
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def broadcast_global_model(bus: MessagingBus, weights):
    msg = ModelUpdate(sender_id="cloud", receiver_id="edges", weights=weights)
    try:
        setattr(msg, "created_ts", float(tf.timestamp()))
    except Exception:
        pass

    print("[cloud] Broadcasting new global model to all edges")
    bus.publish("cloud_to_edges_broadcast", msg)


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cfg", required=True)
    ap.add_argument("--fed_cfg", required=True)
    ap.add_argument("--split_cfg", required=True)
    ap.add_argument("--el_cfg", required=True)
    ap.add_argument("--solver_cfg", required=True)
    ap.add_argument("--rl_cfg", required=True)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_global_seed(args.seed, deterministic_tf=True)

    train_cfg = read_yaml(args.train_cfg)
    fed_cfg = read_yaml(args.fed_cfg)
    split_cfg = read_yaml(args.split_cfg)
    el_cfg = read_yaml(args.el_cfg)
    solver_cfg = read_yaml(args.solver_cfg)
    rl_cfg = read_yaml(args.rl_cfg)

    print("\n================ CONFIG DEBUG ================")
    print("Train cfg:", train_cfg)
    print("Solver cfg:", solver_cfg)
    print("Allowed splits:", split_cfg["allowed"])
    print("Edges:", fed_cfg["edges"])
    print("=============================================\n")

    out_dir = train_cfg.get("out_dir", "runs/vgg16")
    ensure_dir(out_dir)

    train_ds = build_cifar10_dataset(train_cfg["batch_size"])
    train_iter = iter(train_ds)

    full_model = build_vgg16_model(num_classes=train_cfg["num_classes"])
    print("[VGG16] Global model built")

    preg = PriorityRegistry(fed_cfg)
    devs_by_edge = {e["id"]: e["devices"] for e in fed_cfg["edges"]}

    # --------------------------------------------------
    # EDGE RAM
    # --------------------------------------------------

    edge_ram_bytes = {}
    for e in fed_cfg["edges"]:
        edge_ram_bytes[e["id"]] = int(e.get("ram_edge_s", 0))

    print("\n[DEBUG] Edge RAM limits")
    for e, v in edge_ram_bytes.items():
        print(e, v)

    # --------------------------------------------------
    # NEW: EDGE ENERGY BUDGET
    # --------------------------------------------------

    edge_energy_budget = {}
    for e in fed_cfg["edges"]:
        edge_energy_budget[e["id"]] = float(e.get("energy_budget_per_round", 0))

    print("\n[DEBUG] Edge energy budgets")
    for e, v in edge_energy_budget.items():
        print(e, v)

    # --------------------------------------------------

    betas = vgg16_betas()
    allowed = split_cfg["allowed"]

    sp = SolverParams(
        lambda_energy=float(solver_cfg.get("lambda_energy", 1.0)),
        gamma_edge=float(solver_cfg.get("gamma_edge", 1.0)),
        eta=float(solver_cfg.get("eta", 1.0)),
        xi=float(solver_cfg.get("xi", 1.0)),
        mip_gap=float(solver_cfg.get("mip_gap", 0.01)),
        time_limit_s=float(solver_cfg.get("time_limit_s", 30)),
        threads=int(solver_cfg.get("threads", 0)),
    )

    print("\n[DEBUG] Solver params", sp)

    cloud_rl = CloudRLAggregationAgent(
        CloudRLConfig(
            obs_keys=("q_pending_activations", "q_pending_raw", "num_edges_reporting")
        ),
        seed=args.seed,
    )

    bus = MessagingBus()

    edge_models = {}
    edge_root_model = {}
    edge_aggs = {}

    for e in devs_by_edge:
        edge_models[e] = {}
        edge_root_model[e] = tf.keras.models.clone_model(full_model)
        edge_root_model[e].set_weights(full_model.get_weights())

        for d in devs_by_edge[e]:
            m = tf.keras.models.clone_model(full_model)
            m.set_weights(full_model.get_weights())
            edge_models[e][d] = m

        class _Cfg:
            pass

        _cfg = _Cfg()
        _cfg.train_cfg = train_cfg
        _cfg.rl_cfg = rl_cfg

        agg = EdgeAggregator(edge_id=e, bus=bus, cfg=_cfg)
        agg.set_full_model(edge_root_model[e])
        edge_aggs[e] = agg

    cloud_agg = CloudAggregator(cfg=type("C", (), {"rl_cfg": rl_cfg})())
    cloud_agg.bind_bus(bus)

    Distributor(bus=bus, cfg=type("D", (), {})())
    uploader = Uploader(bus=bus)

    for e, devs in devs_by_edge.items():
        for d in devs:
            DeviceTrainer(full_model=edge_models[e][d])

    edge_load = {e: 0 for e in devs_by_edge}
    EDGE_LOAD_ALPHA = 0.002

    Erem_state = {d: cfg["Erem_J"] for d, cfg in el_cfg["devices"].items()}

    for t in range(args.rounds):

        print(f"\n================ ROUND {t} =================")

        device_batches = build_round_batches(
            data_cfg=None,
            fed_cfg=fed_cfg,
            train_cfg=train_cfg,
            priority=preg,
            rng_seed=args.seed + t,
        )

        batches_per_device = {d: len(v) for d, v in device_batches.items()}

        print("\n[DEBUG] batches per device")
        print(batches_per_device)

        print("\n[DEBUG] battery state")
        print(Erem_state)

        devices_static = []
        per_dev_beta = {}

        for d, cfg in el_cfg["devices"].items():

            print(f"\nDEVICE {d}")
            print("Edge:", cfg["edge_id"])
            print("Memory:", cfg["mem_max_bytes"])
            print("Battery:", Erem_state[d])
            print("Latency limit:", cfg["Tmax_s"])

            pb = cfg["per_beta"]

            print("Per-beta parameters")

            for b in range(len(pb["Ecomp_J"])):
                print(
                    f"beta {b+1} -> "
                    f"Ecomp {pb['Ecomp_J'][b]} "
                    f"Ecomm {pb['Ecomm_J'][b]} "
                    f"M {pb['M_bytes'][b]}"
                )

            base_tedge = cfg["Tedge_s"]

            dynamic_tedge = [
                v + EDGE_LOAD_ALPHA * edge_load[cfg["edge_id"]]
                for v in base_tedge
            ]

            devices_static.append(
                DeviceStatic(
                    device_id=d,
                    edge_id=cfg["edge_id"],
                    priority=preg.device_priority(d),
                    mem_max_bytes=cfg["mem_max_bytes"],
                    uplink_rate_Bps=cfg["uplink_rate_Bps"],
                    comm_cost_per_byte_J=cfg["comm_cost_per_byte_J"],
                    Emax_round_J=cfg["Emax_round_J"],
                    Erem_J=Erem_state[d],
                    Umax_bytes=cfg["Umax_bytes"],
                    Tmax_s=cfg["Tmax_s"],
                    RBS_bytes=cfg["RBS_bytes"],
                    Tedge_s=dynamic_tedge,
                    Tsync_s=cfg["Tsync_s"],
                )
            )

            per_dev_beta[d] = BetaCoeffs(
                Ecomp_J=pb["Ecomp_J"],
                Ecomm_J=pb["Ecomm_J"],
                U_bytes=[split_cfg["A_size_bytes"][b] for b in allowed],
                Tcomp_s=pb["Tcomp_s"],
                Tcomm_s=pb["Tcomm_s"],
                M_bytes=pb["M_bytes"],
            )

        rc = build_round_coeffs(devices_static, batches_per_device, per_dev_beta, betas)

        # --------------------------------------------------
        # pass edge constraints to solver
        # --------------------------------------------------

        rc.edge_ram_bytes = edge_ram_bytes
        rc.edge_energy_budget = edge_energy_budget

        print("\n================ RoundCoeffs DEBUG ================")
        print("Devices:", list(rc.statics.keys()))
        print("Transfer energy:", rc.transfer_energy_J)
        print("Transfer time:", rc.transfer_time_s)
        print("===================================================")

        print("\nCalling solver now...")

        sol = RoundMILPSolver(rc, sp).solve()

        summary = summarize_solution(sol)

        print("\n================ SOLVER OUTPUT ================")
        print(summary)

        for d in sol.z:
            print(
                f"{d} -> z={sol.z[d]} k={sol.k[d]} beta={sol.beta_choice.get(d,'')}"
            )

        print("================================================")

        cloud_updated = cloud_agg.maybe_aggregate_and_update(t)

        if cloud_updated:
            gw = cloud_agg.get_weights()
            if gw is not None:
                broadcast_global_model(bus, gw)

        for d, e_spent in sol.energy_J.items():
            Erem_state[d] -= e_spent
            if Erem_state[d] < 0:
                Erem_state[d] = 0

        print("[battery]", Erem_state)

        edge_load = {e: 0 for e in edge_load}

        for d, z in sol.z.items():
            if z == 1:
                e = el_cfg["devices"][d]["edge_id"]
                edge_load[e] += 1

        print("[edge-load]", edge_load)

        print(f"[ROUND {t} COMPLETE]")

    full_model.save(os.path.join(out_dir, "savedmodel"))

    print("Training complete")


if __name__ == "__main__":
    main()
