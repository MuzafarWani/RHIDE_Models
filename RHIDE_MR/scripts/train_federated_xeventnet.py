#!/usr/bin/env python3
from __future__ import annotations

import os, sys, argparse
import numpy as np
import tensorflow as tf

# ================= PATH =================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
sys.path = [SRC_PATH, PROJECT_ROOT] + sys.path
#if PROJECT_ROOT not in sys.path:
#    sys.path.insert(0, PROJECT_ROOT)

# ================= UTILS =================
from src.utils.io_helpers import read_yaml, ensure_dir
from src.utils.seeding import set_global_seed

# ================= DATA =================
from src.data.wrf_reader import WRFReader
from src.data.dataset_builder import build_labeled_arrays
from src.data.normalization import load_stats, apply_channel_stats
from src.data.encoders import SimpleLabelEncoder

# ================= MODEL =================
from src.models.xeventnet import build_model
from src.models.split_graph import create_submodels

# ================= FED =================
from src.federated.priority import PriorityRegistry
from src.federated.messaging.bus import MessagingBus
from src.federated.messaging.messages import ModelUpdate
from src.federated.device_client.uploader import Uploader
from src.federated.edge_server.aggregator import EdgeAggregator
from src.federated.cloud_server.global_aggregator import CloudAggregator
from src.federated.cloud_server.distributor import Distributor
from src.federated.cloud_server.rl_agent import CloudRLAggregationAgent, CloudRLConfig

# ================= SOLVER =================
from src.solver.mappings import default_betas
from src.solver.param_feeds import DeviceStatic, BetaCoeffs, build_round_coeffs
from src.solver.gurobi_solver import RoundMILPSolver, SolverParams
from src.solver.postprocess import summarize_solution

from src.training.round_scheduler import build_round_batches


def broadcast_global_model(bus, weights):
    bus.publish("cloud_to_edges_broadcast",
                ModelUpdate("cloud", "edges", weights))
import inspect
print("[DEBUG MODULE]", build_labeled_arrays.__module__)
print(inspect.getsource(build_labeled_arrays))

def main():

    # ================= ARGS =================
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", required=True)
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

    # ================= CONFIG =================
    data_cfg   = read_yaml(args.data_cfg)
    train_cfg  = read_yaml(args.train_cfg)
    fed_cfg    = read_yaml(args.fed_cfg)
    split_cfg  = read_yaml(args.split_cfg)
    el_cfg     = read_yaml(args.el_cfg)
    solver_cfg = read_yaml(args.solver_cfg)
    rl_cfg     = read_yaml(args.rl_cfg)

    out_dir = train_cfg.get("out_dir", "runs/xeventnet")
    ensure_dir(out_dir)

    # ================= DATA =================
    wrf = WRFReader(data_cfg["wrf_file"], var_map=data_cfg["var_map"]).read_all()
    enc = SimpleLabelEncoder.load(data_cfg["label_encoder_out"])

    X2, X3, y, _ = build_labeled_arrays(
        wrf=wrf,
        locations_csv=data_cfg["locations_train_csv"],
        window_hw=int(data_cfg["window_half_width"]),
        feature2d_keys=data_cfg["feature2d_keys"],
        feature3d_keys=data_cfg["feature3d_keys"],
        classes=data_cfg["classes"], 
        label_encoder=enc
    )

    stats = load_stats(data_cfg["norm_stats_out"])
    X2, X3 = apply_channel_stats(X2, X3, stats)

    N = len(y)

    # ================= MODEL =================
    full_model = build_model(
        tuple(train_cfg["input_shape_2d"]),
        tuple(train_cfg["input_shape_3d"]),
        num_classes=train_cfg["num_classes"],
    )

    # ================= FEDERATION =================
    preg = PriorityRegistry(fed_cfg)
    devs_by_edge = {e["id"]: e["devices"] for e in fed_cfg["edges"]}

    bus = MessagingBus()
    uploader = Uploader(bus=bus)
    Distributor(bus=bus, cfg=type("D", (), {})())

    edge_models, edge_aggs = {}, {}

    for e in devs_by_edge:
        edge_models[e] = {}

        cfg = type("C", (), {"train_cfg": train_cfg, "rl_cfg": rl_cfg})()
        agg = EdgeAggregator(edge_id=e, bus=bus, cfg=cfg)
        agg.set_full_model(tf.keras.models.clone_model(full_model))
        edge_aggs[e] = agg

        for d in devs_by_edge[e]:
            m = tf.keras.models.clone_model(full_model)
            m.set_weights(full_model.get_weights())
            edge_models[e][d] = m
            agg.register_device_model(d, m)

    cloud = CloudAggregator(cfg=type("C", (), {"rl_cfg": rl_cfg})())
    cloud.bind_bus(bus)

    cloud_rl = CloudRLAggregationAgent(
        CloudRLConfig(obs_keys=("q_pending_activations","q_pending_raw","num_edges_reporting")),
        seed=args.seed,
    )

    # ================= MULTI-ROUND STATE =================
    betas = default_betas()
    allowed = split_cfg["allowed"]

    edge_ram_bytes = {e["id"]: e.get("ram_edge_s", None) for e in fed_cfg["edges"]}
    edge_energy_budget = {e["id"]: e.get("energy_budget_per_round", None) for e in fed_cfg["edges"]}

    Erem_state = {d: cfg["Erem_J"] for d, cfg in el_cfg["devices"].items()}
    edge_load = {e: 0 for e in devs_by_edge}
    EDGE_LOAD_ALPHA = 0.002

    alpha_future = float(solver_cfg.get("alpha_future", 0.0))

    # ================= TRAINING LOOP =================
    for t in range(args.rounds):

        print(f"\n[XEVENTNET][ROUND {t}]")

        remaining_rounds = max(1, args.rounds - t)

        sp = SolverParams(
            lambda_energy=float(solver_cfg.get("lambda_energy", 1.0)),
            gamma_edge=float(solver_cfg.get("gamma_edge", 1.0)),
            lambda_comm=float(solver_cfg.get("lambda_comm", 0.0)),
            eta=float(solver_cfg.get("eta", 0.0)),
            xi=float(solver_cfg.get("xi", 0.0)),
            alpha_future=alpha_future,
            remaining_rounds=remaining_rounds,
        )

        device_batches = build_round_batches(
            data_cfg, fed_cfg, train_cfg, preg, args.seed + t
        )
        batches_per_device = {d: len(v) for d, v in device_batches.items()}

        devices_static, per_dev_beta = [], {}

        for d, cfg in el_cfg["devices"].items():

            E_available = Erem_state[d] / remaining_rounds
            Emax_effective = min(cfg["Emax_round_J"], E_available)

            devices_static.append(DeviceStatic(
                device_id=d,
                edge_id=cfg["edge_id"],
                priority=preg.device_priority(d),
                mem_max_bytes=cfg["mem_max_bytes"],
                uplink_rate_Bps=cfg["uplink_rate_Bps"],
                comm_cost_per_byte_J=cfg["comm_cost_per_byte_J"],
                Emax_round_J=Emax_effective,
                Erem_J=Erem_state[d],
                Umax_bytes=cfg["Umax_bytes"],
                Tmax_s=cfg["Tmax_s"],
                RBS_bytes=cfg["RBS_bytes"],
                Tedge_s=cfg["Tedge_s"],
                Tsync_s=cfg["Tsync_s"],
            ))

            pb = cfg["per_beta"]

            per_dev_beta[d] = BetaCoeffs(
                Ecomp_J=pb["Ecomp_J"],
                Ecomm_J=pb["Ecomm_J"],
                U_bytes=[float(split_cfg["A_size_bytes"][b]) for b in allowed],
                Tcomp_s=pb["Tcomp_s"],
                Tcomm_s=pb["Tcomm_s"],
                M_bytes=pb["M_bytes"],
            )

        rc = build_round_coeffs(devices_static, batches_per_device, per_dev_beta, betas)
        rc.edge_ram_bytes = edge_ram_bytes
        rc.edge_energy_budget = edge_energy_budget

        sol = RoundMILPSolver(rc, sp).solve()
        print(summarize_solution(sol))

        # ================= DEVICE =================
        for e, devs in devs_by_edge.items():
            for d in devs:
                if sol.z.get(d, 0) == 0:
                    continue

                beta = sol.beta_choice.get(d, "")
                if not beta:
                    continue

                idx = np.random.choice(N, size=train_cfg["batch_size"])
                X3b = X3[idx]

                device_net, _ = create_submodels(
                    edge_models[e][d], beta,
                    tuple(train_cfg["input_shape_2d"]),
                    tuple(train_cfg["input_shape_3d"]),
                    train_cfg["num_classes"],
                )

                activations = device_net(X3b, training=True)
                if hasattr(activations, "numpy"):
                    activations = activations.numpy()

                uploader.send_activations(
                    device_id=d,
                    edge_id=e,
                    beta=beta,
                    activations={"features": activations},
                )

        # 🔴 FIX: flatten device models
        device_models = {d: m for e in edge_models for d, m in edge_models[e].items()}
        for agg in edge_aggs.values():
            agg.device_models = device_models

        # ================= EDGE =================
        for agg in edge_aggs.values():
            agg.step_round(t)

        # ================= CLOUD =================
        updated = cloud.maybe_aggregate_and_update(t)
        if updated:
            broadcast_global_model(bus, cloud.get_weights())

        # ================= RL =================
        obs = {
            "q_pending_activations": sum(a.state.q_pending_activations for a in edge_aggs.values()),
            "q_pending_raw": sum(a.state.q_pending_raw for a in edge_aggs.values()),
            "num_edges_reporting": len(edge_aggs),
        }

        reward = 1.0 if updated else -0.1
        cloud_rl.learn(reward, obs)

        # ================= BATTERY =================
        for d, e_spent in sol.energy_J.items():
            Erem_state[d] = max(0.0, Erem_state[d] - e_spent)

    full_model.save(os.path.join(out_dir, "savedmodel"))
    print("Training complete")


if __name__ == "__main__":
    main()
