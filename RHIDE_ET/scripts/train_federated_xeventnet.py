# =========================================================
# Train_federated_xeventnet.py (EXECUTION TIME)
# =========================================================

from __future__ import annotations
import os, sys, argparse
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.io_helpers import read_yaml, ensure_dir
from src.utils.seeding import set_global_seed

# DATA
from src.data.wrf_reader import WRFReader
from src.data.dataset_builder import build_labeled_arrays
from src.data.normalization import load_stats, apply_channel_stats
from src.data.encoders import SimpleLabelEncoder

# MODEL
from src.models.xeventnet import build_model
from src.models.split_graph import create_submodels

# FEDERATED
from src.federated.priority import PriorityRegistry
from src.federated.messaging.bus import MessagingBus
from src.federated.messaging.messages import ModelUpdate, Activations
from src.federated.edge_server.aggregator import EdgeAggregator
from src.federated.cloud_server.global_aggregator import CloudAggregator
from src.federated.cloud_server.distributor import Distributor
from src.federated.cloud_server.rl_agent import CloudRLAggregationAgent, CloudRLConfig

# SOLVER
from src.solver.mappings import default_betas
from src.solver.param_feeds import DeviceStatic, BetaCoeffs, build_round_coeffs
from src.solver.gurobi_solver import RoundMILPSolver, SolverParams
from src.solver.postprocess import summarize_solution

from src.training.round_scheduler import build_round_batches


def broadcast_global_model(bus, weights):
    msg = ModelUpdate("cloud", "edges", weights)
    bus.publish("cloud_to_edges_broadcast", msg)


def main():

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

    # ================= LOAD CONFIGS =================
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
        label_encoder=enc,
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

    edge_models = {}
    edge_root_model = {}
    edge_aggs = {}

    for e in devs_by_edge:
        edge_models[e] = {}

        edge_root_model[e] = tf.keras.models.clone_model(full_model)
        edge_root_model[e].set_weights(full_model.get_weights())

        cfg = type("C", (), {"train_cfg": train_cfg, "rl_cfg": rl_cfg})()
        agg = EdgeAggregator(edge_id=e, bus=bus, cfg=cfg)
        agg.set_full_model(edge_root_model[e])
        edge_aggs[e] = agg

        for d in devs_by_edge[e]:
            m = tf.keras.models.clone_model(full_model)
            m.set_weights(full_model.get_weights())
            edge_models[e][d] = m
            agg.register_device_model(d, m)

    cloud = CloudAggregator(cfg=type("C", (), {"rl_cfg": rl_cfg})())
    cloud.bind_bus(bus)

    cloud_rl = CloudRLAggregationAgent(
        CloudRLConfig(obs_keys=("q_pending_activations", "q_pending_raw", "num_edges_reporting")),
        seed=args.seed,
    )

    Distributor(bus=bus, cfg=type("D", (), {})())

    # ================= EDGE CONSTRAINTS =================
    edge_ram_bytes = {e["id"]: int(e.get("ram_edge_s", 0)) for e in fed_cfg["edges"]}
    edge_energy_budget = {e["id"]: float(e.get("energy_budget_per_round", 0)) for e in fed_cfg["edges"]}

    # ================= EDGE LOAD =================
    edge_load = {e: 0 for e in devs_by_edge}
    EDGE_LOAD_ALPHA = 0.002

    # ================= SOLVER =================
    betas = default_betas()
    allowed = split_cfg["allowed"]

    sp = SolverParams(
        eta=solver_cfg.get("eta", 1.0),
        xi=solver_cfg.get("xi", 1.0),
    )

    # ================= TRAINING LOOP =================
    for t in range(args.rounds):

        print(f"\n[XEVENTNET][ROUND {t}]")

        device_batches = build_round_batches(data_cfg, fed_cfg, train_cfg, preg, args.seed + t)
        batches_per_device = {d: len(v) for d, v in device_batches.items()}

        devices_static = []
        per_dev_beta = {}

        for d, cfg in el_cfg["devices"].items():

            dynamic_tedge = [
                v + EDGE_LOAD_ALPHA * edge_load[cfg["edge_id"]]
                for v in cfg["Tedge_s"]
            ]

            devices_static.append(DeviceStatic(
                device_id=d,
                edge_id=cfg["edge_id"],
                priority=preg.device_priority(d),
                mem_max_bytes=cfg["mem_max_bytes"],
                uplink_rate_Bps=cfg["uplink_rate_Bps"],
                comm_cost_per_byte_J=cfg["comm_cost_per_byte_J"],
                Emax_round_J=cfg["Emax_round_J"],
                Erem_J=cfg["Erem_J"],
                Umax_bytes=cfg["Umax_bytes"],
                Tmax_s=cfg["Tmax_s"],
                RBS_bytes=cfg["RBS_bytes"],
                Tedge_s=dynamic_tedge,
                Tsync_s=cfg["Tsync_s"],
            ))

            pb = cfg["per_beta"]

            per_dev_beta[d] = BetaCoeffs(
                Ecomp_J=pb["Ecomp_J"],
                Ecomm_J=pb["Ecomm_J"],
                U_bytes=[split_cfg["A_size_bytes"][b] for b in allowed],
                Tcomp_s=pb["Tcomp_s"],
                Tcomm_s=pb["Tcomm_s"],
                M_bytes=pb["M_bytes"],
            )

        rc = build_round_coeffs(devices_static, batches_per_device, per_dev_beta, betas)
        rc.edge_ram_bytes = edge_ram_bytes
        rc.edge_energy_budget = edge_energy_budget

        sol = RoundMILPSolver(rc, sp).solve()
        print(summarize_solution(sol))

        # ================= DEVICE → EDGE =================
        for e, devs in devs_by_edge.items():
            for d in devs:

                if sol.z.get(d, 0) == 0:
                    continue

                beta = sol.beta_choice.get(d, "")
                if not beta:
                    continue

                idx = np.random.choice(N, size=train_cfg["batch_size"])
                X3b, yb = X3[idx], y[idx]

                device_net, _ = create_submodels(
                    edge_models[e][d], beta,
                    tuple(train_cfg["input_shape_2d"]),
                    tuple(train_cfg["input_shape_3d"]),
                    train_cfg["num_classes"],
                )

                a3 = device_net.predict(X3b, verbose=0)

                msg = Activations(
                    device_id=d,
                    edge_id=e,
                    beta=beta,
                    arrays={"a3d": a3},
                    labels=yb,
                    batch_id=f"{d}_r{t}",
                )

                bus.publish("device_to_edge_activations", msg)

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

        # ================= EDGE LOAD UPDATE =================
        edge_load = {e: 0 for e in edge_load}
        for d, z in sol.z.items():
            if z == 1:
                e = el_cfg["devices"][d]["edge_id"]
                edge_load[e] += 1

    full_model.save(os.path.join(out_dir, "savedmodel"))
    print("Training complete")


if __name__ == "__main__":
    main()
