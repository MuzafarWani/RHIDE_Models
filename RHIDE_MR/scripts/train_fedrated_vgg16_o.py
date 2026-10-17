#!/usr/bin/env python3
# Train_federated_vgg16.py
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

from src.federated.priority import PriorityRegistry
from src.federated.messaging.bus import MessagingBus

from src.solver.mappings_vgg16 import vgg16_betas
from src.solver.param_feeds import DeviceStatic, BetaCoeffs, build_round_coeffs
from src.solver.gurobi_solver import RoundMILPSolver, SolverParams
from src.solver.postprocess import summarize_solution

from src.training.round_scheduler import build_round_batches


# =========================================================
# DATASET
# =========================================================
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


# =========================================================
# MAIN
# =========================================================
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

    out_dir = train_cfg.get("out_dir", "runs/vgg16")
    ensure_dir(out_dir)

    train_ds = build_cifar10_dataset(train_cfg["batch_size"])
    train_iter = iter(train_ds)

    full_model = build_vgg16_model(num_classes=train_cfg["num_classes"])

    preg = PriorityRegistry(fed_cfg)
    devs_by_edge = {e["id"]: e["devices"] for e in fed_cfg["edges"]}

    # =========================================================
    # ✅ EDGE CONSTRAINTS (FIXED & SAFE)
    # =========================================================
    edge_ram_bytes = {}
    edge_energy_budget = {}

    for e in fed_cfg["edges"]:
        eid = e["id"]

        # Use None if not provided (important for solver logic)
        ram_val = e.get("ram_edge_s", None)
        energy_val = e.get("energy_budget_per_round", None)

        edge_ram_bytes[eid] = int(ram_val) if ram_val is not None else None
        edge_energy_budget[eid] = float(energy_val) if energy_val is not None else None

    print("\n========== EDGE CONSTRAINTS ==========")
    print("Edge RAM:", edge_ram_bytes)
    print("Edge Energy Budget:", edge_energy_budget)
    print("=====================================\n")

    # =========================================================
    betas = vgg16_betas()
    allowed = split_cfg["allowed"]

    alpha_future = float(solver_cfg.get("alpha_future", 0.0))
    lambda_comm = float(solver_cfg.get("lambda_comm", 0.0))
    lambda_latency = float(solver_cfg.get("lambda_latency", 0.0))

    edge_load = {e: 0 for e in devs_by_edge}
    EDGE_LOAD_ALPHA = 0.002

    Erem_state = {d: cfg["Erem_J"] for d, cfg in el_cfg["devices"].items()}

    # =========================================================
    # TRAINING LOOP
    # =========================================================
    for t in range(args.rounds):

        print(f"\n================ ROUND {t} =================")

        remaining_rounds = max(1, args.rounds - t)

        # -----------------------------------------------------
        # Solver params
        # -----------------------------------------------------
        sp = SolverParams(
            lambda_energy=float(solver_cfg.get("lambda_energy", 1.0)),
            gamma_edge=float(solver_cfg.get("gamma_edge", 1.0)),
            lambda_comm=lambda_comm,
            lambda_latency=lambda_latency,
            eta=float(solver_cfg.get("eta", 1.0)),
            xi=float(solver_cfg.get("xi", 1.0)),
            alpha_future=alpha_future,
            remaining_rounds=remaining_rounds,
            mip_gap=float(solver_cfg.get("mip_gap", 0.01)),
            time_limit_s=float(solver_cfg.get("time_limit_s", 30)),
            threads=int(solver_cfg.get("threads", 0)),
        )

        # -----------------------------------------------------
        # Data batches
        # -----------------------------------------------------
        device_batches = build_round_batches(
            data_cfg=None,
            fed_cfg=fed_cfg,
            train_cfg=train_cfg,
            priority=preg,
            rng_seed=args.seed + t,
        )

        batches_per_device = {d: len(v) for d, v in device_batches.items()}

        devices_static = []
        per_dev_beta = {}

        # -----------------------------------------------------
        # Build device profiles
        # -----------------------------------------------------
        for d, cfg in el_cfg["devices"].items():

            E_available = Erem_state[d] / remaining_rounds
            Emax_effective = min(cfg["Emax_round_J"], E_available)

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
                    Emax_round_J=Emax_effective,
                    Erem_J=Erem_state[d],
                    Umax_bytes=cfg["Umax_bytes"],
                    Tmax_s=cfg["Tmax_s"],
                    RBS_bytes=cfg["RBS_bytes"],
                    Tedge_s=dynamic_tedge,
                    Tsync_s=cfg["Tsync_s"],
                )
            )

            pb = cfg["per_beta"]

            per_dev_beta[d] = BetaCoeffs(
                Ecomp_J=pb["Ecomp_J"],
                Ecomm_J=pb["Ecomm_J"],
                U_bytes=[split_cfg["A_size_bytes"][b] for b in allowed],
                Tcomp_s=pb["Tcomp_s"],
                Tcomm_s=pb["Tcomm_s"],
                M_bytes=pb["M_bytes"],
            )

        # -----------------------------------------------------
        # Build round coefficients
        # -----------------------------------------------------
        rc = build_round_coeffs(
            devices_static,
            batches_per_device,
            per_dev_beta,
            betas
        )

        # ✅ Attach edge constraints (CRITICAL)
        rc.edge_ram_bytes = edge_ram_bytes
        rc.edge_energy_budget = edge_energy_budget

        # -----------------------------------------------------
        # Solve
        # -----------------------------------------------------
        sol = RoundMILPSolver(rc, sp).solve()

        print(summarize_solution(sol))

        # -----------------------------------------------------
        # Update battery
        # -----------------------------------------------------
        for d, e_spent in sol.energy_J.items():
            Erem_state[d] = max(0.0, Erem_state[d] - e_spent)

        # -----------------------------------------------------
        # Update edge load
        # -----------------------------------------------------
        edge_load = {e: 0 for e in edge_load}
        for d, z in sol.z.items():
            if z == 1:
                e = el_cfg["devices"][d]["edge_id"]
                edge_load[e] += 1

    full_model.save(os.path.join(out_dir, "savedmodel"))
    print("Training complete")


if __name__ == "__main__":
    main()
