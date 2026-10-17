#!/usr/bin/env python3
# scripts/predict_federated_vgg16.py
#python scripts/predict_federated_vgg16.py   --data_cfg configs_vgg16/data.yaml   --train_cfg configs_vgg16/train.yaml   --fed_cfg outvgg/base100/federated.yaml   --split_cfg #configs_vgg16/split.yaml   --el_cfg outvgg/base100/energy_latency.yaml   --out_csv runs/vgg16/predictions_base100.csv   --seed 1
from __future__ import annotations

import os, sys
import argparse
import csv
import json
import numpy as np
import tensorflow as tf

# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

from src.utils.io_helpers import read_yaml, ensure_dir
from src.utils.seeding import set_global_seed

# -----------------------------------------------------------------------------
# VGG16-specific model & split
# -----------------------------------------------------------------------------

from src.models.vgg16 import build_vgg16_model
from src.models.split_graph_vgg16 import create_vgg16_submodels

# -----------------------------------------------------------------------------
# Federated components
# -----------------------------------------------------------------------------

from src.federated.priority import PriorityRegistry
from src.federated.messaging.bus import MessagingBus
from src.federated.edge_server.aggregator import EdgeAggregator

# -----------------------------------------------------------------------------
# Solver / beta selection
# -----------------------------------------------------------------------------

from src.solver.mappings_vgg16 import vgg16_betas
from src.solver.param_feeds import DeviceStatic, BetaCoeffs
from src.solver.beta_selection import select_beta_min_energy


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def make_batches_from_indices(N: int, batch_size: int):
    idxs = np.arange(N)
    return [idxs[i:i + batch_size].tolist() for i in range(0, N, batch_size)]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", required=True)
    ap.add_argument("--train_cfg", required=True)
    ap.add_argument("--fed_cfg", required=True)
    ap.add_argument("--split_cfg", required=True)
    ap.add_argument("--el_cfg", required=True)
    ap.add_argument("--out_csv", default="runs/vgg16/predictions.csv")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    # -------------------------------------------------------------------------
    # Setup
    # -------------------------------------------------------------------------

    set_global_seed(args.seed, deterministic_tf=True)

    data_cfg  = read_yaml(args.data_cfg)
    train_cfg = read_yaml(args.train_cfg)
    fed_cfg   = read_yaml(args.fed_cfg)
    split_cfg = read_yaml(args.split_cfg)
    el_cfg    = read_yaml(args.el_cfg)

    ensure_dir(os.path.dirname(args.out_csv) or ".")

    # -------------------------------------------------------------------------
    # Load inference data
    # -------------------------------------------------------------------------

    if "inference_npz" not in data_cfg:
        raise KeyError("[predict][VGG16] data_cfg must define 'inference_npz'")

    X_all = np.load(data_cfg["inference_npz"])["X"]
    N = len(X_all)

    if N == 0:
        print("[predict][VGG16] No inference samples.")
        return

    # -------------------------------------------------------------------------
    # Build full VGG16 model (ONCE)
    # -------------------------------------------------------------------------

    full_model = build_vgg16_model(
        num_classes=int(train_cfg["num_classes"])
    )

    if train_cfg.get("savedmodel_path") and os.path.exists(train_cfg["savedmodel_path"]):
        full_model = tf.keras.models.load_model(train_cfg["savedmodel_path"])
        print(f"[predict][VGG16] Loaded model from {train_cfg['savedmodel_path']}")

    # -------------------------------------------------------------------------
    # Federated setup
    # -------------------------------------------------------------------------

    preg = PriorityRegistry(fed_cfg)
    devs_by_edge = {e["id"]: e["devices"] for e in fed_cfg["edges"]}

    bus = MessagingBus()
    edge_aggs = {}

    for edge_id in devs_by_edge:
        agg_cfg = type(
            "Cfg",
            (),
            {
                "train_cfg": train_cfg,
                "rl_cfg": {"edge_actions": [1], "exploration_eps": 0.0},
            },
        )()
        agg = EdgeAggregator(edge_id=edge_id, bus=bus, cfg=agg_cfg)
        agg.set_full_model(full_model)
        edge_aggs[edge_id] = agg

    # -------------------------------------------------------------------------
    # Device statics & beta coefficients
    # -------------------------------------------------------------------------

    betas   = vgg16_betas()
    allowed = split_cfg["allowed"]
    A_size  = split_cfg["A_size_bytes"]

    devices_static = []
    per_dev_beta = {}

    for dev_id, dcfg in el_cfg["devices"].items():
        devices_static.append(DeviceStatic(
            device_id=dev_id,
            edge_id=dcfg["edge_id"],
            priority=preg.device_priority(dev_id),
            mem_max_bytes=dcfg["mem_max_bytes"],
            uplink_rate_Bps=dcfg["uplink_rate_Bps"],
            comm_cost_per_byte_J=dcfg["comm_cost_per_byte_J"],
            Emax_round_J=dcfg["Emax_round_J"],
            Erem_J=dcfg["Erem_J"],
            Umax_bytes=dcfg["Umax_bytes"],
            Tmax_s=dcfg["Tmax_s"],
            RBS_bytes=dcfg["RBS_bytes"],
            Tedge_s=dcfg["Tedge_s"],
            Tsync_s=dcfg["Tsync_s"],
        ))

        pb = dcfg["per_beta"]
        per_dev_beta[dev_id] = BetaCoeffs(
            Ecomp_J=pb["Ecomp_J"],
            Ecomm_J=pb["Ecomm_J"],
            U_bytes=[A_size[b] for b in allowed],
            Tcomp_s=pb["Tcomp_s"],
            Tcomm_s=pb["Tcomm_s"],
            M_bytes=pb["M_bytes"],
        )

    # -------------------------------------------------------------------------
    # Assign batches
    # -------------------------------------------------------------------------

    batch_size = int(train_cfg.get("batch_size", 32))
    batches = make_batches_from_indices(N, batch_size)

    device_list = [d for e in devs_by_edge.values() for d in e]
    device_batches = {d: [] for d in device_list}

    for i, idxs in enumerate(batches):
        device_batches[device_list[i % len(device_list)]].append(idxs)

    # -------------------------------------------------------------------------
    # Select β
    # -------------------------------------------------------------------------

    beta_sel = {}
    for dev in devices_static:
        B = len(device_batches.get(dev.device_id, []))
        beta = select_beta_min_energy(per_dev_beta[dev.device_id], dev, betas)
        beta_sel[dev.device_id] = beta if (beta and B > 0) else ""

    # -------------------------------------------------------------------------
    # 🔑 BUILD SPLIT MODELS ONCE PER DEVICE (CRITICAL FIX)
    # -------------------------------------------------------------------------

    split_models = {}

    for dev in devices_static:
        beta = beta_sel[dev.device_id]
        if beta:
            device_net, edge_net = create_vgg16_submodels(
                full_model=full_model,
                beta_name=beta,
            )
            split_models[dev.device_id] = (device_net, edge_net)
        else:
            split_models[dev.device_id] = None

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    out_rows = []

    for dev_id, blist in device_batches.items():
        st = next(d for d in devices_static if d.device_id == dev_id)
        models = split_models[dev_id]

        for idxs in blist:
            Xb = X_all[idxs]

            if models is not None:
                device_net, edge_net = models
                activ = device_net.predict(Xb, verbose=0)
                preds = edge_net.predict(activ, verbose=0)
            else:
                preds = edge_aggs[st.edge_id].trainer.full_model.predict(Xb, verbose=0)

            for i, gi in enumerate(idxs):
                out_rows.append({
                    "sample_id": int(gi),
                    "device": dev_id,
                    "edge": st.edge_id,
                    "pred_class": int(np.argmax(preds[i])),
                    "probs": json.dumps(preds[i].tolist()),
                })

    # -------------------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------------------

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)

    print(f"[predict][VGG16] Wrote {len(out_rows)} predictions → {args.out_csv}")


if __name__ == "__main__":
    main()

