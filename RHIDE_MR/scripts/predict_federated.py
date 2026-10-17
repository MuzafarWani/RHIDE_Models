#!/usr/bin/env python3
# scripts/predict_federated.py

# python scripts/predict_federated.py --data_cfg configs/data.yaml --train_cfg configs/train.yaml --fed_cfg out/adaptive_split_switch/federated.yaml --split_cfg configs/#split.yaml --el_cfg out/adaptive_split_switch/energy_latency.yaml --pred_locations data/labels/snowfall_inference.txt --out_csv runs/expA/predictions_adaptive_split_switch.csv --#seed 1

from __future__ import annotations

import os, sys
import argparse
import csv
import json
import numpy as np
import tensorflow as tf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.io_helpers import read_yaml, ensure_dir
from src.utils.seeding import set_global_seed
from src.data.wrf_reader import WRFReader
from src.data.dataset_builder import build_unlabeled_arrays
from src.data.normalization import load_stats, apply_channel_stats
from src.models.xeventnet import build_model
from src.models.split_graph import create_submodels
from src.models.serialization import make_device_edge_from_full
from src.federated.priority import PriorityRegistry
from src.federated.messaging.bus import MessagingBus
from src.federated.edge_server.aggregator import EdgeAggregator
from src.solver.mappings import default_betas
from src.solver.param_feeds import DeviceStatic, BetaCoeffs
from src.solver.beta_selection import select_beta_min_energy


# ------------------------
# Helpers
# ------------------------

def make_batches_from_indices(N, batch_size):
    idxs = np.arange(N)
    return [idxs[i:i + batch_size].tolist() for i in range(0, N, batch_size)]


def map_activations_to_edge_inputs(edge_net: tf.keras.Model, activ):
    if not isinstance(activ, dict):
        return activ

    input_names = [inp.name.split(":")[0] for inp in edge_net.inputs]
    mapping = {}
    used = set()

    for k, v in activ.items():
        lk = k.lower()
        for name in input_names:
            if name in used:
                continue
            if ("2d" in lk and "2d" in name.lower()) or \
               ("3d" in lk and "3d" in name.lower()) or \
               ("joint" in lk):
                mapping[name] = v
                used.add(name)
                break

    leftovers = [v for v in activ.values() if v not in mapping.values()]
    for name in input_names:
        if name not in mapping and leftovers:
            mapping[name] = leftovers.pop(0)

    return mapping


# ------------------------
# Main
# ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", default="configs/data.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    ap.add_argument("--fed_cfg", default="configs/federated.yaml")
    ap.add_argument("--split_cfg", default="configs/split.yaml")
    ap.add_argument("--el_cfg", default="configs/energy_latency.yaml")
    ap.add_argument("--pred_locations", required=True)
    ap.add_argument("--out_csv", default="runs/pred_out.csv")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_global_seed(args.seed, deterministic_tf=True)

    data_cfg  = read_yaml(args.data_cfg)
    train_cfg = read_yaml(args.train_cfg)
    fed_cfg   = read_yaml(args.fed_cfg)
    split_cfg = read_yaml(args.split_cfg)
    el_cfg    = read_yaml(args.el_cfg)

    ensure_dir(os.path.dirname(args.out_csv) or ".")

    # ------------------------
    # Load data
    # ------------------------

    wrf = WRFReader(data_cfg["wrf_file"], var_map=data_cfg["var_map"]).read_all()
    stats = load_stats(data_cfg["norm_stats_out"])

    X2_all, X3_all, metas = build_unlabeled_arrays(
        wrf=wrf,
        locations_csv=args.pred_locations,
        window_hw=int(data_cfg["window_half_width"]),
        feature2d_keys=data_cfg["feature2d_keys"],
        feature3d_keys=data_cfg["feature3d_keys"],
    )

    X2_all, X3_all = apply_channel_stats(X2_all, X3_all, stats)

    if len(X2_all) == 0:
        print("[predict] No inference samples.")
        return

    # ------------------------
    # Build model
    # ------------------------

    full_model = build_model(
        tuple(train_cfg["input_shape_2d"]),
        tuple(train_cfg["input_shape_3d"]),
        num_classes=int(train_cfg["num_classes"]),
    )

    if train_cfg.get("savedmodel_path") and os.path.exists(train_cfg["savedmodel_path"]):
        full_model = tf.keras.models.load_model(train_cfg["savedmodel_path"])
        print(f"[predict] Loaded model from {train_cfg['savedmodel_path']}")

    # ------------------------
    # Federation
    # ------------------------

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

    # ------------------------
    # Device statics & beta coeffs
    # ------------------------

    betas   = default_betas()
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

    # ------------------------
    # Assign batches
    # ------------------------

    batch_size = int(train_cfg.get("batch_size", 32))
    batches = make_batches_from_indices(len(X2_all), batch_size)

    device_list = [d for e in devs_by_edge.values() for d in e]
    device_batches = {d: [] for d in device_list}

    for i, idxs in enumerate(batches):
        device_batches[device_list[i % len(device_list)]].append(idxs)

    # ------------------------
    # SELECT β (NO SOLVER)
    # ------------------------

    beta_sel = {}
    for dev in devices_static:
        B = len(device_batches.get(dev.device_id, []))
        beta = select_beta_min_energy(per_dev_beta[dev.device_id], dev, betas)
        beta_sel[dev.device_id] = beta if (beta and B > 0) else ""

    # ------------------------
    # Inference execution
    # ------------------------

    out_rows = []

    for dev_id, blist in device_batches.items():
        st = next(d for d in devices_static if d.device_id == dev_id)
        beta = beta_sel[dev_id]

        for idxs in blist:
            X2b, X3b = X2_all[idxs], X3_all[idxs]

            if beta:
                device_net, edge_net = create_submodels(
                    full_model=full_model,
                    beta_name=beta,
                    input_shape_2d=tuple(train_cfg["input_shape_2d"]),
                    input_shape_3d=tuple(train_cfg["input_shape_3d"]),
                    num_classes=int(train_cfg["num_classes"]),
                )
                make_device_edge_from_full(full_model, device_net, edge_net)

                activ_out = device_net.predict(
                    {"input_2d": X2b, "input_3d": X3b}, verbose=0
                )

                activ = {}
                if isinstance(activ_out, (list, tuple)):
                    activ["a2d"], activ["a3d"] = activ_out
                else:
                    activ["joint"] = activ_out

                preds = edge_net.predict(
                    map_activations_to_edge_inputs(edge_net, activ), verbose=0
                )
            else:
                preds = edge_aggs[st.edge_id].trainer.full_model.predict(
                    {"input_2d": X2b, "input_3d": X3b}, verbose=0
                )

            for i, gi in enumerate(idxs):
                meta = metas[gi]
                out_rows.append({
                    "timestep": meta.get("timestep", gi),
                    "XLAT": meta.get("XLAT"),
                    "XLONG": meta.get("XLONG"),
                    "device": dev_id,
                    "edge": st.edge_id,
                    "pred_class": int(np.argmax(preds[i])),
                    "probs": json.dumps(preds[i].tolist()),
                })

    # ------------------------
    # Write output
    # ------------------------

    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)

    print(f"[predict] Wrote {len(out_rows)} predictions → {args.out_csv}")


if __name__ == "__main__":
    main()

