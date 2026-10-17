#!/usr/bin/env python3
# scripts/prepare_data.py
from __future__ import annotations

import os
import argparse

from src.utils.io_helpers import read_yaml, ensure_dir
from src.data.wrf_reader import WRFReader
from src.data.dataset_builder import build_labeled_arrays
from src.data.normalization import compute_channel_stats, save_stats, apply_channel_stats
from src.data.encoders import SimpleLabelEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", default="configs/data.yaml")
    ap.add_argument("--out_dir", default=None, help="override out_dir where stats/encoder/cache will be written")
    args = ap.parse_args()

    cfg = read_yaml(args.data_cfg)
    out_dir = args.out_dir or os.path.dirname(cfg.get("norm_stats_out", "runs/expA/norm_stats.json")) or "runs/expA"
    ensure_dir(out_dir)

    print("[prepare_data] loading WRF:", cfg["wrf_file"])
    wrf = WRFReader(cfg["wrf_file"], var_map=cfg["var_map"]).read_all()

    # Build labeled arrays
    print("[prepare_data] building train arrays ...")
    X2_tr, X3_tr, y_tr, enc = build_labeled_arrays(
        wrf=wrf,
        locations_csv=cfg["locations_train_csv"],
        window_hw=int(cfg["window_half_width"]),
        feature2d_keys=cfg["feature2d_keys"],
        feature3d_keys=cfg["feature3d_keys"],
        classes=cfg["classes"],
    )
    print("[prepare_data] building val arrays ...")
    X2_va, X3_va, y_va, _ = build_labeled_arrays(
        wrf=wrf,
        locations_csv=cfg["locations_val_csv"],
        window_hw=int(cfg["window_half_width"]),
        feature2d_keys=cfg["feature2d_keys"],
        feature3d_keys=cfg["feature3d_keys"],
        classes=cfg["classes"],
        label_encoder=enc,
    )
    print("[prepare_data] building test arrays ...")
    X2_te, X3_te, y_te, _ = build_labeled_arrays(
        wrf=wrf,
        locations_csv=cfg["locations_test_csv"],
        window_hw=int(cfg["window_half_width"]),
        feature2d_keys=cfg["feature2d_keys"],
        feature3d_keys=cfg["feature3d_keys"],
        classes=cfg["classes"],
        label_encoder=enc,
    )

    # Norm stats from train
    print("[prepare_data] computing normalization stats ...")
    stats = compute_channel_stats(X2_tr, X3_tr)
    save_stats(cfg["norm_stats_out"], stats)
    print(f"[prepare_data] saved stats -> {cfg['norm_stats_out']}")

    # Persist label encoder
    enc.save(cfg["label_encoder_out"])
    print(f"[prepare_data] saved label encoder -> {cfg['label_encoder_out']}")

    # Optionally save cached arrays
    cache_cfg = cfg.get("cache_arrays", {})
    if cache_cfg.get("enabled", False):
        cdir = cache_cfg.get("dir", os.path.join(out_dir, "cache"))
        ensure_dir(cdir)
        import numpy as np
        np.save(os.path.join(cdir, "X2_train.npy"), X2_tr)
        np.save(os.path.join(cdir, "X3_train.npy"), X3_tr)
        np.save(os.path.join(cdir, "y_train.npy"), y_tr)
        np.save(os.path.join(cdir, "X2_val.npy"), X2_va)
        np.save(os.path.join(cdir, "X3_val.npy"), X3_va)
        np.save(os.path.join(cdir, "y_val.npy"), y_va)
        np.save(os.path.join(cdir, "X2_test.npy"), X2_te)
        np.save(os.path.join(cdir, "X3_test.npy"), X3_te)
        np.save(os.path.join(cdir, "y_test.npy"), y_te)
        print(f"[prepare_data] cached arrays -> {cdir}")

    print("[prepare_data] done.")


if __name__ == "__main__":
    main()

