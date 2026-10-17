#!/usr/bin/env python3
# scripts/run_inference.py
from __future__ import annotations

import argparse
import os
import math
import json
import csv
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# --- project imports (same as your codebase) ---
from src.utils.io_helpers import read_yaml, ensure_dir
from src.data.wrf_reader import WRFReader
from src.inference.build_grids import build_unlabeled_arrays
from src.data.normalization import load_stats, apply_channel_stats
from src.data.encoders import SimpleLabelEncoder
from src.models.xeventnet import build_model
from src.inference.predictor import SplitPredictor
from src.inference.postprocess import attach_meta_and_save_csv

# Optional β solving
from src.solver.mappings import default_betas
from src.solver.param_feeds import DeviceStatic, BetaCoeffs, build_round_coeffs
from src.solver.gurobi_solver import InferenceBetaSolver, SolverParams

# For gradient calculations
import tensorflow as tf


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

# --- Insert after imports in run_inference.py ---
import math
import numpy as np
from typing import Tuple

def _center_pad_or_crop_2d_array(arr: np.ndarray, expected_h: int, expected_w: int) -> np.ndarray:
    """
    arr: shape (N, H, W, C)
    Return array with spatial dims adjusted to (expected_h, expected_w) using center-crop or symmetric zero-pad.
    """
    if arr.ndim != 4:
        raise ValueError("_center_pad_or_crop_2d_array expects a 4-D array (N,H,W,C)")
    N, H, W, C = arr.shape
    # If already correct
    if H == expected_h and W == expected_w:
        return arr
    # Crop if larger
    top = 0
    left = 0
    if H > expected_h:
        top = (H - expected_h) // 2
    if W > expected_w:
        left = (W - expected_w) // 2
    if H > expected_h or W > expected_w:
        h1 = top
        h2 = top + expected_h
        w1 = left
        w2 = left + expected_w
        arr = arr[:, h1:h2, w1:w2, :]
        N2, H2, W2, C2 = arr.shape
        if H2 == expected_h and W2 == expected_w:
            return arr
    # Pad if smaller
    pad_h = max(0, expected_h - arr.shape[1])
    pad_w = max(0, expected_w - arr.shape[2])
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pads = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    return np.pad(arr, pads, mode="constant", constant_values=0.0)


def _center_pad_or_crop_3d_array(arr: np.ndarray, expected_h: int, expected_w: int) -> np.ndarray:
    """
    arr: expected layout (N, D, H, W, C) or (N, H, W, D, C) or similar.
    We assume spatial H/W are at positions -3 and -2 (i.e., arr.shape[-3], arr.shape[-2]).
    Adjust those dims (center crop or pad) to (expected_h, expected_w) and return array.
    """
    if arr.ndim < 4:
        raise ValueError("_center_pad_or_crop_3d_array expects an array with at least 4 dims")
    shape = list(arr.shape)
    # assume H,W are at -3,-2
    H_idx = -3
    W_idx = -2
    H = shape[H_idx]
    W = shape[W_idx]
    if H == expected_h and W == expected_w:
        return arr
    # We will slice/pad along these axes.
    # If cropping needed:
    slices = [slice(None)] * arr.ndim
    if H > expected_h:
        top = (H - expected_h) // 2
        slices[H_idx] = slice(top, top + expected_h)
    if W > expected_w:
        left = (W - expected_w) // 2
        slices[W_idx] = slice(left, left + expected_w)
    arr = arr[tuple(slices)]
    # After cropping, maybe need padding
    new_shape = list(arr.shape)
    new_H = new_shape[H_idx]
    new_W = new_shape[W_idx]
    pad_h = max(0, expected_h - new_H)
    pad_w = max(0, expected_w - new_W)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # build pads tuple for np.pad
    pads = []
    for i in range(arr.ndim):
        if i == H_idx:
            pads.append((pad_top, pad_bottom))
        elif i == W_idx:
            pads.append((pad_left, pad_right))
        else:
            pads.append((0, 0))
    return np.pad(arr, tuple(pads), mode="constant", constant_values=0.0)


# -----------------------------------------------------
# Helper: Convert simple text file (timestep,lat,lon per line) -> clean CSV
# -----------------------------------------------------
def convert_text_to_temp_csv(txt_path: str) -> str:
    """
    Convert a plain text file (no header) with lines like:
        1,20,69
        2,20.4,68.2
    or tab/space separated variants into a temporary CSV file with header:
        timestep,xlat,xlong
    Returns path to the new temp CSV file (comma-delimited).
    """
    p = Path(txt_path)
    if not p.is_file():
        raise FileNotFoundError(f"locations_txt not found: {txt_path}")

    raw = p.read_text(errors="ignore")

    # strip BOM if present
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Input text file is empty")

    cleaned_rows = []
    for ln in lines:
        # Accept: csv (comma), tsv (tab), or whitespace-separated
        if ',' in ln:
            parts = [x.strip() for x in ln.split(',')]
        elif '\t' in ln:
            parts = [x.strip() for x in ln.split('\t')]
        else:
            parts = [x.strip() for x in ln.split()]

        if len(parts) != 3:
            # skip header-like or malformed lines
            low = ln.lower()
            if low.startswith("timestep") and ("xlat" in low or "xlong" in low):
                # skip header-like line
                continue
            print(f"[warn] skipping malformed line (expected 3 fields): {ln!r}")
            continue

        ts_s, lat_s, lon_s = parts
        # coerce types
        try:
            ts_i = int(float(ts_s))
            lat_f = float(lat_s)
            lon_f = float(lon_s)
        except Exception:
            print(f"[warn] skipping non-numeric line: {ln!r}")
            continue

        cleaned_rows.append((ts_i, lat_f, lon_f))

    if not cleaned_rows:
        raise ValueError("No valid data rows found in input text file after cleaning.")

    # Write a comma-separated CSV (force comma delimiter) with a clean header line
    tmp = tempfile.NamedTemporaryFile(prefix="locs_from_txt_", suffix=".csv", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with tmp_path.open("w", newline='') as wf:
        writer = csv.writer(wf, delimiter=',')
        writer.writerow(["timestep", "xlat", "xlong"])
        for (ts_i, lat_f, lon_f) in cleaned_rows:
            writer.writerow([ts_i, f"{lat_f:.6f}", f"{lon_f:.6f}"])

    # quick preview for debugging
    preview_lines = []
    with tmp_path.open("r", newline='') as rf:
        for i, ln in enumerate(rf):
            if i >= 10:
                break
            preview_lines.append(ln.rstrip("\n"))

    print(f"[inference] Converted text file to temp CSV: {tmp_path}")
    print("[inference] preview of cleaned CSV (first lines):")
    for pl in preview_lines:
        print(" ", pl)
    return str(tmp_path)


# -----------------------------------------------------
# Gradient calculation helper (unchanged)
# -----------------------------------------------------
def compute_per_sample_gradient_score(
    keras_model: tf.keras.Model,
    X2: np.ndarray,
    X3: np.ndarray,
    batch_size: int = 32,
) -> np.ndarray:
    n = X2.shape[0]
    scores = np.zeros(n, dtype=float)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        x2_batch = tf.convert_to_tensor(X2[start:end], dtype=tf.float32)
        x3_batch = tf.convert_to_tensor(X3[start:end], dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x2_batch)
            tape.watch(x3_batch)
            logits = keras_model([x2_batch, x3_batch], training=False)
            top_idx = tf.argmax(logits, axis=1)
            top_logits = tf.gather_nd(
                logits, tf.stack([tf.range(tf.shape(logits)[0]), top_idx], axis=1)
            )

        try:
            g2 = tape.gradient(top_logits, x2_batch)
            g3 = tape.gradient(top_logits, x3_batch)
        except Exception:
            g2, g3 = None, None

        for i in range(end - start):
            norm = 0.0
            if g2 is not None:
                norm += float(np.linalg.norm(g2[i].numpy().ravel(), ord=2))
            if g3 is not None:
                norm += float(np.linalg.norm(g3[i].numpy().ravel(), ord=2))
            scores[start + i] = norm

        del tape

    if scores.max() > 0:
        scores = scores / (scores.max() + 1e-12)
    return scores


def split_indexes_contiguous(n_samples: int, device_ids: List[str]) -> Dict[str, Tuple[int, int]]:
    dcount = len(device_ids)
    if dcount == 0:
        return {}
    per = _ceil_div(n_samples, dcount)
    out = {}
    for i, dev in enumerate(device_ids):
        s = i * per
        e = min(n_samples, (i + 1) * per)
        out[dev] = (s, e)
    return out


def build_device_list_from_elcfg(el_cfg_path: str, fed_cfg_path: Optional[str] = None) -> List[str]:
    el = read_yaml(el_cfg_path)
    device_ids = list(el.get("devices", {}).keys())
    if fed_cfg_path:
        fed = read_yaml(fed_cfg_path)
        if "devices" in fed and isinstance(fed["devices"], dict):
            device_ids = list(fed["devices"].keys())
    return device_ids


# -----------------------------------------------------
# Main entrypoint
# -----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_cfg", default="configs/data.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    ap.add_argument("--locations_txt", required=True, help="Text file with lines: timestep,lat,lon (no header)")
    ap.add_argument("--beta", default=None, help="split cut name; defaults to train.yaml:default_beta")
    ap.add_argument("--out_csv", default="runs/expA/infer_out.csv")
    ap.add_argument("--el_cfg", default="configs/energy_latency.yaml")
    ap.add_argument("--fed_cfg", default=None)
    ap.add_argument("--device_ids", default=None)
    ap.add_argument("--grad_filter_frac", type=float, default=1.0)
    ap.add_argument("--grad_top_k", type=int, default=0)
    ap.add_argument("--grad_batch", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--per_device_outdir", default=None)
    args = ap.parse_args()

    data_cfg = read_yaml(args.data_cfg)
    train_cfg = read_yaml(args.train_cfg)
    beta_arg = args.beta or train_cfg.get("default_beta", "post_pool1")

    # Convert text file → temporary CSV
    print("[inference] converting text file to CSV format...")
    cleaned_csv = convert_text_to_temp_csv(args.locations_txt)

    # Load WRF data
    print("[inference] loading WRF and building grids...")
    try:
        wrf = WRFReader(data_cfg["wrf_file"], var_map=data_cfg["var_map"]).read_all()
    except Exception as e:
        raise RuntimeError(f"Failed to load WRF file ({data_cfg.get('wrf_file')}): {e}")

    # Call build_unlabeled_arrays with the cleaned CSV
    # If it fails with the header-as-data issue, try to repair by rewriting file without any
    # lines whose first token equals 'timestep' (case-insensitive) and retry once.
    def try_build_arrays(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        return build_unlabeled_arrays(
            wrf=wrf,
            locations_csv=csv_path,
            window_hw=int(data_cfg["window_half_width"]),
            feature2d_keys=data_cfg["feature2d_keys"],
            feature3d_keys=data_cfg["feature3d_keys"],
        )

    try:
        X2, X3, meta = try_build_arrays(cleaned_csv)
    except Exception as e:
        # Inspect the cleaned CSV for obvious header-as-data issues and repair
        print("[error] build_unlabeled_arrays failed:", e)
        print("[error] Attempting auto-repair of cleaned CSV (remove stray header-like rows)...")
        repaired_path = _remove_header_rows_from_csv(cleaned_csv)
        print("[inference] Re-run build_unlabeled_arrays with repaired CSV:", repaired_path)
        try:
            X2, X3, meta = try_build_arrays(repaired_path)
            cleaned_csv = repaired_path  # use repaired
        except Exception as e2:
            print("[error] build_unlabeled_arrays still failed after repair:", e2)
            print("[error] Reprinting first 20 lines of cleaned CSV for debugging:")
            with open(cleaned_csv, "r") as f:
                for i, ln in enumerate(f):
                    if i >= 20:
                        break
                    print(" ", ln.rstrip("\n"))
            raise

    n_samples = X2.shape[0]
    print(f"[inference] total samples built: {n_samples}")

    # Normalize inputs
    stats = load_stats(data_cfg["norm_stats_out"])
    X2n, X3n = apply_channel_stats(X2, X3, stats)
    enc = SimpleLabelEncoder.load(data_cfg["label_encoder_out"])

    # Build model
    print("[inference] building model...")
    full = build_model(
        tuple(train_cfg["input_shape_2d"]),
        tuple(train_cfg["input_shape_3d"]),
        num_classes=int(train_cfg["num_classes"]),
    )

    # Decide which devices to emulate
    if args.device_ids:
        device_ids = [d.strip() for d in args.device_ids.split(",") if d.strip()]
    else:
        device_ids = build_device_list_from_elcfg(args.el_cfg, args.fed_cfg)
    if not device_ids:
        raise RuntimeError("[inference] No devices found in el_cfg/fed_cfg or via --device_ids")

    print(f"[inference] Emulating devices: {device_ids}")

    # Predictor
    pred = SplitPredictor(
        full,
        tuple(train_cfg["input_shape_2d"]),
        tuple(train_cfg["input_shape_3d"]),
        num_classes=int(train_cfg["num_classes"]),
    )

    # Optionally compute gradient scores once on full set
    grad_scores = None
    if (args.grad_filter_frac < 1.0) or (args.grad_top_k > 0):
        print("[inference] computing gradient importance scores (this may take a while)...")
        grad_scores = compute_per_sample_gradient_score(full, X2n, X3n, batch_size=args.grad_batch)

    # Split samples among devices (contiguous chunks)
    idx_map = split_indexes_contiguous(X2n.shape[0], device_ids)
    all_preds, all_proba, all_meta = [], [], []

    for dev in device_ids:
        s, e = idx_map.get(dev, (0, 0))
        if s >= e:
            print(f"[inference][{dev}] no samples assigned (range {s}:{e}) -> skipping")
            continue
        dev_X2, dev_X3, dev_meta = X2n[s:e], X3n[s:e], meta[s:e]

        # apply gradient filter (per-device)
        if grad_scores is not None:
            local_scores = grad_scores[s:e]
            if args.grad_top_k and args.grad_top_k > 0:
                k = min(args.grad_top_k, len(local_scores))
                keep_idx = np.argsort(local_scores)[-k:] if k > 0 else np.array([], dtype=int)
            else:
                frac = float(args.grad_filter_frac)
                if frac >= 1.0:
                    keep_idx = np.arange(len(local_scores))
                elif frac <= 0.0:
                    keep_idx = np.array([], dtype=int)
                else:
                    k = max(1, int(math.ceil(frac * len(local_scores))))
                    keep_idx = np.argsort(local_scores)[-k:]
            dev_X2 = dev_X2[keep_idx]
            dev_X3 = dev_X3[keep_idx]
            dev_meta = [dev_meta[i] for i in keep_idx]

        if dev_X2.shape[0] == 0:
            print(f"[inference][{dev}] after filtering zero samples -> skipping")
            continue
        exp_2d = tuple(train_cfg["input_shape_2d"])
        expected_h_2d = int(exp_2d[0]); expected_w_2d = int(exp_2d[1])
        if dev_X2.ndim == 4:
            if (dev_X2.shape[1] != expected_h_2d) or (dev_X2.shape[2] != expected_w_2d):
                dev_X2 = _center_pad_or_crop_2d_array(dev_X2, expected_h_2d, expected_w_2d)
        exp_3d = tuple(train_cfg["input_shape_3d"])
        expected_h_3d = int(exp_3d[1]); expected_w_3d = int(exp_3d[2])
        if dev_X3.ndim >= 4:
           if (dev_X3.shape[-3] != expected_h_3d) or (dev_X3.shape[-2] != expected_w_3d):
               dev_X3 = _center_pad_or_crop_3d_array(dev_X3, expected_h_3d, expected_w_3d)
        
        print(f"[inference][{dev}] running on {dev_X2.shape[0]} samples, beta={beta_arg} (adjusted to {dev_X2.shape[1]}x{dev_X2.shape[2]})")
        yhat, proba = pred.predict(dev_X2, dev_X3, beta_name=beta_arg,
                           batch_size=args.batch_size or int(train_cfg.get("batch_size", 32)),
                           return_proba=True)
        all_preds.append(yhat)
        all_proba.append(proba)
        all_meta.extend(dev_meta)

        if args.per_device_outdir:
            per_out_root = Path(args.per_device_outdir)
            per_out_root.mkdir(parents=True, exist_ok=True)
            dev_csv = per_out_root / f"{dev}_preds.csv"
            attach_meta_and_save_csv(str(dev_csv), meta_rows=dev_meta, preds=yhat, proba=proba, encoder=enc)
            print(f"[inference][{dev}] wrote per-device CSV: {dev_csv}")

    # Merge and save
    if all_preds:
        all_preds_arr = np.concatenate(all_preds, axis=0)
        all_proba_arr = np.concatenate(all_proba, axis=0)
    else:
        all_preds_arr = np.zeros((0,), dtype=int)
        all_proba_arr = np.zeros((0, int(train_cfg["num_classes"])))

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    attach_meta_and_save_csv(args.out_csv, meta_rows=all_meta, preds=all_preds_arr, proba=all_proba_arr, encoder=enc)
    print(f"[inference] Done. Wrote: {args.out_csv}")


# ---------------------
# Small helper used by the try/repair flow above
# ---------------------
def _remove_header_rows_from_csv(csv_path: str) -> str:
    """
    Read csv_path and write a repaired CSV file where any row whose first column
    equals 'timestep' (case-insensitive) is removed. Return path to repaired tmp file.
    This addresses strange cases where the header has accidentally become a data row.
    """
    p = Path(csv_path)
    lines = p.read_text(errors="ignore").splitlines()
    repaired_rows = []
    for ln in lines:
        # strip leading/trailing whitespace
        s = ln.strip()
        if not s:
            continue
        # split by comma (we produced comma-delimited files earlier)
        parts = s.split(",")
        if len(parts) >= 1 and parts[0].strip().lower() == "timestep":
            # skip this header-like line
            continue
        repaired_rows.append(s)

    if not repaired_rows:
        raise ValueError("Repair step removed all rows; nothing left in CSV.")

    tmp = tempfile.NamedTemporaryFile(prefix="locs_repaired_", suffix=".csv", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with tmp_path.open("w", newline='') as wf:
        for r in repaired_rows:
            wf.write(r.rstrip("\n") + "\n")

    print(f"[inference] wrote repaired CSV: {tmp_path} (header-like rows removed)")
    return str(tmp_path)


if __name__ == "__main__":
    main()

