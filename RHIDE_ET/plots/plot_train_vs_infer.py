#!/usr/bin/env python3
"""
Plot average training vs inference times per split-point (beta).

Usage:
  python plot_train_vs_infer.py --csv Train_infer_time.csv --out /home/muzafar/FedML/my_examples/XEenetNet_Real_data/plots/train_vs_infer.png

Input CSV expected header:
avg_beta_train,avg_train_time,avg_beta_infer,avg_infer_time
"""

import argparse
import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_and_validate(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")  # accept comma or tabs
    # expected columns
    expected = {"avg_beta_train", "avg_train_time", "avg_beta_infer", "avg_infer_time"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"CSV missing expected columns. Found: {list(df.columns)}. Need: {sorted(expected)}")
    # Convert to numeric
    for c in ["avg_beta_train", "avg_train_time", "avg_beta_infer", "avg_infer_time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[["avg_train_time", "avg_infer_time"]].isnull().any().any():
        raise ValueError("Found non-numeric values in time columns after conversion.")
    return df


def build_plot(df: pd.DataFrame, out_path: str, show: bool = True):
    # create a unified x axis (use the union of betas, prefer train betas if identical)
    # We'll pair rows by beta if possible; else align by index.
    if np.array_equal(df["avg_beta_train"].values, df["avg_beta_infer"].values):
        betas = df["avg_beta_train"].astype(int).values
        train_times = df["avg_train_time"].values
        infer_times = df["avg_infer_time"].values
    else:
        # try to align by beta value (more robust)
        df_t = df[["avg_beta_train", "avg_train_time"]].rename(columns={"avg_beta_train": "beta", "avg_train_time": "train"})
        df_i = df[["avg_beta_infer", "avg_infer_time"]].rename(columns={"avg_beta_infer": "beta", "avg_infer_time": "infer"})
        merged = pd.merge(df_t, df_i, on="beta", how="outer").sort_values("beta")
        betas = merged["beta"].astype(int).values
        train_times = merged["train"].fillna(0).values
        infer_times = merged["infer"].fillna(0).values

    x = np.arange(len(betas))
    width = 0.35

    plt.figure(figsize=(8, 5))
    ax = plt.gca()

    bars1 = ax.bar(x - width/2, train_times, width, label="Avg train time (s)")
    bars2 = ax.bar(x + width/2, infer_times, width, label="Avg infer time (s)")

    # Labels & ticks
    ax.set_xlabel("Split point")
    ax.set_ylabel("Time (seconds)")
    #ax.set_title("Average training vs inference time per split point")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(b)) for b in betas])
    ax.legend()

    # Annotate bars with values
    def annotate(bars):
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.annotate(f"{h:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4),  # 4 points vertical offset
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    annotate(bars1)
    annotate(bars2)

    plt.tight_layout()
    # Save
    ensure_dir = os.path.dirname(out_path) or "."
    os.makedirs(ensure_dir, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    if show:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot average training vs inference times per split point")
    ap.add_argument("--csv", required=True, help="CSV file with columns: avg_beta_train,avg_train_time,avg_beta_infer,avg_infer_time")
    ap.add_argument("--out", default="train_vs_infer.png", help="Output PNG path")
    ap.add_argument("--no-show", action="store_true", help="Do not display the plot interactively")
    args = ap.parse_args()

    df = load_and_validate(args.csv)
    build_plot(df, args.out, show=not args.no_show)
    print(f"Saved plot to {args.out}")


if __name__ == "__main__":
    main()

