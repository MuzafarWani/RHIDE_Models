#!/usr/bin/env python3
# Usage:
#   python plot_decison_variables.py base15.csv

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# How many rounds to show (evenly spaced)
ROUNDS_TO_PLOT = 5

def main(csv_path):
    df = pd.read_csv(csv_path)

    # Ensure proper ordering
    df["device_num"] = df["device"].str.replace("d", "").astype(int)
    df = df.sort_values(["round", "device_num"])

    all_rounds = sorted(df["round"].unique())

    # Pick evenly spaced rounds
    if len(all_rounds) <= ROUNDS_TO_PLOT:
        rounds_sel = all_rounds
    else:
        idx = np.linspace(0, len(all_rounds) - 1, ROUNDS_TO_PLOT, dtype=int)
        rounds_sel = [all_rounds[i] for i in idx]

    devices = sorted(df["device"].unique(), key=lambda x: int(x[1:]))

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 10), sharex=True, constrained_layout=True
    )

    cmap = plt.get_cmap("tab10")

    for i, r in enumerate(rounds_sel):
        sub = df[df["round"] == r].set_index("device").loc[devices]
        color = cmap(i % 10)

        axes[0].plot(devices, sub["beta"], marker="o", color=color, label=f"Round {r}")
        axes[1].plot(devices, sub["z"], marker="o", color=color, label=f"Round {r}")
        axes[2].plot(devices, sub["K"], marker="o", color=color, label=f"Round {r}")

    axes[0].set_ylabel("β (Split Index)")
    axes[1].set_ylabel("z (Active)")
    axes[2].set_ylabel("K (Batches Uploaded)")
    axes[2].set_xlabel("Device ID")

    axes[0].set_title("SPHERE Decision Variables Across Devices and Rounds")

    # Put legend only once (top plot)
    axes[0].legend(ncol=len(rounds_sel), fontsize=9, loc="upper right")

    plt.savefig("sphere_beta_z_K_multiround.png", dpi=300)
    plt.close()

    print("Saved: sphere_beta_z_K_multiround.png")
    print("Rounds plotted:", rounds_sel)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_decison_variables.py Decision_varaible_filtered.csv")
        sys.exit(1)

    main(sys.argv[1])

