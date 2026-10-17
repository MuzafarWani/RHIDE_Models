#!/usr/bin/env python3
"""
Plot aggregation frequency (O_l, O_e) vs training rounds
for MADRL and SPHERE, with overlap-safe visualization.

- X-axis: training rounds
- Y-axis: aggregation period (O)
- Separate plot per scenario
- Step plots (aggregation changes over time)
- Small vertical offsets to avoid overlapping lines
"""

import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
CSV_FILE = "aggregration_data.csv"   # <-- update if needed
OUT_DIR = "plots_aggregation"
Path(OUT_DIR).mkdir(exist_ok=True)

FIGSIZE = (7, 4)
DPI = 300

# Small vertical offsets (visual only, values unchanged conceptually)
OFFSET = {
    ("MADRL", "O_l"):  0.05,
    ("MADRL", "O_e"):  0.10,
    ("SPHERE", "O_l"): -0.05,
    ("SPHERE", "O_e"): -0.10,
}

LINESTYLE = {
    "MADRL": "--",
    "SPHERE": "-",
}

MARKER = {
    "O_l": "o",
    "O_e": "s",
}

COLOR = {
    ("MADRL", "O_l"): "#1f77b4",   # blue
    ("MADRL", "O_e"): "#ff7f0e",   # orange
    ("SPHERE", "O_l"): "#2ca02c",  # green
    ("SPHERE", "O_e"): "#d62728",  # red
}

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def expand_rounds(round_str):
    """
    Convert "1-400" → list(range(1, 401))
    """
    start, end = map(int, round_str.split("-"))
    return list(range(start, end + 1))


# --------------------------------------------------
# LOAD & EXPAND DATA
# --------------------------------------------------
df = pd.read_csv(CSV_FILE)

rows = []
for _, r in df.iterrows():
    for rd in expand_rounds(r["rounds"]):
        rows.append({
            "round": rd,
            "model": r["model"],
            "layer": r["layer"],
            "frequency": r["frequency"],
            "Scenario": r["Scenario"],
        })

df_exp = pd.DataFrame(rows)

# --------------------------------------------------
# PLOTTING
# --------------------------------------------------
for scenario, sdf in df_exp.groupby("Scenario"):
    plt.figure(figsize=FIGSIZE)

    for (model, layer), g in sdf.groupby(["model", "layer"]):
        offset = OFFSET.get((model, layer), 0.0)

        plt.step(
            g["round"],
            g["frequency"] + offset,
            where="post",
            label=f"{model} {layer}",
            linestyle=LINESTYLE[model],
            linewidth=2.5 if model == "SPHERE" else 2.0,
            marker=MARKER[layer],
            markersize=3,
            color=COLOR[(model, layer)],
        )

    plt.xlabel("Training Rounds")
    plt.ylabel("Aggregation Period (O)")
    #plt.title(f"Aggregation Frequency vs Rounds\n{scenario}")

    plt.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    plt.legend(ncol=2, fontsize=9)

    # ---- IMPORTANT: Explain offset in caption/footnote ----
   # plt.figtext(
   #     0.5, -0.12,
   #     "Note: Small vertical offsets are applied for visualization clarity only.",
   #     ha="center", fontsize=8
   # )

    plt.tight_layout()

    outname = re.sub(r"[^\w\-]+", "_", scenario.lower())
    plt.savefig(f"{OUT_DIR}/{outname}.png", dpi=DPI, bbox_inches="tight")
    plt.close()

print(f"[OK] Plots saved in: {OUT_DIR}/")

