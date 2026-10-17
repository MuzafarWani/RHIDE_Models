# plot_avg_splits_solver_vs_rl.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ User config ============
csv_file_1 = "split_point_avg_high.csv"
csv_file_2 = "split_point_avg_low.csv"   # <-- second CSV
out_fname = "MADRLvsSPHERE_avg_splits_combined.png"
dpi = 300
# =====================================

# -------- Read CSVs --------
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

rounds1 = df1['round'].values
rounds2 = df2['round'].values

avg_solver_1 = df1['avg_beta_1'].values
avg_rl_1     = df1['avg_beta_2'].values

avg_solver_2 = df2['avg_beta_1'].values
avg_rl_2     = df2['avg_beta_2'].values

# -------- Plot --------
plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(8, 3.0))

# --- CSV 1 ---
ax.plot(
    rounds1, avg_solver_1,
    label='SPHERE β (High)',
    color='#1b9e77',
    linewidth=1.8
)
ax.plot(
    rounds1, avg_rl_1,
    label='MADRL β (High)',
    color='#6a3d9a',
    linewidth=1.6
)

# --- CSV 2 ---
ax.plot(
    rounds2, avg_solver_2,
    label='SPHERE β (Low)',
    color='#2f7f7f',
    linestyle='--',
    linewidth=1.8
)
ax.plot(
    rounds2, avg_rl_2,
    label='MADRL β (Low)',
    color='#b2182b',
    linestyle='--',
    linewidth=1.6
)

# -------- Axes / legend --------
ax.set_xlabel('Training round')
ax.set_ylabel('Average split layer (β)')
ax.legend(loc='upper right', fontsize=9, ncol=2)
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(out_fname, dpi=dpi)
print("Saved:", out_fname)

