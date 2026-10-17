# plot_avg_splits_solver_vs_rl.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ User config ============
csv_file = "split_point_avg_high.csv"  # change if necessary
out_fname = "MADRLvsSPHERE_avg_splits_high.png"
dpi = 300
# =====================================

df = pd.read_csv(csv_file)
rounds = df['round'].values
avg_solver = df['avg_beta_1'].values
#std_solver = df['std_beta_solver'].values
avg_rl = df['avg_beta_2'].values
#std_rl = df['std_beta_rl'].values

plt.rcParams.update({'font.size': 10})
fig, ax = plt.subplots(figsize=(8,3.0))

ax.plot(rounds, avg_solver, label='SPHERE β', color='#1f77b4', linewidth=1.8)
ax.fill_between(rounds, avg_solver, avg_solver, color='#1f77b4', alpha=0.18)

ax.plot(rounds, avg_rl, label='MADRL β', color='#ff7f0e', linestyle='--', linewidth=1.6)
ax.fill_between(rounds, avg_rl, avg_rl, color='#ff7f0e', alpha=0.12)

ax.set_xlabel('Training round')
ax.set_ylabel('Average split layer (β)')
#ax.set_title('Average split across devices: Solver vs RL')
ax.legend(loc='upper right', fontsize=9)
ax.grid(axis='y', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(out_fname, dpi=dpi)
print("Saved:", out_fname)

