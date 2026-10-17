import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ User configuration ============
csv_file = "energy_usage_sphere_high_3.csv"
out_fname = "energy_usage_sphere_low.png"
dpi = 300

base_fontsize = 11
x_label_interval = 15

# Bar colors
color_compute = '#4C78A8'    # blue
color_comm = '#F58518'       # orange
color_transfer = '#8B0000'   # red
# ============================================

# ============ Load CSV (SPHERE) ============
df = pd.read_csv(csv_file)

rounds = df['round'].values
E_comp = df['E_comp_J'].values
E_comm = df['E_comm_J'].values
E_transfer = df['E_transfer_J'].values

# Y-axis limit
ymax = np.max(E_comp + E_comm + E_transfer) * 1.05

# ============ Plot: SPHERE only ============
plt.rcParams.update({'font.size': base_fontsize})
fig, ax = plt.subplots(1, 1, figsize=(8, 4))

width = 0.7

ax.bar(rounds, E_comp, width, label='Compute (E_comp)', color=color_compute)
ax.bar(rounds, E_comm, width, bottom=E_comp,
       label='Comm (E_comm)', color=color_comm)
ax.bar(rounds, E_transfer, width, bottom=E_comp + E_comm,
       label='Transfer (E_transfer)', color=color_transfer)

ax.set_xlabel('Training Round', fontsize=base_fontsize + 1)
ax.set_ylabel('Energy (J)', fontsize=base_fontsize + 1)

ax.set_xticks(rounds[::x_label_interval])
ax.set_xticklabels(rounds[::x_label_interval], fontsize=base_fontsize - 1)
ax.tick_params(axis='y', labelsize=base_fontsize - 1)

ax.legend(loc='upper right', fontsize=base_fontsize - 1)
ax.grid(axis='y', linestyle=':', alpha=0.4)
ax.set_title("SPHERE Energy Usage per Round", fontsize=base_fontsize + 2)
ax.set_ylim(0, ymax)

plt.tight_layout()
plt.savefig(out_fname, dpi=dpi, bbox_inches='tight')
print(f"\nSaved figure: {out_fname}\n")

