import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ User configuration ============
csv_file_1 = "energy_usage_sphere_low.csv"       # CSV with E_transfer_J
csv_file_2 = "energy_usage_madrl_low.csv"        # CSV WITHOUT E_transfer_J

out_fname = "low_device_energy_MADRL_SPHERE_split.png"
dpi = 300

# Global font settings
base_fontsize = 11
x_label_interval = 15

# Bar colors
color_compute = '#4C78A8'    # blue
color_comm = '#F58518'       # orange
color_transfer = '#8B0000'   # red
# =============================================

# ============ Load CSV 1 (SPHERE, with E_transfer) ============
df1 = pd.read_csv(csv_file_1)
rounds1 = df1['round'].values
E_comp1 = df1['E_comp_J'].values
E_comm1 = df1['E_comm_J'].values
E_transfer1 = df1['E_transfer_J'].values

# ============ Load CSV 2 (MADRL, no E_transfer_J) ============
df2 = pd.read_csv(csv_file_2)
rounds2 = df2['round'].values
E_comp2 = df2['E_comp_J'].values
E_comm2 = df2['E_comm_J'].values

# ============ Compute global max energy (shared y-scale) ============
max_sphere = np.max(E_comp1 + E_comm1 + E_transfer1)
max_madrl  = np.max(E_comp2 + E_comm2)
ymax = max(max_sphere, max_madrl) * 1.05  # 5% headroom

# ============ Plot: Two stacked subplots ============
plt.rcParams.update({'font.size': base_fontsize})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=False)

width = 0.7

# -----------------------------------------------------
# TOP PLOT: SPHERE (Compute + Comm + Transfer)
# -----------------------------------------------------
ax1.bar(rounds1, E_comp1, width, label='Compute (E_comp)', color=color_compute)
ax1.bar(rounds1, E_comm1, width, bottom=E_comp1,
        label='Comm (E_comm)', color=color_comm)
ax1.bar(rounds1, E_transfer1, width, bottom=E_comp1 + E_comm1,
        label='Transfer (E_transfer)', color=color_transfer)

ax1.set_ylabel('Energy (J)', fontsize=base_fontsize + 1)
ax1.set_xticks(rounds1[::x_label_interval])
ax1.set_xticklabels(rounds1[::x_label_interval], fontsize=base_fontsize - 1)
ax1.tick_params(axis='y', labelsize=base_fontsize - 1)

ax1.legend(loc='upper right', fontsize=base_fontsize - 1)
ax1.grid(axis='y', linestyle=':', alpha=0.4)
ax1.set_title("SPHERE Energy Usage", fontsize=base_fontsize + 2)
ax1.set_ylim(0, ymax)

# -----------------------------------------------------
# BOTTOM PLOT: MADRL (Compute + Comm only)
# -----------------------------------------------------
ax2.bar(rounds2, E_comp2, width, label='Compute (E_comp)', color=color_compute)
ax2.bar(rounds2, E_comm2, width, bottom=E_comp2,
        label='Comm (E_comm)', color=color_comm)

ax2.set_xlabel('Training Round', fontsize=base_fontsize + 1)
ax2.set_ylabel('Energy (J)', fontsize=base_fontsize + 1)
ax2.set_xticks(rounds2[::x_label_interval])
ax2.set_xticklabels(rounds2[::x_label_interval], fontsize=base_fontsize - 1)
ax2.tick_params(axis='y', labelsize=base_fontsize - 1)

ax2.legend(loc='upper right', fontsize=base_fontsize - 1)
ax2.grid(axis='y', linestyle=':', alpha=0.4)
ax2.set_title("MADRL Energy Usage", fontsize=base_fontsize + 2)
ax2.set_ylim(0, ymax)

plt.tight_layout()
plt.savefig(out_fname, dpi=dpi, bbox_inches='tight')
print(f"\nSaved figure: {out_fname}\n")

