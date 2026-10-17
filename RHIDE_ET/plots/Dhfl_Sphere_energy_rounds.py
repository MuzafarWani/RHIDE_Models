import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ User configuration ============
csv_file_1 = "SPHERE_DHFL_energy_roundshigh.csv"     # CSV with E_transfer_J
csv_file_2 = "DHFL_high_energy_rounds.csv"            # CSV WITHOUT E_transfer_J

out_fname = "high_device_energy_DHFL_SPHERE_split.png"
dpi = 300

# Global font settings
base_fontsize = 11

# Dynamic interval based on dataset size
def get_label_interval(n):
    if n > 350: 
        return 25
    if n > 200:
        return 15
    if n > 80:
        return 10
    return 5

# Bar colors
color_compute = '#4C78A8'
color_comm = '#F58518'
color_transfer = '#8B0000'
# =============================================

# Load CSV 1 (SPHERE)
df1 = pd.read_csv(csv_file_1)
rounds1 = df1['round'].values
E_comp1 = df1['E_comp_J'].values
E_comm1 = df1['E_comm_J'].values
E_transfer1 = df1['E_transfer_J'].values

# Load CSV 2 (DHFL)
df2 = pd.read_csv(csv_file_2)
rounds2 = df2['round'].values
E_comp2 = df2['E_comp_J'].values
E_comm2 = df2['E_comm_J'].values

# Determine appropriate label intervals
x_label_interval_1 = get_label_interval(len(rounds1))
x_label_interval_2 = get_label_interval(len(rounds2))

plt.rcParams.update({'font.size': base_fontsize})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# SAME WIDTH FOR BOTH PLOTS
width = 0.6

# -----------------------------------------------------
# TOP PLOT: SPHERE (Compute + Comm + Transfer)
# -----------------------------------------------------
ax1.bar(rounds1, E_comp1, width, label='Compute', color=color_compute, zorder=3)
ax1.bar(rounds1, E_comm1, width, bottom=E_comp1,
        label='Comm', color=color_comm, zorder=3)
ax1.bar(rounds1, E_transfer1, width, bottom=E_comp1 + E_comm1,
        label='Transfer', color=color_transfer, zorder=3)

ax1.set_ylabel('Energy (J)', fontsize=base_fontsize + 1)
ax1.set_xticks(rounds1[::x_label_interval_1])
ax1.set_xticklabels(rounds1[::x_label_interval_1], fontsize=base_fontsize - 1, rotation=45)
ax1.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
ax1.legend(loc='upper right', fontsize=base_fontsize - 1)
ax1.set_title("SPHERE Energy Usage", fontsize=base_fontsize + 3)

# -----------------------------------------------------
# BOTTOM PLOT: DHFL (Compute + Comm)
# -----------------------------------------------------
ax2.bar(rounds2, E_comp2, width, label='Compute', color=color_compute, zorder=3)
ax2.bar(rounds2, E_comm2, width, bottom=E_comp2,
        label='Comm', color=color_comm, zorder=3)

ax2.set_xlabel('Training Round', fontsize=base_fontsize + 1)
ax2.set_ylabel('Energy (J)', fontsize=base_fontsize + 1)
ax2.set_xticks(rounds2[::x_label_interval_2])
ax2.set_xticklabels(rounds2[::x_label_interval_2], fontsize=base_fontsize - 1, rotation=45)
ax2.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
ax2.legend(loc='upper right', fontsize=base_fontsize - 1)
ax2.set_title("DHFL Energy Usage", fontsize=base_fontsize + 3)

plt.tight_layout()
plt.savefig(out_fname, dpi=dpi, bbox_inches='tight')
print(f"\nSaved figure: {out_fname}\n")

