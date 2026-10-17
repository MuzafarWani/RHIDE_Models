import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ User configuration ============
csv_file_1 = "SPHERE_DHFL_energy_roundshigh.csv"
csv_file_2 = "DHFL_high_energy_rounds.csv"

out_fname = "high_device_energy_DHFL_SPHERE_line_logaxis.png"
dpi = 300
base_fontsize = 11

# Colors
color_compute  = '#4C78A8'
color_comm     = '#F58518'
color_transfer = '#8B0000'
# =============================================

# -------- Load CSVs --------
df1 = pd.read_csv(csv_file_1)
df2 = pd.read_csv(csv_file_2)

rounds1 = df1['round'].values
rounds2 = df2['round'].values

E_comp1 = df1['E_comp_J'].values
E_comm1 = df1['E_comm_J'].values
E_transfer1 = df1['E_transfer_J'].values

E_comp2 = df2['E_comp_J'].values
E_comm2 = df2['E_comm_J'].values

# Index-based x
# Use actual training rounds on x-axis
x1 = rounds1
x2 = rounds2

# -------- Compute shared y-axis limits (log scale safe) --------
all_energy_values = np.concatenate([
    E_comp1, E_comm1, E_transfer1,
    E_comp2, E_comm2
])

ymin = np.min(all_energy_values[all_energy_values > 0]) * 0.9
ymax = np.max(all_energy_values) * 1.1

plt.rcParams.update({'font.size': base_fontsize})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=False)

# ------------------ SPHERE ------------------
ax1.plot(x1, E_comp1, label='E_comp_J', color=color_compute, linewidth=1.8)
ax1.plot(x1, E_comm1, label='E_comm_J', color=color_comm, linewidth=1.6, linestyle='--')
ax1.plot(x1, E_transfer1, label='E_transfer_J', color=color_transfer, linewidth=1.6)

ax1.set_yscale('log')
ax1.set_ylabel('Energy (J)', fontsize=base_fontsize + 1)
ax1.set_title('SPHERE Energy Usage', fontsize=base_fontsize + 3)
ax1.legend(loc='upper right', fontsize=base_fontsize - 1)
ax1.grid(axis='y', linestyle=':', alpha=0.4)

ax1.set_xticks(x1[::max(1, len(x1)//20)])
ax1.set_xticklabels(rounds1[::max(1, len(x1)//20)], rotation=45)

ax1.set_ylim(ymin, ymax)

# ------------------ DHFL ------------------
ax2.plot(x2, E_comp2, label='E_comp_J', color=color_compute, linewidth=1.8)
ax2.plot(x2, E_comm2, label='E_comm_J', color=color_comm, linewidth=1.6, linestyle='--')

ax2.set_yscale('log')
ax2.set_xlabel('Training Round', fontsize=base_fontsize + 1)
ax2.set_ylabel('Energy (J)', fontsize=base_fontsize + 1)
ax2.set_title('DHFL Energy Usage', fontsize=base_fontsize + 3)
ax2.legend(loc='upper right', fontsize=base_fontsize - 1)
ax2.grid(axis='y', linestyle=':', alpha=0.4)

# X-ticks every 20 rounds for DHFL
xticks_bottom = np.arange(rounds2.min(), rounds2.max() + 1, 20)
ax2.set_xticks(xticks_bottom)
ax2.set_xticklabels(xticks_bottom, rotation=45)

ax2.set_ylim(ymin, ymax)

# Force both subplots to use the SAME x-axis range
xmin = min(rounds1.min(), rounds2.min())
xmax = max(rounds1.max(), rounds2.max())

ax1.set_xlim(xmin, xmax)
ax2.set_xlim(xmin, xmax)

plt.tight_layout()
plt.savefig(out_fname, dpi=dpi, bbox_inches='tight')
print(f"\nSaved: {out_fname}\n")

