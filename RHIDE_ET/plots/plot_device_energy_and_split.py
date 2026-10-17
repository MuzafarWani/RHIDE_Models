import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============ User config ============
csv_file = "Device_rounds.csv"   # change if needed
out_fname = "fig_device_energy_split_100.png"
dpi = 300

# Font and tick configuration
base_fontsize = 11     # global base font size
x_label_interval = 5  # show label every N rounds (e.g., 20 → 1, 21, 41, …)

# Bar colors
color_compute = '#4C78A8'  # blue
color_comm = '#F58518'     # orange
color_transfer = '#8B0000' # red
# Alternatives: '#B22222' or '#A52A2A'
# =====================================

# Load data
df = pd.read_csv(csv_file)
rounds = df['round'].values
E_comp = df['E_comp_J'].values
E_comm = df['E_comm_J'].values
E_transfer = df['E_transfer_J'].values
beta = df['beta'].values

# Plot
plt.rcParams.update({'font.size': base_fontsize})
fig, ax1 = plt.subplots(figsize=(8, 4.2))

# stacked bars
width = 0.7
ax1.bar(rounds, E_comp, width, label='Compute (E_comp)', color=color_compute)
ax1.bar(rounds, E_comm, width, bottom=E_comp, label='Comm (E_comm)', color=color_comm)
ax1.bar(rounds, E_transfer, width, bottom=E_comp + E_comm, label='Transfer (E_transfer)', color=color_transfer)

ax1.set_xlabel('Training round', fontsize=base_fontsize + 1)
ax1.set_ylabel('Energy (Joules)', fontsize=base_fontsize + 1)
#ax1.set_title('Device energy breakdown per round and chosen split (β)', fontsize=base_fontsize + 2)

# ---- Show only every Nth x-label ----
ax1.set_xticks(rounds[::x_label_interval])
ax1.set_xticklabels(rounds[::x_label_interval], rotation=0, fontsize=base_fontsize - 1)
ax1.tick_params(axis='y', labelsize=base_fontsize - 1)

# secondary axis for split
ax2 = ax1.twinx()
ax2.plot(rounds, beta, '-o', color='k', linewidth=1.5, markersize=5, label='Split (β)')
ax2.set_ylabel('Split layer (β)', fontsize=base_fontsize + 1)
ax2.tick_params(axis='y', labelsize=base_fontsize - 1)
ax2.set_ylim(min(0, beta.min() - 1), max(beta.max() + 1, 1))

# legends
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=base_fontsize - 1)

# optional: light grid for readability
ax1.grid(axis='y', linestyle=':', alpha=0.4)

plt.tight_layout()
plt.savefig(out_fname, dpi=dpi, bbox_inches='tight')
print(f"Saved: {out_fname}")

