import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load data
# -----------------------------
csv_file = "all_scenarios_data.csv"   # update path if needed
df = pd.read_csv(csv_file)

scenarios = df["Scenario"]

# -----------------------------
# Create 4x2 subplot figure
# -----------------------------
fig, axes = plt.subplots(4, 2, figsize=(12, 14))
axes = axes.flatten()

# -----------------------------
# Subplot 1: Energy per round
# -----------------------------
axes[0].bar(scenarios, df["Avegrage_Energy_Round(J)"])
axes[0].set_ylabel("Energy / Round (J)")
axes[0].set_title("Per-Round Energy")

# -----------------------------
# Subplot 2: Device lifetime
# -----------------------------
axes[1].bar(scenarios, df["Average_Rounds_devices_active"])
axes[1].set_ylabel("Active Rounds")
axes[1].set_title("Device Lifetime")

# -----------------------------
# Subplot 3: Split point
# -----------------------------
axes[2].bar(scenarios, df["Average_Split_Point"])
axes[2].set_ylabel("Split Point")
axes[2].set_title("Average Split Point")

# -----------------------------
# Subplot 4: Aggregation frequency
# -----------------------------
axes[3].bar(scenarios, df["Average_aggregration_frequency"])
axes[3].set_ylabel("Rounds / Cloud Sync")
axes[3].set_title("Aggregation Frequency")

# -----------------------------
# Subplot 5: Training time
# -----------------------------
axes[4].bar(scenarios, df["Average_training_time(s)"])
axes[4].set_ylabel("Time (s)")
axes[4].set_title("Training Time")

# -----------------------------
# Subplot 6: Inference time
# -----------------------------
axes[5].bar(scenarios, df["Averege_Inference_time (s)"])
axes[5].set_ylabel("Time (s)")
axes[5].set_title("Inference Time")

# -----------------------------
# Subplot 7: Data transfer rounds
# -----------------------------
axes[6].bar(scenarios, df["Average_data_transfer_rounds"])
axes[6].set_ylabel("Rounds")
axes[6].set_title("Data Transfer Rounds")

# -----------------------------
# Subplot 8: Participation
# -----------------------------
axes[7].bar(scenarios, df["Average_participaion_per_round"])
axes[7].set_ylabel("Devices / Round")
axes[7].set_title("Participation per Round")

# -----------------------------
# Formatting: x-axis labels only on bottom row
# -----------------------------
for i in range(6):
    axes[i].set_xticklabels([])

axes[6].set_xlabel("Scenario")
axes[7].set_xlabel("Scenario")

# -----------------------------
# Final layout and save
# -----------------------------
plt.suptitle("SPHERE System-Level Performance Across Scenarios", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("sphere_summary_8metrics.png", dpi=300)
plt.close()

print("Saved figure: sphere_summary_8metrics.png")

