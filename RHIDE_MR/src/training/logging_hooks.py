# src/training/logging_hooks.py
from __future__ import annotations

import csv
import os
import json
from typing import Any, Dict


class TrainingLogger:
    """
    Logs:
      1) Per-round aggregate metrics (CSV)
      2) Per-device decisions per round (CSV)
      3) Full per-round JSON snapshot (debug / reproducibility)
    """

    # --------------------------------------------------
    # Init
    # --------------------------------------------------
    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)

        # ---------- CSV paths ----------
        self.round_csv = os_toggle_path(run_dir, "training_rounds.csv")
        self.device_csv = os_toggle_path(run_dir, "training_devices.csv")

        # ---------- Init round-level CSV ----------
        if not os.path.exists(self.round_csv):
            with open(self.round_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "round",
                        "mean_energy",
                        "mean_latency",
                        "mean_upload",
                        "round_time_s",
                        "num_devices",
                        "num_participating",
                        "num_raw_transfer",
                    ],
                )
                writer.writeheader()

        # ---------- Init device-level CSV ----------
        if not os.path.exists(self.device_csv):
            with open(self.device_csv, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "round",
                        "device_id",
                        "edge_id",
                        "beta",
                        "participate",
                        "k_transfer",
                        "num_batches",
                    ],
                )
                writer.writeheader()

    # --------------------------------------------------
    # Log one training round
    # --------------------------------------------------
    def log_round(
        self,
        round_idx: int,
        decisions: Dict[str, Dict[str, Any]],
        metrics: Dict[str, Any],
        extras: Dict[str, Any] | None = None,
    ) -> None:
        """
        decisions: per-device execution decisions (from train_loop)
        metrics: aggregate round metrics
        extras: optional (e.g., RL rewards)
        """

        # ---------- Aggregate counts ----------
        num_devices = len(decisions)
        num_participating = sum(1 for d in decisions.values() if d.get("participate"))
        num_raw_transfer = sum(1 for d in decisions.values() if not d.get("participate"))

        # ---------- Round-level CSV ----------
        with open(self.round_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "mean_energy",
                    "mean_latency",
                    "mean_upload",
                    "round_time_s",
                    "num_devices",
                    "num_participating",
                    "num_raw_transfer",
                ],
            )
            writer.writerow({
                "round": round_idx,
                "mean_energy": metrics.get("mean_energy", 0.0),
                "mean_latency": metrics.get("mean_latency", 0.0),
                "mean_upload": metrics.get("mean_upload", 0.0),
                "round_time_s": metrics.get("round_time_s", 0.0),
                "num_devices": num_devices,
                "num_participating": num_participating,
                "num_raw_transfer": num_raw_transfer,
            })

        # ---------- Device-level CSV ----------
        with open(self.device_csv, "a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "round",
                    "device_id",
                    "edge_id",
                    "beta",
                    "participate",
                    "k_transfer",
                    "num_batches",
                ],
            )
            for device_id, dec in decisions.items():
                writer.writerow({
                    "round": round_idx,
                    "device_id": device_id,
                    "edge_id": dec.get("edge_id", ""),
                    "beta": dec.get("beta", ""),
                    "participate": int(bool(dec.get("participate", False))),
                    "k_transfer": float(dec.get("k_transfer", 0.0)),
                    "num_batches": len(dec.get("batch_ids", [])),
                })

        # ---------- Per-round JSON snapshot ----------
        json_path = os.path.join(self.run_dir, f"round_{round_idx:05d}.json")
        payload = {
            "round": round_idx,
            "metrics": metrics,
            "decisions": decisions,
        }
        if extras:
            payload["extras"] = extras

        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)


# --------------------------------------------------
# Small helper (avoids accidental double slashes)
# --------------------------------------------------
def os_toggle_path(base: str, name: str) -> str:
    return os.path.join(base, name)

