# src/federated/edge_server/edge_state.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class EdgeState:
    edge_id: str
    # RL / scheduling
    agg_period_Ol: int = 1              # aggregate every O_l rounds
    last_action_idx: int = 0

    # Queues processed this round (throughput proxies)
    q_activations: int = 0              # count of activation uploads processed this round
    q_raw: int = 0                      # count of raw-batch transfers processed this round

    # Snapshot of pending queues (measured from bus peek)
    q_pending_activations: int = 0      # messages waiting before/after drain
    q_pending_raw: int = 0

    # Realized stats (this round)
    realized_energy_sum: float = 0.0
    realized_latency_sum: float = 0.0
    step_count: int = 0                 # number of device contributions counted

    # Book-keeping of rounds since last aggregation
    rounds_since_agg: int = 0

    # ---------------- NEW (FOR MODEL-DELTA AWARE RL) ----------------
    # Track how much the model changed since last edge aggregation
    last_model_delta: float = 0.0

    # Optional flag if delta crossed threshold (can be useful in RL reward)
    delta_exceeded: float = 0.0
    # ---------------------------------------------------------------

    # Optional extra info
    info: Dict[str, Any] = field(default_factory=dict)

    # ----- mutation helpers -----
    def add_device_metrics(self, energy_j: float, latency_s: float) -> None:
        self.realized_energy_sum += float(energy_j)
        self.realized_latency_sum += float(latency_s)
        self.step_count += 1

    def incr_activation_uploads(self, n: int = 1) -> None:
        self.q_activations += int(n)

    def incr_raw_transfers(self, n: int = 1) -> None:
        self.q_raw += int(n)

    def set_last_action_idx(self, idx: int) -> None:
        self.last_action_idx = int(idx)

    def set_queue_pending(self, activations: int, raw: int) -> None:
        """Update snapshot of pending queue lengths (from bus peek)."""
        self.q_pending_activations = int(activations)
        self.q_pending_raw = int(raw)

    # --------------- NEW: setters for model-delta ---------------
    def set_model_delta(self, delta: float, threshold: float | None = None) -> None:
        """Record latest model delta and whether it crossed threshold."""
        self.last_model_delta = float(delta)
        if threshold is not None:
            self.delta_exceeded = 1.0 if delta >= threshold else 0.0
    # -------------------------------------------------------------

    # ----- round lifecycle -----
    def reset_round_counters(self) -> None:
        self.realized_energy_sum = 0.0
        self.realized_latency_sum = 0.0
        self.q_activations = 0
        self.q_raw = 0
        self.step_count = 0
        # keep q_pending_* as the most recent snapshot (not auto-cleared)

    # ----- observations for RL -----
    def obs_vector(self) -> Dict[str, float]:
        """
        Reward should reflect *current pressure* on the edge. We expose:
          - mean stats for the work processed so far this round
          - queue_depth from the latest peek (pending activations + raw)
          - NEW: model_delta since last aggregation
          - NEW: whether delta exceeded threshold (binary signal)
        """
        n = max(1, self.step_count)
        pending_depth = float(self.q_pending_activations + self.q_pending_raw)

        return {
            "mean_energy": self.realized_energy_sum / n,
            "mean_latency": self.realized_latency_sum / n,
            "queue_depth": pending_depth,
            "last_action_idx": float(self.last_action_idx),
            # ---- NEW FEATURES FOR DELTA-AWARE RL ----
            "model_delta": float(self.last_model_delta),
            "delta_exceeded": float(self.delta_exceeded),
        }

