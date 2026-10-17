# src/federated/cloud_server/global_aggregator.py
from __future__ import annotations

from typing import Any, List, Dict
import numpy as np

from src.federated.messaging.bus import MessagingBus
from src.federated.messaging.messages import ModelUpdate

# Cloud RL agent + delta utility
from src.federated.cloud_server.rl_agent import CloudRLAggregationAgent, CloudRLConfig
from src.federated.utils.model_delta import model_delta


class CloudAggregator:
    """
    Aggregates edge updates (FedAvg by default) and holds the current global weights.

    UPDATED BEHAVIOR (YOUR DESIGN):

    1) Cloud RL agent selects aggregation cadence O_e.
    2) Aggregation is triggered if EITHER:
       - rounds_since_agg >= O_e   (cadence), OR
       - model change (delta) >= tau_cloud (significant drift)
    3) When aggregation happens:
       - Cloud computes FedAvg → ONE global model
       - Cloud BROADCASTS this model to ALL edges
    4) Model delta is a first-class input to the RL agent.
    """

    def __init__(self, cfg: Any):
        self.cfg = cfg
        self.bus = MessagingBus()  # training loop should rebind via bind_bus()

        # Global model state
        self._global_weights: List[np.ndarray] | None = None
        self._updated: bool = False

        # >>> NEW: Cloud RL controller for O_e
        rl_keys = cfg.rl_cfg.get(
            "cloud_obs_keys",
            ("num_edge_updates", "model_delta", "rounds_since_agg"),
        )

        rl_actions = cfg.rl_cfg.get("cloud_actions", (1, 2, 4, 8))

        self.rl = CloudRLAggregationAgent(
            CloudRLConfig(
                obs_keys=tuple(rl_keys),
                actions=tuple(rl_actions),
            )
        )

        # >>> NEW: cloud aggregation state
        self.rounds_since_agg: int = 0
        self._last_global_weights: List[np.ndarray] | None = None

        # >>> NEW: delta threshold for "significant model change"
        self.tau_cloud: float = float(cfg.rl_cfg.get("cloud_tau", 0.05))

    def bind_bus(self, bus: MessagingBus) -> None:
        self.bus = bus

    # ----------------------------------------------------------
    # Main aggregation decision logic (UPDATED)
    # ----------------------------------------------------------
    def maybe_aggregate_and_update(self, round_index: int) -> bool:

        # Drain edge updates
        updates = [
            m for m in self.bus.drain("edge_to_cloud_update")
            if isinstance(m, ModelUpdate)
        ]

        # ---- LOG: no updates received ----
        if not updates:
            print(
                f"[cloud][round {round_index}] "
                f"No edge updates received (rounds_since_agg={self.rounds_since_agg})"
            )
            self._updated = False
            return False

        # ------------------------------------------------------
        # (1) Compute candidate FedAvg weights FIRST
        # ------------------------------------------------------
        ref = updates[0].weights
        k = len(ref)
        stacks = [np.stack([u.weights[i] for u in updates], axis=0) for i in range(k)]
        candidate_weights = [np.mean(s, axis=0) for s in stacks]

        # ------------------------------------------------------
        # (2) Compute model delta vs last global model
        # ------------------------------------------------------
        delta = model_delta(candidate_weights, self._last_global_weights)

        # ------------------------------------------------------
        # (3) Build RL observation INCLUDING model_delta
        # ------------------------------------------------------
        obs: Dict[str, float] = {
            "num_edge_updates": float(len(updates)),
            "model_delta": float(delta),
            "rounds_since_agg": float(self.rounds_since_agg),
        }

        # ------------------------------------------------------
        # (4) Cloud RL chooses aggregation period O_e
        # ------------------------------------------------------
        O_e = int(self.rl.select_action(obs, deterministic=False))
        print(
            f"[cloud][round {round_index}] RL selected aggregation period O_e = {O_e}"
        )

        print(
            f"[cloud][round {round_index}] "
            f"model_delta = {delta:.6f}, tau_cloud = {self.tau_cloud}"
        )

        # ------------------------------------------------------
        # (5) Trigger condition: cadence OR significant drift
        # ------------------------------------------------------
        trigger_by_cadence = self.rounds_since_agg >= O_e
        trigger_by_delta = delta >= self.tau_cloud

        if not (trigger_by_cadence or trigger_by_delta):
            print(
                f"[cloud][round {round_index}] Aggregation SKIPPED "
                f"(rounds_since_agg={self.rounds_since_agg}, O_e={O_e}, "
                f"delta={delta:.6f} < tau_cloud)"
            )
            self.rounds_since_agg += 1
            self._updated = False
            return False

        # ------------------------------------------------------
        # (6) Commit aggregation (FedAvg)
        # ------------------------------------------------------
        reason = []
        if trigger_by_cadence:
            reason.append("cadence")
        if trigger_by_delta:
            reason.append("model-delta")

        print(
            f"[cloud][round {round_index}] "
            f"Aggregating {len(updates)} edge updates using FedAvg "
            f"(trigger: {', '.join(reason)})"
        )

        # Update global model
        self._global_weights = candidate_weights
        self._last_global_weights = candidate_weights
        self._updated = True

        # reset counter
        self.rounds_since_agg = 0

        # ------------------------------------------------------
        # >>> NEW: BROADCAST GLOBAL MODEL TO ALL EDGES
        # ------------------------------------------------------
        msg = ModelUpdate(
            sender_id="cloud",
            receiver_id="edges",
            weights=self._global_weights,
        )

        try:
            setattr(msg, "created_ts", float(round_index))
        except Exception:
            pass

        print(f"[cloud][round {round_index}] Broadcasting new global model to all edges")
        self.bus.publish("cloud_to_edges_broadcast", msg)

        return True

    def get_weights(self) -> List[np.ndarray] | None:
        return self._global_weights

    def was_updated(self) -> bool:
        return bool(self._updated)

