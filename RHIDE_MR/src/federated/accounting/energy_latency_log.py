#src/federated/accounting/energy_latency_log.py
from __future__ import annotations

import csv
import threading
from typing import Dict, Optional, Tuple


class EnergyLatencyLog:
    """
    Thread-safe realized energy/latency accumulator.

    Stores cumulative totals by entity_id (e.g., device_id).
    Optionally publishes snapshots on a bus with topics:
        "energy_used"  -> {entity_id: joules}
        "latency_used" -> {entity_id: seconds}

    >>> NEW (non-breaking) additions:
      - Can also log:
         * model_delta events (edge/cloud)
         * aggregation events (edge/cloud)
         * sync/broadcast events (cloud→edge, edge→device)
    """

    def __init__(self, bus: Optional[object] = None):
        self._lock = threading.Lock()
        self._energy: Dict[str, float] = {}
        self._latency: Dict[str, float] = {}
        self._bus = bus

        # >>> NEW (non-breaking) internal logs for your new design <<<
        self._model_delta_log: Dict[str, float] = {}
        self._agg_event_log: Dict[str, int] = {}
        self._sync_event_log: Dict[str, int] = {}

    # -------- totals --------
    def add_energy(self, entity_id: str, joules: float) -> None:
        with self._lock:
            self._energy[entity_id] = self._energy.get(entity_id, 0.0) + float(joules)
            energy_payload = dict(self._energy)
        if self._bus is not None:
            try:
                self._bus.publish("energy_used", energy_payload)
            except Exception:
                pass

    def add_latency(self, entity_id: str, seconds: float) -> None:
        with self._lock:
            self._latency[entity_id] = self._latency.get(entity_id, 0.0) + float(seconds)
            latency_payload = dict(self._latency)
        if self._bus is not None:
            try:
                self._bus.publish("latency_used", latency_payload)
            except Exception:
                pass

    # -------- convenience: add both at once --------
    def add_totals(self, entity_id: str, joules: float, seconds: float) -> None:
        self.add_energy(entity_id, joules)
        self.add_latency(entity_id, seconds)

    # -------- snapshots / lifecycle --------
    def snapshot(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        with self._lock:
            return dict(self._energy), dict(self._latency)

    def reset(self) -> None:
        with self._lock:
            self._energy.clear()
            self._latency.clear()
            # >>> also reset new logs <<<
            self._model_delta_log.clear()
            self._agg_event_log.clear()
            self._sync_event_log.clear()

    def to_csv(self, path_energy: str, path_latency: str) -> None:
        e_snap, l_snap = self.snapshot()
        with open(path_energy, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "joules"])
            for k, v in e_snap.items():
                w.writerow([k, float(v)])
        with open(path_latency, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "seconds"])
            for k, v in l_snap.items():
                w.writerow([k, float(v)])

    # ------------------------------------------------------------------
    # >>> NEW (ADDED, NOTHING REMOVED) – SUPPORT FOR YOUR NEW DESIGN <<<
    # ------------------------------------------------------------------

    # ---- Log model-delta events (useful for RL + analysis) ----
    def log_model_delta(self, entity_id: str, delta: float) -> None:
        """
        Store the latest model-delta observed at this entity (edge or cloud).
        """
        with self._lock:
            self._model_delta_log[entity_id] = float(delta)
        if self._bus is not None:
            try:
                self._bus.publish("model_delta_log", {entity_id: float(delta)})
            except Exception:
                pass

    # ---- Log aggregation events (edge or cloud) ----
    def log_aggregation(self, entity_id: str) -> None:
        """
        Count how many times aggregation occurred at this entity.
        """
        with self._lock:
            self._agg_event_log[entity_id] = self._agg_event_log.get(entity_id, 0) + 1
        if self._bus is not None:
            try:
                self._bus.publish("aggregation_events", dict(self._agg_event_log))
            except Exception:
                pass

    # ---- Log synchronization (cloud→edge or edge→device) ----
    def log_sync(self, entity_id: str) -> None:
        """
        Count how many times a model sync/broadcast happened.
        """
        with self._lock:
            self._sync_event_log[entity_id] = self._sync_event_log.get(entity_id, 0) + 1
        if self._bus is not None:
            try:
                self._bus.publish("sync_events", dict(self._sync_event_log))
            except Exception:
                pass

    # ---- Optional accessors for analysis ----
    def get_model_deltas(self) -> Dict[str, float]:
        with self._lock:
            return dict(self._model_delta_log)

    def get_aggregation_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._agg_event_log)

    def get_sync_counts(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._sync_event_log)

