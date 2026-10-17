# src/accounting/energy_latency_log.py
from __future__ import annotations

from typing import Dict, Optional, Tuple
from collections import defaultdict
import time

from src.federated.messaging.bus import MessagingBus

class EnergyLatencyLog:
    """
    Lightweight per-device energy/latency accumulator.

    Methods used elsewhere:
      - add_energy(device_id, joules)
      - add_latency(device_id, seconds)
      - snapshot() -> (energy_dict, latency_dict)
    Optional bus telemetry publishes to 'telemetry_energy_latency'.
    """

    def __init__(self, bus: Optional[MessagingBus] = None, publish_topic: str = "telemetry_energy_latency"):
        self.bus = bus
        self.publish_topic = publish_topic
        self._energy_J: Dict[str, float] = defaultdict(float)
        self._latency_s: Dict[str, float] = defaultdict(float)
        self._last_ts = time.time()

    # -------- public API ----------
    def add_energy(self, device_id: str, joules: float) -> None:
        if joules is None:
            return
        j = float(joules)
        if j <= 0.0:
            return
        self._energy_J[device_id] += j
        self._maybe_publish(device_id=device_id, kind="energy", value=j)

    def add_latency(self, device_id: str, seconds: float) -> None:
        if seconds is None:
            return
        s = float(seconds)
        if s <= 0.0:
            return
        self._latency_s[device_id] += s
        self._maybe_publish(device_id=device_id, kind="latency", value=s)

    def snapshot(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        # return shallow copies to avoid external mutation
        return dict(self._energy_J), dict(self._latency_s)

    def reset(self) -> None:
        self._energy_J.clear()
        self._latency_s.clear()

    # -------- internal ----------
    def _maybe_publish(self, *, device_id: str, kind: str, value: float) -> None:
        if self.bus is None:
            return
        try:
            self.bus.publish(self.publish_topic, {
                "ts": time.time(),
                "device_id": device_id,
                "kind": kind,          # "energy" or "latency"
                "value": float(value),
                "totals": {
                    "energy_J": dict(self._energy_J),
                    "latency_s": dict(self._latency_s),
                },
            })
        except Exception:
            # best-effort; never break training due to telemetry
            pass

