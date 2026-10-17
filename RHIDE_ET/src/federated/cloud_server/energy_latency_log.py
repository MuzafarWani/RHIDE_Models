# src/federated/accounting/energy_latency_log.py
from __future__ import annotations

from typing import Dict
from collections import defaultdict

from src.federated.messaging.bus import MessagingBus


class EnergyLatencyLogger:
    """
    Accumulates realized energy/latency per device and publishes dicts:
      "energy_used"  -> {device_id: joules}
      "latency_used" -> {device_id: seconds}

    Other modules (device client, edge) can publish partials; bus.collect merges dicts.
    """

    def __init__(self, bus: MessagingBus):
        self.bus = bus
        self._energy: Dict[str, float] = defaultdict(float)
        self._latency: Dict[str, float] = defaultdict(float)

    def record_energy(self, device_id: str, joules: float) -> None:
        self._energy[str(device_id)] += float(joules)
        self.bus.publish("energy_used", dict(self._energy))

    def record_latency(self, device_id: str, seconds: float) -> None:
        self._latency[str(device_id)] += float(seconds)
        self.bus.publish("latency_used", dict(self._latency))

    def reset(self) -> None:
        self._energy.clear()
        self._latency.clear()

