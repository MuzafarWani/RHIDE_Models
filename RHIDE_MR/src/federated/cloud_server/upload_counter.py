# src/federated/accounting/upload_counter.py
from __future__ import annotations

from typing import Dict
from collections import defaultdict

from src.federated.messaging.bus import MessagingBus


class UploadCounter:
    """
    Simple per-round byte counter published to the bus as a single dict:
      topic: "bytes_uploaded" -> {device_id: total_bytes}
    """

    def __init__(self, bus: MessagingBus):
        self.bus = bus
        self._bytes: Dict[str, int] = defaultdict(int)

    def record_device_upload(self, device_id: str, bytes_count: int) -> None:
        self._bytes[str(device_id)] += int(bytes_count)
        # publish cumulative (merged by bus.collect)
        self.bus.publish("bytes_uploaded", dict(self._bytes))

    def reset(self) -> None:
        self._bytes.clear()

