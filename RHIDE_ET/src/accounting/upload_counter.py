# src/accounting/upload_counter.py
from __future__ import annotations

from typing import Dict, Optional
from collections import defaultdict

from src.federated.messaging.bus import MessagingBus

class UploadCounter:
    """
    Lightweight per-device byte counter.
    - record_device_upload(device_id, bytes_sent)
    - snapshot() -> dict copy
    Optionally publishes a telemetry dict to the bus topic 'telemetry_uploads'.
    """
    def __init__(self, bus: Optional[MessagingBus] = None, publish_topic: str = "telemetry_uploads"):
        self.bus = bus
        self.publish_topic = publish_topic
        self._bytes_by_device: Dict[str, int] = defaultdict(int)
        self._total_bytes: int = 0

    def record_device_upload(self, device_id: str, bytes_sent: int | float) -> None:
        b = int(bytes_sent or 0)
        if b <= 0:
            return
        self._bytes_by_device[device_id] += b
        self._total_bytes += b
        if self.bus is not None:
            # publish a minimal telemetry event (non-blocking, best-effort)
            try:
                self.bus.publish(self.publish_topic, {
                    "device_id": device_id,
                    "bytes": b,
                    "total_bytes": self._total_bytes,
                    "by_device": dict(self._bytes_by_device),
                })
            except Exception:
                pass

    def total_bytes(self) -> int:
        return self._total_bytes

    def bytes_for(self, device_id: str) -> int:
        return int(self._bytes_by_device.get(device_id, 0))

    def snapshot(self) -> Dict[str, int]:
        return dict(self._bytes_by_device)

    def reset(self) -> None:
        self._bytes_by_device.clear()
        self._total_bytes = 0

