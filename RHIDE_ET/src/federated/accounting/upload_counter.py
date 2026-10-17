# src/accounting/upload_counter.py
from __future__ import annotations

import csv
import threading
from typing import Dict, Optional


class UploadCounter:
    """
    Generic byte-upload counter.
    - Thread-safe
    - Works standalone or (optionally) with a MessagingBus-like object
    - If bus is provided, publishes to topic "bytes_uploaded" a dict {entity_id: total_bytes}
    """

    def __init__(self, bus: Optional[object] = None):
        self._lock = threading.Lock()
        self._bytes: Dict[str, int] = {}
        self._bus = bus

    def record(self, entity_id: str, bytes_count: int) -> None:
        with self._lock:
            self._bytes[entity_id] = self._bytes.get(entity_id, 0) + int(bytes_count)
            payload = dict(self._bytes)
        if self._bus is not None:
            try:
                self._bus.publish("bytes_uploaded", payload)
            except Exception:
                # Publishing is best-effort; keep local state regardless.
                pass

    def snapshot(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._bytes)

    def reset(self) -> None:
        with self._lock:
            self._bytes.clear()

    def to_csv(self, path: str) -> None:
        snap = self.snapshot()
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "bytes_uploaded"])
            for k, v in snap.items():
                w.writerow([k, int(v)])

