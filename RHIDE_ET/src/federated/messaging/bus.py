from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Callable, Type


class MessagingBus:
    """
    Very small in-process pub/sub message bus with topic buffers.
    - publish(topic, payload): append to the topic's buffer
    - drain(topic): pop and return all payloads (list)
    - collect(topic, default): if payloads are dicts, merges them into one dict; else returns last item
    Thread-safe (basic) with a per-bus lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._topics: Dict[str, List[Any]] = {}

    def publish(self, topic: str, payload: Any) -> None:
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = []
            self._topics[topic].append(payload)

    # >>> NEW (needed for edge aggregation convenience) <<<
    def publish_many(self, topic: str, payloads: List[Any]) -> None:
        """Publish a list of messages to the same topic."""
        with self._lock:
            if topic not in self._topics:
                self._topics[topic] = []
            self._topics[topic].extend(payloads)

    def drain(self, topic: str) -> List[Any]:
        with self._lock:
            items = self._topics.get(topic, [])
            self._topics[topic] = []
        return items

    def peek(self, topic: str) -> List[Any]:
        with self._lock:
            items = self._topics.get(topic, [])
            return list(items)

    # >>> NEW: alias for compatibility with EdgeAggregator <<<
    def snapshot(self, topic: str) -> List[Any]:
        """Alias of peek() for backward compatibility."""
        return self.peek(topic)

    # >>> NEW: drain only messages of a given type (VERY IMPORTANT) <<<
    def drain_filtered(self, topic: str, msg_type: Type) -> List[Any]:
        """
        Drain only messages of a specific type from a topic.
        Everything else remains in the buffer.
        """
        with self._lock:
            items = self._topics.get(topic, [])
            keep = []
            matched = []
            for m in items:
                if isinstance(m, msg_type):
                    matched.append(m)
                else:
                    keep.append(m)
            self._topics[topic] = keep
        return matched

    def collect(self, topic: str, default: Any = None) -> Any:
        """
        If the topic contains a list of dict payloads, merge them (later entries overwrite).
        If not dicts, return the last payload or `default` when empty.
        """
        items = self.drain(topic)
        if not items:
            return default

        # Merge dicts
        if all(isinstance(x, dict) for x in items):
            merged: Dict[Any, Any] = {}
            for d in items:
                merged.update(d)
            return merged

        # Otherwise return the last item
        return items[-1]

    # Convenience: register a reducer function per topic (optional future extension)
    def reduce(self, topic: str, reducer: Callable[[List[Any]], Any], default: Any = None) -> Any:
        items = self.drain(topic)
        if not items:
            return default
        return reducer(items)

