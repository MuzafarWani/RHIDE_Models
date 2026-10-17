# src/utils/timers.py
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Timer:
    """
    Simple named timers with accumulated timings in seconds.
    """
    totals: Dict[str, float] = field(default_factory=dict)
    _starts: Dict[str, float] = field(default_factory=dict)

    def start(self, name: str) -> None:
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        t0 = self._starts.pop(name, None)
        if t0 is None:
            return 0.0
        dt = time.perf_counter() - t0
        self.totals[name] = self.totals.get(name, 0.0) + dt
        return dt

    @contextmanager
    def span(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)


@contextmanager
def timeit(name: str, sink=None):
    """
    Lightweight context timer:
        with timeit("foo", print): <work>
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        if sink:
            sink(f"[timeit] {name}: {dt:.6f}s")

