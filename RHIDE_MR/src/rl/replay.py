# src/rl/replay.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import collections


class ReplayBuffer:
    """
    Ring buffer for off-policy algorithms.
    Stores (obs, act, rew, next_obs, done).
    """

    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self._o = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._a = np.zeros((capacity,), dtype=np.int32)
        self._r = np.zeros((capacity,), dtype=np.float32)
        self._no = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._d = np.zeros((capacity,), dtype=np.float32)

        self._head = 0
        self.size = 0

    def add(self, obs, act, rew, next_obs, done) -> None:
        i = self._head
        self._o[i] = obs
        self._a[i] = int(act)
        self._r[i] = float(rew)
        self._no[i] = next_obs
        self._d[i] = float(done)
        self._head = (self._head + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        assert self.size > 0, "Empty replay buffer"
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self._o[idx],
            self._a[idx],
            self._r[idx],
            self._no[idx],
            self._d[idx],
        )

