from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


@dataclass
class EnvSpec:
    obs_dim: int
    n_actions: int
    action_values: List[int]


class EdgeAggEnv:

    def __init__(self, spec: EnvSpec, gamma: float = 0.99, max_episode_len: int = 10_000):
        self.spec = spec
        self.gamma = float(gamma)
        self.max_len = int(max_episode_len)
        self._t = 0
        self._obs: Optional[np.ndarray] = None
        self._done = False

    # ---------------------------------------------------------
    # Episode handling
    # ---------------------------------------------------------

    def reset(self, initial_obs: np.ndarray) -> np.ndarray:
        assert initial_obs.shape[-1] == self.spec.obs_dim, \
            f"Bad initial_obs dim. Expected {self.spec.obs_dim}, got {initial_obs.shape[-1]}"
        self._obs = initial_obs.astype(np.float32)
        self._t = 0
        self._done = False
        return self._obs

    def step(self, next_obs: np.ndarray, reward: float):
        assert next_obs.shape[-1] == self.spec.obs_dim, \
            f"Bad next_obs dim. Expected {self.spec.obs_dim}, got {next_obs.shape[-1]}"
        self._t += 1
        self._obs = next_obs.astype(np.float32)
        self._done = self._t >= self.max_len
        return self._obs, float(reward), self._done, {}

    def is_done(self) -> bool:
        return bool(self._done)

    # ---------------------------------------------------------
    # SAFE SCALAR CONVERSION
    # ---------------------------------------------------------

    @staticmethod
    def _to_scalar(x) -> float:
        """
        Safely converts scalars or arrays to a single float.
        If array has multiple values, returns mean.
        """
        if isinstance(x, (int, float, bool)):
            return float(x)

        try:
            arr = np.asarray(x)
            if arr.size == 1:
                return float(arr.item())
            if arr.size > 1:
                return float(arr.mean())
        except Exception:
            pass

        return 0.0

    # ---------------------------------------------------------
    # SAFE ONE-HOT HANDLING
    # ---------------------------------------------------------

    @staticmethod
    def _ensure_onehot(vec, expected_len: int) -> np.ndarray:
        """
        Guarantees we return a proper one-hot vector of correct length.
        If vec invalid → returns zero vector.
        """
        if isinstance(vec, np.ndarray):
            vec = vec.astype(np.float32).reshape(-1)
            if vec.size == expected_len:
                return vec

        # fallback
        return np.zeros((expected_len,), dtype=np.float32)

    # ---------------------------------------------------------
    # OBSERVATION BUILDER
    # ---------------------------------------------------------

    @staticmethod
    def make_observation(
        realized_energy_mean,
        realized_latency_mean,
        queue_depth,
        last_action_onehot,
        model_delta=0.0,
        delta_exceeded=0.0,
        norm_scales: Dict[str, float] | None = None,
    ) -> np.ndarray:
        """
        Final observation structure:

        [ energy_norm,
          latency_norm,
          queue_norm,
          delta_norm,
          delta_flag,
          one_hot_action... ]
        """

        norm_scales = norm_scales or {
            "energy": 1.0,
            "latency": 1.0,
            "queue": 10.0,
            "delta": 1.0,
        }

        # ---- Scalars ----
        e_raw = EdgeAggEnv._to_scalar(realized_energy_mean)
        t_raw = EdgeAggEnv._to_scalar(realized_latency_mean)
        q_raw = EdgeAggEnv._to_scalar(queue_depth)
        d_raw = EdgeAggEnv._to_scalar(model_delta)
        delta_flag = EdgeAggEnv._to_scalar(delta_exceeded)

        e = e_raw / (norm_scales["energy"] + 1e-6)
        t = t_raw / (norm_scales["latency"] + 1e-6)
        q = q_raw / (norm_scales["queue"] + 1e-6)
        d = d_raw / (norm_scales["delta"] + 1e-6)

        base = np.array(
            [e, t, q, d, delta_flag],
            dtype=np.float32,
        )

        # ---- One-hot handling ----
        if isinstance(last_action_onehot, np.ndarray):
            onehot = last_action_onehot.astype(np.float32).reshape(-1)
        else:
            # if accidentally passed wrong type
            onehot = np.zeros((0,), dtype=np.float32)

        obs = np.concatenate([base, onehot], axis=0)

        return obs

