# src/federated/edge_server/rl_agent.py
from __future__ import annotations

from typing import List, Dict, Any
import numpy as np

from src.rl.madrl_env import EnvSpec, EdgeAggEnv
from src.rl.agents import DiscretePPOAgent, PPOConfig


class EdgeRLAggregationAgent:
    """
    Wraps PPO agent to choose O_l (edge aggregation period) each round.

    Observation includes:
        - mean_energy
        - mean_latency
        - queue_depth
        - model_delta
        - delta_exceeded
        - last_action_onehot
    """

    def __init__(
        self,
        edge_id: str,
        action_values: List[int] = [1, 2, 4, 8],
        seed: int = 42,
    ):
        self.edge_id = edge_id
        self.action_values = action_values

        # ------------------------------------------------------------
        # Observation = 5 scalar features + one_hot(action)
        # ------------------------------------------------------------
        self.spec = EnvSpec(
            obs_dim=5 + len(action_values),
            n_actions=len(action_values),
            action_values=action_values,
        )

        self.env = EdgeAggEnv(self.spec)

        self.agent = DiscretePPOAgent(
            PPOConfig(
                obs_dim=self.spec.obs_dim,
                n_actions=self.spec.n_actions,
            )
        )

        self._last_action_idx = 0
        self._last_logp = 0.0

        # ------------------------------------------------------------
        # initial observation
        # IMPORTANT: one-hot must be 4th positional argument
        # ------------------------------------------------------------
        obs0 = EdgeAggEnv.make_observation(
            0.0,  # mean_energy
            0.0,  # mean_latency
            0.0,  # queue_depth
            self._one_hot(self._last_action_idx),  # ✅ one-hot FIRST
            0.0,  # model_delta
            0.0,  # delta_exceeded
        )

        self.env.reset(obs0)

    # ------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------
    def select_action(self, obs: Dict[str, float], deterministic: bool = False) -> int:
        vec = self._pack_obs(obs)
        a_idx, logp = self.agent.act(vec, deterministic=deterministic)
        self._last_action_idx = int(a_idx)
        self._last_logp = float(logp)
        return self.spec.action_values[a_idx]

    # ------------------------------------------------------------
    # PPO training
    # ------------------------------------------------------------
    def learn(self, traj: Dict[str, Any]) -> None:
        """
        Train on a single-step trajectory.
        traj must contain:
            obs, act_idx, logp, reward, next_obs
        """
        obs = np.array([traj["obs"]], dtype=np.float32)
        acts = np.array([traj["act_idx"]], dtype=np.int32)
        logps = np.array([traj["logp"]], dtype=np.float32)
        rews = np.array([traj["reward"]], dtype=np.float32)
        dones = np.array([False], dtype=np.float32)
        next_obs = np.array(traj["next_obs"], dtype=np.float32)

        _ = self.agent.train_on_trajectory(
            obs, acts, logps, rews, dones, next_obs
        )

    def last_action_idx(self) -> int:
        return int(self._last_action_idx)

    def last_logp(self) -> float:
        return float(self._last_logp)

    # ------------------------------------------------------------
    # Observation packing (FIXED ORDER)
    # ------------------------------------------------------------
    def _pack_obs(self, d: Dict[str, float]) -> np.ndarray:
        oa = self._one_hot(self._last_action_idx)

        # IMPORTANT: one-hot must be 4th argument
        return EdgeAggEnv.make_observation(
            d.get("mean_energy", 0.0),
            d.get("mean_latency", 0.0),
            d.get("queue_depth", 0.0),
            oa,                                 # ✅ correct position
            d.get("model_delta", 0.0),
            d.get("delta_exceeded", 0.0),
        )

    # ------------------------------------------------------------
    # One-hot helper
    # ------------------------------------------------------------
    def _one_hot(self, i: int) -> np.ndarray:
        v = np.zeros((self.spec.n_actions,), dtype=np.float32)
        if 0 <= i < self.spec.n_actions:
            v[i] = 1.0
        return v

