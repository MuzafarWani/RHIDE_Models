# rl_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np

from src.rl.agents import DiscretePPOAgent, PPOConfig


def _pack_obs_d(obs_d: Dict[str, float], keys: List[str]) -> np.ndarray:
    """Pack dict features in a stable order."""
    return np.array([float(obs_d.get(k, 0.0)) for k in keys], dtype=np.float32)


@dataclass
class CloudRLConfig:
    obs_keys: Tuple[str, ...]
    actions: Tuple[int, ...] = (1, 2, 4, 8)  # allowed O_e choices
    # PPO net sizes etc. — keep modest for stability
    policy_hidden: Tuple[int, ...] = (64, 64)
    value_hidden: Tuple[int, ...] = (64, 64)
    clip_ratio: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    train_iters: int = 4
    minibatch_size: int = 64
    entropy_coef: float = 0.01


class CloudRLAggregationAgent:
    """
    PPO-driven controller for cloud aggregation cadence O_e.
    - select_action(obs) -> an index in `actions`, and stores (last action, logp, obs)
    - learn(traj)        -> PPO update using the last transition

    >>> UPDATED: Obs can now include model_delta and delta_exceeded.
    """

    def __init__(self, cfg: CloudRLConfig, seed: int = 42):
        self.cfg = cfg
        self.actions = list(cfg.actions)

        # ------------------------------------------------------------
        # >>> UPDATED: obs_keys may now include "model_delta"
        #     and "delta_exceeded" if you configure them in rl_cfg.yaml
        # ------------------------------------------------------------
        self.obs_keys = list(cfg.obs_keys)

        ppo_cfg = PPOConfig(
            obs_dim=len(self.obs_keys),
            n_actions=len(self.actions),
            policy_hidden=cfg.policy_hidden,
            value_hidden=cfg.value_hidden,
            clip_ratio=cfg.clip_ratio,
            gamma=cfg.gamma,
            lam=cfg.lam,
            policy_lr=cfg.policy_lr,
            value_lr=cfg.value_lr,
            train_iters=cfg.train_iters,
            minibatch_size=cfg.minibatch_size,
            entropy_coef=cfg.entropy_coef,
        )
        self.agent = DiscretePPOAgent(ppo_cfg, seed=seed)

        # last transition book-keeping
        self._last_obs: Optional[np.ndarray] = None
        self._last_action_idx: Optional[int] = None
        self._last_logp: Optional[float] = None

    # ---- API ----
    def select_action(self, obs_dict: Dict[str, float], deterministic: bool = False) -> int:
        obs = _pack_obs_d(obs_dict, self.obs_keys)
        a_idx, logp = self.agent.act(obs, deterministic=deterministic)
        self._last_obs = obs
        self._last_action_idx = int(a_idx)
        self._last_logp = float(logp)
        return self.actions[a_idx]  # return O_e value

    def last_action_idx(self) -> Optional[int]:
        return self._last_action_idx

    def last_logp(self) -> Optional[float]:
        return self._last_logp

    def learn(
        self,
        reward: float,
        next_obs_dict: Dict[str, float],
        done: bool = False,
    ) -> Optional[Dict[str, float]]:
        """
        One-step on-policy PPO update using the stored last (obs, act, logp)
        and this round's reward and next_obs.
        """
        if (
            self._last_obs is None
            or self._last_action_idx is None
            or self._last_logp is None
        ):
            return None

        obs = self._last_obs[None, :]
        acts = np.array([self._last_action_idx], dtype=np.int32)
        logps_old = np.array([self._last_logp], dtype=np.float32)
        rews = np.array([float(reward)], dtype=np.float32)
        dones = np.array([1.0 if done else 0.0], dtype=np.float32)

        # >>> UPDATED: pack next_obs using possibly extended obs_keys
        next_obs = _pack_obs_d(next_obs_dict, self.obs_keys)

        stats = self.agent.train_on_trajectory(
            obs=obs,
            acts=acts,
            logps_old=logps_old,
            rews=rews,
            dones=dones,
            next_obs_last=next_obs,
        )
        return stats

    # ---- persistence ----
    def save(self, dirpath: str) -> None:
        # Save Keras sub-models & a small meta.json with obs_keys/actions
        import os, json

        os.makedirs(dirpath, exist_ok=True)
        self.agent.policy.save(os.path.join(dirpath, "policy"))
        self.agent.value.save(os.path.join(dirpath, "value"))

        with open(os.path.join(dirpath, "meta.json"), "w") as f:
            json.dump(
                {"obs_keys": self.obs_keys, "actions": self.actions}, f, indent=2
            )

    def load(self, dirpath: str) -> None:
        import os, json, tensorflow as tf

        pol_p = os.path.join(dirpath, "policy")
        val_p = os.path.join(dirpath, "value")
        if os.path.isdir(pol_p):
            self.agent.policy = tf.keras.models.load_model(pol_p)
        if os.path.isdir(val_p):
            self.agent.value = tf.keras.models.load_model(val_p)

        meta_p = os.path.join(dirpath, "meta.json")
        if os.path.exists(meta_p):
            with open(meta_p, "r") as f:
                meta = json.load(f)
            # keep runtime obs_keys/actions; we don't overwrite to avoid mismatch

