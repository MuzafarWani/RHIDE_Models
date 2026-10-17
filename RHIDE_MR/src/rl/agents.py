#src/rl/agents.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Tuple, Optional

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

from .replay import ReplayBuffer


# ---------------------------
# Utilities
# ---------------------------

def _mlp(input_dim: int, hidden: Tuple[int, ...], n_outputs: int, out_activation=None) -> Model:
    x = inp = layers.Input(shape=(input_dim,), name="obs")
    for h in hidden:
        x = layers.Dense(h, activation="relu")(x)
    x = layers.Dense(n_outputs, activation=out_activation)(x)
    return Model(inp, x, name="mlp")


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _save_optimizer(opt: optimizers.Optimizer, path: str) -> None:
    """Persist optimizer weights to NPZ (best-effort; safe if unset)."""
    try:
        w = opt.get_weights()
        np.savez(path, *w)
    except Exception:
        # Optimizer may be uninitialized (no variables created yet). Ignore.
        pass


def _load_optimizer(opt: optimizers.Optimizer, path: str) -> None:
    """Load optimizer weights if present (best-effort)."""
    try:
        if os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                # Rebuild a list in the original order
                w = [z[k] for k in sorted(z.files, key=lambda x: int(x.replace("arr_", "")))]
            opt.set_weights(w)
    except Exception:
        pass


# ---------------------------
# PPO (discrete)
# ---------------------------

@dataclass
class PPOConfig:
    obs_dim: int
    n_actions: int
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


class DiscretePPOAgent:
    """
    Small PPO implementation for discrete action spaces.
    Uses categorical policy and value head; on-policy (no replay).
    """

    def __init__(self, cfg: PPOConfig, seed: int = 42):
        self.cfg = cfg
        tf.keras.utils.set_random_seed(seed)

        self.policy = _mlp(cfg.obs_dim, cfg.policy_hidden, cfg.n_actions, out_activation=None)
        self.value = _mlp(cfg.obs_dim, cfg.value_hidden, 1, out_activation=None)
        self.pi_opt = optimizers.Adam(learning_rate=cfg.policy_lr)
        self.v_opt = optimizers.Adam(learning_rate=cfg.value_lr)

    # ---------- persistence ----------

    def save(self, dirpath: str) -> None:
        """
        Save policy/value weights, optimizers, and config into `dirpath`.
        Files:
          - config.json
          - policy.weights.h5
          - value.weights.h5
          - pi_opt.npz
          - v_opt.npz
        """
        _ensure_dir(dirpath)
        with open(os.path.join(dirpath, "config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        # Save weights
        self.policy.save_weights(os.path.join(dirpath, "policy.weights.h5"))
        self.value.save_weights(os.path.join(dirpath, "value.weights.h5"))

        # Save optimizer states (best-effort)
        _save_optimizer(self.pi_opt, os.path.join(dirpath, "pi_opt.npz"))
        _save_optimizer(self.v_opt, os.path.join(dirpath, "v_opt.npz"))

    def load(self, dirpath: str, strict: bool = False) -> None:
        """
        Load weights/optimizers if present. If strict=True, raise on missing files.
        """
        def _must(path: str):
            if strict and not os.path.exists(path):
                raise FileNotFoundError(path)

        _must(dirpath)
        cfg_path = os.path.join(dirpath, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    _ = json.load(f)  # we don't mutate self.cfg here; architecture must match
            except Exception:
                if strict:
                    raise

        # Need to build variables before loading optimizer weights
        # Create dummy forward pass
        _ = self.policy(np.zeros((1, self.cfg.obs_dim), dtype=np.float32))
        _ = self.value(np.zeros((1, self.cfg.obs_dim), dtype=np.float32))

        # Load model weights
        pw = os.path.join(dirpath, "policy.weights.h5")
        vw = os.path.join(dirpath, "value.weights.h5")
        if os.path.exists(pw):
            self.policy.load_weights(pw)
        elif strict:
            raise FileNotFoundError(pw)
        if os.path.exists(vw):
            self.value.load_weights(vw)
        elif strict:
            raise FileNotFoundError(vw)

        # Load optimizer weights (best-effort)
        _load_optimizer(self.pi_opt, os.path.join(dirpath, "pi_opt.npz"))
        _load_optimizer(self.v_opt, os.path.join(dirpath, "v_opt.npz"))

    # ---------- acting / learning ----------

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Returns (action_index, log_prob)
        """
        logits = self.policy(obs[None, :])
        logits = tf.squeeze(logits, axis=0)
        
        log_probs = tf.nn.log_softmax(logits)
        num_actions = int(logits.shape[0])
        if deterministic:
            a = int(tf.argmax(logits).numpy())
            #logp = (tf.nn.log_softmax(logits)[a]).numpy().item()
            #return a, logp
        else:
            dist = tf.random.categorical(logits[None, :], num_samples=1)
            a = int(dist.numpy()[0, 0])
            
        if a >= num_actions:
            print(f"[RL WARNING] action {a} out of bounds → clamped to {num_actions - 1}")
            a = num_actions - 1
                 
        logp = (tf.nn.log_softmax(logits)[a]).numpy().item()
        return a, logp

    def train_on_trajectory(
        self,
        obs: np.ndarray,          # (T, obs_dim)
        acts: np.ndarray,         # (T,)
        logps_old: np.ndarray,    # (T,)
        rews: np.ndarray,         # (T,)
        dones: np.ndarray,        # (T,)
        next_obs_last: np.ndarray # (obs_dim,)
    ) -> Dict[str, float]:
        """
        Single PPO update over collected trajectory.
        """
        cfg = self.cfg
        # Compute advantages using GAE(λ)
        vals = tf.squeeze(self.value(obs), axis=1).numpy()
        vals_next_last = float(tf.squeeze(self.value(next_obs_last[None, :]), axis=1).numpy()[0])
        adv, ret = _gae_lambda(rews, dones, vals, vals_next_last, cfg.gamma, cfg.lam)

        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0}
        dataset = _ppo_dataset(obs, acts, logps_old, adv, ret, cfg.minibatch_size)

        for _ in range(cfg.train_iters):
            for batch in dataset:
                o_b, a_b, logp_old_b, adv_b, ret_b = batch

                # Policy update
                with tf.GradientTape() as tape:
                    logits = self.policy(o_b)
                    logp_all = tf.nn.log_softmax(logits)
                    ent = -tf.reduce_mean(tf.reduce_sum(tf.exp(logp_all) * logp_all, axis=-1))
                    idx = tf.stack([tf.range(tf.shape(a_b)[0]), tf.cast(a_b, tf.int32)], axis=1)
                    logp = tf.gather_nd(logp_all, idx)
                    ratio = tf.exp(logp - logp_old_b)
                    adv_norm = (adv_b - tf.reduce_mean(adv_b)) / (tf.math.reduce_std(adv_b) + 1e-8)
                    obj1 = ratio * adv_norm
                    obj2 = tf.clip_by_value(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_norm
                    pi_loss = -tf.reduce_mean(tf.minimum(obj1, obj2)) - cfg.entropy_coef * ent
                grads = tape.gradient(pi_loss, self.policy.trainable_variables)
                self.pi_opt.apply_gradients(zip(grads, self.policy.trainable_variables))

                # Value update
                with tf.GradientTape() as tape:
                    v_pred = tf.squeeze(self.value(o_b), axis=1)
                    v_loss = tf.reduce_mean((ret_b - v_pred) ** 2)
                grads = tape.gradient(v_loss, self.value.trainable_variables)
                self.v_opt.apply_gradients(zip(grads, self.value.trainable_variables))

                stats["pi_loss"] = float(pi_loss.numpy())
                stats["v_loss"] = float(v_loss.numpy())
                stats["entropy"] = float(ent.numpy())
        return stats


def _gae_lambda(rews, dones, vals, last_v, gamma, lam):
    T = len(rews)
    adv = np.zeros(T, dtype=np.float32)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        nextv = last_v if t == T - 1 else vals[t + 1]
        delta = rews[t] + gamma * nextv * nonterminal - vals[t]
        lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
        adv[t] = lastgaelam
    ret = adv + vals
    return adv, ret


def _ppo_dataset(obs, acts, logps_old, adv, ret, batch_size):
    ds = tf.data.Dataset.from_tensor_slices((
        obs.astype(np.float32),
        acts.astype(np.int32),
        logps_old.astype(np.float32),
        adv.astype(np.float32),
        ret.astype(np.float32),
    ))
    ds = ds.shuffle(buffer_size=max(1024, len(acts))).batch(batch_size, drop_remainder=False)
    return ds


# ---------------------------
# DQN (discrete)
# ---------------------------

@dataclass
class DQNConfig:
    obs_dim: int
    n_actions: int
    hidden: Tuple[int, ...] = (128, 128)
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon_start: float = 0.1
    epsilon_end: float = 0.01
    epsilon_decay: float = 1e-4   # per step
    target_tau: float = 0.01      # soft update
    batch_size: int = 64
    replay_capacity: int = 50_000
    min_replay: int = 1_000


class DiscreteDQNAgent:
    """
    Simple DQN agent with soft target updates and epsilon-greedy exploration.
    """

    def __init__(self, cfg: DQNConfig, seed: int = 42):
        self.cfg = cfg
        tf.keras.utils.set_random_seed(seed)

        self.q = _mlp(cfg.obs_dim, cfg.hidden, cfg.n_actions, out_activation=None)
        self.q_tgt = _mlp(cfg.obs_dim, cfg.hidden, cfg.n_actions, out_activation=None)
        self._hard_copy(self.q, self.q_tgt)
        self.opt = optimizers.Adam(learning_rate=cfg.lr)
        self.buffer = ReplayBuffer(capacity=cfg.replay_capacity, obs_dim=cfg.obs_dim)
        self.epsilon = cfg.epsilon_start
        self._steps = 0

    # ---------- persistence ----------

    def save(self, dirpath: str, save_replay: bool = False) -> None:
        """
        Save Q/target weights, optimizer, epsilon/steps, and (optionally) replay buffer.
        Files:
          - config.json
          - q.weights.h5
          - q_tgt.weights.h5
          - opt.npz
          - trainer.json   (epsilon, steps)
          - replay.pkl     (optional, large)
        """
        _ensure_dir(dirpath)
        with open(os.path.join(dirpath, "config.json"), "w") as f:
            json.dump(asdict(self.cfg), f, indent=2)

        self.q.save_weights(os.path.join(dirpath, "q.weights.h5"))
        self.q_tgt.save_weights(os.path.join(dirpath, "q_tgt.weights.h5"))
        _save_optimizer(self.opt, os.path.join(dirpath, "opt.npz"))

        with open(os.path.join(dirpath, "trainer.json"), "w") as f:
            json.dump({"epsilon": float(self.epsilon), "steps": int(self._steps)}, f, indent=2)

        if save_replay:
            try:
                import pickle
                with open(os.path.join(dirpath, "replay.pkl"), "wb") as f:
                    pickle.dump(self.buffer, f)
            except Exception:
                pass

    def load(self, dirpath: str, strict: bool = False, load_replay: bool = False) -> None:
        """
        Load saved components if present. If strict=True, raise on missing core files.
        """
        def _must(p: str):
            if strict and not os.path.exists(p):
                raise FileNotFoundError(p)

        _must(dirpath)

        # Build variables
        _ = self.q(np.zeros((1, self.cfg.obs_dim), dtype=np.float32))
        _ = self.q_tgt(np.zeros((1, self.cfg.obs_dim), dtype=np.float32))

        qw = os.path.join(dirpath, "q.weights.h5")
        tw = os.path.join(dirpath, "q_tgt.weights.h5")
        if os.path.exists(qw):
            self.q.load_weights(qw)
        elif strict:
            raise FileNotFoundError(qw)
        if os.path.exists(tw):
            self.q_tgt.load_weights(tw)
        elif strict:
            raise FileNotFoundError(tw)

        _load_optimizer(self.opt, os.path.join(dirpath, "opt.npz"))

        tjson = os.path.join(dirpath, "trainer.json")
        if os.path.exists(tjson):
            try:
                with open(tjson, "r") as f:
                    td = json.load(f)
                self.epsilon = float(td.get("epsilon", self.epsilon))
                self._steps = int(td.get("steps", self._steps))
            except Exception:
                if strict:
                    raise

        if load_replay:
            rp = os.path.join(dirpath, "replay.pkl")
            if os.path.exists(rp):
                try:
                    import pickle
                    with open(rp, "rb") as f:
                        self.buffer = pickle.load(f)
                except Exception:
                    if strict:
                        raise

    # ---------- acting / learning ----------

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """
        Epsilon-greedy; deterministic ignores epsilon.
        """
        if (not deterministic) and (np.random.rand() < self.epsilon):
            return int(np.random.randint(self.cfg.n_actions))
        qv = self.q(obs[None, :])
        return int(tf.argmax(qv[0]).numpy())

    def observe(self, obs, act, rew, next_obs, done) -> None:
        self.buffer.add(obs, act, rew, next_obs, done)
        self._steps += 1
        # epsilon decay
        self.epsilon = max(
            self.cfg.epsilon_end,
            self.epsilon - self.cfg.epsilon_decay
        )

    def train_step(self) -> Optional[Dict[str, float]]:
        if self.buffer.size < self.cfg.min_replay:
            return None
        batch = self.buffer.sample(self.cfg.batch_size)
        o, a, r, no, d = batch

        # Compute targets: r + gamma * max_a' Q_tgt(no, a') * (1-d)
        q_next = self.q_tgt(no)
        q_next_max = tf.reduce_max(q_next, axis=1)
        target = r + (1.0 - d) * self.cfg.gamma * q_next_max

        with tf.GradientTape() as tape:
            q_pred_all = self.q(o)
            idx = tf.stack([tf.range(tf.shape(a)[0]), tf.cast(a, tf.int32)], axis=1)
            q_pred = tf.gather_nd(q_pred_all, idx)
            loss = tf.reduce_mean((target - q_pred) ** 2)
        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))

        # soft target update
        self._soft_update(self.q, self.q_tgt, tau=self.cfg.target_tau)
        return {"loss": float(loss.numpy()), "epsilon": float(self.epsilon)}

    @staticmethod
    def _hard_copy(src: tf.keras.Model, dst: tf.keras.Model):
        dst.set_weights(src.get_weights())

    @staticmethod
    def _soft_update(src: tf.keras.Model, dst: tf.keras.Model, tau: float):
        new_w = []
        for w_src, w_dst in zip(src.get_weights(), dst.get_weights()):
            new_w.append(tau * w_src + (1 - tau) * w_dst)
        dst.set_weights(new_w)

