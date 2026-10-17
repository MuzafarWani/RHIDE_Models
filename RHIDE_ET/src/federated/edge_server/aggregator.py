# src/federated/edge_server/aggregator.py
from __future__ import annotations

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

from src.federated.messaging.bus import MessagingBus
from src.federated.messaging.messages import Activations, RawBatch, ModelUpdate
from src.federated.edge_server.trainer import EdgeTrainer
from src.federated.edge_server.edge_state import EdgeState
from src.federated.edge_server.rl_agent import EdgeRLAggregationAgent
from src.accounting.upload_counter import UploadCounter
from src.accounting.energy_latency_log import EnergyLatencyLog
from src.federated.messaging.codecs import checksum_arrays

from src.federated.utils.model_delta import model_delta


# --- Minimal container for cut-gradients published back to devices ---
@dataclass
class CutGradients:
    device_id: str
    edge_id: str
    beta: str
    grads: Dict[str, np.ndarray]


class EdgeAggregator:
    """
    SAME DESIGN — NO CHANGE

    Only update:
    ✔ activation handling for 3D-only split (XEventNet)
    ✔ keep VGG16 unchanged
    """

    def __init__(self, edge_id: str, bus: MessagingBus, cfg: Any):
        self.edge_id = edge_id
        self.bus = bus
        self.cfg = cfg

        tc = cfg.train_cfg

        self.trainer = EdgeTrainer(
            edge_id=edge_id,
            full_model=None,
            input_shape_2d=tuple(tc["input_shape_2d"]),
            input_shape_3d=tuple(tc["input_shape_3d"]),
            num_classes=int(tc.get("num_classes", 5)),
        )

        self.state = EdgeState(edge_id=edge_id)
        self.rl = EdgeRLAggregationAgent(
            edge_id=edge_id,
            action_values=cfg.rl_cfg.get("edge_actions", [1, 2, 4, 8])
        )

        self.uploader = UploadCounter(bus)
        self.acc_logger = EnergyLatencyLog(bus)

        self.edge_root_model = None
        self.device_models: Dict[str, Any] = {}
        self.devices_trained_this_round: List[str] = []

        self._last_sent_weights: Optional[List[np.ndarray]] = None
        self.tau_edge: float = float(cfg.rl_cfg.get("edge_tau", 0.05))

        self._last_cloud_update_ts: float = 0.0
        self._pending_losses: List[float] = []

    # -------------------------------------------------
    def set_full_model(self, model) -> None:
        self.edge_root_model = model
        self.trainer.full_model = model

    def register_device_model(self, device_id: str, model) -> None:
        self.device_models[device_id] = model

    # -------------------------------------------------
    def _apply_cloud_broadcasts(self) -> bool:
        b_msgs = [m for m in self.bus.drain("cloud_to_edges_broadcast")
                  if isinstance(m, ModelUpdate)]

        per_edge_topic = f"cloud_to_edge_{self.edge_id}"
        e_msgs = [m for m in self.bus.drain(per_edge_topic)
                  if isinstance(m, ModelUpdate)]

        all_msgs = b_msgs + e_msgs
        if not all_msgs or self.edge_root_model is None:
            return False

        newest = max(all_msgs, key=lambda m: getattr(m, "created_ts", 0.0))
        ts = float(getattr(newest, "created_ts", 0.0))
        if ts <= self._last_cloud_update_ts:
            return False

        self.edge_root_model.set_weights(newest.weights)

        for d, m in self.device_models.items():
            m.set_weights(newest.weights)
            #self.uploader.send_model_to_device(edge_id=self.edge_id, device_id=d, weights=newest.weights)

        if hasattr(self.trainer, "_edge_nets"):
            try:
                self.trainer._edge_nets.clear()
            except Exception:
                self.trainer._edge_nets = {}

        self._last_cloud_update_ts = ts
        print(f"[edge:{self.edge_id}] applied new global model from cloud (ts={ts})")
        return True

    # -------------------------------------------------
    def _peek_pending_for_edge(self) -> tuple[int, int]:
        pending_a = pending_r = 0
        for topic, kind in (("device_to_edge_activations", "act"),
                            ("device_to_edge_raw", "raw")):
            msgs = None
            for fn in ("peek", "snapshot"):
                peek_fn = getattr(self.bus, fn, None)
                if callable(peek_fn):
                    try:
                        msgs = peek_fn(topic)
                        break
                    except Exception:
                        msgs = None
            if msgs is None:
                continue
            if kind == "act":
                pending_a = sum(
                    1 for m in msgs
                    if isinstance(m, Activations) and m.edge_id == self.edge_id
                )
            else:
                pending_r = sum(
                    1 for m in msgs
                    if isinstance(m, RawBatch) and m.edge_id == self.edge_id
                )
        return pending_a, pending_r

    # -------------------------------------------------
    def step_round(self, round_index: int) -> None:
        if self.edge_root_model is None:
            return

        self._apply_cloud_broadcasts()
        self.devices_trained_this_round.clear()

        pend_a_before, pend_r_before = self._peek_pending_for_edge()
        self.state.set_queue_pending(pend_a_before, pend_r_before)

        acts_msgs = [m for m in self.bus.drain("device_to_edge_activations")
                     if isinstance(m, Activations) and m.edge_id == self.edge_id]
        raw_msgs = [m for m in self.bus.drain("device_to_edge_raw")
                    if isinstance(m, RawBatch) and m.edge_id == self.edge_id]

        self.state.q_activations += len(acts_msgs)
        self.state.q_raw += len(raw_msgs)

        # ============================================================
        # TRAIN ON ACTIVATIONS
        # ============================================================
        for m in acts_msgs:
            d = m.device_id
            self.devices_trained_this_round.append(d)

            self.trainer.full_model = self.device_models[d]
            self.uploader.record_device_upload(d, getattr(m, "bytes", 0))

            acts = m.arrays

            # 🔥 FIX: handle both VGG16 and XEventNet
            if isinstance(acts, dict) and "a3d" in acts and "a2d" not in acts:
                activations = acts["a3d"]   # XEventNet
            elif isinstance(acts, dict) and "a2d" in acts and "a3d" in acts:
                activations = acts          # VGG16
            else:
                activations = acts          # fallback

            out = self.trainer.train_from_activations(
                beta_name=m.beta,
                activations=activations,
                labels=getattr(m, "labels", None),
                batch_size=int(self.cfg.train_cfg.get("batch_size", 8)),
                return_cut_grads=True,
            )

            cut_grads = out.get("cut_grads")
            if cut_grads:
                cg = CutGradients(
                    device_id=d,
                    edge_id=self.edge_id,
                    beta=m.beta,
                    grads=cut_grads,
                )
                self.bus.publish("edge_to_device_cutgrads", cg)

            try:
                self._pending_losses.append(float(out.get("loss", 0.0)))
            except Exception:
                pass

        # ============================================================
        # RAW TRAINING
        # ============================================================
        for m in raw_msgs:
            d = m.device_id
            self.devices_trained_this_round.append(d)

            self.trainer.full_model = self.device_models[d]
            self.uploader.record_device_upload(d, getattr(m, "bytes", 0))

            stat = self.trainer.train_from_raw(arrays=m.arrays, labels=m.labels)
            self._pending_losses.append(stat["loss"])

        # ============================================================
        # REMAINING CODE
        # ============================================================
        pend_a_after, pend_r_after = self._peek_pending_for_edge()
        self.state.set_queue_pending(pend_a_after, pend_r_after)

        obs_now = self.state.obs_vector()
        O_l = int(self.rl.select_action(obs_now, deterministic=False))
        self.state.agg_period_Ol = O_l

        current_weights = [w for w in self.edge_root_model.get_weights()]
        delta = model_delta(current_weights, self._last_sent_weights)

        self.state.rounds_since_agg += 1

        if self.state.rounds_since_agg >= O_l or delta >= self.tau_edge:
            self._aggregate_device_models()
            self._send_update_to_cloud()
            self.state.rounds_since_agg = 0
            self._last_sent_weights = current_weights

        self.state.reset_round_counters()

    # -------------------------------------------------
    def _aggregate_device_models(self):
        if not self.devices_trained_this_round:
            return

        device_weights = [
            self.device_models[d].get_weights()
            for d in self.devices_trained_this_round
        ]

        avg_weights = []
        for i in range(len(device_weights[0])):
            avg_weights.append(np.mean([w[i] for w in device_weights], axis=0))

        self.edge_root_model.set_weights(avg_weights)

        for d, m in self.device_models.items():
            m.set_weights(avg_weights)

    # -------------------------------------------------
    def _send_update_to_cloud(self) -> None:
        weights = [w for w in self.edge_root_model.get_weights()]

        msg = ModelUpdate(
            sender_id=self.edge_id,
            receiver_id="cloud",
            weights=weights,
        )

        self.bus.publish("edge_to_cloud_update", msg)

    # -------------------------------------------------
    def rl_learn(self, reward: float) -> None:
        next_obs_d = self.state.obs_vector()
        traj = {
            "obs": self.rl._pack_obs(self.state.obs_vector()),
            "act_idx": self.rl.last_action_idx(),
            "logp": self.rl.last_logp(),
            "reward": float(reward),
            "next_obs": self.rl._pack_obs(next_obs_d),
        }
        self.rl.learn(traj)

    # -------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        return {
            "agg_period_Ol": self.state.agg_period_Ol,
            "rounds_since_agg": self.state.rounds_since_agg,
        }

