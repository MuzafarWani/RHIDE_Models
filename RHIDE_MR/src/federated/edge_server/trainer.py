from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf

from src.models.split_graph_vgg16 import create_vgg16_submodels
from src.models.split_graph import create_submodels
from src.models.serialization import transfer_weights_by_name


class EdgeTrainer:
    """
    Edge-side training:
      - train from Activations (β < L): needs labels and edge_net
      - train from RawBatch      (z=0): trains the full model

    ✅ VGG16: unchanged
    ✅ XEventNet: supports 3D-only activations safely
    """

    def __init__(self, edge_id: str, full_model: tf.keras.Model,
                 input_shape_2d, input_shape_3d, num_classes: int):
        self.edge_id = edge_id
        self.full_model = full_model
        self.input_shape_2d = input_shape_2d
        self.input_shape_3d = input_shape_3d
        self.num_classes = num_classes
        self._edge_nets: Dict[str, tf.keras.Model] = {}

        self._device_full_models: Dict[str, tf.keras.Model] = {}

    # ------------------------ helpers ------------------------

    def _clone_optimizer_from_full(self):
        opt = getattr(self.full_model, "optimizer", None)
        if opt is None:
            return tf.keras.optimizers.Adam(1e-3)
        try:
            return tf.keras.optimizers.deserialize(tf.keras.optimizers.serialize(opt))
        except Exception:
            return tf.keras.optimizers.Adam(1e-3)

    def _pick_loss_from_full(self):
        loss = getattr(self.full_model, "loss", None)

        if isinstance(loss, (list, tuple)) and len(loss) > 0:
            loss = loss[0]

        if loss is None:
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        try:
            return tf.keras.losses.deserialize(tf.keras.losses.serialize(loss))
        except Exception:
            return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    def _get_device_model(self, device_id):
        if device_id is None:
            return self.full_model

        if device_id not in self._device_full_models:
            cloned = tf.keras.models.clone_model(self.full_model)
            cloned.set_weights(self.full_model.get_weights())
            self._device_full_models[device_id] = cloned

        return self._device_full_models[device_id]

    # ------------------------ nets ------------------------

    def _get_edge_net(self, beta_name, device_id=None):
        key = f"{device_id}:{beta_name}"

        if key in self._edge_nets:
            return self._edge_nets[key]

        base_model = self._get_device_model(device_id)

        if beta_name.startswith("block"):
            _device_net, edge_net = create_vgg16_submodels(base_model, beta_name)
        else:
            _device_net, edge_net = create_submodels(
                full_model=base_model,
                beta_name=beta_name,
                input_shape_2d=self.input_shape_2d,
                input_shape_3d=self.input_shape_3d,
                num_classes=self.num_classes,
            )

        transfer_weights_by_name(base_model, edge_net, strict=False)

        opt = self._clone_optimizer_from_full()
        loss = self._pick_loss_from_full()

        edge_net.compile(optimizer=opt, loss=loss, metrics=None)

        self._edge_nets[key] = edge_net
        return edge_net

    # ============================================================
    # TRAIN FROM ACTIVATIONS
    # ============================================================

    def train_from_activations(
        self,
        beta_name: str,
        activations,
        labels,
        batch_size=8,
        return_cut_grads=True,
        device_id=None,
    ):

        if labels is None or len(labels) == 0:
            return {"loss": 0.0, "n": 0}

        labels = np.asarray(labels).reshape(-1)

        edge_net = self._get_edge_net(beta_name, device_id)

        # ============================================================
        # 🔥 FIX 1: Normalize activations format
        # ============================================================
        if isinstance(activations, np.ndarray):
            activations = {"a3d": activations}

        if not isinstance(activations, dict):
            raise ValueError(f"Invalid activations type: {type(activations)}")

        # ============================================================
        # INPUT ROUTING (UNCHANGED FOR VGG16)
        # ============================================================
        if "joint" in activations:
            X_in = activations["joint"]

        elif "a2d" in activations and "a3d" in activations:
            X_in = [activations["a2d"], activations["a3d"]]

        elif "a3d" in activations:
            X_in = activations["a3d"]

        else:
            raise ValueError(f"Unsupported activation format: {list(activations.keys())}")

        # ---------------- TRAIN ----------------
        hist = edge_net.fit(X_in, labels, epochs=1, batch_size=batch_size, verbose=0)

        out = {
            "loss": float(hist.history.get("loss", [0.0])[-1]),
            "n": int(len(labels)),
        }

        # ---------------- GRADIENTS ----------------
        if return_cut_grads:

            with tf.GradientTape() as tape:

                if isinstance(X_in, list):
                    xin = [tf.convert_to_tensor(x) for x in X_in]
                    for t in xin:
                        tape.watch(t)
                    preds = edge_net(xin, training=True)

                else:
                    xin = tf.convert_to_tensor(X_in)
                    tape.watch(xin)
                    preds = edge_net(xin, training=True)

                loss_obj = edge_net.loss
                loss_val = loss_obj(tf.convert_to_tensor(labels), preds)

            if isinstance(X_in, list):
                grads = tape.gradient(loss_val, xin)
                cut_grads = {}

                if "a2d" in activations and grads[0] is not None:
                    cut_grads["a2d"] = grads[0].numpy()

                if "a3d" in activations and grads[1] is not None:
                    cut_grads["a3d"] = grads[1].numpy()

            elif "a3d" in activations:
                grad = tape.gradient(loss_val, xin)
                cut_grads = {"a3d": grad.numpy() if grad is not None else np.zeros_like(X_in)}

            else:
                grad = tape.gradient(loss_val, xin)
                cut_grads = {"joint": grad.numpy() if grad is not None else np.zeros_like(X_in)}

            out["cut_grads"] = cut_grads

        return out

    # ============================================================
    # RAW TRAIN (UNCHANGED)
    # ============================================================

    def train_from_raw(self, arrays, labels, batch_size=8, device_id=None):

        X2 = arrays["X2"]
        X3 = arrays["X3"]
        labels = np.asarray(labels)

        model = self._get_device_model(device_id)

        hist = model.fit(
            {"input_2d": X2, "input_3d": X3},
            labels,
            epochs=1,
            batch_size=batch_size,
            verbose=0,
        )

        return {
            "loss": float(hist.history.get("loss", [0.0])[-1]),
            "n": int(len(labels)),
        }

