# src/federated/device_client/trainer.py
from __future__ import annotations

from typing import Tuple, Dict, Any, List, Optional

import numpy as np
import tensorflow as tf

from src.models.split_graph import create_submodels
from src.models.serialization import make_device_edge_from_full


class DeviceTrainer:
    """
    Two modes:
      1) β = L (full local): run train_on_batch and produce a ModelUpdate (weights list).
      2) β < L (split): forward handled by Splitter; supports applying cut-layer gradients
         (d(loss)/d(cut_output)) to update device-side subgraph via backprop.

    >>> UPDATED:
      - Devices can now expose their own updated weights to the Edge.
      - Edge can later sync aggregated weights back to the device.
    """

    def __init__(self, full_model: tf.keras.Model):
        self.full_model = full_model
        self._device_nets: Dict[str, tf.keras.Model] = {}
        self._last_inputs: Dict[str, Dict[str, np.ndarray]] = {}
        # Optimizer for device heads (if full_model has one, mirror its config)
        if getattr(self.full_model, "optimizer", None) is not None:
            self._opt = tf.keras.optimizers.deserialize(
                tf.keras.optimizers.serialize(self.full_model.optimizer)
            )
        else:
            self._opt = tf.keras.optimizers.Adam(1e-3)

        # >>> NEW: track last weights for delta computation if needed
        self._last_sent_weights: Optional[List[np.ndarray]] = None

    def _get_device_net(self, beta_name: str) -> tf.keras.Model:
        if beta_name in self._device_nets:
            return self._device_nets[beta_name]

        # Infer shapes from the full model inputs/outputs
        in2_shape = None
        in3_shape = None
        try:
            # full_model.input can be a list of KerasTensors
            if isinstance(self.full_model.input, (list, tuple)) and len(self.full_model.input) >= 2:
                in2_shape = self.full_model.input[0].shape[1:]
                in3_shape = self.full_model.input[1].shape[1:]
            else:
                in2_shape = self.full_model.input.shape[1:]
        except Exception:
            pass

        nout = 1
        try:
            nout = int(self.full_model.output_shape[-1])
        except Exception:
            pass

        device_net, _edge_net = create_submodels(
            full_model=self.full_model,
            beta_name=beta_name,
            input_shape_2d=in2_shape,
            input_shape_3d=in3_shape,
            num_classes=nout,
        )
        # Sync weights into device_net from the full model
        try:
            make_device_edge_from_full(self.full_model, device_net, None)
        except TypeError:
            # Older helper signature (device_net, edge_net) – pass device twice as safe fallback
            make_device_edge_from_full(self.full_model, device_net, device_net)

        self._device_nets[beta_name] = device_net
        return device_net

    # ----------- helper: cache raw inputs used to create activations -----------

    def cache_last_inputs(self, beta_name: str, inputs: Dict[str, np.ndarray]) -> None:
        """
        Store the raw *device* inputs used for the most recent forward-to-cut.
        inputs: {"input_2d": X2, "input_3d": X3} or {"input_2d": X2} depending on model.
        """
        self._last_inputs[beta_name] = inputs

    # ----------- full local training (β = L) -----------

    def train_full_step(
        self,
        X2_batch: np.ndarray,
        X3_batch: np.ndarray,
        y_batch: np.ndarray,
    ) -> Tuple[float, List[np.ndarray]]:
        """
        Returns (loss_value, weights_list).
        """
        metrics = self.full_model.train_on_batch(
            {"input_2d": X2_batch, "input_3d": X3_batch},
            y_batch,
            return_dict=True
        )
        # Return a flat list of arrays for transport
        weights_list = [w for w in self.full_model.get_weights()]
        return float(metrics.get("loss", 0.0)), weights_list

    # ----------- cut-backward (β < L): apply upstream gradients at cut -----------

    def apply_cut_gradients(
        self,
        beta_name: str,
        cut_grads: Dict[str, np.ndarray]
    ) -> None:
        """
        Applies gradients received from the edge at the cut layer to update
        the device-side subgraph parameters.
        """
        if beta_name not in self._last_inputs:
            raise RuntimeError(
                f"[DeviceTrainer] No cached inputs for beta='{beta_name}'. "
                "Call cache_last_inputs(beta_name, inputs) when you do the forward-to-cut."
            )

        device_net = self._get_device_net(beta_name)
        cached = self._last_inputs[beta_name]

        # Build the same input structure as device_net expects
        if "input_3d" in cached:
            dev_inputs = [cached["input_2d"], cached["input_3d"]]
        else:
            dev_inputs = cached["input_2d"]

        # Forward to get the cut outputs produced by device_net and backprop using upstream grads
        with tf.GradientTape() as tape:
            outputs = device_net(dev_inputs, training=True)

            # Build a "virtual loss" whose gradient w.r.t outputs equals cut_grads.
            if isinstance(outputs, (list, tuple)):
                g2d = tf.convert_to_tensor(cut_grads.get("a2d")) if "a2d" in cut_grads else None
                g3d = tf.convert_to_tensor(cut_grads.get("a3d")) if "a3d" in cut_grads else None
                virtual_loss = 0.0
                if g2d is not None:
                    virtual_loss = virtual_loss + tf.reduce_sum(outputs[0] * g2d)
                if g3d is not None:
                    virtual_loss = virtual_loss + tf.reduce_sum(outputs[1] * g3d)
            else:
                gj = tf.convert_to_tensor(cut_grads.get("joint"))
                virtual_loss = tf.reduce_sum(outputs * gj)

        # Backprop to device parameters and apply update
        vars_ = device_net.trainable_variables
        grads = tape.gradient(virtual_loss, vars_)
        if any(g is None for g in grads):
            grads = [tf.zeros_like(v) if g is None else g for g, v in zip(grads, vars_)]
        self._opt.apply_gradients(zip(grads, vars_))

    # ------------------------------------------------------------
    # >>> NEW METHODS FOR YOUR NEW DESIGN (ADDED BELOW)
    # ------------------------------------------------------------

    def get_device_weights(self) -> List[np.ndarray]:
        """
        Return current device model weights as a list of numpy arrays.
        Used when sending per-device model to Edge for aggregation.
        """
        return [w for w in self.full_model.get_weights()]

    def set_device_weights(self, weights: List[np.ndarray]) -> None:
        """
        Apply updated (aggregated) weights received from Edge/Cloud.
        This enables cloud → edge → device synchronization.
        """
        self.full_model.set_weights(weights)

    def get_model_delta(self, prev_weights: Optional[List[np.ndarray]]) -> float:
        """
        Compute L2 norm of weight difference vs previous version.
        Useful for RL or drift detection.
        """
        if prev_weights is None:
            return 0.0

        curr = self.get_device_weights()
        total = 0.0
        for w1, w0 in zip(curr, prev_weights):
            total += float(np.linalg.norm(w1 - w0))
        return total

