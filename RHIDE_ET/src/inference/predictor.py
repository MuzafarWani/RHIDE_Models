# src/inference/predictor.py
from __future__ import annotations

from typing import List, Tuple, Optional, Dict

import numpy as np
import tensorflow as tf

from src.models.split_graph import create_submodels
from src.models.serialization import make_device_edge_from_full


class SplitPredictor:
    """
    Runs inference using either:
      - full model (beta="output"), or
      - split device/edge models (beta in allowed cut names).
    """

    def __init__(
        self,
        full_model: tf.keras.Model,
        input_shape_2d: Tuple[int, int, int],
        input_shape_3d: Tuple[int, int, int, int],
        num_classes: int,
        dtype_activations: tf.dtypes.DType = tf.float32,
    ):
        self.full_model = full_model
        self.input_shape_2d = input_shape_2d
        self.input_shape_3d = input_shape_3d
        self.num_classes = num_classes
        self.dtype_activations = dtype_activations

    def predict(
        self,
        X2: np.ndarray,
        X3: np.ndarray,
        beta_name: str,
        batch_size: int = 32,
        return_proba: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Args:
            X2: (N, H, W, C2)
            X3: (N, Z, H, W, C3)
            beta_name: split choice ("output" -> no split)
            batch_size: inference batch size
            return_proba: if True, also return softmax probabilities

        Returns:
            preds: (N,) int class indices
            proba: (N, num_classes) or None
        """
        if beta_name == "output":
            proba = self.full_model.predict({"input_2d": X2, "input_3d": X3}, batch_size=batch_size, verbose=0)
            preds = np.argmax(proba, axis=-1)
            return preds, proba if return_proba else None

        # Build split submodels for this β and transfer weights
        device_net, edge_net = create_submodels(
            full_model=self.full_model,
            beta_name=beta_name,
            input_shape_2d=self.input_shape_2d,
            input_shape_3d=self.input_shape_3d,
            num_classes=self.num_classes,
        )
        make_device_edge_from_full(self.full_model, device_net, edge_net)

        # Stage 1: device forward to cut
        # Outputs can be one tensor (joint cut) or two tensors (paired cut)
        device_out = device_net.predict({"input_2d": X2, "input_3d": X3}, batch_size=batch_size, verbose=0)

        # Ensure list form
        if isinstance(device_out, list) or isinstance(device_out, tuple):
            cut_payload = [tf.convert_to_tensor(arr).numpy().astype(self.dtype_activations.as_numpy_dtype) for arr in device_out]
        else:
            cut_payload = tf.convert_to_tensor(device_out).numpy().astype(self.dtype_activations.as_numpy_dtype)

        # Stage 2: edge completes inference
        if isinstance(cut_payload, list):
            proba = edge_net.predict(cut_payload, batch_size=batch_size, verbose=0)
        else:
            proba = edge_net.predict(cut_payload, batch_size=batch_size, verbose=0)

        preds = np.argmax(proba, axis=-1)
        return preds, proba if return_proba else None

