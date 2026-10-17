# src/federated/device_client/splitter.py
from __future__ import annotations

from typing import Tuple, Dict, Any, Optional

import numpy as np
import tensorflow as tf

from src.models.split_graph import create_submodels
from src.models.serialization import make_device_edge_from_full


class Splitter:
    """
    Builds and caches device-side submodels for a chosen split β.
    Runs forward pass up to the cut and returns activation tensors.

    IMPORTANT CONTRACT:
      - beta_name == ""  → splitter MUST NOT be called
      - beta_name != ""  → splitter runs forward_to_cut
    """

    def __init__(
        self,
        full_model: tf.keras.Model,
        input_shape_2d: Tuple[int, int, int],
        input_shape_3d: Tuple[int, int, int, int],
        num_classes: int,
        cast_dtype: Optional[np.dtype] = None,   # e.g., np.float16 to reduce payload size
    ):
        self.full_model = full_model
        self.input_shape_2d = input_shape_2d
        self.input_shape_3d = input_shape_3d
        self.num_classes = num_classes
        self.cast_dtype = cast_dtype

        # Cache device-side submodels per β
        self._cache: Dict[str, tf.keras.Model] = {}

    # --------------------------------------------------
    # Internal: build or fetch device-side submodel
    # --------------------------------------------------
    def _get_device_net(self, beta_name: str) -> tf.keras.Model:
        if not beta_name:
            raise ValueError(
                "Splitter called with empty beta_name. "
                "This indicates a solver/execution contract violation."
            )

        if beta_name in self._cache:
            return self._cache[beta_name]

        device_net, edge_net = create_submodels(
            full_model=self.full_model,
            beta_name=beta_name,
            input_shape_2d=self.input_shape_2d,
            input_shape_3d=self.input_shape_3d,
            num_classes=self.num_classes,
        )

        # Copy weights from full model → device submodel
        # (edge_net is None here; edge will rebuild its own copy)
        make_device_edge_from_full(
            full_model=self.full_model,
            device_net=device_net,
            edge_net=None,
        )

        self._cache[beta_name] = device_net
        return device_net

    # --------------------------------------------------
    # Public: forward pass up to split β
    # --------------------------------------------------
    def forward_to_cut(
        self,
        beta_name: str,
        X2_batch: np.ndarray,
        X3_batch: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Runs forward pass on the device up to the chosen split β.

        Returns:
          - {"a2d": ..., "a3d": ...} for paired cuts
          - {"joint": ...}          for joint cuts

        NOTE:
          - This function MUST NOT be called if beta_name == ""
          - Raw-transfer-only devices should bypass the splitter entirely
        """

        if not beta_name:
            raise ValueError(
                "forward_to_cut called with empty beta_name. "
                "Raw-transfer-only devices must bypass the Splitter."
            )

        device_net = self._get_device_net(beta_name)

        # Forward pass
        outputs = device_net.predict(
            {"input_2d": X2_batch, "input_3d": X3_batch},
            verbose=0,
        )

        # Optional dtype cast to reduce payload size
        def _maybe_cast(arr: np.ndarray) -> np.ndarray:
            if self.cast_dtype is not None:
                return arr.astype(self.cast_dtype, copy=False)
            return arr

        # Handle multi-output vs single-output cuts
        if isinstance(outputs, (list, tuple)):
            # Paired outputs (e.g., separate 2D and 3D activations)
            return {
                "a2d": _maybe_cast(outputs[0]),
                "a3d": _maybe_cast(outputs[1]),
            }
        else:
            # Joint representation
            return {
                "joint": _maybe_cast(outputs),
            }

