from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Literal
import time
import numpy as np


def _sizeof_arrays(arrs: Dict[str, np.ndarray]) -> int:
    total = 0
    for a in arrs.values():
        total += int(a.nbytes)
    return total


# -----------------------------
# Split-learning: Activations
# -----------------------------

@dataclass
class Activations:
    """
    Payload for device→edge when β < L (split learning).

    Required routing fields:
      - device_id, edge_id: origin and target edge
      - beta: the cut name
      - batch_id: ID to correlate with labels when labels are sent separately.

    Data:
      - arrays: cut activations; for two-branch cuts, use keys {"a2d", "a3d"}; for single-branch, {"joint"}.
      - labels (optional): attach labels directly during training phases for simplicity.
      - checksum (optional): checksum of the pre-compression raw-npz (or any agreed encoding).
      - bytes: payload size actually sent over the wire (filled by the uploader when known).
    """
    device_id: str
    edge_id: str
    beta: str
    batch_id: str
    arrays: Dict[str, np.ndarray]
    labels: Optional[np.ndarray] = None
    checksum: str = ""
    bytes: int = 0
    created_ts: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        if not self.bytes:
            sz = _sizeof_arrays(self.arrays)
            if self.labels is not None:
                sz += int(self.labels.nbytes)
            self.bytes = int(sz)


# -----------------------------
# Optional: Labels as a separate message
# -----------------------------

@dataclass
class Labels:
    """
    Lightweight label message keyed by batch_id to pair with Activations.
    Useful if you want to route labels independently (e.g., privacy/logging policies).
    """
    device_id: str
    edge_id: str
    batch_id: str
    labels: np.ndarray
    bytes: int = 0
    created_ts: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        if not self.bytes:
            self.bytes = int(self.labels.nbytes)


# -----------------------------
# Raw batch transfer (z=0, k>0)
# -----------------------------

@dataclass
class RawBatch:
    """
    Payload for device→edge when z=0 and k>0 (raw-data transfer).
    'arrays' may include both 2D and 3D grids keyed, e.g.:
      {"X2": <np.ndarray>, "X3": <np.ndarray>}
    'labels' is np.ndarray of shape (N,)
    """
    device_id: str
    edge_id: str
    arrays: Dict[str, np.ndarray]
    labels: np.ndarray
    checksum: str = ""
    bytes: int = 0
    created_ts: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        if not self.bytes:
            sz = _sizeof_arrays(self.arrays) + int(self.labels.nbytes)
            self.bytes = int(sz)


# ===========================================================
#   NEW DESIGN: CLEARLY SEPARATED MODEL UPDATES
# ===========================================================

# -----------------------------
# Base model update (kept for backward compatibility)
# -----------------------------

@dataclass
class ModelUpdate:
    """
    Generic model update payload.
    Used mainly for:
      - Edge → Cloud (aggregated edge model)
      - Cloud → Edges (global broadcast)

    'weights' is a list of numpy arrays mirroring Keras layer weights.
    """
    sender_id: str
    receiver_id: str
    weights: List[np.ndarray]
    bytes: int = 0
    created_ts: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        if not self.bytes:
            total = 0
            for w in self.weights:
                total += int(w.nbytes)
            self.bytes = int(total)


# -----------------------------
# 🔹 Device → Edge model update (per-device model)
# -----------------------------

@dataclass
class DeviceModelUpdate(ModelUpdate):
    """
    Payload for:
      Device → Edge

    Each IoT device sends ITS OWN partial/full model to the Edge.
    Edge will later aggregate these into ONE model.
    """
    device_id: str = ""  # explicitly track originating device


# -----------------------------
# 🔹 Edge → Cloud aggregated model update
# -----------------------------

@dataclass
class EdgeModelUpdate(ModelUpdate):
    """
    Payload for:
      Edge → Cloud

    Represents ONE aggregated model from all devices at that edge.
    """
    edge_id: str = ""  # explicitly track originating edge


# -----------------------------
# Cut gradients (edge→device), optional for full split backprop
# -----------------------------

@dataclass
class CutGradients:
    """
    Payload for edge→device with gradients at the cut outputs:
      - For two-branch cuts: {"a2d": grad2d, "a3d": grad3d}
      - For single-branch:   {"joint": grad}
    Include lightweight shape metadata to sanity-check on device.
    """
    device_id: str
    edge_id: str
    beta: str
    grads: Dict[str, np.ndarray]
    shape_meta: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    bytes: int = 0
    created_ts: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        if not self.shape_meta:
            self.shape_meta = {k: tuple(v.shape) for k, v in self.grads.items()}
        if not self.bytes:
            self.bytes = _sizeof_arrays(self.grads)

