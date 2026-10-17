# src/data/normalization.py
from __future__ import annotations

import os
import json
from typing import Dict, Any, Tuple

import numpy as np


def compute_channel_stats(
    X2: np.ndarray, X3: np.ndarray, eps: float = 1e-6
) -> Dict[str, Any]:
    """
    Computes per-channel mean/std for 2D and 3D branches.
    X2 shape: (N, H, W, C2)
    X3 shape: (N, Z, H, W, C3)
    """
    assert X2.ndim == 4 and X3.ndim == 5, "Bad shapes for normalization stats."

    stats = {}
    # 2D
    C2 = X2.shape[-1]
    means2 = []
    stds2 = []
    for c in range(C2):
        x = X2[..., c]
        means2.append(float(x.mean()))
        stds2.append(float(x.std() + eps))
    stats["2d"] = {"mean": means2, "std": stds2}

    # 3D
    C3 = X3.shape[-1]
    means3 = []
    stds3 = []
    for c in range(C3):
        x = X3[..., c]
        means3.append(float(x.mean()))
        stds3.append(float(x.std() + eps))
    stats["3d"] = {"mean": means3, "std": stds3}

    return stats


def apply_channel_stats(X2: np.ndarray, X3: np.ndarray, stats: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes inputs in-place-like (returns new arrays) using provided stats.
    """
    X2n = X2.astype(np.float32, copy=True)
    X3n = X3.astype(np.float32, copy=True)

    m2 = np.array(stats["2d"]["mean"], dtype=np.float32)
    s2 = np.array(stats["2d"]["std"], dtype=np.float32)
    m3 = np.array(stats["3d"]["mean"], dtype=np.float32)
    s3 = np.array(stats["3d"]["std"], dtype=np.float32)

    X2n = (X2n - m2) / s2
    X3n = (X3n - m3) / s3
    return X2n, X3n


def save_stats(stats: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

