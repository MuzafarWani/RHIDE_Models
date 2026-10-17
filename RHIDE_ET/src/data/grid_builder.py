# src/data/grid_builder.py
from __future__ import annotations

from typing import Tuple, Dict, Any
import numpy as np


def find_nearest_grid_point(lat: float, lon: float, lats_2d: np.ndarray, lons_2d: np.ndarray) -> Tuple[int, int]:
    """
    Returns (i,j) indices of nearest grid cell to (lat,lon) on a 2D lat/lon mesh.
    """
    # lats_2d/lons_2d shape: (H,W)
    dlat = np.abs(lats_2d - lat)
    dlon = np.abs(lons_2d - lon)
    dist = np.sqrt(dlat**2 + dlon**2)
    idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    return int(idx[0]), int(idx[1])


def extract_window_2d(arr: np.ndarray, t: int, i: int, j: int, half: int, pad_mode: str = "reflect") -> np.ndarray:
    """
    Extracts a 2D spatial window centered at (i,j) for a given timestep t with safe padding.
    arr shape: (T, H, W) or (T, H, W, C)
    Returns shape: (2*half, 2*half) or (2*half, 2*half, C)
    """
    if arr.ndim == 3:
        # (T,H,W)
        slice_2d = arr[t]
    elif arr.ndim == 4:
        # (T,H,W,C)
        slice_2d = arr[t]
    else:
        raise ValueError(f"Unsupported 2D array rank: {arr.ndim}")

    return _crop_with_padding(slice_2d, i, j, half, pad_mode)


def extract_window_3d(arr: np.ndarray, t: int, i: int, j: int, half: int, pad_mode: str = "reflect") -> np.ndarray:
    """
    Extracts a 3D spatial window (with vertical dimension preserved) centered at (i,j) for timestep t.
    arr shape: (T, Z, H, W) or (T, Z, H, W, C)
    Returns shape: (Z, 2*half, 2*half) or (Z, 2*half, 2*half, C)
    """
    if arr.ndim == 4:
        # (T, Z, H, W)
        slice_3d = arr[t]  # (Z,H,W)
        return _crop3d_with_padding(slice_3d, i, j, half, pad_mode)
    elif arr.ndim == 5:
        # (T, Z, H, W, C)
        slice_3d = arr[t]  # (Z,H,W,C)
        return _crop3d_with_padding(slice_3d, i, j, half, pad_mode)
    else:
        raise ValueError(f"Unsupported 3D array rank: {arr.ndim}")


def _crop_with_padding(img: np.ndarray, i: int, j: int, half: int, pad_mode: str) -> np.ndarray:
    """
    Pads a 2D/3D tensor on H,W then crops a (2*half)x(2*half)[xC] window centered at (i,j).
    Accepts (H,W) or (H,W,C).
    """
    H, W = img.shape[:2]
    pad = ((half, half), (half, half)) + ((0, 0),) if img.ndim == 3 else ((half, half), (half, half))
    padded = np.pad(img, pad_width=pad, mode=pad_mode)
    # shift indices because of padding
    ci, cj = i + half, j + half
    return padded[ci - half:ci + half, cj - half:cj + half, ...]


def _crop3d_with_padding(vol: np.ndarray, i: int, j: int, half: int, pad_mode: str) -> np.ndarray:
    """
    Pads along H,W only; keeps Z as-is.
    Accepts (Z,H,W) or (Z,H,W,C).
    """
    Z, H, W = vol.shape[:3]
    if vol.ndim == 3:
        pad = ((0, 0), (half, half), (half, half))
    else:
        pad = ((0, 0), (half, half), (half, half), (0, 0))
    padded = np.pad(vol, pad_width=pad, mode=pad_mode)
    ci, cj = i + half, j + half
    return padded[:, ci - half:ci + half, cj - half:cj + half, ...]

