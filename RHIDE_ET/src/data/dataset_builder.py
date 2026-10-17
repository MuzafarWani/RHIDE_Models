#dataset_builder.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# We expect WRFReader.read_all() to produce a dict of numpy arrays keyed by var names.
# Examples:
#   wrf["RAINC"]   -> shape (T, Y, X)
#   wrf["T"]       -> shape (T, Z, Y, X)
#   wrf["XLAT"]    -> shape (Y, X) or (T, Y, X)
#   wrf["XLONG"]   -> shape (Y, X) or (T, Y, X)

def _ensure_latlon_2d(wrf: Dict[str, np.ndarray], t_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return 2D XLAT, XLONG arrays of shape (Y, X). If time-dim is present, select at t_idx."""
    lat = wrf.get("XLAT")
    lon = wrf.get("XLONG")
    if lat is None or lon is None:
        raise ValueError("XLAT/XLONG not found in WRF reader output.")
    if lat.ndim == 2:
        return lat, lon
    if lat.ndim == 3:
        return lat[t_idx], lon[t_idx]
    raise ValueError(f"Unexpected XLAT/XLONG dims: {lat.shape}, {lon.shape}")

def _closest_ij(lat2d: np.ndarray, lon2d: np.ndarray, lat: float, lon: float) -> Tuple[int, int]:
    """Find nearest grid index (i=j_row, j=col) to given lat/lon."""
    # Handle possible wrap-around with lon in [-180,180] vs [0,360]
    lons = lon2d
    if lons.max() > 180.0 and lon < 0:
        lon = (lon + 360.0)
    if lons.max() <= 180.0 and lon > 180.0:
        lon = lon - 360.0
    d2 = (lat2d - lat) ** 2 + (lons - lon) ** 2
    idx = int(d2.argmin())
    i = idx // lat2d.shape[1]
    j = idx % lat2d.shape[1]
    return i, j

def _crop_hw(arr: np.ndarray, i: int, j: int, hw: int) -> np.ndarray:
    """Crop a (Y,X) array to window centered at (i,j) with half-width=hw. Pads by clamping edges."""
    Y, X = arr.shape[-2], arr.shape[-1]
    i0 = max(0, i - hw); i1 = min(Y, i + hw + 1)
    j0 = max(0, j - hw); j1 = min(X, j + hw + 1)
    crop = arr[..., i0:i1, j0:j1]
    # If near borders, pad by edge values to reach full window size
    target_h = 2 * hw + 1
    pad_h = target_h - crop.shape[-2]
    pad_w = target_h - crop.shape[-1]
    if pad_h > 0 or pad_w > 0:
        pad_before_h = max(0, hw - (i - i0))
        pad_after_h = max(0, target_h - crop.shape[-2] - pad_before_h)
        pad_before_w = max(0, hw - (j - j0))
        pad_after_w = max(0, target_h - crop.shape[-1] - pad_before_w)
        pad_spec = [(0,0)] * (crop.ndim - 2) + [(pad_before_h, pad_after_h), (pad_before_w, pad_after_w)]
        crop = np.pad(crop, pad_spec, mode="edge")
    return crop

def _stack_2d(chs: List[np.ndarray]) -> np.ndarray:
    # Each element is (H, W); return (H, W, C)
    return np.stack(chs, axis=-1).astype(np.float32)

def _stack_3d(chs: List[np.ndarray]) -> np.ndarray:
    # Each element is (Z, H, W); return (Z, H, W, C)
    Zs = {c.shape[0] for c in chs}
    if len(Zs) != 1:
        raise ValueError(f"3D feature channels have different Z: {Zs}")
    return np.stack(chs, axis=-1).astype(np.float32)

def _encode_labels(labels: List[str], classes: List[str], label_encoder) -> np.ndarray:
    if label_encoder is not None:
        return np.asarray(label_encoder.transform(labels), dtype=np.int64)
    # Fallback: map via classes list
    cls2i = {c: i for i, c in enumerate(classes)}
    return np.asarray([cls2i.get(l, -1) for l in labels], dtype=np.int64)

def build_labeled_arrays(
    *,
    wrf: Dict[str, np.ndarray],
    locations_csv: str,
    window_hw: int,
    feature2d_keys: List[str],
    feature3d_keys: List[str],
    classes: List[str],
    label_encoder,   # SimpleLabelEncoder or None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """
    Build (X2, X3, y, meta) from a WRF variable dict and a label CSV.
      X2: (N, H, W, C2) for 2D channels
      X3: (N, H, W, Z, C3) for 3D channels  <-- NOTE: final layout after transpose
      y:  (N,)
      meta: list of dicts for traceability: {"t": t, "i": i, "j": j}
    CSV columns required: timestep, xlat, xlong, label
    - timestep should be 0-based. If your file is 1-based, subtract 1 before training.
    """
    df = pd.read_csv(locations_csv)
    # Be generous with column names (lowercase matching)
    cols = {c.lower(): c for c in df.columns}
    for need in ("timestep", "xlat", "xlong", "label"):
        if need not in cols:
            raise ValueError(f"Label CSV missing required column '{need}'. Found: {list(df.columns)}")

    t_col = cols["timestep"]; lat_col = cols["xlat"]; lon_col = cols["xlong"]; lab_col = cols["label"]

    # Shapes
    # Determine T, Y, X, and Z (if any)
    # Use first 2D var to infer T,Y,X
    T = None; Y = None; X = None
    for k in feature2d_keys:
        if k in wrf:
            v = wrf[k]
            if v.ndim != 3:
                raise ValueError(f"2D var {k} must be (T,Y,X); got {v.shape}")
            T, Y, X = v.shape
            break
    if T is None:
        # if no 2D, look into a 3D var
        for k in feature3d_keys:
            if k in wrf:
                v = wrf[k]
                if v.ndim != 4:
                    raise ValueError(f"3D var {k} must be (T,Z,Y,X); got {v.shape}")
                T, Z, Y, X = v.shape
                break
    if T is None:
        raise ValueError("Cannot infer (T,Y,X) from wrf dict; no known 2D/3D variables present.")

    # Collect samples
    X2_list: List[np.ndarray] = []
    X3_list: List[np.ndarray] = []
    y_list:  List[str] = []
    meta:    List[dict] = []

    for _, row in df.iterrows():
        t = int(row[t_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])
        lab = str(row[lab_col])

        if not (0 <= t < T):
            # Skip out-of-range sample
            continue

        lat2d, lon2d = _ensure_latlon_2d(wrf, t)
        i, j = _closest_ij(lat2d, lon2d, lat, lon)

        # 2D features
        two_d_channels = []
        for k in feature2d_keys:
            arr = wrf.get(k)
            if arr is None:
                continue
            if arr.ndim != 3:
                raise ValueError(f"2D var {k} expected shape (T,Y,X); got {arr.shape}")
            tile = _crop_hw(arr[t], i, j, window_hw)  # (H,W)
            two_d_channels.append(tile)
        X2 = _stack_2d(two_d_channels) if two_d_channels else np.zeros((2*window_hw+1, 2*window_hw+1, 0), dtype=np.float32)

        # 3D features
        three_d_channels = []
        z_depth = None
        for k in feature3d_keys:
            arr = wrf.get(k)
            if arr is None:
                continue
            if arr.ndim != 4:
                raise ValueError(f"3D var {k} expected shape (T,Z,Y,X); got {arr.shape}")
            _, Z, _, _ = arr.shape
            if z_depth is None:
                z_depth = Z
            elif Z != z_depth:
                raise ValueError(f"Inconsistent Z among 3D vars: expected {z_depth}, got {Z} for {k}")
            tile3d = _crop_hw(arr[t], i, j, window_hw)  # (Z,H,W)
            three_d_channels.append(tile3d)
        if three_d_channels:
            X3 = _stack_3d(three_d_channels)  # (Z,H,W,C3)
        else:
            X3 = np.zeros((0, 2*window_hw+1, 2*window_hw+1, 0), dtype=np.float32)

        X2_list.append(X2)
        X3_list.append(X3)
        y_list.append(lab)
        meta.append({"t": t, "i": i, "j": j})

    if not X2_list and not X3_list:
        raise ValueError("No samples built from the label CSV; check timestep range and coordinates.")

    # Stack to arrays
    X2_arr = np.stack(X2_list, axis=0)  # (N,H,W,C2)
    # For X3, ensure consistent Z,H,W across samples
    if any(x.shape != X3_list[0].shape for x in X3_list):
        raise ValueError("3D crops differ in shape across samples (near borders?). Check padding logic.")
    if X3_list and X3_list[0].size > 0:
        X3_arr = np.stack(X3_list, axis=0)  # (N,Z,H,W,C3)
        # ---- Transpose to (N, H, W, Z, C3) to match model IO ----
        X3_arr = np.transpose(X3_arr, (0, 2, 3, 1, 4))
    else:
        X3_arr = np.zeros((len(X2_arr), 0, 2*window_hw+1, 2*window_hw+1, 0), dtype=np.float32)

    y_arr = _encode_labels(y_list, classes, label_encoder)

    return X2_arr, X3_arr, y_arr, meta
    
def build_unlabeled_arrays(
    *,
    wrf: Dict[str, np.ndarray],
    locations_csv: str,
    window_hw: int,
    feature2d_keys: List[str],
    feature3d_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    Build (X2, X3, meta) from a WRF variable dict and an inference CSV.
      X2: (N, H, W, C2)
      X3: (N, H, W, Z, C3)
      meta: list of dicts for traceability (timestep, lat, lon, grid indices)

    CSV columns required:
      timestep, XLAT, XLONG
    """

    df = pd.read_csv(locations_csv)

    # Be generous with column names
    cols = {c.lower(): c for c in df.columns}
    for need in ("timestep", "xlat", "xlong"):
        if need not in cols:
            raise ValueError(
                f"Inference CSV missing required column '{need}'. Found: {list(df.columns)}"
            )

    t_col = cols["timestep"]
    lat_col = cols["xlat"]
    lon_col = cols["xlong"]

    # Infer T,Y,X from available variables
    T = None
    for k in feature2d_keys:
        if k in wrf:
            v = wrf[k]
            if v.ndim != 3:
                raise ValueError(f"2D var {k} must be (T,Y,X); got {v.shape}")
            T, _, _ = v.shape
            break

    if T is None:
        for k in feature3d_keys:
            if k in wrf:
                v = wrf[k]
                if v.ndim != 4:
                    raise ValueError(f"3D var {k} must be (T,Z,Y,X); got {v.shape}")
                T, _, _, _ = v.shape
                break

    if T is None:
        raise ValueError("Cannot infer time dimension T from WRF data.")

    X2_list: List[np.ndarray] = []
    X3_list: List[np.ndarray] = []
    meta: List[dict] = []

    for _, row in df.iterrows():
        t = int(row[t_col])
        lat = float(row[lat_col])
        lon = float(row[lon_col])

        if not (0 <= t < T):
            continue

        lat2d, lon2d = _ensure_latlon_2d(wrf, t)
        i, j = _closest_ij(lat2d, lon2d, lat, lon)

        # -----------------
        # 2D FEATURES
        # -----------------
        two_d_channels = []
        for k in feature2d_keys:
            arr = wrf.get(k)
            if arr is None:
                continue
            tile = _crop_hw(arr[t], i, j, window_hw)
            two_d_channels.append(tile)

        X2 = (
            _stack_2d(two_d_channels)
            if two_d_channels
            else np.zeros((2 * window_hw + 1, 2 * window_hw + 1, 0), dtype=np.float32)
        )

        # -----------------
        # 3D FEATURES
        # -----------------
        three_d_channels = []
        z_depth = None

        for k in feature3d_keys:
            arr = wrf.get(k)
            if arr is None:
                continue
            _, Z, _, _ = arr.shape
            if z_depth is None:
                z_depth = Z
            elif Z != z_depth:
                raise ValueError("Inconsistent Z dimension across 3D variables.")

            tile3d = _crop_hw(arr[t], i, j, window_hw)
            three_d_channels.append(tile3d)

        if three_d_channels:
            X3 = _stack_3d(three_d_channels)  # (Z,H,W,C)
        else:
            X3 = np.zeros((0, 2 * window_hw + 1, 2 * window_hw + 1, 0), dtype=np.float32)

        X2_list.append(X2)
        X3_list.append(X3)
        meta.append({
            "timestep": t,
            "XLAT": lat,
            "XLONG": lon,
            "i": i,
            "j": j,
        })

    if not X2_list:
        raise ValueError("No inference samples built from CSV.")

    # Stack arrays
    X2_arr = np.stack(X2_list, axis=0)

    if X3_list and X3_list[0].size > 0:
        X3_arr = np.stack(X3_list, axis=0)        # (N,Z,H,W,C)
        X3_arr = np.transpose(X3_arr, (0, 2, 3, 1, 4))  # → (N,H,W,Z,C)
    else:
        X3_arr = np.zeros((len(X2_arr), 0, 2 * window_hw + 1, 2 * window_hw + 1, 0), dtype=np.float32)

    return X2_arr, X3_arr, meta
    

