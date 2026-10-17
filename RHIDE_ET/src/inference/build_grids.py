# src/inference/build_grids.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.grid_builder import find_nearest_grid_point, extract_window_2d, extract_window_3d


@dataclass(frozen=True)
class InferenceMetaRow:
    row_id: int
    timestep: int
    xlat: float
    xlong: float
    grid_i: int
    grid_j: int


def build_unlabeled_arrays(
    wrf: Dict[str, np.ndarray],
    locations_csv: str,
    window_hw: int = 10,
    feature2d_keys: List[str] | None = None,
    feature3d_keys: List[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[InferenceMetaRow]]:
    """
    Create inference tensors (X2, X3) for locations listed in a CSV:
        columns: timestep, xlat, xlong
    Returns:
        X2: (N, H, W, C2)
        X3: (N, Z, H, W, C3)
        meta_rows: list with per-sample metadata (timestep, lat/lon, grid indices)
    """
    if feature2d_keys is None:
        feature2d_keys = ["RAINC", "RAINNC", "OLR", "SNOW", "SNOWH", "PSFC"]
    if feature3d_keys is None:
        feature3d_keys = ["P", "U", "V", "T", "PB", "QVAPOR"]

    df = pd.read_csv(locations_csv, header=None, names=["timestep", "xlat", "xlong"])

    XLAT = wrf["XLAT"]
    XLONG = wrf["XLONG"]
    # unify to (T,H,W) if static lat/lon
    if XLAT.ndim == 2:
        T = int(max(df["timestep"]) + 1)
        XLAT = np.broadcast_to(XLAT[None, ...], (T,) + XLAT.shape)
        XLONG = np.broadcast_to(XLONG[None, ...], (T,) + XLONG.shape)

    X2_list: List[np.ndarray] = []
    X3_list: List[np.ndarray] = []
    meta_rows: List[InferenceMetaRow] = []

    for ridx, row in df.iterrows():
        t = int(row["timestep"])
        lat, lon = float(row["xlat"]), float(row["xlong"])
        i, j = find_nearest_grid_point(lat, lon, XLAT[t], XLONG[t])

        # 2D features
        feats2 = []
        for k in feature2d_keys:
            feats2.append(extract_window_2d(wrf[k], t, i, j, half=window_hw))
        X2_list.append(np.stack(feats2, axis=-1))

        # 3D features
        feats3 = []
        for k in feature3d_keys:
            feats3.append(extract_window_3d(wrf[k], t, i, j, half=window_hw))
        X3_list.append(np.stack(feats3, axis=-1))

        meta_rows.append(InferenceMetaRow(
            row_id=int(ridx),
            timestep=t,
            xlat=lat,
            xlong=lon,
            grid_i=i,
            grid_j=j
        ))

    X2 = np.asarray(X2_list)
    X3 = np.asarray(X3_list)
    return X2, X3, meta_rows

