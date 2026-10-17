# src/inference/postprocess.py
from __future__ import annotations

import os
import csv
from typing import List, Optional, Sequence

import numpy as np

from src.inference.build_grids import InferenceMetaRow
from src.data.encoders import SimpleLabelEncoder


def attach_meta_and_save_csv(
    out_path: str,
    meta_rows: List[InferenceMetaRow],
    preds: Sequence[int],
    proba: Optional[np.ndarray],
    encoder: Optional[SimpleLabelEncoder] = None,
    include_proba: bool = True,
) -> None:
    """
    Write predictions + metadata to CSV.
    Columns:
      row_id, timestep, xlat, xlong, grid_i, grid_j, pred_idx, pred_label, [p0, p1, ...]
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["row_id", "timestep", "xlat", "xlong", "grid_i", "grid_j", "pred_idx", "pred_label"]
        if include_proba and proba is not None:
            header += [f"p_{i}" for i in range(proba.shape[1])]
        writer.writerow(header)

        for mr, p_idx in zip(meta_rows, preds):
            label = encoder.inverse_transform([int(p_idx)])[0] if encoder is not None else str(p_idx)
            row = [mr.row_id, mr.timestep, f"{mr.xlat:.6f}", f"{mr.xlong:.6f}", mr.grid_i, mr.grid_j, int(p_idx), label]
            if include_proba and proba is not None:
                row += [f"{float(x):.6f}" for x in proba[int(mr.row_id)]]
            writer.writerow(row)

