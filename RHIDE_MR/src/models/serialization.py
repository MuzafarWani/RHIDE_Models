# src/models/serialization.py
from __future__ import annotations

import os
from typing import Dict, Optional

import tensorflow as tf
from tensorflow.keras import Model


def save_model(model: Model, path: str) -> None:
    """
    Saves a Keras model in SavedModel format at `path`.
    """
    os.makedirs(path, exist_ok=True)
    # `save_format="tf"` is default for directories; keep explicit for clarity.
    model.save(path, include_optimizer=True, save_format="tf")


def load_model(path: str) -> Model:
    """
    Loads a Keras model saved via `save_model`.
    """
    return tf.keras.models.load_model(path, compile=True)


def _layer_map_by_name(model: Optional[Model]) -> Dict[str, tf.keras.layers.Layer]:
    """
    Return a {layer_name: layer} map. Safe if model is None.
    """
    if model is None:
        return {}
    return {l.name: l for l in model.layers}


def transfer_weights_by_name(
    source: Optional[Model],
    target: Optional[Model],
    strict: bool = False
) -> None:
    """
    Copies weights into `target` from `source` by matching layer names.
    If either source or target is None, this is a no-op.
    Layers not present in either side are ignored unless strict=True.
    """
    if source is None or target is None:
        return  # no-op if either side is missing

    src = _layer_map_by_name(source)
    tgt = _layer_map_by_name(target)
    missing = []
    for name, tl in tgt.items():
        sl = src.get(name)
        if sl is None:
            if strict:
                missing.append(name)
            continue
        try:
            tl.set_weights(sl.get_weights())
        except Exception:
            if strict:
                missing.append(name)
    if strict and missing:
        raise ValueError(f"Could not transfer weights for layers: {missing}")


def make_device_edge_from_full(
    full_model: Optional[Model],
    device_net: Optional[Model],
    edge_net: Optional[Model]
) -> None:
    """
    Transfers weights from full_model into device_net and/or edge_net (if present),
    using exact layer name matches. Safe if any argument is None.
    """
    transfer_weights_by_name(full_model, device_net, strict=False)
    transfer_weights_by_name(full_model, edge_net,   strict=False)

