# src/data/tf_dataset.py
from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    raise ImportError("Please install tensorflow (or tf-cpu) to use tf_dataset pipelines.") from e


def make_tf_datasets(
    X2: np.ndarray,
    X3: np.ndarray,
    y: np.ndarray,
    batch_size: int = 8,
    shuffle: bool = True,
    val_split: float = 0.1,
    test_split: float = 0.1,
    prefetch: int = 2,
    cache: bool = True,
    seed: int = 42
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Creates tf.data.Dataset splits from numpy arrays for two-input model.
    Returns: (train_ds, val_ds, test_ds)
    """
    assert len(X2) == len(X3) == len(y), "Mismatched lengths."

    N = len(y)
    idx = np.arange(N)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_test = int(round(test_split * N))
    n_val = int(round(val_split * N))
    n_train = max(0, N - n_test - n_val)

    idx_train = idx[:n_train]
    idx_val = idx[n_train:n_train + n_val]
    idx_test = idx[n_train + n_val:]

    def make_ds(ix):
        ds = tf.data.Dataset.from_tensor_slices((
            {"input_2d": X2[ix], "input_3d": X3[ix]},
            y[ix]
        ))
        if cache:
            ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=max(32, len(ix)), seed=seed, reshuffle_each_iteration=True)
        ds = ds.batch(batch_size)
        if prefetch:
            ds = ds.prefetch(prefetch)
        return ds

    return make_ds(idx_train), make_ds(idx_val), make_ds(idx_test)

