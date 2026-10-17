# src/utils/seeding.py
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import tensorflow as tf  # optional
except Exception:  # pragma: no cover
    tf = None


def set_global_seed(seed: int, deterministic_tf: bool = False) -> None:
    """
    Sets seeds for random, numpy, and (optionally) TensorFlow.
    If deterministic_tf=True, also toggles TF deterministic ops where possible.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if tf is not None:
        try:
            tf.keras.utils.set_random_seed(seed)
            if deterministic_tf:
                os.environ["TF_DETERMINISTIC_OPS"] = "1"
                os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        except Exception:
            pass

