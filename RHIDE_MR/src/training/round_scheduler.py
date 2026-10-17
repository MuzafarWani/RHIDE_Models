#src/training/round_schedular.py
from __future__ import annotations

from typing import Any, Dict, List
import random


def build_round_batches(
    data_cfg: Dict[str, Any],
    fed_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    priority,
    rng_seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns: device_id -> list[ batch_desc ]
    batch_desc: {"id": str, "edge_id": str, "size": int}

    Heuristic:
      - Start with a base number of batches per device.
      - Add up to (priority * random_int(0, max_extra)) extra batches.
      - Batch 'size' equals train_cfg.batch_size.

    NOTE: This is a scheduler only; it doesn't pick *which* samples.
          The training loop will sample 'size' windows per descriptor.
    """
    random.seed(rng_seed)
    device_batches: Dict[str, List[Dict[str, Any]]] = {}

    base_batch = int(train_cfg.get("device_base_batches", 2))
    max_extra = int(train_cfg.get("device_extra_batches", 2))
    bsize = int(train_cfg.get("batch_size", 8))

    for edge in fed_cfg["edges"]:
        edge_id = edge["id"]
        p = float(edge.get("priority", 1.0))
        device_ids = edge.get("devices", [])

        for d in device_ids:
            n_batches = base_batch + int(p * random.randint(0, max_extra))
            lst: List[Dict[str, Any]] = []
            for _ in range(n_batches):
                lst.append({
                    "id": f"{d}_r{random.randint(0,10**9)}",
                    "edge_id": edge_id,
                    "size": bsize,
                })
            device_batches[d] = lst
    return device_batches

