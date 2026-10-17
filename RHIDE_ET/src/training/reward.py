# src/training/reward.py
from __future__ import annotations

from typing import Dict, Any
import math


def compute_rl_reward(
    realized_energy: Dict[str, float],
    realized_latency: Dict[str, float],
    fed_cfg: Dict[str, Any],
    rl_cfg: Dict[str, Any],
) -> Dict[str, float]:
    """
    Returns per-edge reward used by the RL agents controlling O_ℓ.
    reward = - (w_e * mean_energy_edge + w_t * mean_latency_edge)
    """
    w_e = float(rl_cfg.get("w_energy", 1.0))
    w_t = float(rl_cfg.get("w_latency", 1.0))

    # Build device->edge map
    dev2edge = {}
    for edge in fed_cfg["edges"]:
        for d in edge["devices"]:
            dev2edge[d] = edge["id"]

    # Aggregate by edge
    sum_e = {}
    sum_t = {}
    count = {}
    for dev, e in realized_energy.items():
        edge = dev2edge.get(dev)
        if edge is None:
            continue
        sum_e[edge] = sum_e.get(edge, 0.0) + float(e)
        count[edge] = count.get(edge, 0) + 1

    for dev, t in realized_latency.items():
        edge = dev2edge.get(dev)
        if edge is None:
            continue
        sum_t[edge] = sum_t.get(edge, 0.0) + float(t)
        count[edge] = count.get(edge, 0) + 1

    rewards = {}
    for edge in {e["id"] for e in fed_cfg["edges"]}:
        n = max(1, count.get(edge, 1))
        mean_e = sum_e.get(edge, 0.0) / n
        mean_t = sum_t.get(edge, 0.0) / n
        rewards[edge] = - (w_e * mean_e + w_t * mean_t)
    return rewards

