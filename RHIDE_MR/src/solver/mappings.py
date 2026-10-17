# src/solver/mappings.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class BetaSpec:
    """Definition of a cut point β."""
    name: str
    # Optional anchors (informational; param_feeds may use these to sum layer costs)
    anchor_2d: str | None
    anchor_3d: str | None
    is_joint: bool  # if True, device sends one joint tensor (e.g., "concat" or later)


def default_betas() -> List[BetaSpec]:
    """
    Canonical β set matching the model graph.
    Order matters (index used in solver).
    """
    return [
        BetaSpec("early_prepool",       "2d_conv2",       "3d_conv2",       False),
        BetaSpec("post_pool1",          "2d_pool1",       "3d_pool1",       False),
        BetaSpec("post_conv3",          "2d_conv3",       "3d_conv3",       False),
        BetaSpec("global_pooled",       "2d_global_pool", "3d_global_pool", False),
        BetaSpec("branch_dense",        "2d_dense",       "3d_dense",       False),
        BetaSpec("concat",              None,             None,             True),
        BetaSpec("combined_dense",      None,             None,             True),
        #BetaSpec("output",              None,             None,             True),  # β = L (fully local)
    ]


def beta_index_map(betas: List[BetaSpec]) -> Dict[str, int]:
    return {b.name: i for i, b in enumerate(betas)}

