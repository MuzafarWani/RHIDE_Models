# src/solver/postprocess.py
from __future__ import annotations

from typing import Dict, Any

from .gurobi_solver import SolverOutput


def summarize_solution(sol: SolverOutput) -> Dict[str, Any]:
    """
    Returns a compact dict useful for logs and RL reward computation.

    In addition to the usual aggregates (energy, latency, etc.),
    this version also propagates per-device *reasons* explaining
    why a device did or did not participate.
    """
    total_e = sum(sol.energy_J.values())
    total_t = sum(sol.latency_s.values())
    total_u = sum(sol.upload_bytes.values())
    active = sum(sol.z.values())
    total_batches = sum(sol.batches_used.values())

    # Be defensive: older SolverOutput objects may not have "reasons"
    reasons = getattr(sol, "reasons", {})

    return {
        "status": sol.status,
        "obj": sol.obj_value,
        "active_devices": active,
        "total_energy_J": total_e,
        "total_latency_s": total_t,
        "total_upload_bytes": total_u,
        "total_batches": total_batches,

        # Per-device decisions
        "beta_choice": sol.beta_choice,
        "z": sol.z,
        "k": sol.k,
        "batches_used": sol.batches_used,
        "energy_J": sol.energy_J,
        "latency_s": sol.latency_s,
        "upload_bytes": sol.upload_bytes,

        ## Tag 1
        # e.g., "TRAIN", "NO_FEASIBLE_SPLIT", "DROPPED_BY_EDGE_CAPACITY", etc.
        "reasons": reasons,
    }

