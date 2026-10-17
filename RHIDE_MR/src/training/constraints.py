#src/training/constraints.py
from __future__ import annotations

from typing import Any, Dict


def assert_feasible_decisions(
    decisions: Dict[str, Dict[str, Any]],
    meta: Dict[str, Dict[str, Any]],
    energy_latency_cfg: Dict[str, Any],
    split_cfg: Dict[str, Any],
) -> None:
    """
    Post-solver sanity checks.

    These checks enforce the *solver contract*, not the optimization constraints
    (which are already guaranteed by the MILP).
    """

    for dev, dec in decisions.items():
        # -------------------------------
        # Required keys
        # -------------------------------
        assert "beta" in dec, f"[{dev}] missing 'beta'"
        assert "participate" in dec, f"[{dev}] missing 'participate'"
        assert "k_transfer" in dec, f"[{dev}] missing 'k_transfer'"

        beta = dec["beta"]
        z = int(bool(dec["participate"]))
        k = float(dec["k_transfer"])

        # -------------------------------
        # Basic domain checks
        # -------------------------------
        assert z in (0, 1), f"[{dev}] participate must be bool-like"
        assert k >= 0.0, f"[{dev}] k_transfer must be non-negative"

        B_it = int(meta[dev]["B"])

        # -------------------------------
        # Solver contract checks
        # -------------------------------

        # Case 1: Device participates → must have a valid beta, no raw transfer
        if z == 1:
            assert beta != "", (
                f"[{dev}] participate=1 but beta is empty (solver contract violation)"
            )
            assert k == 0.0, (
                f"[{dev}] participate=1 but k_transfer={k} (must be 0)"
            )

        # Case 2: Device does not participate → beta must be empty
        else:
            assert beta == "", (
                f"[{dev}] participate=0 but beta='{beta}' (must be empty)"
            )
            assert k <= B_it + 1e-6, (
                f"[{dev}] k_transfer={k} exceeds B={B_it}"
            )

        # -------------------------------
        # Allowed beta check (ONLY if beta exists)
        # -------------------------------
        if beta != "":
            allowed = meta[dev].get("allowed_betas", [])
            assert beta in allowed, (
                f"[{dev}] beta='{beta}' not in allowed_betas={allowed}"
            )

