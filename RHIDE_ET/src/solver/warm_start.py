# src/solver/warm_start.py
from __future__ import annotations

from typing import Dict, Any, Iterable
import json
import os


# ==========================================================
# Warm-start contract (NEW)
# ----------------------------------------------------------
# Warm start ONLY applies to MILP variables:
#   - z : participation indicator
#   - k : raw-transfer batches
#
# beta is NOT warm-started because it is now
# deterministically selected outside the MILP.
# ==========================================================


def pack_warm_start(
    z: Dict[str, int],
    k: Dict[str, float],
) -> Dict[str, Any]:
    """
    Build a warm-start payload for the solver.

    Expected structure:
      {
        "z": {device_id: 0|1, ...},
        "k": {device_id: float_batches, ...}
      }
    """
    return {
        "z": dict(z),
        "k": dict(k),
    }


# ----------------------------------------------------------
# Persistence helpers
# ----------------------------------------------------------

def save_warm_start(path: str, ws: Dict[str, Any]) -> None:
    """
    Persist warm-start JSON to disk.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(ws, f, indent=2)


def load_warm_start(path: str) -> Dict[str, Any]:
    """
    Load warm-start JSON from disk.
    Raises if file missing or invalid.
    """
    with open(path, "r") as f:
        data = json.load(f)
    validate_warm_start(data)
    return data


# ----------------------------------------------------------
# Safe / convenience APIs
# ----------------------------------------------------------

def try_load_warm_start(path: str) -> Dict[str, Any]:
    """
    Safe loader: returns {} if file not found or invalid.
    Also tolerates legacy warm-start files that include beta_choice.
    """
    try:
        if not os.path.exists(path):
            return {}

        with open(path, "r") as f:
            data = json.load(f)

        # Backward compatibility: drop beta_choice if present
        if isinstance(data, dict) and "beta_choice" in data:
            data = {
                "z": data.get("z", {}),
                "k": data.get("k", {}),
            }

        validate_warm_start(data)
        return data

    except Exception:
        # Keep training resilient
        return {}


def prune_warm_start(
    ws: Dict[str, Any],
    allowed_devices: Iterable[str],
) -> Dict[str, Any]:
    """
    Keep warm-start entries only for currently active devices.
    """
    if not ws:
        return {}

    allowed = set(allowed_devices)

    z = {
        dev: v
        for dev, v in ws.get("z", {}).items()
        if dev in allowed
    }
    k = {
        dev: v
        for dev, v in ws.get("k", {}).items()
        if dev in allowed
    }

    return {"z": z, "k": k}


# ----------------------------------------------------------
# Validation
# ----------------------------------------------------------

def validate_warm_start(ws: Dict[str, Any]) -> None:
    """
    Lightweight schema validation.
    Enforces the NEW warm-start contract.
    """
    if not isinstance(ws, dict):
        raise ValueError("warm start must be a dict")

    # Required keys
    for key in ("z", "k"):
        if key not in ws:
            raise ValueError(f"warm start missing key '{key}'")

    if not isinstance(ws["z"], dict):
        raise ValueError("warm start 'z' must be a dict")
    if not isinstance(ws["k"], dict):
        raise ValueError("warm start 'k' must be a dict")

    # Validate z
    for dev, v in ws["z"].items():
        if not isinstance(dev, str):
            raise ValueError("warm start z keys must be device_id strings")
        if v not in (0, 1):
            raise ValueError(f"warm start z[{dev}] must be 0 or 1")

    # Validate k
    for dev, v in ws["k"].items():
        if not isinstance(dev, str):
            raise ValueError("warm start k keys must be device_id strings")
        try:
            float(v)
        except Exception as e:
            raise ValueError(f"warm start k[{dev}] must be numeric") from e

