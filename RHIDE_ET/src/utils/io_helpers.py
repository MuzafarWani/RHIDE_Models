# src/utils/io_helpers.py
from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    import yaml  # pip install pyyaml
except Exception:  # pragma: no cover
    yaml = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, obj: Any, indent: int = 2) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def read_yaml(path: str) -> Dict[str, Any]:
    if yaml is None:
        raise ImportError("pyyaml not installed; cannot read YAML.")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, obj: Dict[str, Any]) -> None:
    if yaml is None:
        raise ImportError("pyyaml not installed; cannot write YAML.")
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def list_device_ids_from_fed(fed_cfg: Dict[str, Any]) -> list[str]:
    """Flatten edges[].devices from federated.yaml into a unique, ordered list."""
    seen = set()
    out = []
    for e in fed_cfg.get("edges", []):
        for d in e.get("devices", []):
            sd = str(d)
            if sd not in seen:
                seen.add(sd)
                out.append(sd)
    return out

def list_device_ids_from_energy_latency(el_cfg: Dict[str, Any]) -> list[str]:
    """List device ids (keys) from energy_latency.yaml:devices."""
    return list((el_cfg.get("devices") or {}).keys())

