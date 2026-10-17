#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from typing import Dict, Any, List, Set

from src.utils.io_helpers import read_yaml


def _err(msg: str, errors: List[str]) -> None:
    errors.append(msg)


def _require_keys(d: Dict[str, Any], keys: List[str], ctx: str, errors: List[str]) -> None:
    for k in keys:
        if k not in d:
            _err(f"[{ctx}] missing required key: '{k}'", errors)


def _as_set_devices_from_fed(fed_cfg: Dict[str, Any]) -> Set[str]:
    devices = set()
    for e in fed_cfg.get("edges", []):
        for d in e.get("devices", []):
            devices.add(str(d))
    return devices


def _edge_map_from_fed(fed_cfg: Dict[str, Any]) -> Dict[str, str]:
    """device_id -> edge_id (based on federated topology)"""
    m = {}
    for e in fed_cfg.get("edges", []):
        eid = e.get("id")
        for d in e.get("devices", []):
            m[str(d)] = eid
    return m


def validate_split_cfg(split_cfg: Dict[str, Any], errors: List[str]) -> List[str]:
    allowed = split_cfg.get("allowed", [])
    if not isinstance(allowed, list) or not allowed:
        _err("[split.yaml] 'allowed' must be a non-empty list of β names.", errors)
        return allowed
    a_sizes = split_cfg.get("A_size_bytes", {})
    if not isinstance(a_sizes, dict) or not a_sizes:
        _err("[split.yaml] 'A_size_bytes' must be a non-empty mapping β -> bytes.", errors)
    else:
        missing = [b for b in allowed if b not in a_sizes]
        extra = [b for b in a_sizes.keys() if b not in allowed]
        if missing:
            _err(f"[split.yaml] A_size_bytes missing entries for β {missing}", errors)
        if extra:
            _err(f"[split.yaml] A_size_bytes contains unknown β {extra}", errors)
    return allowed


def validate_train_vs_data(train_cfg: Dict[str, Any], data_cfg: Dict[str, Any], errors: List[str]) -> None:
    _require_keys(train_cfg, ["input_shape_2d", "input_shape_3d", "num_classes"], "train.yaml", errors)
    _require_keys(data_cfg, ["feature2d_keys", "feature3d_keys"], "data.yaml", errors)
    if errors:
        return

    ish2 = train_cfg["input_shape_2d"]  # [H, W, C2]
    ish3 = train_cfg["input_shape_3d"]  # [Z, H, W, C3]
    if not (isinstance(ish2, list) and len(ish2) == 3):
        _err("[train.yaml] input_shape_2d must be [H, W, C2]", errors)
    if not (isinstance(ish3, list) and len(ish3) == 4):
        _err("[train.yaml] input_shape_3d must be [Z, H, W, C3]", errors)

    if isinstance(ish2, list):
        C2 = ish2[-1]
        if C2 != len(data_cfg["feature2d_keys"]):
            _err(f"[train.yaml] input_shape_2d last dim C2={C2} "
                 f"does not match len(data.feature2d_keys)={len(data_cfg['feature2d_keys'])}", errors)
    if isinstance(ish3, list):
        C3 = ish3[-1]
        if C3 != len(data_cfg["feature3d_keys"]):
            _err(f"[train.yaml] input_shape_3d last dim C3={C3} "
                 f"does not match len(data.feature3d_keys)={len(data_cfg['feature3d_keys'])}", errors)


def validate_devices_alignment(fed_cfg: Dict[str, Any], el_cfg: Dict[str, Any], errors: List[str]) -> None:
    fed_devices = _as_set_devices_from_fed(fed_cfg)
    el_devices = set((el_cfg.get("devices") or {}).keys())

    if not fed_devices:
        _err("[federated.yaml] edges[].devices is empty or missing.", errors)
    if not el_devices:
        _err("[energy_latency.yaml] devices is empty or missing.", errors)

    missing_in_el = sorted(list(fed_devices - el_devices))
    extra_in_el = sorted(list(el_devices - fed_devices))
    if missing_in_el:
        _err(f"[devices] present in federated.yaml but missing in energy_latency.yaml: {missing_in_el}", errors)
    if extra_in_el:
        _err(f"[devices] present in energy_latency.yaml but not declared in federated.yaml: {extra_in_el}", errors)

    # Edge membership consistency
    dev_to_edge = _edge_map_from_fed(fed_cfg)
    for d, dc in (el_cfg.get("devices") or {}).items():
        want_edge = dev_to_edge.get(d)
        have_edge = dc.get("edge_id")
        if want_edge and have_edge and want_edge != have_edge:
            _err(f"[topology] device '{d}' edge mismatch: federated.yaml={want_edge} vs energy_latency.yaml={have_edge}", errors)


def validate_per_beta_lengths(el_cfg: Dict[str, Any], allowed: List[str], errors: List[str]) -> None:
    L = len(allowed)
    for dev_id, dcfg in (el_cfg.get("devices") or {}).items():
        pb = dcfg.get("per_beta")
        if not isinstance(pb, dict):
            _err(f"[energy_latency.yaml] device '{dev_id}' missing 'per_beta' mapping.", errors)
            continue
        for key in ("Ecomp_J", "Ecomm_J", "Tcomp_s", "Tcomm_s", "M_bytes"):
            arr = pb.get(key)
            if not isinstance(arr, list) or len(arr) != L:
                _err(
                    f"[energy_latency.yaml] device '{dev_id}' per_beta.{key} must be a list of length "
                    f"len(split.allowed)={L}; got {None if arr is None else len(arr)}", errors
                )


def main():
    ap = argparse.ArgumentParser(description="Validate configuration consistency across YAML files.")
    ap.add_argument("--data_cfg", default="configs/data.yaml")
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    ap.add_argument("--fed_cfg", default="configs/federated.yaml")
    ap.add_argument("--split_cfg", default="configs/split.yaml")
    ap.add_argument("--el_cfg", default="configs/energy_latency.yaml")
    ap.add_argument("--strict", action="store_true", help="Treat warnings as errors (currently all are errors).")
    args = ap.parse_args()

    errors: List[str] = []

    # Load YAMLs
    try:
        data_cfg = read_yaml(args.data_cfg)
        train_cfg = read_yaml(args.train_cfg)
        fed_cfg = read_yaml(args.fed_cfg)
        split_cfg = read_yaml(args.split_cfg)
        el_cfg = read_yaml(args.el_cfg)
    except Exception as ex:
        print(f"[FATAL] Failed to read YAMLs: {ex}")
        sys.exit(2)

    # 1) split.yaml
    allowed = validate_split_cfg(split_cfg, errors)

    # 2) train.yaml vs data.yaml shapes
    validate_train_vs_data(train_cfg, data_cfg, errors)

    # 3) device ID alignment & topology consistency
    validate_devices_alignment(fed_cfg, el_cfg, errors)

    # 4) per-β arrays length checks
    if allowed:
        validate_per_beta_lengths(el_cfg, allowed, errors)

    # Summary / exit
    if errors:
        print("\nCONFIG VALIDATION FAILED:")
        for e in errors:
            print(" -", e)
        print(f"\nTotal issues: {len(errors)}")
        sys.exit(1)

    print("All config checks passed ✅")
    sys.exit(0)


if __name__ == "__main__":
    main()

