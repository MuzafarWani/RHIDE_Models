#!/usr/bin/env python3
# scripts/profile_energy_latency.py
from __future__ import annotations

import argparse
import numpy as np

from src.utils.io_helpers import read_yaml
from src.solver.mappings import default_betas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--el_cfg", default="configs/energy_latency.yaml")
    ap.add_argument("--split_cfg", default="configs/split.yaml")
    ap.add_argument("--device", default=None, help="device id to profile; default = all")
    args = ap.parse_args()

    el = read_yaml(args.el_cfg)
    sp = read_yaml(args.split_cfg)
    betas = [b for b in sp["allowed"]]

    def prof_one(dev_id, dcfg):
        print(f"\n[profile] device={dev_id} uplink={dcfg['uplink_rate_Bps']} B/s Emax={dcfg['Emax_round_J']}J Umax={dcfg['Umax_bytes']}B Tmax={dcfg['Tmax_s']}s")
        A_size = sp["A_size_bytes"]
        pb = dcfg["per_beta"]
        for name, ecomp, ecomm, tc, tm, m in zip(
            betas, pb["Ecomp_J"], pb["Ecomm_J"], pb["Tcomp_s"], pb["Tcomm_s"], pb["M_bytes"]
        ):
            U = A_size.get(name, 0)
            print(f"  β={name:15s}  E_comp={ecomp:.3f}J  E_comm={ecomm:.3f}J  U={U/1e3:.1f}KB  "
                  f"T_comp={tc:.3f}s  T_comm={tm:.3f}s  M={m/1e6:.1f}MB")
        ttr = dcfg["RBS_bytes"] / max(1.0, dcfg["uplink_rate_Bps"])
        etr = dcfg["RBS_bytes"] * dcfg["comm_cost_per_byte_J"]
        print(f"  transfer_per_batch:  T={ttr:.4f}s  E={etr:.6f}J  (RBS={dcfg['RBS_bytes']}B)")

    if args.device:
        prof_one(args.device, el["devices"][args.device])
    else:
        for dev_id, dcfg in el["devices"].items():
            prof_one(dev_id, dcfg)

    print("\n[profile] done.")


if __name__ == "__main__":
    main()

