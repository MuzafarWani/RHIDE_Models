#!/usr/bin/env python3
"""
Generate scenario folders for XEventNet split federated learning using
the resource levels and scenario matrix reported in the RHiDE paper.

Usage:
python3 generate_yamls_xeventnet_paper.py --scenarios scenarios_xeventnet.csv --outdir outxev
"""

import argparse, csv, math, random
from pathlib import Path
from typing import Any, Dict, List

# ============================================================
# SPLITS (unchanged)
# ============================================================

SPLITS = [
    "early_prepool",
    "post_pool1",
    "post_conv3",
    "global_pooled",
    "branch_dense",
    "concat",
    "combined_dense",
]

# ============================================================
# MEASURED VALUES (unchanged)
# ============================================================

LAYER_TABLE = {
    "early_prepool":  (590401765, 6.15895699, 0.010),
    "post_pool1":     (594755452, 8.33915411, 0.015),
    "post_conv3":     (610103459, 10.64529441, 0.020),
    "global_pooled":  (616984215, 13.35524961, 0.030),
    "branch_dense":   (624603168, 15.86877712, 0.035),
    "concat":         (627855851, 17.13266731, 0.040),
    "combined_dense": (633499287, 18.43577127, 0.050),
}

TEDGE_PER_BETA = [0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.05]

# Baseline communication energy and time values supplied for XEventNet.
Ecomm_BASE = [
    0.4194304, 0.1572864, 0.3145728,
    0.0004096, 0.0004096, 0.0004096, 0.0002048,
]
Tcomm_BASE = [
    16.868622, 16.993013, 17.431527,
    17.628120, 17.845805, 17.938739, 18.099980,
]
COMM_J_PER_BYTE = 5.0e-9


# ============================================================
# HELPERS
# ============================================================

def q(s: str) -> str:
    return f'"{s}"'


def emit_list(lst: List[float]) -> str:
    return "[" + ", ".join(f"{x}" for x in lst) + "]"


# ============================================================
# DEVICE BUILD
# ============================================================

def build_devices(
    num_devices,
    num_edges,
    *,
    Emax_round_J,
    Erem_low,
    Erem_high,
    uplink_low,
    uplink_high,
    Tmax_low,
    Tmax_high,
    mem_max_bytes=None,
    mem_low=None,
    mem_high=None,
    heterogeneous=False,
    Umax_bytes,
    harvest_J_per_round,
    Tsync_s,
):
    dpe = max(1, math.ceil(num_devices / num_edges))
    devs = []

    # Homogeneous scenarios use one shared resource realization across all
    # devices. Heterogeneous scenarios sample device resources independently.
    shared_Erem = random.uniform(Erem_low, Erem_high)
    shared_mem = (
        random.randint(mem_low, mem_high)
        if mem_low is not None and mem_high is not None
        else int(mem_max_bytes)
    )
    shared_uplink = random.randint(uplink_low, uplink_high)
    shared_Tmax = random.uniform(Tmax_low, Tmax_high)

    activ = [LAYER_TABLE[b][0] for b in SPLITS]
    Ecomp = [LAYER_TABLE[b][1] for b in SPLITS]
    Tcomp = [LAYER_TABLE[b][2] for b in SPLITS]

    for i in range(num_devices):
        edge_idx = min(i // dpe, num_edges - 1)

        if heterogeneous:
            uplink = random.randint(uplink_low, uplink_high)
            Erem = random.uniform(Erem_low, Erem_high)
            mem_bytes = (
                random.randint(mem_low, mem_high)
                if mem_low is not None and mem_high is not None
                else int(mem_max_bytes)
            )
            Tmax = random.uniform(Tmax_low, Tmax_high)
        else:
            uplink = shared_uplink
            Erem = shared_Erem
            mem_bytes = shared_mem
            Tmax = shared_Tmax

        # ---------------------------------------------------------
        # PER-DEVICE SPLIT METRICS
        # ---------------------------------------------------------
        # Keep baseline arrays unchanged for homogeneous scenarios.
        # For heterogeneous scenarios (S7-S11), add small independent
        # variations to every split entry, including timing, communication
        # energy, and edge execution time.
        if heterogeneous:
            Tcomp_device = [
                round(t * random.uniform(1.00, 1.08), 6)
                for t in Tcomp
            ]
            Tcomm_device = [
                round(t * random.uniform(0.97, 1.05), 6)
                for t in Tcomm_BASE
            ]
            Ecomp_device = [
                round(e * random.uniform(0.98, 1.04), 8)
                for e in Ecomp
            ]
            Ecomm_device = [
                round(e * random.uniform(0.97, 1.03), 9)
                for e in Ecomm_BASE
            ]
            Tedge_device = [
                round(t * random.uniform(0.97, 1.05), 6)
                for t in TEDGE_PER_BETA
            ]
            device_Umax_bytes = max(
                1, int(Umax_bytes * random.uniform(0.95, 1.05))
            )
        else:
            Tcomp_device = Tcomp[:]
            Tcomm_device = Tcomm_BASE[:]
            Ecomp_device = Ecomp[:]
            Ecomm_device = Ecomm_BASE[:]
            Tedge_device = TEDGE_PER_BETA[:]
            device_Umax_bytes = Umax_bytes

        devs.append({
            "idx": i,
            "edge_id": f"e{edge_idx}",
            "mem_max_bytes": int(mem_bytes),
            "uplink_rate_Bps": uplink,
            "comm_cost_per_byte_J": COMM_J_PER_BYTE,
            "Emax_round_J": Emax_round_J,
            "Erem_J": round(Erem, 6),
            "harvest_J_per_round": harvest_J_per_round,
            "Umax_bytes": device_Umax_bytes,
            "Tmax_s": round(Tmax, 6),
            "RBS_bytes": activ[0],
            "Tedge_s": Tedge_device,
            "Tsync_s": Tsync_s,
            "per_beta": {
                "Ecomp_J": Ecomp_device,
                "Ecomm_J": Ecomm_device,
                "Tcomp_s": Tcomp_device,
                "Tcomm_s": Tcomm_device,
                "M_bytes": activ,
            },
        })

    return devs


# ============================================================
# ENERGY LATENCY YAML (original structure and parameter names)
# ============================================================

def emit_energy_latency(devs):
    L = ["devices:\n"]

    for d in devs:
        i = d["idx"]
        pb = d["per_beta"]

        L += [
            f"  d{i}:",
            f"    edge_id: {q(d['edge_id'])}",
            f"    mem_max_bytes: {d['mem_max_bytes']}",
            f"    uplink_rate_Bps: {d['uplink_rate_Bps']}",
            f"    comm_cost_per_byte_J: {d['comm_cost_per_byte_J']}",
            f"    Emax_round_J: {d['Emax_round_J']}",
            f"    Erem_J: {d['Erem_J']}",
            f"    harvest_J_per_round: {d['harvest_J_per_round']}",
            f"    Umax_bytes: {d['Umax_bytes']}",
            f"    Tmax_s: {d['Tmax_s']}",
            f"    RBS_bytes: {d['RBS_bytes']}",
            f"    Tedge_s: {emit_list(d['Tedge_s'])}",
            f"    Tsync_s: {d['Tsync_s']}",
            "",
            f"    per_beta:",
            f"      Ecomp_J: {emit_list(pb['Ecomp_J'])}",
            f"      Ecomm_J: {emit_list(pb['Ecomm_J'])}",
            f"      Tcomp_s: {emit_list(pb['Tcomp_s'])}",
            f"      Tcomm_s: {emit_list(pb['Tcomm_s'])}",
            f"      M_bytes: {emit_list(pb['M_bytes'])}",
            "",
        ]

    return "\n".join(L)


# ============================================================
# FEDERATED YAML (original structure and parameter names)
# ============================================================

def emit_federated(
    num_devices, num_edges,
    ram_edge_s=None, edge_energy_budget=None,
    *, edge_mem_low_bytes=None, edge_mem_high_bytes=None,
    edge_energy_low=None, edge_energy_high=None,
    heterogeneous=False
):
    dpe = max(1, math.ceil(num_devices / num_edges))
    lines = ["edges:"]

    # Homogeneous scenarios share the same edge resource values. Heterogeneous
    # scenarios draw each edge's resources independently within paper ranges.
    shared_mem = (
        random.randint(edge_mem_low_bytes, edge_mem_high_bytes)
        if edge_mem_low_bytes is not None and edge_mem_high_bytes is not None
        else int(ram_edge_s)
    )
    shared_energy = (
        random.uniform(edge_energy_low, edge_energy_high)
        if edge_energy_low is not None and edge_energy_high is not None
        else float(edge_energy_budget)
    )

    for e in range(num_edges):
        devs = [f"d{i}" for i in range(e * dpe, min((e + 1) * dpe, num_devices))]
        if not devs:
            continue

        if heterogeneous:
            edge_mem = (
                random.randint(edge_mem_low_bytes, edge_mem_high_bytes)
                if edge_mem_low_bytes is not None and edge_mem_high_bytes is not None
                else int(ram_edge_s)
            )
            edge_energy = (
                random.uniform(edge_energy_low, edge_energy_high)
                if edge_energy_low is not None and edge_energy_high is not None
                else float(edge_energy_budget)
            )
        else:
            edge_mem = shared_mem
            edge_energy = shared_energy

        lines += [
            f"  - id: {q(f'e{e}')}",
            f"    priority: {2.0 if e == 0 else 1.0}",
            f"    ram_edge_s: {edge_mem}",
            f"    energy_budget_per_round: {round(edge_energy, 6)}",
            f"    devices: [{', '.join(q(d) for d in devs)}]",
        ]

    lines += ["", "devices:"]

    for i in range(num_devices):
        edge = f"e{min(i // dpe, num_edges - 1)}"
        lines += [
            f"  d{i}:",
            f"    edge: {q(edge)}",
        ]

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--outdir", default="outxev")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.scenarios, newline="") as f:
        rdr = csv.DictReader(f)

        for row in rdr:
            random.seed(int(row.get("seed", 1)))

            num_devices = int(row["devices"])
            num_edges = int(row["edges"])
            heterogeneous = row.get("heterogeneous", "false").strip().lower() in ("1", "true", "yes")

            mem_low = int(row["mem_low_bytes"])
            mem_high = int(row["mem_high_bytes"])
            edge_mem_low = int(row["edge_mem_low_bytes"])
            edge_mem_high = int(row["edge_mem_high_bytes"])

            devs = build_devices(
                num_devices,
                num_edges,
                Emax_round_J=float(row["Emax_round_J"]),
                Erem_low=float(row["Erem_low"]),
                Erem_high=float(row["Erem_high"]),
                uplink_low=int(row["uplink_low"]),
                uplink_high=int(row["uplink_high"]),
                Tmax_low=float(row["Tmax_low"]),
                Tmax_high=float(row["Tmax_high"]),
                mem_max_bytes=int(row["mem_max_bytes"]),
                mem_low=mem_low,
                mem_high=mem_high,
                heterogeneous=heterogeneous,
                Umax_bytes=int(row["Umax_bytes"]),
                harvest_J_per_round=float(row["harvest_J_per_round"]),
                Tsync_s=float(row["sync_overhead_s"]),
            )

            scen = out_root / row["scenario"]
            scen.mkdir(parents=True, exist_ok=True)
            (scen / "energy_latency.yaml").write_text(emit_energy_latency(devs))

            (scen / "federated.yaml").write_text(
                emit_federated(
                    num_devices,
                    num_edges,
                    int(row["ram_edge_s"]),
                    float(row["energy_budget_per_round"]),
                    edge_mem_low_bytes=edge_mem_low,
                    edge_mem_high_bytes=edge_mem_high,
                    edge_energy_low=float(row["edge_energy_low"]),
                    edge_energy_high=float(row["edge_energy_high"]),
                    heterogeneous=heterogeneous,
                )
            )

            print(f"[OK] Scenario written: {scen}")


if __name__ == "__main__":
    main()
