#!/usr/bin/env python3

"""
Generate scenario folders for XEvenetNet split federated learning (MULTI-ROUND VERSION)

Usage:
python3 generate_yamls_xeventnet.py --scenarios scenarios.csv --outdir outxeventnet
"""

import argparse, csv, math, random
from pathlib import Path
from typing import Any, Dict, List

# ============================================================
# XEvenetNet SPLITS (ORDER MATTERS)
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
# MEASURED VALUES (DO NOT CHANGE)
# ============================================================

LAYER_TABLE = {
    "early_prepool":  (590401765, 2.540521, 0.010),
    "post_pool1":     (594755452, 5.552049, 0.015),
    "post_conv3":     (610103459, 9.018532, 0.020),
    "global_pooled":  (616984215, 12.465544, 0.030),
    "branch_dense":   (624603168, 16.341937, 0.035),
    "concat":         (627855851, 21.190190, 0.040),
    "combined_dense": (633499287, 26.1282897, 0.050),
}

# Edge latency per split
TEDGE_PER_BETA = [0.20, 0.16, 0.13, 0.10, 0.08, 0.06, 0.05]

# Communication energy (fixed measured)
ECOMM_FIXED = [
    0.4194304,
    0.1572864,
    0.3145728,
    0.0004096,
    0.0004096,
    0.0004096,
    0.0002048,
]

# REQUIRED for solver
COMM_J_PER_BYTE = 5.0e-9


# ============================================================
# HELPERS
# ============================================================

def q(s: str) -> str:
    return f"\"{s}\""


def emit_list_inline(lst: List[float]) -> str:
    return "[" + ", ".join(f"{x}" for x in lst) + "]"


# ============================================================
# DEVICE GENERATION
# ============================================================

def build_devices(
    num_devices: int,
    num_edges: int,
    *,
    Emax_round_J: float,
    Erem_low: float,
    Erem_high: float,
    uplink_low: int,
    uplink_high: int,
    Tmax_low: float,
    Tmax_high: float,
    mem_max_bytes: int,
    Umax_bytes: int,
    harvest_J_per_round: float,
    Tsync_s: float,
) -> List[Dict[str, Any]]:

    dpe = max(1, math.ceil(num_devices / max(1, num_edges)))
    devs = []

    for i in range(num_devices):

        edge_idx = min(i // dpe, num_edges - 1)
        uplink = random.randint(uplink_low, uplink_high)

        activ = [LAYER_TABLE[b][0] for b in SPLITS]

        # IMPORTANT: fixed measured values
        Ecomp = [
            6.15895699,
            8.33915411,
            10.64529441,
            13.35524961,
            15.86877712,
            17.13266731,
            18.43577127,
        ]

        Tcomp = [0.010, 0.015, 0.020, 0.030, 0.035, 0.040, 0.050]

        Ecomm = ECOMM_FIXED
        Tcomm = [round(a / uplink, 6) for a in activ]

        devs.append({
            "idx": i,
            "edge_id": f"e{edge_idx}",
            "mem_max_bytes": int(mem_max_bytes + random.randint(-5_000_000, 5_000_000)),
            "uplink_rate_Bps": uplink,
            "Emax_round_J": Emax_round_J,
            "Erem_J": round(random.uniform(Erem_low, Erem_high), 6),
            "harvest_J_per_round": harvest_J_per_round,
            "Umax_bytes": Umax_bytes,
            "Tmax_s": round(random.uniform(Tmax_low, Tmax_high), 6),
            "RBS_bytes": activ[0],
            "Tedge_s": TEDGE_PER_BETA,
            "Tsync_s": Tsync_s,
            "per_beta": {
                "Ecomp_J": Ecomp,
                "Ecomm_J": Ecomm,
                "Tcomp_s": Tcomp,
                "Tcomm_s": Tcomm,
                "M_bytes": activ,
            },
        })

    return devs


# ============================================================
# YAML EMITTERS
# ============================================================

def emit_energy_latency(devs: List[Dict[str, Any]]) -> str:

    L = []
    L.append("# XEvenetNet per-device energy/latency constraints")
    L.append("devices:\n")

    for d in devs:
        i = d["idx"]
        pb = d["per_beta"]

        L += [
            f"  d{i}:",
            f"    edge_id: {q(d['edge_id'])}",
            f"    mem_max_bytes: {d['mem_max_bytes']}",
            f"    uplink_rate_Bps: {d['uplink_rate_Bps']}",
            f"    comm_cost_per_byte_J: {COMM_J_PER_BYTE}",
            f"    Emax_round_J: {d['Emax_round_J']}",
            f"    Erem_J: {d['Erem_J']}",
            f"    harvest_J_per_round: {d['harvest_J_per_round']}",
            f"    Umax_bytes: {d['Umax_bytes']}",
            f"    Tmax_s: {d['Tmax_s']}",
            f"    RBS_bytes: {d['RBS_bytes']}",
            f"    Tedge_s: {emit_list_inline(d['Tedge_s'])}",
            f"    Tsync_s: {d['Tsync_s']}",
            "",
            f"    per_beta:",
            f"      Ecomp_J: {emit_list_inline(pb['Ecomp_J'])}",
            f"      Ecomm_J: {emit_list_inline(pb['Ecomm_J'])}",
            f"      Tcomp_s: {emit_list_inline(pb['Tcomp_s'])}",
            f"      Tcomm_s: {emit_list_inline(pb['Tcomm_s'])}",
            f"      M_bytes: {emit_list_inline(pb['M_bytes'])}",
            "",
        ]

    return "\n".join(L)


def emit_federated(num_devices: int, num_edges: int, ram_edge_s: int) -> str:

    dpe = max(1, math.ceil(num_devices / num_edges))
    lines = []

    # -------- Edges --------
    lines.append("edges:")
    for e in range(num_edges):

        devs = [f"d{i}" for i in range(e * dpe, min((e + 1) * dpe, num_devices))]
        if not devs:
            continue

        lines += [
            f"  - id: {q(f'e{e}')}",
            f"    priority: {2.0 if e == 0 else 1.0}",
            f"    ram_edge_s: {ram_edge_s}",
            f"    devices: [{', '.join(q(d) for d in devs)}]",
        ]

    # -------- Device mapping --------
    lines += ["", "devices:"]
    for i in range(num_devices):
        edge = f"e{min(i // dpe, num_edges - 1)}"
        lines += [
            f"  d{i}:",
            f"    edge: {q(edge)}",
        ]

    # -------- Defaults + messaging --------
    lines += [
        "",
        "batches_per_device_default: 1",
        "min_batches_if_active: 1",
        "",
        "topics:",
        "  activations: \"device_to_edge_activations\"",
        "  raw:         \"device_to_edge_raw\"",
        "  edge2cloud:  \"edge_to_cloud_update\"",
        "  cloud2edges: \"cloud_to_edges_broadcast\"",
        "",
        "links:",
        "  device_to_edge:",
        "    use_zstd: true",
        "    zstd_level: 7",
        "    zstd_level_activ: 7",
        "    zstd_level_raw: 5",
        "    cast_dtype: \"float16\"",
        "    activations_dtype: \"float16\"",
        "    raw_dtype: \"float32\"",
        "    use_zstd_activ: true",
        "    use_zstd_raw: true",
        "",
        "  edge_to_cloud:",
        "    use_zstd: false",
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

    with open(args.scenarios) as f:
        rdr = csv.DictReader(f)

        for row in rdr:
            random.seed(int(row.get("seed", 1)))

            ram_edge_s = int(row["ram_edge_s"])

            devs = build_devices(
                int(row["devices"]),
                int(row["edges"]),
                Emax_round_J=float(row["Emax_round_J"]),
                Erem_low=float(row["Erem_low"]),
                Erem_high=float(row["Erem_high"]),
                uplink_low=int(row["uplink_low"]),
                uplink_high=int(row["uplink_high"]),
                Tmax_low=float(row["Tmax_low"]),
                Tmax_high=float(row["Tmax_high"]),
                mem_max_bytes=int(row["mem_max_bytes"]),
                Umax_bytes=int(row["Umax_bytes"]),
                harvest_J_per_round=float(row["harvest_J_per_round"]),
                Tsync_s=float(row["sync_overhead_s"]),
            )

            scen = out_root / row["scenario"]
            scen.mkdir(parents=True, exist_ok=True)

            (scen / "energy_latency.yaml").write_text(
                emit_energy_latency(devs)
            )

            (scen / "federated.yaml").write_text(
                emit_federated(
                    int(row["devices"]),
                    int(row["edges"]),
                    ram_edge_s
                )
            )

            print(f"[ok] XEvenetNet scenario written: {scen}")


if __name__ == "__main__":
    main()
