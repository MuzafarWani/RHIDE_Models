#!/usr/bin/env python3

"""
Generate scenario folders for VGG16 split federated learning.

python3 generate_yamls_vgg16.py --scenarios scenarios.csv --outdir outvgg

"""

import argparse, csv, math, random
from pathlib import Path
from typing import Any, Dict, List

# ============================================================
# VGG16 8 SPLIT POINTS (ORDER MATTERS)
# ============================================================

SPLITS = [
    "block1_pool",
    "block2_conv1",
    "block3_conv1",
    "block3_conv2",
    "block3_conv3",
    "block4_conv3",
    "block5_conv2",
    "block5_conv3",
]

# Measured table (filtered for the above splits)
# layer -> (activation_bytes, Ecomp_J, Tcomp_s)
LAYER_TABLE = {
    "block1_pool":  (811409080, 4.337879, 0.111706245760433),
    "block2_conv1": (824791007, 9.354467, 0.147883336385712),
    "block3_conv1": (903638679, 16.941728, 0.225169905181974),
    "block3_conv2": (932798529, 25.633234, 0.273592969833408),
    "block3_conv3": (964603936, 35.592254, 0.321575720387045),
    "block4_conv3": (992923877, 48.996152, 0.441691204905510),
    "block5_conv2": (1070366457,  64.029379, 0.462340303207748),
    "block5_conv3": (1071202172,  79.286847, 0.478222534398083),
}

# Per-β edge latency (seconds)
TEDGE_PER_BETA = [0.22, 0.18, 0.15, 0.11, 0.09, 0.07, 0.05, 0.03]

# Communication energy: 50 nJ per byte
COMM_J_PER_BYTE = 5.0e-9


def q(s: str) -> str:
    return f"\"{s}\""


def emit_list_inline(lst: List[float]) -> str:
    return "[" + ", ".join(f"{x}" for x in lst) + "]"


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
    Umax_bytes: int,      # <<< KEPT (unchanged, still per-device)
    harvest_J_per_round: float,
    Tsync_s: float,
) -> List[Dict[str, Any]]:

    dpe = max(1, math.ceil(num_devices / max(1, num_edges)))
    devs = []

    for i in range(num_devices):
        edge_idx = min(i // dpe, num_edges - 1)
        uplink = random.randint(uplink_low, uplink_high)

        # Build per-β vectors from measured data
        activ = [LAYER_TABLE[b][0] for b in SPLITS]
        Ecomp = [LAYER_TABLE[b][1] for b in SPLITS]
        Tcomp = [LAYER_TABLE[b][2] for b in SPLITS]

        Ecomm = [round(a * COMM_J_PER_BYTE, 9) for a in activ]
        Tcomm = [round(a / uplink, 6) for a in activ]

        devs.append({
            "idx": i,
            "edge_id": f"e{edge_idx}",
            "mem_max_bytes": int(mem_max_bytes + random.randint(-5_000_000, 5_000_000)),
            "uplink_rate_Bps": uplink,
            "Emax_round_J": Emax_round_J,
            "Erem_J": round(random.uniform(Erem_low, Erem_high), 6),
            "harvest_J_per_round": harvest_J_per_round,
            "Umax_bytes": Umax_bytes,   # <<< KEPT (unchanged)
            "Tmax_s": round(random.uniform(Tmax_low, Tmax_high), 6),
            "RBS_bytes": activ[0],  # raw = first split input
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


def emit_energy_latency(devs: List[Dict[str, Any]]) -> str:
    L = []
    L.append("# VGG16 per-device energy/latency constraints")
    L.append("devices:\n")

    for d in devs:
        i = d["idx"]
        pb = d["per_beta"]
        L += [
            f"  d{i}:",
            f"    edge_id: {q(d['edge_id'])}",
            f"    mem_max_bytes: {d['mem_max_bytes']:}",
            f"    uplink_rate_Bps: {d['uplink_rate_Bps']:}",
            f"    comm_cost_per_byte_J: {COMM_J_PER_BYTE}",
            f"    Emax_round_J: {d['Emax_round_J']}",
            f"    Erem_J: {d['Erem_J']}",
            f"    harvest_J_per_round: {d['harvest_J_per_round']}",
            f"    Umax_bytes: {d['Umax_bytes']:}",   # <<< KEPT
            f"    Tmax_s: {d['Tmax_s']}",
            f"    RBS_bytes: {d['RBS_bytes']:}",
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
    # >>> NEW: pass ram_edge_s from CSV into this function

    dpe = max(1, math.ceil(num_devices / num_edges))
    lines = []

    # -------- Edges --------
    lines.append("edges:")
    for e in range(num_edges):
        devs = [f"d{i}" for i in range(e * dpe, min((e + 1) * dpe, num_devices))]
        if not devs:
            continue
        prio = 2.0 if e == 0 else 1.0

        lines += [
            f"  - id: {q(f'e{e}')}",
            f"    priority: {prio}",
            f"    ram_edge_s: {ram_edge_s}",   # >>> NEW FIELD (EDGE MEMORY)
            f"    devices: [{', '.join(q(d) for d in devs)}]",
        ]

    # -------- Optional device metadata --------
    lines += ["", "devices:"]
    for i in range(num_devices):
        edge = f"e{min(i // dpe, num_edges - 1)}"
        lines += [
            f"  d{i}:",
            f"    edge: {q(edge)}",
        ]

    # -------- Defaults and links --------
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--outdir", default="outvgg")
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.scenarios) as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            random.seed(int(row.get("seed", 1)))

            # >>> NEW: read ram_edge_s from CSV
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
                Umax_bytes=int(row["Umax_bytes"]),  # <<< KEPT
                harvest_J_per_round=float(row["harvest_J_per_round"]),
                Tsync_s=float(row["sync_overhead_s"]),
            )

            scen = out_root / row["scenario"]
            scen.mkdir(parents=True, exist_ok=True)

            (scen / "energy_latency.yaml").write_text(emit_energy_latency(devs))

            (scen / "federated.yaml").write_text(
                emit_federated(
                    int(row["devices"]),
                    int(row["edges"]),
                    ram_edge_s   # >>> NEW
                )
            )

            print(f"[ok] VGG16 scenario written: {scen}")


if __name__ == "__main__":
    main()

