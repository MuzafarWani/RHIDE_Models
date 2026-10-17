#!/usr/bin/env python3

"""
Generate scenario folders for VGG16 split federated learning.

Supports:
    1. Homogeneous scenarios: S1-S6
    2. Heterogeneous scenarios: S7-S12
    3. Resource levels from Table III of the RHiDE paper
    4. Legacy CSV files with explicit numerical parameters
    5. Small per-device timing and transfer-limit variations for S7-S12

Usage:
    python3 generate_yamls_vgg16.py --scenarios scenarios.csv --outdir outvgg

The generated structure is:

    outvgg/
        S1/
            energy_latency.yaml
            federated.yaml
        ...
        S12/
            energy_latency.yaml
            federated.yaml
"""

import argparse
import csv
import math
import random

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# VGG16 SPLITS
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

LAYER_TABLE = {
    "block1_pool":  (811409080, 4.337879, 1.51706245760433),
    "block2_conv1": (824791007, 9.354467, 2.147883336385712),
    "block3_conv1": (903638679, 16.941728, 2.525169905181974),
    "block3_conv2": (932798529, 25.633234, 3.173592969833408),
    "block3_conv3": (964603936, 35.592254, 3.321575720387045),
    "block4_conv3": (992923877, 48.996152, 4.041691204905510),
    "block5_conv2": (1070366457, 64.029379, 4.562340303207748),
    "block5_conv3": (1071202172, 79.286847, 4.9378222534398083),
}

TEDGE_PER_BETA = [3.22, 3.08, 2.85, 2.31, 1.8, 1.3, 0.95, 0.53]

COMM_J_PER_BYTE = 5.0e-9


# ============================================================
# RESOURCE RANGES
# Table III: VGG16 configurations
# ============================================================

# Device energy ranges are in joules.
DEVICE_ENERGY_RANGES = {
    "LOW": (100.0, 200.0),
    "MEDIUM": (200.0, 300.0),
    "HIGH": (300.0, 2000.0),
}

# Device memory ranges are in MB.
DEVICE_MEMORY_RANGES_MB = {
    "LOW": (780.0, 800.0),
    "MEDIUM": (800.0, 900.0),
    "HIGH": (900.0, 1000.0),
}

# Edge energy ranges are in joules.
EDGE_ENERGY_RANGES = {
    "LOW": (400.0, 500.0),
    "MEDIUM": (500.0, 800.0),
    "HIGH": (800.0, 1000.0),
}

# Edge memory ranges are in GB.
EDGE_MEMORY_RANGES_GB = {
    "LOW": (9.0, 10.4),
    "MEDIUM": (10.5, 11.5),
    "HIGH": (11.5, 15.0),
}

RESOURCE_LEVELS = ("LOW", "MEDIUM", "HIGH")


# ============================================================
# SCENARIO DEFINITIONS
# Table II: Experimental scenarios
# ============================================================

SCENARIO_TABLE = {
    "S1": {
        "devices": 100,
        "device_energy": "HIGH",
        "device_memory": "HIGH",
        "edge_energy": "HIGH",
        "edge_memory": "HIGH",
        "heterogeneous_devices": False,
        "heterogeneous_edges": False,
    },
    "S2": {
        "devices": 100,
        "device_energy": "HIGH",
        "device_memory": "HIGH",
        "edge_energy": "HIGH",
        "edge_memory": "LOW",
        "heterogeneous_devices": False,
        "heterogeneous_edges": False,
    },
    "S3": {
        "devices": 100,
        "device_energy": "HIGH",
        "device_memory": "HIGH",
        "edge_energy": "LOW",
        "edge_memory": "HIGH",
        "heterogeneous_devices": False,
        "heterogeneous_edges": False,
    },
    "S4": {
        "devices": 100,
        "device_energy": "HIGH",
        "device_memory": "LOW",
        "edge_energy": "HIGH",
        "edge_memory": "HIGH",
        "heterogeneous_devices": False,
        "heterogeneous_edges": False,
    },
    "S5": {
        "devices": 100,
        "device_energy": "LOW",
        "device_memory": "HIGH",
        "edge_energy": "HIGH",
        "edge_memory": "HIGH",
        "heterogeneous_devices": False,
        "heterogeneous_edges": False,
    },
    "S6": {
        "devices": 500,
        "device_energy": "HIGH",
        "device_memory": "HIGH",
        "edge_energy": "HIGH",
        "edge_memory": "HIGH",
        "heterogeneous_devices": False,
        "heterogeneous_edges": False,
    },
    "S7": {
        "devices": 100,
        "device_energy": "HIGH",
        "device_memory": "HETEROGENEOUS",
        "edge_energy": "HETEROGENEOUS",
        "edge_memory": "HETEROGENEOUS",
        "heterogeneous_devices": True,
        "heterogeneous_edges": True,
    },
    "S8": {
        "devices": 100,
        "device_energy": "HIGH",
        "device_memory": "HETEROGENEOUS",
        "edge_energy": "HETEROGENEOUS",
        "edge_memory": "LOW",
        "heterogeneous_devices": True,
        "heterogeneous_edges": True,
    },
    "S9": {
        "devices": 100,
        "device_energy": "HIGH",
        "device_memory": "HETEROGENEOUS",
        "edge_energy": "LOW",
        "edge_memory": "HETEROGENEOUS",
        "heterogeneous_devices": True,
        "heterogeneous_edges": True,
    },
    "S10": {
        "devices": 100,
        "device_energy": "LOW",
        "device_memory": "HETEROGENEOUS",
        "edge_energy": "HETEROGENEOUS",
        "edge_memory": "HETEROGENEOUS",
        "heterogeneous_devices": True,
        "heterogeneous_edges": True,
    },
    "S11": {
        "devices": 100,
        "device_energy": "LOW",
        "device_memory": "HETEROGENEOUS",
        "edge_energy": "HETEROGENEOUS",
        "edge_memory": "HETEROGENEOUS",
        "heterogeneous_devices": True,
        "heterogeneous_edges": True,
    },
    "S12": {
        "devices": 500,
        "device_energy": "HIGH",
        "device_memory": "HETEROGENEOUS",
        "edge_energy": "HETEROGENEOUS",
        "edge_memory": "HETEROGENEOUS",
        "heterogeneous_devices": True,
        "heterogeneous_edges": True,
    },
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def q(s: str) -> str:
    """Quote a YAML string."""
    return f'"{s}"'


def emit_list_inline(lst: List[float]) -> str:
    """Format a list for YAML."""
    return "[" + ", ".join(f"{x}" for x in lst) + "]"


def normalize_level(level: str) -> str:
    """Normalize a resource-level name."""
    return level.strip().upper().replace("-", "_")


def random_value(
    low: float,
    high: float,
    heterogeneous: bool,
    homogeneous_value: Optional[float] = None,
) -> float:
    """
    Generate a resource value.

    Homogeneous:
        Use the same value for every entity.

    Heterogeneous:
        Sample a value independently for each entity.
    """
    if heterogeneous:
        return random.uniform(low, high)

    if homogeneous_value is not None:
        return homogeneous_value

    return (low + high) / 2.0


def get_range(
    level: str,
    ranges: Dict[str, Tuple[float, float]],
) -> Tuple[float, float]:
    """Get the numerical range for a resource level."""
    level = normalize_level(level)

    if level not in ranges:
        raise ValueError(
            f"Unsupported resource level '{level}'. "
            f"Expected one of {list(ranges.keys())}."
        )

    return ranges[level]


def choose_heterogeneous_range(
    ranges: Dict[str, Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Select a resource range for a heterogeneous entity.

    The entity may receive a Low, Medium, or High configuration.
    """
    level = random.choice(RESOURCE_LEVELS)
    return ranges[level]


def resolve_entity_range(
    level: str,
    ranges: Dict[str, Tuple[float, float]],
    heterogeneous: bool,
) -> Tuple[float, float]:
    """
    Resolve a resource level into a numerical range.

    A Heterogeneous level means that individual entities can
    receive different Low, Medium, or High configurations.
    """
    level = normalize_level(level)

    if level == "HETEROGENEOUS":
        if not heterogeneous:
            raise ValueError(
                "HETEROGENEOUS resource level requires "
                "heterogeneous configuration."
            )

        return choose_heterogeneous_range(ranges)

    return get_range(level, ranges)


# ============================================================
# BUILD DEVICES
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
    mem_low: int = None,
    mem_high: int = None,
    mem_max_bytes: int = None,
    Umax_bytes: int = None,
    harvest_J_per_round: float = 0,
    Tsync_s: float = 0,
    heterogeneous_devices: bool = False,
) -> List[Dict[str, Any]]:

    dpe = max(1, math.ceil(num_devices / max(1, num_edges)))
    devs = []

    # Fixed values for homogeneous configurations.
    homogeneous_uplink = random.randint(uplink_low, uplink_high)
    homogeneous_Erem = random.uniform(Erem_low, Erem_high)
    homogeneous_Tmax = random.uniform(Tmax_low, Tmax_high)

    if mem_low is not None and mem_high is not None:
        homogeneous_memory = random.randint(mem_low, mem_high)
    elif mem_max_bytes is not None:
        homogeneous_memory = int(mem_max_bytes)
    else:
        raise ValueError(
            "Either mem_max_bytes or both mem_low and mem_high "
            "must be specified."
        )

    for i in range(num_devices):

        edge_idx = min(i // dpe, num_edges - 1)

        # -----------------------------
        # BANDWIDTH
        # -----------------------------
        if heterogeneous_devices:
            uplink = random.randint(uplink_low, uplink_high)
        else:
            uplink = homogeneous_uplink

        # -----------------------------
        # MEMORY
        # -----------------------------
        if mem_low is not None and mem_high is not None:
            if heterogeneous_devices:
                mem_bytes = random.randint(mem_low, mem_high)
            else:
                mem_bytes = homogeneous_memory
        else:
            mem_bytes = int(mem_max_bytes)

        # -----------------------------
        # ENERGY AND EXECUTION BUDGET
        # -----------------------------
        if heterogeneous_devices:
            Erem = random.uniform(Erem_low, Erem_high)
            Tmax = random.uniform(Tmax_low, Tmax_high)
        else:
            Erem = homogeneous_Erem
            Tmax = homogeneous_Tmax

        # -----------------------------
        # PER-LAYER METRICS
        # -----------------------------
        activ = [LAYER_TABLE[b][0] for b in SPLITS]
        Ecomp = [LAYER_TABLE[b][1] for b in SPLITS]
        Tcomp = [LAYER_TABLE[b][2] for b in SPLITS]

        # ---------------------------------------------------------
        # HETEROGENEOUS PER-DEVICE VARIATIONS
        # ---------------------------------------------------------
        # Keep the measured LAYER_TABLE values unchanged for S1-S6.
        # For heterogeneous scenarios (S7-S12), model small device-level
        # differences in computation time, communication time, edge time,
        # upload limits, and memory/bandwidth-related parameters.
        if heterogeneous_devices:
            # Example: 1.517062... s may become approximately 1.60-1.62 s
            # on another device. Each device/layer gets its own small factor.
            Tcomp = [
                round(t * random.uniform(1.00, 1.07), 15)
                for t in Tcomp
            ]

            # Slight device-specific variation in transfer time in addition
            # to the variation already caused by each device's uplink rate.
            Tcomm = [
                round((a / uplink) * random.uniform(0.97, 1.05), 6)
                for a in activ
            ]

            # Slightly different edge processing speed for each device.
            Tedge = [
                round(t * random.uniform(0.97, 1.05), 6)
                for t in TEDGE_PER_BETA
            ]

            # Each device has its own transfer-size limit, varied by +/- 5%.
            device_Umax_bytes = max(
                1, int(Umax_bytes * random.uniform(0.95, 1.05))
            )
        else:
            Tcomm = [
                round(a / uplink, 6)
                for a in activ
            ]
            Tedge = TEDGE_PER_BETA
            device_Umax_bytes = Umax_bytes

        Ecomm = [
            round(a * COMM_J_PER_BYTE, 9)
            for a in activ
        ]

        devs.append({
            "idx": i,
            "edge_id": f"e{edge_idx}",
            "mem_max_bytes": mem_bytes,
            "uplink_rate_Bps": uplink,
            "Emax_round_J": Emax_round_J,
            "Erem_J": round(Erem, 6),
            "harvest_J_per_round": harvest_J_per_round,
            "Umax_bytes": device_Umax_bytes,
            "Tmax_s": round(Tmax, 6),
            "RBS_bytes": activ[0],
            "Tedge_s": Tedge,
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
# ENERGY LATENCY YAML
# ============================================================

def emit_energy_latency(
    devs: List[Dict[str, Any]]
) -> str:

    L = []
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
            "    per_beta:",
            f"      Ecomp_J: {emit_list_inline(pb['Ecomp_J'])}",
            f"      Ecomm_J: {emit_list_inline(pb['Ecomm_J'])}",
            f"      Tcomp_s: {emit_list_inline(pb['Tcomp_s'])}",
            f"      Tcomm_s: {emit_list_inline(pb['Tcomm_s'])}",
            f"      M_bytes: {emit_list_inline(pb['M_bytes'])}",
            "",
        ]

    return "\n".join(L)


# ============================================================
# FEDERATED YAML
# ============================================================

def emit_federated(
    num_devices: int,
    num_edges: int,
    ram_edge_s: int,
    edge_energy_budget: float,
    *,
    heterogeneous_edges: bool = False,
    edge_memory_values: Optional[List[float]] = None,
    edge_energy_values: Optional[List[float]] = None,
) -> str:

    dpe = max(1, math.ceil(num_devices / num_edges))
    lines = []

    lines.append("edges:")

    for e in range(num_edges):

        devs = [
            f"d{i}"
            for i in range(
                e * dpe,
                min((e + 1) * dpe, num_devices)
            )
        ]

        if not devs:
            continue

        prio = 2.0 if e == 0 else 1.0

        if edge_memory_values is not None:
            edge_memory = edge_memory_values[e]
        else:
            edge_memory = ram_edge_s

        if edge_energy_values is not None:
            edge_energy = edge_energy_values[e]
        else:
            edge_energy = edge_energy_budget

        lines += [
            f"  - id: {q(f'e{e}')}",
            f"    priority: {prio}",
            f"    ram_edge_s: {edge_memory}",
            f"    energy_budget_per_round: {edge_energy}",
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
# SCENARIO CONFIGURATION
# ============================================================

def build_scenario_configuration(
    row: Dict[str, str],
) -> Dict[str, Any]:
    """
    Construct a scenario configuration.

    If resource-level columns are supplied, use the resource
    ranges defined in Table III.

    Otherwise, retain compatibility with the original CSV,
    which supplies explicit numerical parameters.
    """

    scenario = row["scenario"].strip()

    # --------------------------------------------------------
    # LEGACY CSV FORMAT
    # --------------------------------------------------------
    if not row.get("device_energy_level"):

        return {
            "scenario": scenario,
            "devices": int(row["devices"]),
            "edges": int(row["edges"]),
            "Emax_round_J": float(row["Emax_round_J"]),
            "Erem_low": float(row["Erem_low"]),
            "Erem_high": float(row["Erem_high"]),
            "uplink_low": int(row["uplink_low"]),
            "uplink_high": int(row["uplink_high"]),
            "Tmax_low": float(row["Tmax_low"]),
            "Tmax_high": float(row["Tmax_high"]),
            "mem_low": (
                int(row["mem_low"])
                if row.get("mem_low")
                else None
            ),
            "mem_high": (
                int(row["mem_high"])
                if row.get("mem_high")
                else None
            ),
            "mem_max_bytes": (
                int(row["mem_max_bytes"])
                if row.get("mem_max_bytes")
                else None
            ),
            "Umax_bytes": int(row["Umax_bytes"]),
            "harvest_J_per_round": float(
                row["harvest_J_per_round"]
            ),
            "Tsync_s": float(row["sync_overhead_s"]),
            "ram_edge_s": int(row["ram_edge_s"]),
            "edge_energy_budget": float(
                row["energy_budget_per_round"]
            ),
            "heterogeneous_devices": bool(
                row.get("mem_low") and row.get("mem_high")
            ),
            "heterogeneous_edges": False,
            "edge_memory_values": None,
            "edge_energy_values": None,
        }

    # --------------------------------------------------------
    # RESOURCE-LEVEL CSV FORMAT
    # --------------------------------------------------------
    if scenario not in SCENARIO_TABLE:
        raise ValueError(
            f"Unknown scenario '{scenario}'. "
            f"Expected one of {list(SCENARIO_TABLE.keys())}."
        )

    definition = SCENARIO_TABLE[scenario]

    devices = int(row.get("devices") or definition["devices"])
    edges = int(row.get("edges") or 10)

    seed = int(row.get("seed") or 1)
    random.seed(seed)

    hetero_devices = definition["heterogeneous_devices"]
    hetero_edges = definition["heterogeneous_edges"]

    device_energy_level = definition["device_energy"]
    device_memory_level = definition["device_memory"]
    edge_energy_level = definition["edge_energy"]
    edge_memory_level = definition["edge_memory"]

    # Allow explicit level overrides from the CSV.
    device_energy_level = row.get(
        "device_energy_level", device_energy_level
    ) or definition["device_energy"]

    device_memory_level = row.get(
        "device_memory_level", device_memory_level
    ) or definition["device_memory"]

    edge_energy_level = row.get(
        "edge_energy_level", edge_energy_level
    ) or definition["edge_energy"]

    edge_memory_level = row.get(
        "edge_memory_level", edge_memory_level
    ) or definition["edge_memory"]

    # -----------------------------
    # DEVICE ENERGY
    # -----------------------------
    e_low, e_high = resolve_entity_range(
        device_energy_level,
        DEVICE_ENERGY_RANGES,
        heterogeneous=False,
    )

    if not hetero_devices:
        device_energy = random.uniform(e_low, e_high)
        Erem_low = Erem_high = device_energy
    else:
        Erem_low, Erem_high = e_low, e_high

    # Preserve the existing per-round energy field.
    # This is intentionally kept separate from Erem_J.
    Emax_round_J = float(
        row.get("Emax_round_J") or 18.0
    )

    # -----------------------------
    # DEVICE MEMORY
    # -----------------------------
    if normalize_level(device_memory_level) == "HETEROGENEOUS":

        # Heterogeneous devices receive different memory
        # configurations selected from Table III.
        memory_ranges = DEVICE_MEMORY_RANGES_MB

        sampled_memory_ranges = [
            choose_heterogeneous_range(memory_ranges)
            for _ in range(devices)
        ]

        memory_values = []

        for low, high in sampled_memory_ranges:
            memory_mb = random.uniform(low, high)
            memory_values.append(int(memory_mb * 1024 * 1024))

        # build_devices() accepts one memory range. To preserve
        # device-specific values, the actual values are applied
        # after constructing the devices.
        mem_low = mem_high = None
        mem_max_bytes = max(memory_values)

    else:
        m_low, m_high = get_range(
            device_memory_level,
            DEVICE_MEMORY_RANGES_MB,
        )

        if hetero_devices:
            mem_low = int(m_low * 1024 * 1024)
            mem_high = int(m_high * 1024 * 1024)
            mem_max_bytes = None
            memory_values = None
        else:
            memory_mb = random.uniform(m_low, m_high)
            mem_low = mem_high = None
            mem_max_bytes = int(memory_mb * 1024 * 1024)
            memory_values = None

    # -----------------------------
    # EDGE ENERGY
    # -----------------------------
    edge_energy_values = []

    for _ in range(edges):

        low, high = resolve_entity_range(
            edge_energy_level,
            EDGE_ENERGY_RANGES,
            heterogeneous=hetero_edges,
        )

        if hetero_edges:
            edge_energy = random.uniform(low, high)
        else:
            edge_energy = (low + high) / 2.0

        edge_energy_values.append(round(edge_energy, 6))

    # -----------------------------
    # EDGE MEMORY
    # -----------------------------
    edge_memory_values = []

    for _ in range(edges):

        low, high = resolve_entity_range(
            edge_memory_level,
            EDGE_MEMORY_RANGES_GB,
            heterogeneous=hetero_edges,
        )

        if hetero_edges:
            edge_memory_gb = random.uniform(low, high)
        else:
            edge_memory_gb = (low + high) / 2.0

        # Use the existing YAML field name ram_edge_s. Convert the Table III
        # GB capacity to bytes; legacy CSV rows retain their supplied value
        # (including 2147483648) through the legacy configuration path.
        edge_memory_values.append(int(edge_memory_gb * (1024 ** 3)))

    # -----------------------------
    # OTHER PARAMETERS
    # -----------------------------
    # Preserve existing CSV controls when present.
    uplink_low = int(row.get("uplink_low") or 2_000_000)
    uplink_high = int(row.get("uplink_high") or 20_000_000)

    Tmax_low = float(row.get("Tmax_low") or 100.0)
    Tmax_high = float(row.get("Tmax_high") or 100.0)

    Umax_bytes = int(row.get("Umax_bytes") or 10**12)

    harvest_J_per_round = float(
        row.get("harvest_J_per_round") or 0
    )

    Tsync_s = float(row.get("sync_overhead_s") or 0)

    # Retain the first edge's value in the original scalar fields.
    ram_edge_s = int(edge_memory_values[0])
    edge_energy_budget = float(edge_energy_values[0])

    return {
        "scenario": scenario,
        "devices": devices,
        "edges": edges,
        "Emax_round_J": Emax_round_J,
        "Erem_low": Erem_low,
        "Erem_high": Erem_high,
        "uplink_low": uplink_low,
        "uplink_high": uplink_high,
        "Tmax_low": Tmax_low,
        "Tmax_high": Tmax_high,
        "mem_low": mem_low,
        "mem_high": mem_high,
        "mem_max_bytes": mem_max_bytes,
        "memory_values": memory_values,
        "Umax_bytes": Umax_bytes,
        "harvest_J_per_round": harvest_J_per_round,
        "Tsync_s": Tsync_s,
        "ram_edge_s": ram_edge_s,
        "edge_energy_budget": edge_energy_budget,
        "heterogeneous_devices": hetero_devices,
        "heterogeneous_edges": hetero_edges,
        "edge_memory_values": edge_memory_values,
        "edge_energy_values": edge_energy_values,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--scenarios",
        required=True,
        help="CSV containing scenario configurations",
    )

    ap.add_argument(
        "--outdir",
        default="outvgg",
        help="Output directory for generated YAML files",
    )

    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(args.scenarios, newline="") as f:

        rdr = csv.DictReader(f)

        for row in rdr:

            scenario = row["scenario"].strip()

            seed = int(row.get("seed") or 1)
            random.seed(seed)

            config = build_scenario_configuration(row)

            devs = build_devices(
                config["devices"],
                config["edges"],
                Emax_round_J=config["Emax_round_J"],
                Erem_low=config["Erem_low"],
                Erem_high=config["Erem_high"],
                uplink_low=config["uplink_low"],
                uplink_high=config["uplink_high"],
                Tmax_low=config["Tmax_low"],
                Tmax_high=config["Tmax_high"],
                mem_low=config["mem_low"],
                mem_high=config["mem_high"],
                mem_max_bytes=config["mem_max_bytes"],
                Umax_bytes=config["Umax_bytes"],
                harvest_J_per_round=config["harvest_J_per_round"],
                Tsync_s=config["Tsync_s"],
                heterogeneous_devices=config["heterogeneous_devices"],
            )

            # Apply device-specific memory configurations for
            # heterogeneous resource-level scenarios.
            memory_values = config.get("memory_values")

            if memory_values is not None:
                for d, memory_bytes in zip(devs, memory_values):
                    d["mem_max_bytes"] = memory_bytes

            scen = out_root / scenario
            scen.mkdir(parents=True, exist_ok=True)

            # -----------------------------
            # ENERGY/LATENCY YAML
            # -----------------------------
            energy_latency_path = scen / "energy_latency.yaml"

            energy_latency_path.write_text(
                emit_energy_latency(devs)
            )

            # -----------------------------
            # FEDERATED YAML
            # -----------------------------
            federated_path = scen / "federated.yaml"

            federated_path.write_text(
                emit_federated(
                    config["devices"],
                    config["edges"],
                    config["ram_edge_s"],
                    config["edge_energy_budget"],
                    heterogeneous_edges=config["heterogeneous_edges"],
                    edge_memory_values=config["edge_memory_values"],
                    edge_energy_values=config["edge_energy_values"],
                )
            )

            print(f"[OK] Scenario written: {scen}")
            print(f"     Devices: {config['devices']}")
            print(f"     Edges:   {config['edges']}")
            print(f"     Seed:    {seed}")


if __name__ == "__main__":
    main()
