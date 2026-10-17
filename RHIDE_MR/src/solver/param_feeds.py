# param_feeds.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import copy

from .mappings import BetaSpec, default_betas, beta_index_map


# ----------------------------------------------------------------------
# Device static
# ----------------------------------------------------------------------
@dataclass
class DeviceStatic:
    device_id: str
    edge_id: str
    priority: float
    mem_max_bytes: float
    uplink_rate_Bps: float
    comm_cost_per_byte_J: float
    Emax_round_J: float
    Erem_J: float
    Umax_bytes: float
    Tmax_s: float
    RBS_bytes: float
    Tedge_s: List[float]
    Tsync_s: float


# ----------------------------------------------------------------------
# Beta coefficients
# ----------------------------------------------------------------------
@dataclass
class BetaCoeffs:
    Ecomp_J: List[float]
    Ecomm_J: List[float]
    U_bytes: List[float]
    Tcomp_s: List[float]
    Tcomm_s: List[float]
    M_bytes: List[float]


# ----------------------------------------------------------------------
# Round coefficients (solver input)
# ----------------------------------------------------------------------
@dataclass
class RoundCoeffs:
    betas: List[BetaSpec]
    beta_idx: Dict[str, int]
    beta_coeffs: Dict[str, BetaCoeffs]
    statics: Dict[str, DeviceStatic]
    transfer_time_s: Dict[str, float]
    transfer_energy_J: Dict[str, float]
    batches_per_device: Dict[str, int]

    # ✅ NEW: precomputed grouping (important)
    devices_by_edge: Dict[str, List[str]]

    # Edge constraints
    edge_ram_bytes: Optional[Dict[str, float]] = None
    edge_energy_budget: Optional[Dict[str, float]] = None


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------
def _validate_beta_coeff_lengths(device_id: str, bc: "BetaCoeffs", n_betas: int) -> None:
    fields = {
        "Ecomp_J": bc.Ecomp_J,
        "Ecomm_J": bc.Ecomm_J,
        "U_bytes": bc.U_bytes,
        "Tcomp_s": bc.Tcomp_s,
        "Tcomm_s": bc.Tcomm_s,
        "M_bytes": bc.M_bytes,
    }
    bad = {k: len(v) for k, v in fields.items() if len(v) != n_betas}
    if bad:
        raise ValueError(
            f"[param_feeds] Device '{device_id}' beta length mismatch: {bad}"
        )


def _validate_tedge_lengths(device_id: str, d: DeviceStatic, n_betas: int) -> None:
    if len(d.Tedge_s) != n_betas:
        raise ValueError(
            f"[param_feeds] Device '{device_id}' Tedge_s length mismatch"
        )


# ----------------------------------------------------------------------
# Build round coefficients
# ----------------------------------------------------------------------
def build_round_coeffs(
    devices: List[DeviceStatic],
    batches_per_device: Dict[str, int],
    per_device_beta_coeffs: Dict[str, BetaCoeffs],
    betas: List[BetaSpec] | None = None
) -> RoundCoeffs:

    if betas is None:
        betas = default_betas()

    bidx = beta_index_map(betas)

    statics = {d.device_id: d for d in devices}

    # ✅ COPY (avoid mutation across rounds)
    beta_coeffs = {
        d: copy.deepcopy(per_device_beta_coeffs[d])
        for d in per_device_beta_coeffs
    }

    transfer_time_s: Dict[str, float] = {}
    transfer_energy_J: Dict[str, float] = {}

    # --------------------------------------------------
    # Precompute grouping (IMPORTANT)
    # --------------------------------------------------
    devices_by_edge: Dict[str, List[str]] = {}
    for d in devices:
        devices_by_edge.setdefault(d.edge_id, []).append(d.device_id)

    # --------------------------------------------------
    # Build coefficients
    # --------------------------------------------------
    for d in devices:
        rbs = float(d.RBS_bytes)
        uplink = float(d.uplink_rate_Bps)
        cost = float(d.comm_cost_per_byte_J)

        t_per_batch = rbs / max(1.0, uplink)
        e_per_batch = rbs * cost

        transfer_time_s[d.device_id] = t_per_batch
        transfer_energy_J[d.device_id] = e_per_batch

        B_i = int(batches_per_device.get(d.device_id, 0))
        bc = beta_coeffs[d.device_id]

        _validate_beta_coeff_lengths(d.device_id, bc, len(betas))
        _validate_tedge_lengths(d.device_id, d, len(betas))

        # ✅ scale per batch → per round (SAFE COPY)
        bc.Ecomp_J = [float(v) * B_i for v in bc.Ecomp_J]
        bc.Ecomm_J = [float(v) * B_i for v in bc.Ecomm_J]
        bc.U_bytes = [float(v) * B_i for v in bc.U_bytes]
        bc.Tcomp_s = [float(v) * B_i for v in bc.Tcomp_s]
        bc.Tcomm_s = [float(v) * B_i for v in bc.Tcomm_s]

        d.Tedge_s = [float(v) * B_i for v in d.Tedge_s]

    # --------------------------------------------------
    # Return
    # --------------------------------------------------
    return RoundCoeffs(
        betas=betas,
        beta_idx=bidx,
        beta_coeffs=beta_coeffs,
        statics=statics,
        transfer_time_s=transfer_time_s,
        transfer_energy_J=transfer_energy_J,
        batches_per_device=batches_per_device,
        devices_by_edge=devices_by_edge,  # ✅ NEW
        edge_ram_bytes=None,
        edge_energy_budget=None,
    )
