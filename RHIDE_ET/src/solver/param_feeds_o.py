# param_feeds.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .mappings import BetaSpec, default_betas, beta_index_map


# ----------------------------------------------------------------------
# Static parameters of a device (do NOT change inside a round)
# Each instance corresponds to one IoT device i.
# ----------------------------------------------------------------------
@dataclass
class DeviceStatic:
    """Static, per-device parameters not changing within a round."""
    device_id: str                   # "d0", "d1", ...
    edge_id: str                     # which edge server this device belongs to
    priority: float                  # p_i (used in objective)
    mem_max_bytes: float             # M_max: device RAM budget
    uplink_rate_Bps: float           # R_i: uplink rate in bytes/sec
    comm_cost_per_byte_J: float      # energy cost per transmitted byte (J/B)
    Emax_round_J: float              # max energy allowed this round
    Erem_J: float                    # remaining battery energy
    Umax_bytes: float                # max upload allowed this round (KEEP - not removed)
    Tmax_s: float                    # latency budget for this round
    RBS_bytes: float                 # size of one raw batch (bytes)

    # Edge execution latency is β-dependent
    # Tedge_s[b] = time spent on edge if split β=b is used
    Tedge_s: List[float]

    Tsync_s: float                   # synchronization overhead (scalar)


# ----------------------------------------------------------------------
# Per-β coefficients for one device.
# Each list has length = number of β split points.
# Values are *per batch* before scaling.
# ----------------------------------------------------------------------
@dataclass
class BetaCoeffs:
    Ecomp_J: List[float]   # compute energy on device for each β
    Ecomm_J: List[float]   # communication energy for each β
    U_bytes: List[float]   # bytes uploaded for each β
    Tcomp_s: List[float]   # compute time for each β
    Tcomm_s: List[float]   # comm time for each β
    M_bytes: List[float]   # memory footprint for each β


# ----------------------------------------------------------------------
# All numbers that the MILP sees for *one round*
# ----------------------------------------------------------------------
@dataclass
class RoundCoeffs:
    betas: List[BetaSpec]                 # ordered list of β
    beta_idx: Dict[str, int]              # name -> index
    beta_coeffs: Dict[str, BetaCoeffs]    # "d0" -> BetaCoeffs
    statics: Dict[str, DeviceStatic]      # "d0" -> DeviceStatic
    transfer_time_s: Dict[str, float]     # raw transfer time per batch
    transfer_energy_J: Dict[str, float]   # raw transfer energy per batch
    batches_per_device: Dict[str, int]    # B_i for this round

    # >>> NEW FIELD: EDGE-SERVER RAM CAPACITY (from federated.yaml)
    edge_ram_bytes: Optional[Dict[str, float]] = None
    # maps: "e0" -> ram_edge_s (bytes)


# ----------------------------------------------------------------------
# Sanity checks: every per-β vector must match |betas|
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
        details = ", ".join(f"{k}={l}" for k, l in bad.items())
        raise ValueError(
            f"[param_feeds] Device '{device_id}' per-β length mismatch(s): {details}; "
            f"expected all == {n_betas}. Check split.allowed order and per_beta arrays."
        )


def _validate_tedge_lengths(device_id: str, d: DeviceStatic, n_betas: int) -> None:
    if len(d.Tedge_s) != n_betas:
        raise ValueError(
            f"[param_feeds] Device '{device_id}' Tedge_s length = {len(d.Tedge_s)} "
            f"but expected {n_betas} (must match split.allowed)."
        )


# ----------------------------------------------------------------------
# Build all MILP inputs for one round.
# This converts *per-batch* quantities into *per-round* totals
# by multiplying by B_i.
# ----------------------------------------------------------------------
def build_round_coeffs(
    devices: List[DeviceStatic],
    batches_per_device: Dict[str, int],
    per_device_beta_coeffs: Dict[str, BetaCoeffs],
    betas: List[BetaSpec] | None = None
) -> RoundCoeffs:

    # If caller did not provide β list, use defaults from mappings
    if betas is None:
        betas = default_betas()

    # Map "block3_conv1" -> 2, etc.
    bidx = beta_index_map(betas)

    # Convert list of DeviceStatic into dict keyed by device_id
    statics = {d.device_id: d for d in devices}
    beta_coeffs = per_device_beta_coeffs

    # Raw transfer cost per batch (device -> edge)
    transfer_time_s: Dict[str, float] = {}
    transfer_energy_J: Dict[str, float] = {}

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

        _validate_beta_coeff_lengths(d.device_id, bc, n_betas=len(betas))
        _validate_tedge_lengths(d.device_id, d, n_betas=len(betas))

        # Convert per-batch → per-round totals
        bc.Ecomp_J = [float(v) * B_i for v in bc.Ecomp_J]
        bc.Ecomm_J = [float(v) * B_i for v in bc.Ecomm_J]
        bc.U_bytes = [float(v) * B_i for v in bc.U_bytes]
        bc.Tcomp_s = [float(v) * B_i for v in bc.Tcomp_s]
        bc.Tcomm_s = [float(v) * B_i for v in bc.Tcomm_s]

        # Edge latency also scales with number of batches
        d.Tedge_s = [float(v) * B_i for v in d.Tedge_s]

    # Package everything into one object for the MILP
    return RoundCoeffs(
        betas=betas,
        beta_idx=bidx,
        beta_coeffs=beta_coeffs,
        statics=statics,
        transfer_time_s=transfer_time_s,
        transfer_energy_J=transfer_energy_J,
        batches_per_device=batches_per_device,
        edge_ram_bytes=None,  # will be filled in train_federated_vgg16.py
    )

