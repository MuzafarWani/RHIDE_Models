#src/federated/device_client/energy_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EnergyModelCfg:
    """
    Per-device energy config.

    Fields:
      e_per_layer: Joules per batch for each layer name up to the cut.
                   Example: { "2d_conv1": 1.2e-3, ... }
      comm_cost_per_byte: Joules per byte for uplink (matches el_cfg.comm_cost_per_byte_J)
      downlink_cost_per_byte: Joules per byte for downlink (unused for now)
    """
    e_per_layer: Dict[str, float]
    comm_cost_per_byte: float
    downlink_cost_per_byte: float = 0.0


class EnergyModel:
    """
    Computes realized per-round energy:
      - local compute for the device head up to β
      - uplink comm for activations/model update (β<L) or 0 for β=output
      - raw transfer uplink when z=0 and k>0 (semantically same as comm uplink)
    """

    def __init__(self, cfg: EnergyModelCfg):
        self.cfg = cfg

    @staticmethod
    def from_device_static(
        e_per_layer: Dict[str, float],
        comm_cost_per_byte_J: float,
        downlink_cost_per_byte_J: float = 0.0,
    ) -> "EnergyModel":
        """
        Convenience ctor when you have DeviceStatic fields from energy_latency.yaml.
        """
        return EnergyModel(EnergyModelCfg(
            e_per_layer=dict(e_per_layer),
            comm_cost_per_byte=float(comm_cost_per_byte_J),
            downlink_cost_per_byte=float(downlink_cost_per_byte_J),
        ))

    def compute_energy_for_beta(self, beta_layers: list[str], batches: int) -> float:
        """
        Sum per-layer energy up to the cut, across B batches.
        """
        e_sum = sum(float(self.cfg.e_per_layer.get(name, 0.0)) for name in beta_layers)
        return float(batches) * e_sum

    def comm_energy_for_upload(self, bytes_uploaded: float | int) -> float:
        return float(bytes_uploaded) * float(self.cfg.comm_cost_per_byte)

    def transfer_energy_for_raw(self, bytes_uploaded: float | int) -> float:
        # Same cost model as uplink comm; kept separate for clarity/future policies.
        return self.comm_energy_for_upload(bytes_uploaded)

    def total_energy_round(self, e_comp: float, e_comm: float, e_transfer: float) -> float:
        return float(e_comp + e_comm + e_transfer)

