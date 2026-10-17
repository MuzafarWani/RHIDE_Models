#src/federated/device_client/latency_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class LatencyModelCfg:
    """
    Per-device latency config.

    Fields:
      t_per_layer: seconds per batch for each layer name up to the cut.
                   Example: { "2d_conv1": 7.5e-3, ... }
      uplink_rate_Bps: bytes per second for uplink (matches el_cfg.uplink_rate_Bps)
      edge_time_overhead: fixed seconds (edge processing tail + pipeline overhead)
      sync_overhead: fixed seconds (round synchronization barrier)
    """
    t_per_layer: Dict[str, float]
    uplink_rate_Bps: float
    edge_time_overhead: float
    sync_overhead: float


class LatencyModel:
    def __init__(self, cfg: LatencyModelCfg):
        self.cfg = cfg

    @staticmethod
    def from_device_static(
        t_per_layer: Dict[str, float],
        uplink_rate_Bps: float,
        edge_time_overhead_s: float,
        sync_overhead_s: float,
    ) -> "LatencyModel":
        """
        Convenience ctor when you have DeviceStatic fields from energy_latency.yaml.
        """
        return LatencyModel(LatencyModelCfg(
            t_per_layer=dict(t_per_layer),
            uplink_rate_Bps=float(uplink_rate_Bps),
            edge_time_overhead=float(edge_time_overhead_s),
            sync_overhead=float(sync_overhead_s),
        ))

    def compute_time_for_beta(self, beta_layers: list[str], batches: int) -> float:
        t_sum = sum(float(self.cfg.t_per_layer.get(n, 0.0)) for n in beta_layers)
        return float(batches) * t_sum

    def comm_time_for_bytes(self, bytes_uploaded: float | int) -> float:
        rate = max(1.0, float(self.cfg.uplink_rate_Bps))
        return float(bytes_uploaded) / rate

    def total_round_time(self, t_compute: float, t_comm: float, include_edge_sync: bool = True) -> float:
        extra = (self.cfg.edge_time_overhead + self.cfg.sync_overhead) if include_edge_sync else 0.0
        return float(t_compute + t_comm + extra)

