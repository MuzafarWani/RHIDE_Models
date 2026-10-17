# src/federated/device_client/memory_model.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class MemoryModelCfg:
    """
    memory tables (bytes) for layers / activations up to cut.
    Provided by param_feeds/mappings or precomputed in configs/split.yaml.
      weights_cum: cumulative weights bytes up to each anchor name
      activ_cum:   cumulative activations bytes retained up to each anchor
    """
    weights_cum: Dict[str, int]
    activ_cum: Dict[str, int]
    mem_max_bytes: int


class MemoryModel:
    def __init__(self, cfg: MemoryModelCfg):
        self.cfg = cfg

    def mem_for_beta(self, anchor_name: str) -> int:
        w = int(self.cfg.weights_cum.get(anchor_name, 0))
        a = int(self.cfg.activ_cum.get(anchor_name, 0))
        return w + a

    def fits(self, anchor_name: str) -> bool:
        return self.mem_for_beta(anchor_name) <= int(self.cfg.mem_max_bytes)

