# src/federated/priority.py
from __future__ import annotations

from typing import Dict, Any


class PriorityRegistry:
    """
    Holds edge-level priorities and provides device→edge→priority lookup.
    fed_cfg format expectation:
      {
        "edges": [
          {"id": "e0", "priority": 1.0, "devices": ["d0","d1"]},
          {"id": "e1", "priority": 2.0, "devices": ["d2"]}
        ]
      }
    """

    def __init__(self, fed_cfg: Dict[str, Any]):
        self._edge_priority: Dict[str, float] = {}
        self._dev2edge: Dict[str, str] = {}

        edges = fed_cfg.get("edges", [])
        for edge in edges:
            eid = str(edge["id"])
            pr = float(edge.get("priority", 1.0))
            self._edge_priority[eid] = pr
            for d in edge.get("devices", []):
                self._dev2edge[str(d)] = eid

    def edge_priority(self, edge_id: str) -> float:
        return float(self._edge_priority.get(edge_id, 1.0))

    def device_priority(self, device_id: str) -> float:
        edge = self._dev2edge.get(device_id)
        return self.edge_priority(edge) if edge is not None else 1.0

    def device_edge(self, device_id: str) -> str | None:
        return self._dev2edge.get(device_id)

    # alias for solver objective: p_i := p_edge(device)
    def get_p(self, device_id: str) -> float:
        return self.device_priority(device_id)

