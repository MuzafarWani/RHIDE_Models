from __future__ import annotations

from typing import Any, List, Optional
import numpy as np

from src.federated.messaging.bus import MessagingBus
from src.federated.messaging.messages import ModelUpdate


class Distributor:
    """
    Broadcasts updated global weights from cloud to edges (and devices, if needed).
    """

    def __init__(self, bus: MessagingBus, cfg: Any):
        self.bus = bus
        self.cfg = cfg

    def broadcast_to_edges(self, weights: List[np.ndarray] | None) -> None:
        if weights is None:
            return

        # --------- ORIGINAL BEHAVIOR (KEPT) ---------
        # Single broadcast topic; edges will listen and set_weights on receipt.
        msg = ModelUpdate(sender_id="cloud", receiver_id="all_edges", weights=weights)
        self.bus.publish("cloud_to_edges_broadcast", msg)

        # --------- NEW BEHAVIOR (ADDED, NOT REMOVING ANYTHING) ---------
        # Also publish a "sync signal" so edges know:
        #   → "A new global model is available and should be pushed to devices."
        #
        # This is IMPORTANT for your new design:
        # Cloud → Edge (model)
        # Edge  → Device (model)

        signal_msg = ModelUpdate(
            sender_id="cloud",
            receiver_id="edges_sync_signal",
            weights=weights,
        )
        self.bus.publish("cloud_to_edges_sync_signal", signal_msg)

    # >>> NEW (needed for your future step) <<<
    def broadcast_to_single_edge(self, edge_id: str, weights: List[np.ndarray]) -> None:
        """
        Optional helper for explicit per-edge sync.
        (You may or may not use this later.)
        """
        msg = ModelUpdate(
            sender_id="cloud",
            receiver_id=edge_id,
            weights=weights,
        )
        topic = f"cloud_to_edge_{edge_id}"
        self.bus.publish(topic, msg)

