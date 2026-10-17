from __future__ import annotations

from typing import Dict, Any, List, Optional
import time
import numpy as np

from src.federated.messaging.bus import MessagingBus
from src.federated.messaging.messages import Activations, RawBatch, ModelUpdate
from src.federated.messaging.codecs import (
    dtype_cast, compress_arrays, estimate_bytes_arrays
)


class Uploader:
    """
    Packs and publishes payloads to the in-proc bus.
    Returns size accounting (uncompressed vs payload bytes).
    Embeds the payload byte count in the message so edge/cloud can account energy/latency
    on the compressed size actually transmitted.

    Per-link compression config:
      - activations_dtype: np.float16 / np.float32, etc. (cast before compression)
      - raw_dtype: dtype for raw-batch arrays (usually keep native)
      - zstd_level_activ: zstd level for activations
      - zstd_level_raw: zstd level for raw batches
      - use_zstd_activ / use_zstd_raw: toggles per link
    """

    def __init__(
        self,
        bus: MessagingBus,
        *,
        # global default toggles
        use_zstd: bool = True,
        cast_dtype: Optional[np.dtype] = None,
        # per-link overrides (optional)
        activations_dtype: Optional[np.dtype] = None,
        raw_dtype: Optional[np.dtype] = None,
        zstd_level_activ: int = 3,
        zstd_level_raw: int = 3,
        use_zstd_activ: Optional[bool] = None,
        use_zstd_raw: Optional[bool] = None,
    ):
        self.bus = bus

        # Backward-compatible defaults
        self.default_use_zstd = bool(use_zstd)
        self.default_cast_dtype = cast_dtype  # legacy global cast (if set)

        # Per-link configs
        self.activations_dtype = activations_dtype
        self.raw_dtype = raw_dtype
        self.zstd_level_activ = int(zstd_level_activ)
        self.zstd_level_raw = int(zstd_level_raw)
        self.use_zstd_activ = self.default_use_zstd if use_zstd_activ is None else bool(use_zstd_activ)
        self.use_zstd_raw = self.default_use_zstd if use_zstd_raw is None else bool(use_zstd_raw)

    # ----------- Activations (β < L) -----------

    def send_activations(
        self,
        device_id: str,
        edge_id: str,
        beta: str,
        activations: Dict[str, np.ndarray],
        *,
        topic: str = "device_to_edge_activations",
        # per-call overrides (optional)
        dtype: Optional[np.dtype] = None,
        use_zstd: Optional[bool] = None,
        zstd_level: Optional[int] = None,
        npz_compress: bool = True,
        batch_id: Optional[str] = None,
        labels: Optional[np.ndarray] = None,
    ) -> Dict[str, int]:
        """
        Publishes an Activations object and returns byte accounting:
          {"bytes_uncompressed": int, "bytes_payload": int}
        The message includes m.bytes = payload_size for downstream accounting.

        Per-call overrides let you experiment per device/link without re-instantiating the uploader.
        """
        # 0) batch_id (generate if absent, since Activations requires it)
        if not batch_id:
            batch_id = f"{device_id}-{int(time.time() * 1e6)}"

        # 1) dtype cast preference: call override > per-link > legacy global > no cast
        cast_to = dtype or self.activations_dtype or self.default_cast_dtype
        arrs = dtype_cast(activations, cast_to) if cast_to is not None else activations

        # 2) sizes and compression choice
        bytes_uncompressed = estimate_bytes_arrays(arrs) + (int(labels.nbytes) if labels is not None else 0)
        use_z = self.use_zstd_activ if use_zstd is None else bool(use_zstd)
        level = self.zstd_level_activ if zstd_level is None else int(zstd_level)

        _payload, bytes_payload, checksum = compress_arrays(
            arrs, use_zstd=use_z, zstd_level=level, npz_compress=npz_compress
        )

        # 3) publish message (arrays inline for simplicity)
        msg = Activations(
            device_id=device_id,
            edge_id=edge_id,
            beta=beta,
            batch_id=batch_id,
            arrays=arrs,
            labels=labels,
            checksum=checksum,
            bytes=int(bytes_payload),
        )
        self.bus.publish(topic, msg)
        return {"bytes_uncompressed": bytes_uncompressed, "bytes_payload": bytes_payload}

    # ----------- Raw batches (z=0, k>0) -----------

    def send_rawbatch(
        self,
        device_id: str,
        edge_id: str,
        arrays: Dict[str, np.ndarray],
        labels: np.ndarray,
        *,
        topic: str = "device_to_edge_raw",
        dtype: Optional[np.dtype] = None,
        use_zstd: Optional[bool] = None,
        zstd_level: Optional[int] = None,
        npz_compress: bool = True,
        batch_id: Optional[str] = None,
    ) -> Dict[str, int]:
        # dtype cast for raw (usually keep native, but allow override)
        cast_to = dtype or self.raw_dtype or self.default_cast_dtype
        darrs = dtype_cast(arrays, cast_to) if cast_to is not None else arrays

        bytes_uncompressed = estimate_bytes_arrays(darrs) + int(labels.nbytes)
        use_z = self.use_zstd_raw if use_zstd is None else bool(use_zstd)
        level = self.zstd_level_raw if zstd_level is None else int(zstd_level)

        _payload, bytes_payload, checksum = compress_arrays(
            darrs, use_zstd=use_z, zstd_level=level, npz_compress=npz_compress
        )

        msg = RawBatch(
            device_id=device_id,
            edge_id=edge_id,
            arrays=darrs,
            labels=labels,
            checksum=checksum,
            bytes=int(bytes_payload),
        )
        # Optional convenience tag; RawBatch doesn't declare batch_id but it's harmless to set.
        if batch_id:
            try:
                setattr(msg, "batch_id", batch_id)
            except Exception:
                pass

        self.bus.publish(topic, msg)
        return {"bytes_uncompressed": bytes_uncompressed, "bytes_payload": bytes_payload}

    # ----------- Model update (β = L, or edge→cloud) -----------

    def send_model_update(
        self,
        sender_id: str,
        receiver_id: str,
        weights: List[np.ndarray],
        *,
        topic: str = "edge_to_cloud_update",
        # future: allow weights compression toggles if needed
    ) -> Dict[str, int]:
        # For weights, we keep it uncompressed in this MVP (shape/dtype-safe).
        bytes_uncompressed = sum(int(w.nbytes) for w in weights)
        msg = ModelUpdate(
            sender_id=sender_id,
            receiver_id=receiver_id,
            weights=weights,
            bytes=int(bytes_uncompressed),
        )
        self.bus.publish(topic, msg)
        return {"bytes_uncompressed": bytes_uncompressed, "bytes_payload": bytes_uncompressed}

    # ------------------------------------------------------------------
    # NEW (ADDED, NOTHING REMOVED)
    # ------------------------------------------------------------------
    # Needed for your new design:
    # Edge → Device model synchronization after cloud aggregation
    # ------------------------------------------------------------------

    def send_model_to_device(
        self,
        edge_id: str,
        device_id: str,
        weights: List[np.ndarray],
        *,
        topic_prefix: str = "edge_to_device_model",
    ) -> Dict[str, int]:
        """
        Send updated (global) model weights from Edge → Device.

        This will be used AFTER:
          Cloud aggregates → broadcasts to Edge
          Edge receives new global model → pushes to its devices

        Topic will be:
          edge_to_device_model_<device_id>
        """
        topic = f"{topic_prefix}_{device_id}"

        bytes_uncompressed = sum(int(w.nbytes) for w in weights)

        msg = ModelUpdate(
            sender_id=edge_id,
            receiver_id=device_id,
            weights=weights,
            bytes=int(bytes_uncompressed),
        )

        self.bus.publish(topic, msg)

        return {
            "bytes_uncompressed": bytes_uncompressed,
            "bytes_payload": bytes_uncompressed,
        }

