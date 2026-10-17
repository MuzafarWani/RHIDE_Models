#src/federated/messaging/codecs.py
from __future__ import annotations

import io
import hashlib
from typing import Dict, Tuple, Any

import numpy as np

# zstd is optional; we fall back gracefully
try:
    import zstandard as zstd  # pip install zstandard
except Exception:  # pragma: no cover
    zstd = None


# ---------------------------
# Core (de)serialization APIs
# ---------------------------

def arrays_to_npz_bytes(arrs: Dict[str, np.ndarray], compress: bool = True) -> bytes:
    """
    Serialize dict of arrays to npz bytes.
    - compress=True uses numpy's zip compression (np.savez_compressed).
    - compress=False stores raw arrays (larger, but stable for checksums and zstd).
    """
    buf = io.BytesIO()
    if compress:
        np.savez_compressed(buf, **arrs)
    else:
        np.savez(buf, **arrs)
    return buf.getvalue()


def arrays_from_npz_bytes(data: bytes) -> Dict[str, np.ndarray]:
    """Inverse of arrays_to_npz_bytes."""
    buf = io.BytesIO(data)
    with np.load(buf, allow_pickle=False) as z:
        out = {k: z[k] for k in z.files}
    return out


def arrays_to_zstd(arrs: Dict[str, np.ndarray], level: int = 3) -> Tuple[bytes, str]:
    """
    Serialize arrays to *raw* npz (uncompressed), then zstd-compress.
    Returns (compressed_bytes, checksum_hex) where checksum is over the raw npz bytes.
    This makes the checksum independent of chosen compression level.
    """
    raw = arrays_to_npz_bytes(arrs, compress=False)
    if zstd is None:
        # fallback: return raw with checksum; caller can inspect size/codec availability
        return raw, blake2s_hex(raw)
    cctx = zstd.ZstdCompressor(level=int(level))
    comp = cctx.compress(raw)
    return comp, blake2s_hex(raw)


def arrays_from_zstd(data: bytes) -> Dict[str, np.ndarray]:
    """Inverse for arrays_to_zstd if you ever transport the compressed bytes."""
    if zstd is None:
        # assume data is raw npz if zstd not available
        return arrays_from_npz_bytes(data)
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(data)
    return arrays_from_npz_bytes(raw)


# -------------
# Small helpers
# -------------

def blake2s_hex(b: bytes) -> str:
    return hashlib.blake2s(b).hexdigest()


def estimate_bytes_arrays(arrs: Dict[str, np.ndarray]) -> int:
    """
    Uncompressed size in bytes (sum of .nbytes).
    Useful for energy/latency accounting vs payload bytes.
    """
    return sum(int(a.nbytes) for a in arrs.values())


def dtype_cast(arrs: Dict[str, np.ndarray], dtype: Any) -> Dict[str, np.ndarray]:
    """
    Return a new dict with all arrays cast to dtype (e.g., np.float16 to reduce size).
    """
    return {k: np.asarray(v, dtype=dtype) for k, v in arrs.items()}


# ---------------------------------
# Checksum & unified compression
# ---------------------------------

def checksum_arrays(arrs: Dict[str, np.ndarray]) -> str:
    """
    Deterministic checksum for integrity verification/logging.
    We compute blake2s over the *uncompressed* npz serialization so the hash
    is independent of transport (zstd level, etc.).
    """
    raw = arrays_to_npz_bytes(arrs, compress=False)
    return blake2s_hex(raw)


def compress_arrays(
    arrs: Dict[str, np.ndarray],
    *,
    use_zstd: bool = True,
    zstd_level: int = 3,
    npz_compress: bool = True,
) -> Tuple[bytes, int, str]:
    """
    Unified compression wrapper.

    Returns: (payload_bytes, payload_size, checksum_hex)

    - If use_zstd=True (preferred), produces zstd over raw-npz and a checksum of the raw npz.
    - Else, produces .npz (compressed if npz_compress=True). The checksum is still over raw npz
      for consistency with the zstd path.
    """
    # Always compute checksum over the raw npz form for consistency.
    raw_npz = arrays_to_npz_bytes(arrs, compress=False)
    checksum = blake2s_hex(raw_npz)

    if use_zstd:
        if zstd is None:
            # fallback to raw npz if zstd not installed
            payload = raw_npz
            return payload, len(payload), checksum
        cctx = zstd.ZstdCompressor(level=int(zstd_level))
        payload = cctx.compress(raw_npz)
        return payload, len(payload), checksum
    else:
        payload = arrays_to_npz_bytes(arrs, compress=bool(npz_compress))
        return payload, len(payload), checksum

