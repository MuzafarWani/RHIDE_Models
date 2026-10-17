# src/utils/typing_aliases.py
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Callable, Optional, Iterable, Sequence, Mapping

DeviceID = str
EdgeID = str
Weights = List["np.ndarray"]  # forward-declared; import numpy as needed at call sites

JSONDict = Dict[str, Any]
ConfigDict = Dict[str, Any]

Batch = Dict[str, Any]
BatchesByDevice = Dict[DeviceID, List[Batch]]

Reducer = Callable[[Iterable[Any]], Any]

