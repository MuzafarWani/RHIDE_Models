# src/data/encoders.py
from __future__ import annotations

import os
import json
import pickle
from typing import List, Dict, Any


class SimpleLabelEncoder:
    """
    Minimal label encoder with explicit class list persistence.
    """
    def __init__(self, classes: List[str] | None = None):
        self.classes_: List[str] = list(classes) if classes else []
        self._index: Dict[str, int] = {c: i for i, c in enumerate(self.classes_)}

    def fit(self, classes: List[str]) -> None:
        uniq = sorted(set(classes))
        self.classes_ = uniq
        self._index = {c: i for i, c in enumerate(self.classes_)}

    def transform(self, labels: List[str]) -> List[int]:
        return [self._index[l] for l in labels]

    def inverse_transform(self, idxs: List[int]) -> List[str]:
        return [self.classes_[i] for i in idxs]

    def to_dict(self) -> Dict[str, Any]:
        return {"classes": self.classes_}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimpleLabelEncoder":
        return cls(classes=d["classes"])

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SimpleLabelEncoder":
        with open(path, "r") as f:
            d = json.load(f)
        return cls.from_dict(d)

