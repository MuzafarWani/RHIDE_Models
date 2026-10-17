# src/eval/metrics.py
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> np.ndarray:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if num_classes is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1) if len(y_true) else 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int | None = None) -> Dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    macro = {
        "precision": float(np.mean(precision)) if len(precision) else 0.0,
        "recall": float(np.mean(recall)) if len(recall) else 0.0,
        "f1": float(np.mean(f1)) if len(f1) else 0.0,
    }
    per_class = {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": cm.sum(axis=1).astype(int).tolist(),
    }
    return {"macro": macro, "per_class": per_class, "confusion_matrix": cm}


def topk_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int = 3) -> float:
    """
    y_proba: (N, C)
    """
    y_true = np.asarray(y_true).reshape(-1)
    topk = np.argpartition(-y_proba, kth=min(k, y_proba.shape[1]-1), axis=1)[:, :k]
    hits = np.any(topk == y_true[:, None], axis=1)
    return float(np.mean(hits))

