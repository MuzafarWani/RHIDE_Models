# src/eval/reports.py
from __future__ import annotations

import os
from typing import Dict, Any, Optional

import numpy as np

from .metrics import accuracy, precision_recall_f1


def make_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Returns a dict with accuracy, macro PRF1, per-class PRF1, and confusion matrix.
    """
    acc = accuracy(y_true, y_pred)
    prf = precision_recall_f1(y_true, y_pred, None)
    if class_names and len(class_names) == len(prf["per_class"]["support"]):
        prf["class_names"] = class_names
    return {"accuracy": acc, **prf}


def save_report_markdown(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    lines = []
    lines.append(f"# Classification Report\n")
    lines.append(f"- Accuracy: **{report['accuracy']:.4f}**")
    macro = report["macro"]
    lines.append(f"- Macro Precision: **{macro['precision']:.4f}**")
    lines.append(f"- Macro Recall: **{macro['recall']:.4f}**")
    lines.append(f"- Macro F1: **{macro['f1']:.4f}**\n")

    # Per-class table
    names = report.get("class_names", [f"class_{i}" for i in range(len(report['per_class']['support']))])
    pc = report["per_class"]
    lines.append("| Class | Support | Precision | Recall | F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for i, name in enumerate(names):
        lines.append(
            f"| {name} | {pc['support'][i]} | {pc['precision'][i]:.4f} | {pc['recall'][i]:.4f} | {pc['f1'][i]:.4f} |"
        )
    lines.append("")

    # Confusion matrix
    cm = np.asarray(report["confusion_matrix"])
    lines.append("## Confusion Matrix")
    header = "| |" + "|".join(names) + "|"
    sep = "|:--|" + "|".join([":--:" for _ in names]) + "|"
    lines.append(header)
    lines.append(sep)
    for i, name in enumerate(names):
        row = "| " + name + " | " + " | ".join(str(int(x)) for x in cm[i]) + " |"
        lines.append(row)

    with open(path, "w") as f:
        f.write("\n".join(lines))


def save_report_json(report: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    import json
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

