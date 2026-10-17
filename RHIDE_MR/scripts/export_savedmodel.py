#!/usr/bin/env python3
# scripts/export_savedmodel.py
from __future__ import annotations

import os
import argparse
import tensorflow as tf

from src.utils.io_helpers import read_yaml, ensure_dir
from src.models.xeventnet import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_cfg", default="configs/train.yaml")
    ap.add_argument("--weights_path", default="runs/expA/savedmodel", help="Keras SavedModel or .h5")
    ap.add_argument("--export_dir", default="runs/expA/exported_savedmodel")
    args = ap.parse_args()

    train_cfg = read_yaml(args.train_cfg)
    ensure_dir(args.export_dir)

    # Try to load as SavedModel first; else rebuild architecture and load weights
    try:
        model = tf.keras.models.load_model(args.weights_path, compile=False)
        print(f"[export] loaded model from SavedModel: {args.weights_path}")
    except Exception:
        model = build_model(tuple(train_cfg["input_shape_2d"]), tuple(train_cfg["input_shape_3d"]), num_classes=int(train_cfg["num_classes"]))
        model.load_weights(args.weights_path)
        print(f"[export] loaded weights into fresh model: {args.weights_path}")

    tf.saved_model.save(model, args.export_dir)
    print(f"[export] SavedModel exported -> {args.export_dir}")


if __name__ == "__main__":
    main()

