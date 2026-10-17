#!/usr/bin/env python3
"""
Download CIFAR-10, resize to 224x224, and store in
data/images/{train,val,test}/<class_name>/

Run from project root:
python scripts/prepare_cifar10_vgg16.py
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10

# ----------------------------
# Configuration
# ----------------------------
OUT_ROOT = "data/images"
IMG_SIZE = (224, 224)
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1
SEED = 1337

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ----------------------------
# Helpers
# ----------------------------
def ensure_dirs():
    for split in ["train", "val", "test"]:
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(OUT_ROOT, split, cls), exist_ok=True)

def save_images(X, y, split):
    counters = {c: 0 for c in CLASS_NAMES}
    for img, label in zip(X, y):
        cls = CLASS_NAMES[int(label)]
        counters[cls] += 1
        fname = f"{cls}_{counters[cls]:05d}.jpg"
        path = os.path.join(OUT_ROOT, split, cls, fname)

        im = Image.fromarray(img)
        im = im.resize(IMG_SIZE, Image.BILINEAR)
        im.save(path, quality=95)

# ----------------------------
# Main
# ----------------------------
def main():
    print("[INFO] Downloading CIFAR-10...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    y_train = y_train.squeeze()
    y_test = y_test.squeeze()

    print("[INFO] Creating train/val/test split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=VAL_SPLIT,
        random_state=SEED,
        stratify=y_train
    )

    ensure_dirs()

    print("[INFO] Saving TRAIN images...")
    save_images(X_train, y_train, "train")

    print("[INFO] Saving VAL images...")
    save_images(X_val, y_val, "val")

    print("[INFO] Saving TEST images...")
    save_images(X_test, y_test, "test")

    print("[DONE] CIFAR-10 prepared for VGG16")
    print(f"[PATH] {OUT_ROOT}")

if __name__ == "__main__":
    main()

