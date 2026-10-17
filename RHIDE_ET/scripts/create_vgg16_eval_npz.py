#!/usr/bin/env python3
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

ROOT = "data/images/val"
IMG_SIZE = (224, 224)
CLASS_NAMES = sorted(os.listdir(ROOT))

X, y = [], []

for label, cls in enumerate(CLASS_NAMES):
    cls_dir = os.path.join(ROOT, cls)
    for fname in os.listdir(cls_dir):
        path = os.path.join(cls_dir, fname)
        img = load_img(path, target_size=IMG_SIZE)
        arr = img_to_array(img) / 255.0
        X.append(arr)
        y.append(label)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y, dtype=np.int64)

np.savez("data/vgg16_val.npz", X=X, y=y, class_names=CLASS_NAMES)

print("Saved:", X.shape, y.shape)

