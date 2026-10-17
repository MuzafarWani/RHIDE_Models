#!/usr/bin/env python3
"""
Create VGG16 inference NPZ from folder-structured images.

Expected structure:
  root/
    class1/
      img1.jpg
      img2.png
    class2/
      ...

Output:
  NPZ with:
    X          -> (N, 224, 224, 3)
    filenames  -> image file names
    classes    -> folder name (optional, not used for inference)

Example:
  python create_vgg16_inference_npz.py \
    --image_root data/images/val \
    --out data/vgg16_inference.npz
"""

import argparse
import os
import numpy as np
from PIL import Image


IMG_EXTS = (".jpg", ".jpeg", ".png")


def load_images_from_folders(root_dir, img_size):
    X = []
    filenames = []
    classes = []

    for cls in sorted(os.listdir(root_dir)):
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue

        for fname in sorted(os.listdir(cls_path)):
            if not fname.lower().endswith(IMG_EXTS):
                continue

            path = os.path.join(cls_path, fname)

            img = Image.open(path).convert("RGB")
            img = img.resize((img_size, img_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0

            X.append(arr)
            filenames.append(fname)
            classes.append(cls)

    if not X:
        raise RuntimeError(f"No images found in {root_dir}")

    return (
        np.stack(X, axis=0),
        np.array(filenames),
        np.array(classes),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_root", required=True,
                    help="Path like data/images/val or data/images/test")
    ap.add_argument("--out", required=True,
                    help="Output NPZ file")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    X, filenames, classes = load_images_from_folders(
        args.image_root, args.img_size
    )

    np.savez(
        args.out,
        X=X,
        filenames=filenames,
        classes=classes,
    )

    print(f"[OK] Saved inference NPZ → {args.out}")
    print(f"     X shape      : {X.shape}")
    print(f"     Num samples  : {len(X)}")
    print(f"     Classes seen : {sorted(set(classes.tolist()))}")


if __name__ == "__main__":
    main()

