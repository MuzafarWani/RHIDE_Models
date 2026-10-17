import csv
import json
import numpy as np

pred_csv = "runs/vgg16/predictions_base100.csv"
npz_file = "data/vgg16_val.npz"

y_true = np.load(npz_file)["y"]

y_pred = []

with open(pred_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        y_pred.append(int(row["pred_class"]))

y_pred = np.asarray(y_pred)

assert len(y_true) == len(y_pred)

acc = (y_true == y_pred).mean()

print(f"Accuracy: {acc * 100:.2f}%")

