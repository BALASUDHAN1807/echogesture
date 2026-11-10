"""Measure the scored metrics on a held-out landmark set and write results/results.csv.

    python evaluate.py --test ../data/test.npz --device "Galaxy S25" --runtime "ORT INT8 CPU"
avg_power needs a real phone (e.g. Android battery stats / Snapdragon Profiler): pass --power-mw.
"""
import argparse
import csv
import json
import os
import time
import numpy as np

from augment import degrade
from config import MODEL_PATH, CLASSES_PATH
from features import add_velocity
from main import predict_from_features


def accuracy(X, y, classes, transform=None):
    idx = {c: i for i, c in enumerate(classes)}
    top1 = top5 = n = 0
    lat = []
    for x, lab in zip(X, y):
        if lab not in idx:
            continue
        if transform:
            x = transform(x)
        p, ms = predict_from_features(add_velocity(x))
        order = np.argsort(-p)
        top1 += order[0] == idx[lab]
        top5 += idx[lab] in order[:5]
        lat.append(ms)
        n += 1
    return 100 * top1 / n, 100 * top5 / n, lat


def thermal_drift(X, minutes):
    """% change of median model latency between first and last minute of sustained inference."""
    x = add_velocity(X[0])
    end = time.time() + minutes * 60
    first, last, t0 = [], [], time.time()
    while time.time() < end:
        _, ms = predict_from_features(x)
        (first if time.time() - t0 < minutes * 6 else last).append(ms)   # first/last 10 % windows
        if time.time() > end - minutes * 6:
            last.append(ms)
    first = first or [1.0]
    last = last or first
    return 100 * (np.median(last) - np.median(first)) / np.median(first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="../results/results.csv")
    ap.add_argument("--device", default="")
    ap.add_argument("--runtime", default="ONNX Runtime INT8 CPU")
    ap.add_argument("--power-mw", type=float, default=None)
    ap.add_argument("--thermal-minutes", type=float, default=10)
    a = ap.parse_args()

    d = np.load(a.test)
    classes = json.loads(CLASSES_PATH.read_text())
    t1, t5, lat = accuracy(d["X"], d["y"], classes)
    t1d, _, _ = accuracy(d["X"], d["y"], classes, lambda x: degrade(x))
    size = os.path.getsize(MODEL_PATH) / 1e6
    drift = thermal_drift(d["X"], a.thermal_minutes) if a.thermal_minutes > 0 else None
    p95 = float(np.percentile(lat, 95))

    def row(metric, target, val, unit, ok):
        return [metric, target, "" if val is None else f"{val:.2f}", unit, a.device, a.runtime,
                "" if val is None else ("PASS" if ok(val) else "FAIL")]

    rows = [
        row("top1_accuracy", ">=70", t1, "%", lambda v: v >= 70),
        row("latency_p95", "<=50", p95, "ms", lambda v: v <= 50),
        row("model_size", "<=25", size, "MB", lambda v: v <= 25),
        row("accuracy_drop_robustness", "<=5", t1 - t1d, "pt", lambda v: v <= 5),
        row("avg_power", "<=500", a.power_mw, "mW", lambda v: v <= 500),
        row("thermal_drift_10min", "<=20", drift, "%", lambda v: v <= 20),
    ]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        w.writerows(rows)
    print(f"top1 {t1:.1f}%  top5 {t5:.1f}%  degraded top1 {t1d:.1f}%  p95 {p95:.2f} ms (model only)  size {size:.2f} MB")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
