"""Turn a labelled video list into a landmark dataset (data/features.npz).

Usage:
    python extract_landmarks.py --csv data/train.csv --out ../data/train.npz
CSV columns: path,label   (label = gloss/word; case-insensitive, stored uppercase)
Raw video is read once and never copied.
"""
import argparse
import numpy as np
from tqdm import tqdm

from features import extract_raw, normalize_frame, resample
from config import FEAT_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".", help="folder that CSV paths are relative to")
    a = ap.parse_args()

    import csv
    from pathlib import Path
    rows = list(csv.DictReader(open(a.csv)))
    X, y, kept = [], [], 0
    for r in tqdm(rows):
        try:
            raw = extract_raw(Path(a.root) / r["path"])
        except Exception as e:  # corrupt clip
            print("skip", r["path"], e)
            continue
        if not raw:
            continue
        seq = np.stack([normalize_frame(*f) for f in raw])
        X.append(resample(seq))
        y.append(r["label"].strip().upper())
        kept += 1
    np.savez_compressed(a.out, X=np.stack(X).astype(np.float32), y=np.array(y))
    print(f"saved {kept}/{len(rows)} clips -> {a.out}  (FEAT_DIM={FEAT_DIM})")


if __name__ == "__main__":
    main()
