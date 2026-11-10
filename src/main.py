"""EchoGesture - on-device sign recognition.

predict(video_path) -> dict   (the function the evaluator calls)
CLI:  python main.py --clips <dir or manifest.csv> --out ../results/predictions.jsonl

Everything runs locally: MediaPipe landmarks -> INT8 ONNX model. No network access, no video is saved.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from config import MODEL_PATH, CLASSES_PATH

_session = None
_classes = None


def _load():
    global _session, _classes
    if _session is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        _session = ort.InferenceSession(str(MODEL_PATH), so, providers=["CPUExecutionProvider"])
        _classes = json.loads(CLASSES_PATH.read_text())
    return _session, _classes


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def predict_from_features(x):
    """x: [T, MODEL_IN_DIM] float32 -> (probs over classes, model-only latency in ms)."""
    sess, classes = _load()
    t0 = time.perf_counter()
    logits = sess.run(None, {"input": x[None].astype(np.float32)})[0][0]
    ms = (time.perf_counter() - t0) * 1000
    return _softmax(logits), ms


def predict(video_path: str) -> dict:
    """Return the predicted gloss/word for one video clip. Runs fully on-device."""
    from features import video_to_model_input   # lazy: keeps import cheap
    t0 = time.perf_counter()
    x = video_to_model_input(video_path)
    probs, model_ms = predict_from_features(x)
    _, classes = _load()
    order = np.argsort(-probs)
    top5 = [classes[i].upper() for i in order[:5]]
    total_ms = (time.perf_counter() - t0) * 1000
    # ECHO_LATENCY=model reports model-only time; default reports the honest full pipeline time.
    latency = model_ms if os.environ.get("ECHO_LATENCY") == "model" else total_ms
    return {
        "clip_id": Path(video_path).stem,
        "prediction": top5[0],
        "top5": top5,
        "confidence": round(float(probs[order[0]]), 4),
        "latency_ms": round(float(latency), 1),
    }


def _clip_list(src):
    src = Path(src)
    if src.is_dir():
        return [(p.stem, p) for p in sorted(src.glob("*")) if p.suffix.lower() in {".mp4", ".mov", ".avi", ".webm", ".mkv"}]
    import csv
    rows = list(csv.DictReader(open(src)))
    base = src.parent
    return [(r["clip_id"], base / r["path"]) for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, help="folder of clips or CSV with clip_id,path")
    ap.add_argument("--out", default="../results/predictions.jsonl")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for cid, path in _clip_list(a.clips):
            r = predict(str(path))
            r["clip_id"] = cid                      # must match the test manifest exactly
            f.write(json.dumps(r) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
