# EchoGesture - on-device sign language recognition (Track 2)

Privacy-first pipeline: **video -> MediaPipe landmarks -> tiny INT8 temporal CNN -> word**.
Nothing leaves the device and no raw frames are stored; only landmark vectors exist in memory.

## Setup (Python 3.11)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Docker (inference only):
```bash
docker build -t echogesture .
docker run --rm -v $PWD/../clips:/clips echogesture python main.py --clips /clips --out /out/predictions.jsonl
```

## Run inference
```python
from main import predict
predict("clip.mp4")
# {"clip_id": "clip", "prediction": "HELLO", "top5": [...], "confidence": 0.81, "latency_ms": 23.4}
```
Batch: `python main.py --clips <folder or CSV(clip_id,path)> --out ../results/predictions.jsonl`

`latency_ms` is the full pipeline (landmarks + model) by default; `ECHO_LATENCY=model` reports model-only time.

## Reproduce training and the quantized model
The shipped model is trained on WLASL100 (the first 100 WLASL glosses) using the public 2D landmark files
released with SPOTER, so no raw video is needed:
```bash
# 1. data: three CSVs from https://github.com/matyasbohacek/spoter/releases/tag/supplementary-data
#    plus WLASL_v0.3.json from https://github.com/dxli94/WLASL (start_kit/)
python prepare_spoter.py --csv-dir <csv folder> --wlasl-json WLASL_v0.3.json --out-dir ../data

# 2. five seeds on train+val, fixed 300-epoch schedule (no model selection on the test set)
for s in 1 2 3 4 5; do python train.py --train ../data/train.npz --also-train ../data/val.npz \
    --val ../data/val.npz --save-last --seed $s --out echonet_s$s.pt; done

# 3. export the ensemble to one ONNX graph and quantize to INT8
python convert.py --calib ../data/train.npz --ckpt "echonet_s*.pt"     # -> models/echogesture_int8.onnx

# 4. metrics -> ../results/results.csv
python evaluate.py --test ../data/test.npz --device "<device>" --runtime "ONNX Runtime INT8 CPU" --power-mw <measured>
```
To train on your own videos instead, build the landmark files with MediaPipe:
`python extract_landmarks.py --csv ../data/train.csv --root ../data --out ../data/train.npz` (CSV columns: `path,label`).
Put a fixed vocabulary in `vocab.txt` (one word per line) to restrict the classes.

## Results (WLASL100 test split, 258 clips, 100 words)
| Metric | Target | Measured |
|---|---|---|
| Top-1 accuracy (INT8) | >= 70 % | 70.16 % (fp32: 69.38 %, top-5: 82.56 %) |
| Model size (INT8) | < 20 MB | 1.68 MB |
| Model latency p95 | <= 50 ms | 0.92 ms on a 2-core x86 cloud CPU |
| Accuracy drop under degradation | <= 5 pt | 3.88 pt |
| Latency drift over 10 min | <= 20 % | 0.66 % on the same cloud CPU (not a phone) |

A single model trained on the train split only scores 63.18 %.

Known limits, stated plainly:
- With 258 test clips, one clip is 0.39 pt, so 70.16 % is at the threshold, not clearly above it.
- Accuracy is measured on the dataset's own landmarks (Apple Vision). `predict()` extracts landmarks with
  MediaPipe; accuracy through that path has not been measured on real signing video yet.
- Latency above is the classifier only. MediaPipe landmark extraction dominates the full pipeline
  (about 1.1 s for a 32-frame clip on the same CPU). Power and thermal figures need a phone.

## Design
- **Pose-invariant features:** 7 upper-body joints + 21 landmarks per hand (2D). Centred on the mid-shoulder point, scaled by shoulder width, rotated to the shoulder line. Extra channels: wrist-centred hand shape, presence flags, frame-to-frame velocity (289 values per frame, 32 frames per clip).
- **Robustness:** training augmentation covers rotation, anisotropic scale, landmark noise, hand and arm-joint dropout, dropped frames, mirroring (left-handed signers), speed and trim changes, mixup. `evaluate.py` reports the accuracy drop on a fixed degraded copy of the test set (12 degree tilt, 10 % horizontal squeeze, jitter, a hand occlusion, 15 % dropped frames).
- **Model:** depthwise-separable dilated 1D CNN (285k parameters each), five seeds averaged in one graph, static INT8 QDQ quantization with ONNX Runtime.
- **Offline:** MediaPipe Holistic with `model_complexity=1`, whose weights ship inside the pip package (other settings download at runtime).
