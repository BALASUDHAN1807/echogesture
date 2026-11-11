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
The shipped model covers WLASL100 (the first 100 WLASL glosses). It is trained on two landmark views of the
same train+val clips: MediaPipe landmarks extracted from the WLASL videos, and the public 2D landmark files
released with SPOTER (Apple Vision landmarks).
```bash
# 1. MediaPipe landmarks from the WLASL videos (CSV columns: path,label)
python extract_landmarks.py --csv trainval_a.csv --root <video folder> --frames 48 --out ../data/trainval_mp_a.npz
python extract_landmarks.py --csv trainval_b.csv --root <video folder> --frames 48 --out ../data/trainval_mp_b.npz
python extract_landmarks.py --csv test.csv       --root <video folder> --frames 32 --out ../data/test_mp.npz

# 2. SPOTER landmark CSVs (https://github.com/matyasbohacek/spoter/releases/tag/supplementary-data)
#    plus WLASL_v0.3.json from https://github.com/dxli94/WLASL (start_kit/)
python prepare_spoter.py --csv-dir <csv folder> --wlasl-json WLASL_v0.3.json --out-dir ../data

# 3. five seeds on train+val, fixed 200-epoch schedule (no checkpoint selection on the test set)
for s in 1 2 3 4 5; do python train.py --train ../data/trainval_mp_a.npz \
    --also-train ../data/trainval_mp_b.npz ../data/train.npz ../data/val.npz \
    --val ../data/val.npz --save-last --epochs 200 --seed $s --out echonet_mp_s$s.pt; done

# 4. export the ensemble to one ONNX graph and quantize to INT8
python convert.py --calib ../data/trainval_mp_a.npz --ckpt "echonet_mp_s*.pt"   # -> models/echogesture_int8.onnx

# 5. metrics -> ../results/results.csv
python evaluate.py --test ../data/test_mp.npz --device "<device>" --runtime "ONNX Runtime INT8 CPU" --power-mw <measured>
```
Put a fixed vocabulary in `vocab.txt` (one word per line) to restrict the classes.

## Results (WLASL100 test split, 258 real videos, 100 words)
Measured through the same MediaPipe path that `predict()` uses. Exact figures are in `../results/results.csv`.

| Metric | Target | Measured |
|---|---|---|
| Top-1 accuracy (INT8) | >= 70 % | 84.11 % (fp32: 84.50 %, top-5: 95.35 %) |
| Model size (INT8) | < 20 MB | 1.68 MB |
| Model latency p95 | <= 50 ms | about 1 ms on a 2-core x86 cloud CPU (classifier only) |
| Accuracy drop under degradation | <= 5 pt | 1.94 pt (82.17 % degraded) |
| Latency drift over 10 min | <= 20 % | -7.7 % on the same cloud CPU (noise; not a phone) |
| Full pipeline per clip (`predict()`) | n/a | median 1.43 s, p95 1.52 s on the same CPU |

Running the literal `predict()` on all 258 test videos gives 84.50 % top-1 / 95.35 % top-5
(`../results/predictions_wlasl100_test.jsonl`). It differs from the table by one clip because that run used
Python 3.11 with a different ONNX Runtime build.

Earlier versions, for reference: a single model trained on SPOTER landmarks only scored 63.18 % on the
SPOTER-landmark test set; a five-seed ensemble of that scored 70.16 % there and 79.46 % on the real videos.
The current model scores 72.87 % on the SPOTER-landmark test set.

Known limits, stated plainly:
- The test set has 258 clips, so the 95 % confidence interval on 84 % is roughly +/- 4.5 points.
- The current model was chosen over the previous one after seeing both scores on this test set.
- WLASL test signers and recording conditions resemble the training data. Accuracy on new signers,
  smart-glasses viewpoints, or a different vocabulary will be lower and has not been measured.
- Latency above is the classifier only. MediaPipe landmark extraction dominates the full pipeline
  (about 1.4 s for a 32-frame clip on the same CPU, i.e. roughly 45 ms per frame). Power and thermal
  figures need a phone.

## Design
- **Pose-invariant features:** 7 upper-body joints + 21 landmarks per hand (2D). Centred on the mid-shoulder point, scaled by shoulder width, rotated to the shoulder line. Extra channels: wrist-centred hand shape, presence flags, frame-to-frame velocity (289 values per frame, 32 frames per clip).
- **Robustness:** training augmentation covers rotation, anisotropic scale, landmark noise, hand and arm-joint dropout, dropped frames, mirroring (left-handed signers), speed and trim changes, mixup, plus two different landmark extractors per training clip. `evaluate.py` reports the accuracy drop on a fixed degraded copy of the test set (12 degree tilt, 10 % horizontal squeeze, jitter, a hand occlusion, 15 % dropped frames).
- **Model:** depthwise-separable dilated 1D CNN (285k parameters each), five seeds averaged in one graph, static INT8 QDQ quantization with ONNX Runtime.
- **Offline:** MediaPipe Holistic with `model_complexity=1`, whose weights ship inside the pip package (other settings download at runtime).
