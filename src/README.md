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
```bash
python extract_landmarks.py --csv ../data/train.csv --root ../data --out ../data/train.npz   # also val / test
python train.py   --train ../data/train.npz --val ../data/val.npz
python convert.py --calib ../data/train.npz        # -> models/echogesture_int8.onnx (<20 MB, INT8)
python evaluate.py --test ../data/test.npz --device "<phone>" --runtime "ONNX Runtime INT8 CPU" --power-mw <measured>
```
Datasets: WLASL (+ How2Sign). Put the organizers' vocabulary in `vocab.txt` (one word per line) before training.

## Design
- **Pose-invariant features:** 7 upper-body pose points + 21 landmarks per hand, centred on the shoulders, scaled by shoulder width, rotated to the shoulder axis.
- **Robustness:** training augmentation covers rotation, scale, noise, dropped hands, dropped frames, mirroring (left-handed signers), time shifts. `evaluate.py` reports the accuracy drop on a fixed degraded set.
- **Model:** depthwise-separable dilated 1D CNN (~0.1M params, 0.14 MB after INT8 in the smoke test), static INT8 QDQ quantization.
