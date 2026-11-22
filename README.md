# EchoGesture: on-device sign language recognition

Samsung PRISM XR Hackathon 2026, Track 2.

EchoGesture recognises isolated sign-language words from a video clip, entirely on the device.
MediaPipe turns each frame into body and hand landmarks, the frames are discarded, and a small INT8
temporal CNN predicts the word from the landmarks alone. No video is stored or sent anywhere.

## Results (WLASL100 test split, 258 real videos, 100 words)

| Metric | Target | Measured |
|---|---|---|
| Top-1 accuracy (INT8) | >= 70 % | 84.11 % (top-5: 95.35 %) |
| Model size (INT8) | < 20 MB | 1.68 MB |
| Model latency p95 | <= 50 ms | 1.09 ms, classifier only, 2-core x86 cloud CPU |
| Accuracy drop under degradation | <= 5 pt | 1.94 pt |
| Average power | <= 500 mW | not measured (needs a phone) |

Full metrics are in [`results/results.csv`](results/results.csv). Limits and the full-pipeline latency
(about 1.4 s per clip on the same CPU, mostly MediaPipe) are described in [`src/README.md`](src/README.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/main.py` | `predict(video_path) -> dict`, the inference entry point |
| `src/` | feature extraction, augmentation, model, training, INT8 conversion, evaluation |
| `src/models/echogesture_int8.onnx` | the quantized model (5-seed ensemble, 1.68 MB) |
| `src/README.md` | setup, inference, how to reproduce training, design notes |
| `src/Dockerfile` | inference-only container |
| `results/results.csv` | measured metrics |
| `results/predictions_wlasl100_test.jsonl` | `predict()` output for the 258 WLASL100 test videos |
| `docs/SRMIST_EchoGesture_demo.mp4` | 2-minute demo video: real test clips captioned by the model, plus a stress test |
| `docs/SRMIST_EchoGesture.pptx`, `.pdf` | presentation |
| [Canva version](https://www.canva.com/d/swOz12dXGJAmoNb) | the same presentation, designed in Canva (view link) |
| `AI_DISCLOSURE.md` | how AI tools were used in this project |

## Quick start (Python 3.11)

```bash
cd src
pip install -r requirements.txt
python main.py --clips <folder of clips> --out ../results/predictions.jsonl
```
