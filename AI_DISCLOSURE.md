# AI Disclosure

## AI assistant used

**Claude** (Anthropic), used through the Claude app.

## What it did

- Proposed the overall approach: landmark-only recognition with MediaPipe, a small temporal CNN, and INT8 quantization.
- Wrote the source code in `src/`: feature extraction and normalisation, augmentation, model, training,
  ONNX export and INT8 quantization, evaluation and `predict()`.
- Found the training data, ran landmark extraction, training, quantization and evaluation, and produced
  `results/results.csv` and `results/predictions_wlasl100_test.jsonl`.
- Made the demo video and the presentation in `docs/`.
- Wrote the README files and this disclosure.

## What the team did

- Chose Track 2 and the project direction, and directed the work.
- Reviewed and submitted the project.

## Third-party models, data and libraries

- **MediaPipe Holistic** (Google): pretrained landmark detector, used unchanged.
- **WLASL** (Li et al., 2020): word list, splits and videos for the first 100 glosses
  (videos obtained from the GitHub repository `4n-ngth/WLASL100_I3D`).
- **SPOTER supplementary data** (Boháček and Hrúz, 2022): public 2D landmark files for WLASL100.
- PyTorch, ONNX and ONNX Runtime for training and inference.

No pretrained sign-language recognition model was used. The classifier was trained from scratch.
