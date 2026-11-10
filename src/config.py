"""Shared constants. Keep feature layout identical for training and inference."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "echogesture_int8.onnx"
CLASSES_PATH = MODEL_DIR / "classes.json"
VOCAB_PATH = ROOT / "vocab.txt"

T = 32                      # frames per clip after temporal resampling
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]   # nose, shoulders, elbows, wrists (MediaPipe pose)
N_POSE = len(POSE_IDX)
N_HAND = 21
N_POINTS = N_POSE + 2 * N_HAND           # 49 landmarks
COORDS = 3
# per frame: landmarks (49*3) + 2 hand-present flags
FEAT_LANDMARK = N_POINTS * COORDS
FEAT_DIM = FEAT_LANDMARK + 2
# model input adds velocity of landmark part -> FEAT_LANDMARK extra
MODEL_IN_DIM = FEAT_DIM + FEAT_LANDMARK

SEED = 42


def load_vocab():
    """Optional fixed vocabulary (uppercase, one word per line). Empty list = use labels found in data."""
    if not VOCAB_PATH.exists():
        return []
    return [w.strip().upper() for w in VOCAB_PATH.read_text().splitlines() if w.strip()]
