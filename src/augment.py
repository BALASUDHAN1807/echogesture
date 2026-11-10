"""Landmark-space augmentation + degradation used for training and robustness evaluation.

All functions take/return [T, FEAT_DIM] arrays (before velocity is appended).
"""
import numpy as np

from config import N_POSE, N_HAND, FEAT_LANDMARK, COORDS

_L = slice(N_POSE * COORDS, (N_POSE + N_HAND) * COORDS)
_R = slice((N_POSE + N_HAND) * COORDS, FEAT_LANDMARK)


def _pts(x):
    return x[:, :FEAT_LANDMARK].reshape(len(x), -1, COORDS)


def rotate(x, deg):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    out = x.copy()
    out[:, :FEAT_LANDMARK] = (_pts(x) @ R.T).reshape(len(x), -1)
    return out


def scale(x, f):
    out = x.copy()
    out[:, :FEAT_LANDMARK] *= f
    return out


def noise(x, sigma):
    out = x.copy()
    n = np.random.randn(len(x), FEAT_LANDMARK).astype(np.float32) * sigma
    mask = np.ones_like(n)
    mask[:, _L.start:_L.stop] *= x[:, FEAT_LANDMARK:FEAT_LANDMARK + 1]
    mask[:, _R.start:_R.stop] *= x[:, FEAT_LANDMARK + 1:FEAT_LANDMARK + 2]
    out[:, :FEAT_LANDMARK] += n * mask
    return out


def drop_hand(x, which, start=None, length=None):
    """Simulate occlusion: zero one hand (0=left, 1=right) over a time window."""
    out = x.copy()
    t = len(x)
    length = length or t
    start = 0 if start is None else start
    sl = slice(start, min(t, start + length))
    block = _L if which == 0 else _R
    out[sl, block] = 0
    out[sl, FEAT_LANDMARK + which] = 0
    return out


def drop_frames(x, p):
    """Simulate dropped frames: repeat the previous frame with probability p."""
    out = x.copy()
    for i in range(1, len(x)):
        if np.random.rand() < p:
            out[i] = out[i - 1]
    return out


def mirror(x):
    """Left/right-handed signers: flip x and swap hands + paired pose points."""
    out = x.copy()
    pts = _pts(out)
    pts[:, :, 0] *= -1
    for a, b in [(1, 2), (3, 4), (5, 6)]:
        pts[:, [a, b]] = pts[:, [b, a]]
    L, R = pts[:, N_POSE:N_POSE + N_HAND].copy(), pts[:, N_POSE + N_HAND:].copy()
    pts[:, N_POSE:N_POSE + N_HAND], pts[:, N_POSE + N_HAND:] = R, L
    out[:, :FEAT_LANDMARK] = pts.reshape(len(x), -1)
    out[:, FEAT_LANDMARK:FEAT_LANDMARK + 2] = x[:, FEAT_LANDMARK:FEAT_LANDMARK + 2][:, ::-1]
    return out


def time_warp(x, max_shift=3):
    s = np.random.randint(-max_shift, max_shift + 1)
    return np.roll(x, s, axis=0)


def random_augment(x):
    if np.random.rand() < 0.5:
        x = rotate(x, np.random.uniform(-20, 20))
    if np.random.rand() < 0.5:
        x = scale(x, np.random.uniform(0.85, 1.15))
    if np.random.rand() < 0.5:
        x = noise(x, np.random.uniform(0.002, 0.02))
    if np.random.rand() < 0.3:
        t = len(x)
        x = drop_hand(x, np.random.randint(2), np.random.randint(0, t // 2), np.random.randint(4, t // 2))
    if np.random.rand() < 0.3:
        x = drop_frames(x, np.random.uniform(0.05, 0.25))
    if np.random.rand() < 0.2:
        x = mirror(x)
    if np.random.rand() < 0.3:
        x = time_warp(x)
    return x


def degrade(x, seed=0):
    """Fixed harsh-condition transform for the robustness metric (angled signer, occlusion, jitter, drops)."""
    rng_state = np.random.get_state()
    np.random.seed(seed)
    x = rotate(x, 15)
    x = noise(x, 0.01)
    x = drop_hand(x, np.random.randint(2), len(x) // 3, len(x) // 4)
    x = drop_frames(x, 0.15)
    np.random.set_state(rng_state)
    return x
