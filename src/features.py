"""Video -> pose-invariant landmark sequence. Raw pixels are never stored or returned."""
import numpy as np
import cv2

from config import T, POSE_IDX, N_POSE, N_HAND, N_POINTS, FEAT_LANDMARK, FEAT_DIM

_holistic = None


def _get_holistic():
    global _holistic
    if _holistic is None:
        import mediapipe as mp
        _holistic = mp.solutions.holistic.Holistic(
            # complexity 1 ships inside the mediapipe wheel; 0 and 2 download at runtime (not offline-safe)
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.4, min_tracking_confidence=0.4)
    return _holistic


def _lm_to_array(lms, idx=None):
    if lms is None:
        return None
    pts = lms.landmark if idx is None else [lms.landmark[i] for i in idx]
    return np.array([[p.x, p.y, p.z] for p in pts], dtype=np.float32)


def extract_raw(video_path, max_frames=T * 3):
    """Return list of (pose[7,3] | None, left[21,3] | None, right[21,3] | None).

    Frames are sampled uniformly so long clips stay cheap on-device.
    """
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    keep = None
    if n > max_frames:
        keep = set(np.linspace(0, n - 1, max_frames).astype(int).tolist())
    hol = _get_holistic()
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if keep is None or i in keep:
            res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            out.append((_lm_to_array(res.pose_landmarks, POSE_IDX),
                        _lm_to_array(res.left_hand_landmarks),
                        _lm_to_array(res.right_hand_landmarks)))
        i += 1
    cap.release()
    return out


def normalize_frame(pose, left, right):
    """Shoulder-centred, shoulder-width-scaled, in-plane-rotation-aligned landmarks.

    Returns feature vector of length FEAT_DIM.
    """
    lm = np.zeros((N_POINTS, 3), dtype=np.float32)
    flags = np.zeros(2, dtype=np.float32)
    if pose is not None:
        lm[:N_POSE] = pose
    if left is not None:
        lm[N_POSE:N_POSE + N_HAND] = left
        flags[0] = 1
    if right is not None:
        lm[N_POSE + N_HAND:] = right
        flags[1] = 1

    present = np.ones(N_POINTS, dtype=bool)
    present[N_POSE:N_POSE + N_HAND] = bool(flags[0])
    present[N_POSE + N_HAND:] = bool(flags[1])
    present[:N_POSE] = pose is not None

    if pose is not None:
        ls, rs = lm[1], lm[2]                      # shoulders (indices into POSE_IDX order)
        origin = (ls + rs) / 2
        d = rs - ls
        scale = float(np.linalg.norm(d[:2])) or 1.0
        ang = np.arctan2(d[1], d[0])
    else:
        origin, scale, ang = np.zeros(3, np.float32), 1.0, 0.0
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    lm = (lm - origin) / scale
    lm = lm @ R.T
    lm[~present] = 0.0
    return np.concatenate([lm.reshape(-1), flags]).astype(np.float32)


def resample(seq, t=T):
    """Resample [N, F] to [t, F] by linear index interpolation (nearest-neighbour on frames)."""
    n = len(seq)
    if n == 0:
        return np.zeros((t, FEAT_DIM), dtype=np.float32)
    idx = np.linspace(0, n - 1, t).round().astype(int)
    return seq[idx]


def add_velocity(seq):
    """[T, FEAT_DIM] -> [T, FEAT_DIM + FEAT_LANDMARK] (frame-to-frame delta of landmarks)."""
    lm = seq[:, :FEAT_LANDMARK]
    vel = np.zeros_like(lm)
    vel[1:] = lm[1:] - lm[:-1]
    return np.concatenate([seq, vel], axis=1).astype(np.float32)


def video_to_model_input(video_path):
    raw = extract_raw(video_path)
    seq = np.stack([normalize_frame(*r) for r in raw]) if raw else np.zeros((0, FEAT_DIM), np.float32)
    return add_velocity(resample(seq))
