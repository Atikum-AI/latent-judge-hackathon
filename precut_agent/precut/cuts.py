"""Camera-stop detection for usable ranges (spec step 5).

PySceneDetect when it is installed; otherwise the built-in numpy detector: frames decoded at 12 fps,
downscaled to 160x90 grey, and a cut is where both the mean absolute difference and the histogram
distance spike (median + k*MAD). The fallback needs no downloads.
"""
from __future__ import annotations

import subprocess

import numpy as np

try:  # optional: the spec's detector (needs a Caio-approved pip install; see STATUS.md)
    from scenedetect import ContentDetector, SceneManager, open_video
    _HAVE_SD = True
except ImportError:
    _HAVE_SD = False

HAVE_SCENEDETECT = _HAVE_SD
SD_THRESHOLD = 27.0          # ContentDetector default
FALLBACK_FPS = 12
FALLBACK_WH = (160, 90)


def _cuts_scenedetect(path: str) -> list[float]:
    video = open_video(path)
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=SD_THRESHOLD))
    sm.detect_scenes(video, show_progress=False)
    return [float(t.get_seconds()) for t, _ in sm.get_scene_list() if t.get_seconds() > 0]


def _frames_gray(path: str) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf",
         f"fps={FALLBACK_FPS},scale={FALLBACK_WH[0]}:{FALLBACK_WH[1]}:flags=area", "-f", "rawvideo",
         "-pix_fmt", "gray", "-"],
        capture_output=True, check=True).stdout
    w, h = FALLBACK_WH
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)


def _cuts_numpy(path: str) -> list[float]:
    fr = _frames_gray(path)
    if len(fr) < 3:
        return []
    diff = np.abs(np.diff(fr, axis=0)).mean(axis=(1, 2))          # mean abs diff per frame pair
    hists = np.stack([np.histogram(f, bins=32, range=(0, 256), density=True)[0] for f in fr])
    hdist = np.abs(np.diff(hists, axis=0)).sum(axis=1)            # histogram L1 per frame pair
    def spikes(x: np.ndarray, k: float) -> np.ndarray:
        med, mad = np.median(x), np.median(np.abs(x - np.median(x))) + 1e-9
        return np.where(x > med + k * mad)[0]
    cand = set(spikes(diff, 12)) & set(spikes(hdist, 12))
    return sorted(round(float(i + 1) / FALLBACK_FPS, 3) for i in cand)


def find_cuts(path: str) -> tuple[list[float], str]:
    """Cut times in seconds and the detector used ('pyscenedetect' or 'numpy')."""
    if _HAVE_SD:
        try:
            return _cuts_scenedetect(path), "pyscenedetect"
        except Exception:
            pass  # fall through to the built-in detector; the caller logs which ran
    return _cuts_numpy(path), "numpy"
