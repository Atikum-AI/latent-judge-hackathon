"""Pre-cut agent steps 2-3 - audio sync and the drift check.

  python -m precut.sync shoot_<date> [--search-s 25]

The external recorder's file is matched to the phone's scratch audio by cross-correlating the clap
(GCC-PHAT, robust across different mics); timecode is used first when both files carry it. The offset
is measured at the START and at the END of the take; if it moved, the external audio is resampled
(a copy in precut_cache/, the recorder's file is never touched) and the drift is logged per take.

offset_ms > 0 means the external file's content runs LATE by that much (a sound at scratch time T sits
at ext time T + offset):  scratch_t = ext_t - offset.  drift_ms is how much the offset grows from take
start to take end (positive: the recorder's clock runs slow).
"""
from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

from . import common as C

SR = 16000
WINDOW_S = 12.0          # correlation window at each end of the take
SUB = 8                  # sub-sample peak refinement factor (parabolic is enough; kept simple)


def audio_f32(path: str, cache: str | None = None) -> np.ndarray:
    """Decode to 16 kHz mono float32. Cached under precut_cache/ when a cache dir is given."""
    if cache:
        key = os.path.join(cache, C.sha256_file(path)[:16] + ".npy")
        if os.path.exists(key):
            return np.load(key)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32).copy()
    if cache:
        np.save(key, a)
    return a


def _phat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """GCC-PHAT cross-correlation of b against a (full)."""
    n = 1 << (len(a) + len(b) - 1).bit_length()
    A, B = np.fft.rfft(a, n), np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    return np.fft.irfft(R, n)


def offset_samples(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, float]:
    """Lag of b relative to a at the best GCC-PHAT peak: b = a delayed by `lag` samples (b late =
    positive). Only valid lags are searched: [-(len(b)-1), len(a)-1] (each side is limited by the
    OTHER signal's length)."""
    cc = _phat(a, b)
    lo, hi = max(-max_lag, -(len(b) - 1)), min(max_lag, len(a) - 1)
    lags = np.arange(lo, hi + 1)
    vals = np.array([cc[l % len(cc)] for l in lags])
    i = int(np.argmax(vals))
    lag = float(lags[i])
    if 0 < i < len(vals) - 1:  # parabolic sub-sample refinement (lags are ascending here)
        y0, y1, y2 = vals[i - 1], vals[i], vals[i + 1]
        den = y0 - 2 * y1 + y2
        if abs(den) > 1e-12:
            lag += float(np.clip(0.5 * (y0 - y2) / den, -1, 1))
    weight = float(vals[i] / (np.abs(vals).mean() + 1e-12))
    return lag, weight


def correlate(take_audio: np.ndarray, ext_audio: np.ndarray, search_s: float) -> tuple[float, float]:
    """Offset in seconds: ext_pos = scratch_pos + offset (ext content late = positive)."""
    max_lag = int(search_s * SR)
    lag, w = offset_samples(take_audio, ext_audio, max_lag)
    return -lag / SR, w


def offset_at(scratch: np.ndarray, ext_audio: np.ndarray, t_s: float, w_s: float, est_s: float,
              slack_s: float = 3.0) -> tuple[float, float]:
    """The offset at one point of the take: correlate the scratch window around t_s against the ext
    window around its expected position t_s + est. Returns (offset_s, weight) in scratch-time terms."""
    a = _window(scratch, t_s, w_s)
    centre = t_s + est_s
    b = _window(ext_audio, centre, w_s + 2 * slack_s)
    b_start = max(0.0, centre - (w_s / 2 + slack_s))
    max_lag = int((w_s / 2 + slack_s + 0.5) * SR)
    lag, w = offset_samples(a, b, max_lag)
    # PHAT convention (measured): lag = -(delay of a's content inside b). a's window starts at scratch
    # t_s - w_s/2, so the offset is (ext position of a's start) - (scratch position of a's start).
    off = (b_start - lag / SR) - (t_s - w_s / 2)
    return off, w


def _window(a: np.ndarray, center_s: float, width_s: float) -> np.ndarray:
    lo = max(0, int((center_s - width_s / 2) * SR))
    hi = min(len(a), lo + int(width_s * SR))
    return a[lo:hi]


def run(d: str, search_s: float = 25.0) -> dict:
    m = C.load_manifest(d)
    C.clear_decisions(m, "sync")
    snd_dir = os.path.join(d, "sound")
    externals = sorted(n for n in os.listdir(snd_dir) if os.path.splitext(n)[1].lower() in C.AUDIO_EXT | C.VIDEO_EXT) \
        if os.path.isdir(snd_dir) else []
    if not m.get("takes"):
        raise SystemExit("no takes in the manifest yet - run takes.py first")
    if not externals:
        for t in m["takes"]:
            t["sync"] = {"source": "none", "offset_ms": None, "drift_ms": None, "drift_frames": 0.0,
                         "resampled": False, "confidence": None}
        C.decide(m, None, "sync", "no external sound", "sound/ is empty; takes carry scratch audio only")
        C.save_manifest(d, m)
        print("sound/ is empty - sync skipped, offsets null")
        return m

    cache = C.cache_dir(d, "audio")
    ext_audio = {name: audio_f32(os.path.join(snd_dir, name), cache) for name in externals}
    ext_tc = {}
    for name in externals:
        if os.path.splitext(name)[1].lower() in C.VIDEO_EXT:
            try:
                j = C.ffprobe(os.path.join(snd_dir, name))
                tags = j.get("format", {}).get("tags") or {}
                ext_tc[name] = tags.get("timecode") or next(
                    ((s.get("tags") or {}).get("timecode") for s in j["streams"] if (s.get("tags") or {}).get("timecode")), None)
            except Exception:
                ext_tc[name] = None

    for t in m["takes"]:
        f = next((x for x in m["files"] if x["name"] == t["file"]), None)
        if f is None or not f["checks"]["ok"]:
            continue
        path = os.path.join(d, f["rel_path"])
        scratch = audio_f32(path, cache)
        tc_scratch = (f.get("probe") or {}).get("timecode")
        best = None  # (weight, name, offset_s)
        for name, ea in ext_audio.items():
            off, w = correlate(scratch, ea, search_s)
            if best is None or w > best[0]:
                best = (w, name, off)
        if best is None:
            continue
        weight, ext_name, off_s = best
        source = "clap gcc-phat"
        if tc_scratch and ext_tc.get(ext_name):
            # spec step 2: timecode first when both sides carry it
            tc_off = (C.tc_to_s(ext_tc[ext_name]) - C.tc_to_s(tc_scratch))
            C.decide(m, t["id"], "sync", f"timecode offset {round(tc_off * 1000, 1)} ms",
                     f"both files carry timecode ({tc_scratch} vs {ext_tc[ext_name]}); the clap "
                     f"correlation ({round(off_s * 1000, 1)} ms) is the cross-check")
            off_s, source = tc_off, "timecode"

        # drift: the offset again at the END of the take; how much it moved is the drift (spec step 3)
        dur = len(scratch) / SR
        drift_s: float | None = None
        w_s = min(WINDOW_S, max(2.0, dur * 0.3))               # short takes get short windows
        if dur >= 6.0:
            ea = ext_audio[ext_name]
            ext_dur = len(ea) / SR
            t0, t1 = w_s / 2, dur - w_s / 2                  # start and end of the take
            if t1 + off_s + w_s / 2 > ext_dur + 0.5:
                C.decide(m, t["id"], "sync", "drift not measured",
                         f"the external file ends at {ext_dur:.1f} s, before the take's end window "
                         f"({t1 + off_s:.1f} s with the offset) - recorder stopped early?")
            else:
                off0, w0 = offset_at(scratch, ea, t0, w_s, off_s)
                off1, w1 = offset_at(scratch, ea, t1, w_s, off_s)
                if min(w0, w1) < 10:
                    C.decide(m, t["id"], "sync", "drift not measured",
                             f"correlation peak too weak at an end (weights {w0:.1f}/{w1:.1f}); "
                             "the offset stands, the drift is unknown")
                else:
                    drift_s = off1 - off0
        else:
            C.decide(m, t["id"], "sync", "drift not measured", f"take is {dur:.1f} s; too short for a "
                       "start/end offset comparison (needs >= 6 s)")
        drift_ms = None if drift_s is None else round(drift_s * 1000, 1)

        # timecode was already honoured above when both sides carry it
        offset_ms = round(off_s * 1000, 1)
        resampled = False
        if drift_s is not None and abs(drift_ms) > C.FRAME_MS / 2:  # > half a frame: resample the copy
            ratio = (dur + drift_s) / dur
            out = os.path.join(C.cache_dir(d, "sync"), f"{os.path.splitext(ext_name)[0]}.resampled.wav")
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", os.path.join(snd_dir, ext_name),
                            "-af", f"asetrate={SR}*{1 / ratio:.9f},aresample={SR}", "-ar", str(SR), out],
                           check=True)
            resampled = True
            C.decide(m, t["id"], "sync", f"resampled {ext_name} by {1 / ratio:.6f}",
                     f"offset moved {drift_ms} ms over {dur:.0f} s (device clock drift); the copy in "
                     f"precut_cache/sync/ is corrected, the recorder's file is untouched")
        t["sync"] = {"source": source,
                     "external_file": ext_name, "offset_ms": offset_ms, "drift_ms": drift_ms,
                     "drift_frames": None if drift_ms is None else round(abs(drift_ms) / C.FRAME_MS, 2),
                     "resampled": resampled, "confidence": round(weight, 1)}
        C.decide(m, t["id"], "sync", f"{ext_name} matched at {offset_ms} ms, drift {drift_ms} ms",
                 f"GCC-PHAT peak weight {weight:.1f}; measured at take start and end")
        print(f"{t['id']} {t['file']}: ext={ext_name} offset {offset_ms} ms, drift {drift_ms} ms "
              f"({t['sync']['drift_frames']} f), resampled={resampled}")

    C.save_manifest(d, m)
    return m


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    if len(args) != 1:
        raise SystemExit("usage: python -m precut.sync shoot_<date> [--search-s=25]")
    run(C.shoot_dir(args[0]), float(opts.get("search-s", 25)))
