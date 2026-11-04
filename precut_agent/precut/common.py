"""Shared pieces of the pre-cut agent: paths, manifest io, ffprobe, small helpers.

The agent is non-destructive by rule (lab-hq/plan/LONG_TAKES_AND_PRECUT.md): originals are never moved,
renamed or trimmed. Everything the agent learns lives in shoot_<date>/manifest.json; analysis caches
(decoded audio, downscaled frames) live in shoot_<date>/precut_cache/ and can be deleted freely.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone

SCHEMA = "shoot/manifest@1"
FPS = 24.0                      # exactly 24, never 23.976 (the plan is binding)
WIDTH, HEIGHT = 1280, 720       # capture size; the working copy is 1280x704 (crop 8 px top and bottom)
WORK_W, WORK_H = 1280, 704
WINDOW_FRAMES = 241             # 10 s at 24 fps; LTX needs 8k+1 frames
STRIDE_FRAMES = 192             # overlap ~49 frames
FRAME_MS = 1000.0 / FPS         # 41.7 ms, one frame

VIDEO_EXT = {".mov", ".mp4", ".m4v"}
AUDIO_EXT = {".wav", ".aif", ".aiff", ".mp3", ".m4a", ".caf"}


def shoot_dir(path: str) -> str:
    d = os.path.abspath(path)
    if not os.path.isdir(d):
        raise SystemExit(f"no such shoot folder: {d}")
    return d


def manifest_path(d: str) -> str:
    return os.path.join(d, "manifest.json")


def load_manifest(d: str) -> dict:
    p = manifest_path(d)
    if not os.path.exists(p):
        raise SystemExit(f"no manifest.json in {d} - create the folder with ./new_shoot.sh <date>")
    m = json.load(open(p))
    if m.get("schema") != SCHEMA:
        raise SystemExit(f"{p}: schema is {m.get('schema')!r}, expected {SCHEMA!r}")
    return m


def save_manifest(d: str, m: dict) -> None:
    """Atomic write (tmp + rename), so a crash never leaves half a manifest."""
    p = manifest_path(d)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".manifest.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")
    os.replace(tmp, p)


def utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def fps_of(stream: dict) -> tuple[float, str]:
    """(fps as float, exact string) = the CODEC frame rate (r_frame_rate). avg_frame_rate lies on
    Blackmagic files (time_base 1/480 rounds the container duration at stop, so 234 frames at 24 fps
    read as 24.005). Verified on real files 29 Oct. avg is exposed separately for the warning path."""
    raw = stream.get("r_frame_rate") or stream.get("avg_frame_rate")
    if raw and raw != "0/0":
        num, den = (int(x) for x in raw.split("/"))
        return num / den, raw
    return 0.0, "0/0"


def avg_fps_of(stream: dict) -> str:
    return stream.get("avg_frame_rate") or "0/0"


def s_to_tc(s: float) -> str:
    """Seconds -> HH:MM:SS (the ingest contract's tc pattern)."""
    s = max(0, int(round(s)))
    return f"{s // 3600:02d}:{s // 60 % 60:02d}:{s % 60:02d}"


def tc_to_s(tc: str) -> float:
    """HH:MM:SS[:FF] -> seconds (frames dropped: sync uses timecode to the second, the clap refines)."""
    parts = [int(x) for x in tc.replace(";", ":").split(":")][:3]
    while len(parts) < 3:
        parts.append(0)
    return float(parts[0] * 3600 + parts[1] * 60 + parts[2])


def cache_dir(d: str, sub: str) -> str:
    p = os.path.join(d, "precut_cache", sub)
    os.makedirs(p, exist_ok=True)
    return p


def decide(m: dict, take_id: str | None, step: str, decision: str, reason: str) -> None:
    """Every agent decision is logged with its reason (spec step 6: 'every decision is logged with its reason')."""
    m.setdefault("decisions", []).append(
        {"at": utcnow(), "take_id": take_id, "step": step, "decision": decision, "reason": reason})


def clear_decisions(m: dict, step: str) -> None:
    """A re-run of a step replaces its earlier decisions (the log describes the CURRENT manifest)."""
    m["decisions"] = [d for d in m.get("decisions", []) if d.get("step") != step]
