"""Synthetic test shoot for the pre-cut agent: no phone needed.

  python -m precut.make_test_fixtures [shoot_2099-01-15]

Creates shoot_2099-01-15/ with three "takes" (1280x720, exactly 24 fps, HEVC 10-bit, BT.2020 tags):
  take1.mov   slate "scene 1 shot 1 A take 1", clap, action, dialogue "You said you'd be here by six.", cut
  take2.mov   same, take 2, plus "circle that" after cut      (the keeper)
  take3.mov   same, take 3, with a hard picture cut at 6.0 s  (camera stop inside the range)
  bad_2398.mov  23.976 fps - ingest must refuse it with a warning and keep it in the files list
  sound/take2_ext.wav   take 2's audio 350 ms late and 500 ppm slow (device drift), for sync.py

Speech is macOS `say`; the clap is a 1 kHz beep. Nothing here is footage of anyone.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

from . import common as C

SR = 48000
DURATION = {"take1.mov": 9.6, "take2.mov": 9.8, "take3.mov": 12.0}
DIALOGUE = {"take1.mov": "You said you'd be here by six.", "take2.mov": "You said you'd be here by six.",
            "take3.mov": "Then we start again."}
TAKE_NO = {"take1.mov": 1, "take2.mov": 2, "take3.mov": 3}
# slate at 0.3 s, clap beep at 2.8 s, action 3.6 s, dialogue 4.2 s, cut 7.8 s, circle-that 8.6 s (take 2)


def say_segment(text: str, tmp: str) -> np.ndarray:
    aiff = os.path.join(tmp, "seg.aiff")
    subprocess.run(["say", "-o", aiff, text], check=True)
    raw = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", aiff, "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    a = np.frombuffer(raw, np.float32)
    peak = np.abs(a).max()
    return a / peak * 0.35 if peak > 0 else a


def clap() -> np.ndarray:
    t = np.arange(int(0.08 * SR)) / SR
    ramp = np.minimum(1.0, np.arange(len(t)) / (0.005 * SR))
    return (0.9 * np.sin(2 * np.pi * 1000 * t) * ramp).astype(np.float32)


def build_audio(name: str, tmp: str) -> np.ndarray:
    dur = DURATION[name]
    out = np.zeros(int(dur * SR), np.float32)
    out += (np.random.default_rng(TAKE_NO[name]).normal(0, 0.0015, len(out))).astype(np.float32)  # room tone
    def put(seg: np.ndarray, at_s: float):
        i = int(at_s * SR)
        out[i:i + len(seg)] += seg[: max(0, min(len(seg), len(out) - i))]
    n = TAKE_NO[name]
    put(say_segment(f"Scene 1, shot 1 A, take {n}.", tmp), 0.3)
    put(clap(), 2.8)
    put(say_segment("Action!", tmp), 3.6)
    put(say_segment(DIALOGUE[name], tmp), 4.2)
    put(say_segment("Cut!", tmp), 7.8)
    if name == "take2.mov":
        put(say_segment("Circle that one.", tmp), 8.6)
    return out


def write_wav(path: str, a: np.ndarray) -> None:
    import wave
    pcm = (np.clip(a, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def make_video(name: str, audio_wav: str, out_path: str) -> None:
    dur = DURATION[name]
    base = ["ffmpeg", "-y", "-v", "error"]
    enc = ["-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p10le",
           "-color_primaries", "bt2020", "-color_trc", "bt2020-10", "-colorspace", "bt2020nc",
           "-tag:v", "hvc1", "-timecode", "00:00:00:00", "-c:a", "aac", "-shortest"]
    if name == "take3.mov":  # hard cut at 6.0 s: two sources concatenated
        cmd = base + ["-f", "lavfi", "-i", f"testsrc2=size={C.WIDTH}x{C.HEIGHT}:rate=24:duration=6.0",
                      "-f", "lavfi", "-i", f"smptebars=size={C.WIDTH}x{C.HEIGHT}:rate=24:duration={dur - 6.0}",
                      "-i", audio_wav,
                      "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", "-map", "2:a"] + enc + [out_path]
    else:
        cmd = base + ["-f", "lavfi", "-i", f"testsrc2=size={C.WIDTH}x{C.HEIGHT}:rate=24:duration={dur}",
                      "-i", audio_wav, "-map", "0:v", "-map", "1:a"] + enc + [out_path]
    subprocess.run(cmd, check=True)


def make_bad(out_path: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"testsrc2=size={C.WIDTH}x{C.HEIGHT}:rate=24000/1001:duration=4", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=4", "-c:v", "libx265", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1", "-c:a", "aac", "-map", "0:v", "-map", "1:a",
                    "-shortest", out_path], check=True)


def run(d: str) -> str:
    if os.path.exists(d):
        raise SystemExit(f"{d} already exists; refusing to overwrite a shoot folder")
    for sub in ("originals", "sound", "stream", "latents", "developed"):
        os.makedirs(os.path.join(d, sub))
    with tempfile.TemporaryDirectory() as tmp:
        for name in DURATION:
            print(f"building {name} …", flush=True)
            wav = os.path.join(tmp, name + ".wav")
            write_wav(wav, build_audio(name, tmp))
            make_video(name, wav, os.path.join(d, "originals", name))
        print("building bad_2398.mov (refusal-path check) …", flush=True)
        make_bad(os.path.join(d, "originals", "bad_2398.mov"))
        # external sound for take 2: 350 ms late, 500 ppm slow
        wav2 = os.path.join(tmp, "take2.mov.wav")
        write_wav(wav2, build_audio("take2.mov", tmp))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav2, "-af",
                        "adelay=350|350,asetrate=47976,aresample=48000",
                        os.path.join(d, "sound", "take2_ext.wav")], check=True)
    m = {"schema": C.SCHEMA, "shoot_date": os.path.basename(d)[6:], "created_at": C.utcnow(), "files": [],
         "takes": [], "plan": None,
         "approval": {"approved": False, "approved_by": None, "approved_at": None,
                      "approved_take_ids": [], "encoding": "not_started"},
         "meta_calls_file": "meta_calls.jsonl"}
    C.save_manifest(d, m)
    print(f"\nfixture shoot ready: {d}")
    print(f"next: ./precut.sh all {d}   (or ./precut.sh all {d} --offline for no Meta API calls)")
    return d


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "shoot_2099-01-15"
    run(os.path.abspath(target))
