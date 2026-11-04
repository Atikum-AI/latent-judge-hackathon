"""Pre-cut agent steps 4-5 - find takes and usable ranges.

  python -m precut.takes shoot_<date> [--no-transcribe]

Take boundaries come from file boundaries (one take per recording, the on-set protocol) and from the
spoken slate: Muse Voice Transcribe (Meta API) reads the scratch audio; "scene X, shot Y, take Z",
"action", "cut" and "circle that" are found in it. A file that holds several slates is split into
several takes. Usable ranges are action..cut plus 1 s handles, snapped to frames; PySceneDetect (or the
built-in numpy detector) logs camera stops inside the range.

--no-transcribe skips the Meta API (offline dry run): events fall back to the clap spike and energy
onsets and every range is marked estimated, with the reason logged.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import numpy as np

from . import common as C
from .cuts import find_cuts
from .metaapi import TRANSCRIBE, MetaApi
from .sync import SR, audio_f32

HANDLE_S = 1.0
WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
         "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}
# "scene one, shot one A, take two" and "scene 1, shot 1A, take 2" both parse; the letter suffix sticks
# to the shot ("1A").
SLATE_RE = re.compile(r"scene\s+(\w+)[,.\s]+shot\s+(\w+)(?:\s+([A-Za-z])(?=[,.\s]))?[,.]?\s+take\s+(\w+)", re.I)
TIME_RE = re.compile(r"\[(\d{1,2}):(\d{2})(?:[.:](\d{1,3}))?\]")   # [MM:SS] or [MM:SS.mmm] markers
TRANSCRIBE_PROMPT = ("Transcribe this audio verbatim. Before every spoken phrase write its start time in the "
                     "audio as [MM:SS.mmm]. Output only the timestamped transcript. Keep the words 'action', "
                     "'cut' and 'circle that' exactly as spoken.")


def norm_num(w: str) -> str:
    w = w.lower().strip(".,;")
    if w in WORDS:
        return str(WORDS[w])
    if w.isdigit():
        return str(int(w))
    return w.upper()  # letters like "1A" shot suffixes survive as-is


def clap_times(audio: np.ndarray) -> list[float]:
    """Sharp onsets: short-window RMS spikes well above the floor. A hand clap is impulsive - the
    above-threshold run is short; speech sustains. Peaks that sustain past 0.25 s are not claps."""
    w = int(0.02 * SR)
    if len(audio) < 4 * w:
        return []
    env = np.sqrt(np.convolve(audio ** 2, np.ones(w) / w, "same"))
    floor = np.median(env) + 1e-9
    peaks = np.where(env > max(6 * floor, env.max() * 0.35))[0]
    out, last = [], -10 * SR
    for p in peaks:
        if p - last <= 0.5 * SR:
            continue
        run = p
        while run < len(env) and env[run] > max(6 * floor, env.max() * 0.2):
            run += 1
        if run - p <= int(0.25 * SR):          # impulsive: a clap, not a speech run
            out.append(round(p / SR, 3))
            last = p
    return out


def parse_phrases(text: str) -> list[dict]:
    """The transcript as an ordered phrase list: [{model_t, text}]. Markers are stripped from the text."""
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = TIME_RE.match(line)
        t = None
        if m:
            t = int(m.group(1)) * 60 + int(m.group(2)) + int((m.group(3) or "0").ljust(3, "0")) / 1000
            line = line[m.end():].strip()
        if line:
            out.append({"model_t": t, "text": line})
    return out


def speech_segments(audio: np.ndarray) -> list[tuple[float, float]]:
    """Energy VAD at 25 ms: speech runs >= 0.15 s, gaps < 0.3 s merged. The 80 ms clap beep is too
    short to count as speech."""
    w = int(0.025 * SR)
    env = np.sqrt(np.convolve(audio ** 2, np.ones(w) / w, "same"))
    floor = np.percentile(env, 20) + 1e-9
    on = env > max(5 * floor, 0.02)
    segs, i = [], 0
    while i < len(on):
        if on[i]:
            j = i
            while j < len(on) and (on[j] or (j + int(0.3 * SR) < len(on) and on[j:j + int(0.3 * SR)].any())):
                j += 1
            segs.append((i / SR, j / SR))
            i = j
        else:
            i += 1
    return [(a, b) for a, b in segs if b - a >= 0.15]


def align_events(phrases: list[dict], audio: np.ndarray, claps: list[float], m: dict, take_id: str) -> dict:
    """Event times from speech-segment onsets, anchored by the on-set protocol order
    (slate - CLAP - action - dialogue - cut - 'circle that'):
    the clap splits slate from action; the model's words say WHAT each phrase is and its [MM:SS.mmm]
    marker only picks among the segments the protocol order allows. A 'circle that' the audio does not
    corroborate (a segment >= 0.6 s after 'cut', near the model's time) is dropped and logged - the
    transcribe model sometimes invents it, and circling stays a human act anyway."""
    ev = {"slates": [], "action_s": None, "cut_s": None, "circle_that_s": None}
    segs = speech_segments(audio)
    clap = claps[0] if claps else None
    used: set[int] = set()

    def pick(model_t: float | None, allowed: list[int], min_len: float = 0.0, window: float = 2.0):
        cand = [(k, abs(segs[k][0] - model_t)) for k in allowed
                if k not in used and (segs[k][1] - segs[k][0]) >= min_len
                and (model_t is None or abs(segs[k][0] - model_t) <= window)]
        if not cand:
            return None
        k = min(cand, key=lambda x: x[1])[0]
        used.add(k)
        return k

    def note(text: str, model_t: float | None, t: float, what: str):
        if model_t is not None and abs(t - model_t) > 0.75:
            C.decide(m, take_id, "takes", f"{what} placed by speech onset, not the model clock",
                     f"model said {model_t:.2f} s, the speech segment starts at {t:.2f} s; the onset wins")

    # 1) slate: nearest segment before the clap (timing not critical, identity is)
    for ph in phrases:
        sm = SLATE_RE.search(ph["text"])
        if not sm:
            continue
        before = [k for k in range(len(segs)) if clap is None or segs[k][0] < clap - 0.15]
        k = pick(ph["model_t"], before) or (before[0] if before else None)
        t = segs[k][0] if k is not None else ph["model_t"]
        if k is not None:
            used.add(k)
        shot = norm_num(sm.group(2)) + (sm.group(3).upper() if sm.group(3) else "")
        ev["slates"].append({"scene": norm_num(sm.group(1)), "shot": shot, "take": norm_num(sm.group(4)),
                             "t_s": t})
    first_slate_t = ev["slates"][0]["t_s"] if ev["slates"] else None

    # 2) action: first segment after the clap (or after the slate), near its marker
    for ph in phrases:
        if not re.search(r"\baction\b", ph["text"], re.I):
            continue
        after = [k for k in range(len(segs))
                 if (clap is not None and segs[k][0] > clap + 0.25)
                 or (clap is None and first_slate_t is not None and segs[k][0] > first_slate_t + 0.3)]
        k = pick(ph["model_t"], after)
        if k is not None:
            ev["action_s"] = segs[k][0]
            note(ph["text"], ph["model_t"], segs[k][0], "'action'")
        else:
            ev["action_s"] = ph["model_t"]
            if ph["model_t"] is not None:
                C.decide(m, take_id, "takes", "'action' placed by the model clock",
                         "no speech segment after the clap corroborates it; the marker is the estimate")
        break

    # 3) cut: nearest segment after action's, near its marker
    a = ev["action_s"]
    for ph in phrases:
        if not re.search(r"\bcut\b", ph["text"], re.I):
            continue
        after = [k for k in range(len(segs)) if a is None or segs[k][0] > a + 0.5]
        k = pick(ph["model_t"], after)
        if k is not None:
            ev["cut_s"] = segs[k][0]
            note(ph["text"], ph["model_t"], segs[k][0], "'cut'")
        else:
            ev["cut_s"] = ph["model_t"]
            if ph["model_t"] is not None:
                C.decide(m, take_id, "takes", "'cut' placed by the model clock",
                         "no speech segment after 'action' corroborates it; the marker is the estimate")
        break

    # 4) circle that: ONLY with a free segment >= 0.6 s after cut, near the marker; else dropped.
    #    The model clock runs loose (seen at +-2 s), so the window is wider here; the after-cut,
    #    free-segment and duration rules do the real guarding.
    for ph in phrases:
        if not re.search(r"circle that", ph["text"], re.I):
            continue
        c = ev["cut_s"]
        after = [k for k in range(len(segs)) if c is None or segs[k][0] > c + 0.3]
        k = pick(ph["model_t"], after, min_len=0.6, window=2.5)
        if k is None:
            C.decide(m, take_id, "takes", "a 'circle that' in the transcript was IGNORED",
                     "no free speech segment >= 0.6 s after 'cut' sits near the marker - likely a "
                     "transcription hallucination; circling stays a human act anyway")
        else:
            ev["circle_that_s"] = segs[k][0]
        break
    return ev


def transcribe(api: MetaApi, wav_path: str, take_id: str) -> tuple[str, str]:
    """Transcribe one take's scratch audio as base64 audio_url blocks (the shape this API surface
    serves). muse-voice-transcribe-1.0 is tried first; while Meta does not serve it here (404
    model_not_found), muse-wonder-1.0 does the transcription - the model that ran is logged per call."""
    import base64
    b64 = base64.b64encode(open(wav_path, "rb").read()).decode()
    for model in TRANSCRIBE:
        try:
            text, _ = api.respond(
                model,
                [{"type": "input_text", "text": TRANSCRIBE_PROMPT},
                 {"type": "input_audio", "audio_url": f"data:audio/wav;base64,{b64}"}],
                step="transcribe", take=take_id)
            return text, model
        except Exception as e:
            if "model_not_found" in str(e) or "404" in str(e):
                continue
            raise
    raise RuntimeError(f"no transcription model answered (tried {', '.join(TRANSCRIBE)})")


def usable_range(dur_s: float, ev: dict, claps: list[float], take_id: str, m: dict) -> tuple[float, float, str]:
    """action..cut + 1 s handles, clamped and snapped to 24 fps frames. Falls back, loudly, when the
    transcript carries no usable words or no timestamps."""
    a, c = ev.get("action_s"), ev.get("cut_s")
    if a is not None and c is not None and c > a:
        start, end, how = a - HANDLE_S, c + HANDLE_S, "action..cut from the transcript"
    elif a is not None:
        start, end, how = a - HANDLE_S, dur_s, "action from the transcript, no 'cut' heard: range runs to file end"
    elif claps:
        start, end, how = max(0.0, claps[0]), dur_s, "no action/cut in the transcript: from the clap to file end (estimated)"
    else:
        start, end, how = 0.0, dur_s, "no transcript events and no clap: the whole file (estimated)"
    start_f, end_f = int(start * C.FPS), min(int(round(dur_s * C.FPS)), int(np.ceil(end * C.FPS)))
    start = max(0.0, start_f / C.FPS)
    end = min(dur_s, end_f / C.FPS)
    if "estimated" in how or "no 'cut'" in how:
        C.decide(m, take_id, "takes", "usable range is a fallback", how)
    return start, end, how


def run(d: str, use_transcribe: bool = True) -> dict:
    m = C.load_manifest(d)
    C.clear_decisions(m, "takes")
    usable = [f for f in m.get("files", [])
              if f.get("checks", {}).get("ok") and not f.get("superseded_by")]
    if not usable:
        raise SystemExit("no usable originals - run ingest.py first (refused files stay out of the take list)")
    api = MetaApi(shoot=d) if use_transcribe else None
    cache = C.cache_dir(d, "audio")
    tcache = C.cache_dir(d, "transcripts")
    takes = []
    n = 0
    for f in usable:
        path = os.path.join(d, f["rel_path"])
        dur = f["probe"]["duration_s"]
        scratch = audio_f32(path, cache)
        claps = clap_times(scratch)
        text, how = "", "transcribe skipped (--no-transcribe)"
        if api:
            wav = os.path.join(cache, f["sha256"][:16] + ".wav")
            if not os.path.exists(wav):
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-vn", "-ac", "1", "-ar", str(SR),
                                wav], check=True)
            tc = os.path.join(tcache, f["sha256"][:16] + ".json")
            if os.path.exists(tc):  # transcripts are cached per file hash: re-runs spend nothing
                saved = json.load(open(tc))
                text, how = saved["text"], saved["model"] + " (cached)"
            else:
                try:
                    text, how = transcribe(api, wav, f"file:{f['name']}")
                    low = text.lower()
                    if "no audio" in low and "transcribe" in low or "can't transcribe" in low:
                        # the model refused (seen live 29 Oct): that is a failure, not a transcript
                        json.dump({"text": "", "model": how + " (refused)"}, open(tc, "w"))
                        raise ValueError("the model returned a refusal instead of a transcript")
                    json.dump({"text": text, "model": how}, open(tc, "w"))
                except Exception as e:
                    text = ""
                    how = f"transcribe failed ({type(e).__name__}); fell back to claps and energy"
                    C.decide(m, None, "takes", "transcription failed", f"{f['name']}: {e}")
        ev = align_events(parse_phrases(text), scratch, claps, m, f["name"]) if text else \
            {"slates": [], "action_s": None, "cut_s": None, "circle_that_s": None}
        slates = ev["slates"] or [{}]
        if len(slates) > 1:
            C.decide(m, None, "takes", f"{f['name']} holds {len(slates)} slates",
                     "a rolling take is split at each spoken slate (file stays untouched)")
        cuts, detector = find_cuts(path)
        for slate in slates:
            n += 1
            tid = f"tk_{n:04d}"
            start, end, how_range = usable_range(dur, ev, claps, tid, m)
            inner = [c for c in cuts if start < c < end]
            if inner:
                C.decide(m, tid, "takes", f"camera stop(s) inside the range at {inner}",
                         f"{detector} detected a cut; the range is kept, the encode plan marks the windows")
            takes.append({
                "id": tid, "file": f["name"], "source": f.get("source", "original"),
                "scene": slate.get("scene"), "shot": slate.get("shot"),
                "take": slate.get("take"), "slate_t_s": slate.get("t_s"),
                "usable": {"start_s": round(start, 3), "end_s": round(end, 3), "handles_s": HANDLE_S,
                           "how": how_range},
                "events": {"claps_s": claps, "action_s": ev.get("action_s"), "cut_s": ev.get("cut_s"),
                           "circle_that": ev.get("circle_that_s") is not None},
                "cuts_in_range_s": inner, "cut_detector": detector,
                "transcript": text.strip()[:2000], "transcript_by": how,
                "duration_s": dur, "included": True,
            })
            print(f"{tid} {f['name']}: usable {start:.2f}..{end:.2f} ({how_range}); slate "
                  f"{slate.get('scene')}/{slate.get('shot')}/{slate.get('take') or 'not heard'}; "
                  f"claps {len(claps)}; cuts {inner or '-'}")
    m["takes"] = takes
    C.decide(m, None, "takes", f"{len(takes)} takes from {len(usable)} files",
             "file boundaries + spoken slate; ranges are action..cut with 1 s handles" if api else
             "file boundaries; ranges are ESTIMATES (--no-transcribe)")
    C.save_manifest(d, m)
    return m


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit("usage: python -m precut.takes shoot_<date> [--no-transcribe]")
    run(C.shoot_dir(args[0]), use_transcribe="--no-transcribe" not in sys.argv)
