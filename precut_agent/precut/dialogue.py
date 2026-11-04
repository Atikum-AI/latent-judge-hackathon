"""Pre-cut agent step - dialogue. Muse Voice Transcribe writes down what is said between "action" and
"cut", line by line, with a start time and a speaker per line.

  python -m precut.dialogue shoot_<date> [--no-transcribe] [--replay] [--scratch]

Runs after `sync` (it uses the external recorder when sync matched one) and before `label` (Spark gets
the lines as evidence, and the take list shows them). Per take:

1. The audio: the external recorder's file when sync matched it (the drift-corrected copy if sync
   resampled it), else the camera's scratch audio. Sync can match one recorder file to every take (a
   clap is a clap), so the recorder's clip is first checked against the camera's clip of the same
   moment (it must cover the range and line up by GCC-PHAT in every 1 s window with sound), and the
   scratch audio is used, and logged, when it does not.
   `--scratch` forces the scratch audio. The clip is
   action..cut from the takes step (the usable range without its handles when those are missing).
   A 16 kHz mono clip is cut into precut_cache/dialogue/ (originals and the recorder's files are
   never touched).
2. The model: muse-voice-transcribe-1.0 is tried first. While Meta does not serve it on our account
   (404 model_not_found, verified 29 Oct), muse-wonder-1.0 transcribes. The model that ran is stored
   per take and shown in the take list, so the report never claims Voice Transcribe when it was Wonder.
3. The checks (code, not the model): speakers become person_1, person_2 ... in order of first line
   (anything else the model calls a speaker is replaced and logged); crew calls (slate, "action",
   "cut", "circle that") are dropped, because the takes step owns those; times are clamped to the clip;
   each line is marked corroborated only if a speech segment of the clip starts near it, and when the
   clip holds no speech at all every line is dropped as a likely hallucination. Text that looks like a
   full name is flagged in the decision log for a human check before anything is shown.

Answers are cached in precut_cache/dialogue.json (per take, file hash, range and audio source):
re-runs spend nothing, and `--replay` uses only that file (no API call, no key needed). A take with
no saved answer is left without dialogue and logged. `--no-transcribe` skips the step offline.

Writes t["dialogue"] on every take (an additive field: the manifest stays shoot/manifest@1):
  {"lines": [{"start_s", "speaker", "text", "corroborated"}], "text", "speakers", "by", "source",
   "range_s", "status"}
Only our own footage goes to the Meta API. No transcript is ever committed to git (shoot folders are
outside the repo, and precut_cache/ is git-ignored).
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys

import numpy as np

from . import common as C
from .metaapi import TRANSCRIBE, MetaApi
from .sync import SR, audio_f32, offset_samples
from .takes import HANDLE_S, SLATE_RE, TIME_RE, speech_segments

PROMPT = """Transcribe the dialogue in this audio clip from a film take. Reply with JSON lines only, one
object per spoken line, in time order, no prose:
{"t": <start time in seconds from the start of this clip>, "speaker": "person_1", "text": "<the words>"}
Tell speakers apart by voice and call them person_1, person_2, ... in order of first appearance. Never
use or guess a real name for a speaker. Write the words exactly as spoken. Leave out crew calls
("action", "cut", "circle that", the slate). If nobody speaks, reply with nothing."""
CREW = re.compile(r"^\W*(action|cut|circle that( one)?|rolling|speed|background action|reset|"
                  r"back to one|that'?s a (wrap|print))\W*$", re.I)
NAME_IN_TEXT = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
SPEAKER_RE = re.compile(r"^person_(\d+)$")
NEAR_S = 1.5                     # a line is corroborated when a speech segment starts within this
MIN_CLIP_S = 0.5
MATCH_WIN_S, MATCH_LAG_S, MATCH_WEIGHT = 1.0, 0.05, 10.0   # recorder-vs-camera check (weight 10 as in sync)
REFUSAL = re.compile(r"(can'?t|cannot|unable to) (transcribe|process|hear)|no (audio|speech) (was )?(provided|found)",
                     re.I)


# ---------------------------------------------------------------------- the clip
def dialogue_range(t: dict) -> tuple[float, float, str]:
    """action..cut when the takes step heard both; else the usable range without its handles."""
    ev, u = t.get("events") or {}, t["usable"]
    a, c = ev.get("action_s"), ev.get("cut_s")
    if a is not None and c is not None and c > a:
        return float(a), float(c), "action..cut"
    hs = u.get("handles_s", HANDLE_S)
    start, end = u["start_s"] + (hs if a is not None else 0.0), u["end_s"]
    if end - start < MIN_CLIP_S:
        start, end = u["start_s"], u["end_s"]
    return float(start), float(end), "usable range (no action..cut heard)"


def audio_source(d: str, t: dict, f: dict, force_scratch: bool = False) -> tuple[str, float, str]:
    """(audio file, seconds to add to take time to get file time, label). The external recorder wins
    when sync matched it; its drift-corrected copy wins over the raw file."""
    s = t.get("sync") or {}
    ext = s.get("external_file")
    if ext and not force_scratch and s.get("offset_ms") is not None:
        raw = os.path.join(d, "sound", ext)
        fixed = os.path.join(d, "precut_cache", "sync", f"{os.path.splitext(ext)[0]}.resampled.wav")
        path = fixed if s.get("resampled") and os.path.exists(fixed) else raw
        if os.path.exists(path):
            return path, s["offset_ms"] / 1000.0, f"external:{ext}" + (" (drift-corrected)" if path == fixed else "")
    return os.path.join(d, f["rel_path"]), 0.0, "scratch"


def recorder_matches(scratch: np.ndarray, ext: np.ndarray) -> tuple[bool, str]:
    """Is the recorder's clip the same moment as the camera's clip? Sync can match one recorder file to
    every take (a clap is a clap), so before its words are trusted: the recorder clip must cover the
    range, and GCC-PHAT (the cross-mic measure sync uses) must find the two clips aligned, within
    50 ms, in EVERY 1 s window (half-overlapped) where both carry sound. One window of different words
    fails it. The windows are short on purpose: the same crew call ("action!") in two takes lines up,
    and a 3 s window that holds it hides the different line next to it (seen live on 3 Nov)."""
    if len(ext) < 0.9 * len(scratch):
        return False, f"the recorder file covers {len(ext) / max(len(scratch), 1):.0%} of the range"
    win, hop = int(MATCH_WIN_S * SR), int(MATCH_WIN_S * SR / 2)
    n = min(len(scratch), len(ext))
    checked = 0
    for lo in range(0, max(n - win, 0) + 1, hop):
        a, b = scratch[lo:lo + win], ext[lo:lo + win]
        if np.sqrt((a ** 2).mean()) < 0.01 or np.sqrt((b ** 2).mean()) < 0.01:
            continue                                   # room tone on either side: nothing to compare
        lag, w = offset_samples(a, b, int(0.25 * SR))
        checked += 1
        if abs(lag) > MATCH_LAG_S * SR or w < MATCH_WEIGHT:
            return False, (f"at {lo / SR:.1f}-{(lo + win) / SR:.1f} s of the clip the two do not line up "
                           f"(lag {lag / SR * 1000:+.0f} ms, peak weight {w:.1f})")
    if not checked:
        return False, "no window with sound on both sides to compare"
    return True, f"aligned in {checked} window(s)"


def cut_clip(src: str, start_s: float, end_s: float, out: str) -> str:
    if not os.path.exists(out):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{max(0.0, start_s):.3f}", "-to", f"{end_s:.3f}",
                        "-i", src, "-vn", "-ac", "1", "-ar", str(SR), out], check=True)
    return out


# ---------------------------------------------------------------------- the model
def transcribe(api: MetaApi, wav: str, take_id: str) -> tuple[str, str]:
    """Voice Transcribe first, Wonder while Voice Transcribe is not served. -> (text, model that ran)."""
    b64 = base64.b64encode(open(wav, "rb").read()).decode()
    content = [{"type": "input_text", "text": PROMPT},
               {"type": "input_audio", "audio_url": f"data:audio/wav;base64,{b64}"}]
    for model in TRANSCRIBE:
        try:
            text, _ = api.respond(model, content, step="dialogue", take=take_id)
            return text, model
        except Exception as e:
            if "model_not_found" in str(e) or "404" in str(e):
                continue
            raise
    raise RuntimeError(f"no transcription model answered (tried {', '.join(TRANSCRIBE)})")


def parse_lines(text: str) -> list[dict]:
    """JSON lines (the asked format); a JSON array, or '[MM:SS.mmm] text' lines, are accepted too.
    -> [{t, speaker, text}] with raw values; check_lines cleans them."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?|```$", "", text).strip()
    if text.startswith("["):
        try:
            arr = json.loads(text)
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except ValueError:
            pass
    out = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line:
            continue
        if line.startswith("{"):
            try:
                j = json.loads(line)
                if isinstance(j, dict):
                    out.append(j)
                continue
            except ValueError:
                pass
        m = TIME_RE.match(line)
        if m:
            t = int(m.group(1)) * 60 + int(m.group(2)) + int((m.group(3) or "0").ljust(3, "0")) / 1000
            rest = line[m.end():].strip()
            sp = re.match(r"(person_\d+|speaker\s*\d+|[A-Z][\w ]{0,20}):\s*", rest, re.I)
            out.append({"t": t, "speaker": sp.group(1) if sp else None, "text": rest[sp.end():] if sp else rest})
    return out


def check_lines(raw: list[dict], clip_start: float, clip_dur: float, segs: list[tuple[float, float]]
                ) -> tuple[list[dict], list[str]]:
    """The code's checks on the model's lines. Times come back in TAKE seconds. -> (lines, notes)."""
    notes, lines, speakers = [], [], {}
    for x in raw:
        words = " ".join(str(x.get("text") or "").split())[:400]
        if not words:
            continue
        if CREW.match(words) or SLATE_RE.search(words):
            notes.append(f"dropped a crew call {words[:40]!r} (the takes step owns those)")
            continue
        try:
            t = float(x.get("t"))
        except (TypeError, ValueError):
            t = None
        if t is None or not (0.0 <= t <= clip_dur + 0.25):
            notes.append(f"line {words[:30]!r}: time {x.get('t')!r} is outside the clip, set to the "
                         f"previous line's time")
            t = lines[-1]["start_s"] - clip_start if lines else 0.0
        t = min(max(t, 0.0), clip_dur)
        name = str(x.get("speaker") or "").strip()
        if name not in speakers:
            speakers[name] = f"person_{len(speakers) + 1}"
            if name and not SPEAKER_RE.match(name):
                notes.append("a speaker the model named was replaced with " + speakers[name])
        near = any(abs(a - t) <= NEAR_S or a - 0.25 <= t <= b for a, b in segs)
        lines.append({"start_s": round(clip_start + t, 3), "speaker": speakers[name], "text": words,
                      "corroborated": bool(near)})
        if NAME_IN_TEXT.search(words):
            notes.append(f"{speakers[name]}'s line at {clip_start + t:.2f} s may contain a name: check before "
                         "showing it")
    if lines and not segs:
        notes.append(f"all {len(lines)} line(s) dropped: the clip holds no speech, so the model invented them")
        return [], notes
    loose = [ln for ln in lines if not ln["corroborated"]]
    if loose:
        notes.append(f"{len(loose)} line(s) not corroborated by a speech onset within {NEAR_S} s: kept, marked")
    lines.sort(key=lambda ln: ln["start_s"])
    return lines, notes


# ---------------------------------------------------------------------- the step
def _cache_path(d: str) -> str:
    return os.path.join(C.cache_dir(d, ""), "dialogue.json")


def _load(d: str) -> dict:
    p = _cache_path(d)
    return json.load(open(p)) if os.path.exists(p) else {}


def _save(d: str, answers: dict) -> None:
    with open(_cache_path(d), "w") as f:
        json.dump(answers, f, indent=1)


def empty(status: str, **kw) -> dict:
    return {"lines": [], "text": None, "speakers": 0, "by": None, "source": None, "range_s": None,
            "status": status, **kw}


def run(d: str, use_transcribe: bool = True, replay: bool = False, force_scratch: bool = False) -> dict:
    m = C.load_manifest(d)
    C.clear_decisions(m, "dialogue")
    if not m.get("takes"):
        raise SystemExit("no takes yet - run takes.py first")
    if not use_transcribe and not replay:
        for t in m["takes"]:
            t["dialogue"] = empty("skipped (--no-transcribe)")
        C.decide(m, None, "dialogue", "dialogue not transcribed", "--no-transcribe: offline dry run")
        C.save_manifest(d, m)
        print("dialogue skipped (--no-transcribe)")
        return m
    api = None if replay else MetaApi(shoot=d)
    answers, fresh = _load(d), 0
    files = {f["name"]: f for f in m.get("files", [])}
    cache = C.cache_dir(d, "dialogue")
    for t in m["takes"]:
        f = files.get(t["file"])
        if f is None:
            t["dialogue"] = empty("no source file in the manifest")
            continue
        start, end, how = dialogue_range(t)
        if end - start < MIN_CLIP_S:
            t["dialogue"] = empty(f"range too short ({end - start:.2f} s)")
            continue
        sha = (f.get("sha256") or "")[:12]
        stem = os.path.join(cache, f"{t['id']}_{sha}_{start:.3f}-{end:.3f}")
        wav = cut_clip(os.path.join(d, f["rel_path"]), start, end, stem + "_scr.wav")
        src, shift, source = audio_source(d, t, f, force_scratch)
        if source != "scratch":
            ext_wav = cut_clip(src, start + shift, end + shift, stem + "_ext.wav")
            ok, why = recorder_matches(audio_f32(wav), audio_f32(ext_wav))
            if ok:
                wav = ext_wav
            else:
                C.decide(m, t["id"], "dialogue", "used the scratch audio, not the recorder",
                         f"the {source} clip is not this take's moment ({why}): sync likely matched the "
                         "wrong file")
                source = "scratch"
        key = f"{t['id']}:{sha}:{start:.3f}-{end:.3f}:{source}"
        if key in answers:
            text, model = answers[key]["text"], answers[key]["model"]
            if replay:
                C.decide(m, t["id"], "dialogue", "replayed a saved transcript", "--replay: no API call")
        elif replay:
            t["dialogue"] = empty("no saved transcript to replay", source=source, range_s=[start, end])
            C.decide(m, t["id"], "dialogue", "no dialogue", "--replay and no saved transcript for this take")
            continue
        else:
            try:
                text, model = transcribe(api, wav, t["id"])
                if REFUSAL.search(text) and not text.lstrip().startswith(("{", "[")):
                    raise ValueError("the model returned a refusal instead of a transcript")
                answers[key] = {"text": text, "model": model}
                fresh += 1
            except Exception as e:
                t["dialogue"] = empty(f"transcription failed ({type(e).__name__})", source=source,
                                      range_s=[start, end])
                C.decide(m, t["id"], "dialogue", "transcription failed", f"{type(e).__name__}: {e}"[:300])
                continue
        clip = audio_f32(wav)
        segs = speech_segments(clip) if len(clip) else []
        lines, notes = check_lines(parse_lines(text), start, end - start, segs)
        for note in notes:
            C.decide(m, t["id"], "dialogue", "transcript corrected", note)
        t["dialogue"] = {"lines": lines, "text": " ".join(ln["text"] for ln in lines) or None,
                         "speakers": len({ln["speaker"] for ln in lines}), "by": model, "source": source,
                         "range_s": [round(start, 3), round(end, 3)],
                         "status": "ok" if lines else "no dialogue heard"}
        C.decide(m, t["id"], "dialogue",
                 f"{len(lines)} line(s), {t['dialogue']['speakers']} speaker(s)" if lines else "no dialogue heard",
                 f"{model} on {source} audio, {how} {start:.2f}-{end:.2f} s")
        print(f"{t['id']} {t['file']}: {len(lines)} line(s) by {model} from {source} audio "
              f"({start:.2f}..{end:.2f} s)")
    if fresh:
        _save(d, answers)
    C.save_manifest(d, m)
    return m


def srt(t: dict) -> str:
    """One take's dialogue as SubRip, times from the start of the FILE (each line runs to the next one,
    at most 6 s)."""
    def ts(s: float) -> str:
        ms = int(round(max(0.0, s) * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    lines = (t.get("dialogue") or {}).get("lines") or []
    end_of = (t.get("dialogue") or {}).get("range_s") or [0, 0]
    out = []
    for i, ln in enumerate(lines):
        nxt = lines[i + 1]["start_s"] if i + 1 < len(lines) else end_of[1]
        stop = min(max(nxt, ln["start_s"] + 0.8), ln["start_s"] + 6.0)
        out += [str(i + 1), f"{ts(ln['start_s'])} --> {ts(stop)}", f"{ln['speaker']}: {ln['text']}", ""]
    return "\n".join(out)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit("usage: python -m precut.dialogue shoot_<date> [--no-transcribe] [--replay] [--scratch]")
    run(C.shoot_dir(args[0]), use_transcribe="--no-transcribe" not in sys.argv, replay="--replay" in sys.argv,
        force_scratch="--scratch" in sys.argv)
