"""Pre-cut agent step 6 - labels. Code measures; Muse Spark decides, and every decision is logged
with its reason.

  python -m precut.label shoot_<date> [--no-spark]

Measurements (deterministic, numpy): focus (Laplacian spread), clipping (% luma at the top code),
shake (frame-to-frame translation). Spark gets the numbers, the transcript and the slate, and returns
strict JSON: scene/shot/take, dialogue, a best-take GUESS and flags. "circle that?" is a label only -
circling stays a human act, in Review.

--no-spark: deterministic fallback labels (heuristics), each marked "fallback" in the decision log.
--require-spark: the competition mode - judgments come from Meta models or the step FAILS. A fallback
label is never acceptable in that mode (offline dev only).

Picture labels (merged from the old top-level precut/ labeller): Spark also sees a few small key frames
of the usable range (10 %, 50 %, 90 % by default; --frames N, --frames 0 for text only) and describes the
picture: shot type, camera move, subjects, action, location, lighting, picture issues, a usable guess
and which frames each claim came from. The code keeps only known values, renames anything that looks
like a real name to person_1, person_2 ..., and drops evidence that points at frames it was not sent.
Key frames live in shoot_<date>/precut_cache/keyframes/ (deletable). Only our own shoots go to the
Meta API.

--replay: reuse the Spark answers saved in precut_cache/spark_labels.json (no API call, no cost);
a take with no saved answer falls back (or fails with --require-spark).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import base64
import numpy as np

from . import common as C
from .metaapi import SPARK, MetaApi

SAMPLE_FPS = 2
SMALL = (320, 180)
KEYFRAMES = 3                    # key frames per take sent to Spark (0 = text only)
KEYFRAME_W = 768                 # small JPEGs: enough to read framing, cheap to send

SHOT_TYPES = ("extreme_wide", "wide", "medium_wide", "medium", "medium_close", "close", "extreme_close", "insert")
CAMERA = ("static", "pan", "tilt", "dolly", "truck", "handheld", "crane", "zoom", "mixed")
ISSUES = ("out_of_focus", "overexposed", "underexposed", "clipped_highlights", "crushed_blacks", "motion_blur",
          "camera_shake", "boom_in_frame", "slate_in_frame", "crew_in_frame", "flicker", "framing")
NAME_LIKE = re.compile(r"^[A-Z][a-z]+(?: [A-Z][a-z]+)+$")   # "Jane Doe": looks like a person's name
NAME_IN_TEXT = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
PROMPT = """You are the labelling step of a film pre-cut agent. You get one take's evidence as JSON and reply
with ONE JSON object, no prose, exactly these keys:
{"scene": str, "shot": str, "take": str, "dialogue": str|null, "best_take_guess": bool, "flags": [...], "reason": str,
 "picture": {...}|null}
- scene/shot/take: from the spoken slate ("scene 1, shot 1A, take 2"). Keep the slate's own words
  ("1A" stays "1A"). If no slate was heard, use the passed file order guess and say so in reason.
- dialogue: the line spoken between action and cut, else null. When the evidence carries
  dialogue_lines (from the transcription step, speakers already person_N), use those words.
- best_take_guess: true only when the crew said "circle that" (or clearly praised the take). It is a
  LABEL, never a selection. The evidence's circle_that_corroborated tells you whether the audio actually
  contains "circle that": the transcription model sometimes invents it. If false, best_take_guess = false.
- flags: subset of ["focus","clipping","shake"] using the measurements: focus true when laplacian_var
  is low for this shoot (soft), clipping true when clipped_pct is visibly high (>1%), shake true when
  translate_px is high (>1.5 px at 320 px wide). Explain each flag in reason.
- reason: one sentence per decision, plain words.
If key frames follow the evidence (frames 1..N in time order), also add a "picture" object describing
only what the frames show:
{"shot_type": one of SHOT_TYPES, "camera_move": one of CAMERA, "subjects": ["person_1", "a red car", ...],
 "action": one sentence, "location": "interior/exterior, place, time of day", "lighting": short phrase,
 "issues": [zero or more of ISSUES], "usable": true|false, "why": one sentence,
 "evidence": {"<claim>": [frame numbers, 1-based]}}
Never name or guess the identity of a person: call people person_1, person_2, ... in order of prominence.
Without key frames, set "picture" to null.
Evidence:
"""
PROMPT = (PROMPT.replace("SHOT_TYPES", json.dumps(list(SHOT_TYPES))).replace("CAMERA", json.dumps(list(CAMERA)))
          .replace("ISSUES", json.dumps(list(ISSUES))))


def frames_gray(path: str, start_s: float, end_s: float) -> np.ndarray:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start_s:.3f}", "-to", f"{end_s:.3f}", "-i", path, "-vf",
         f"fps={SAMPLE_FPS},scale={SMALL[0]}:{SMALL[1]}:flags=area", "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True).stdout
    w, h = SMALL
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w)


def measure(path: str, start_s: float, end_s: float) -> dict:
    fr = frames_gray(path, start_s, end_s).astype(np.float32)
    if len(fr) < 2:
        return {"laplacian_var": None, "clipped_pct": None, "translate_px": None}
    lap = (-4 * fr[:, 1:-1, 1:-1] + fr[:, :-2, 1:-1] + fr[:, 2:, 1:-1] + fr[:, 1:-1, :-2] + fr[:, 1:-1, 2:])
    lap_var = float(np.median(lap.var(axis=(1, 2))))
    clipped = float((fr >= 250).mean() * 100)
    # translation: phase correlation on 64x64 centres, median over pairs
    centres = [f[58:122, 128:192] for f in fr]
    shifts = []
    for a, b in zip(centres, centres[1:]):
        A, B = np.fft.rfft2(a), np.fft.rfft2(b)
        R = A * np.conj(B)
        R /= np.abs(R) + 1e-9
        peak = np.unravel_index(np.argmax(np.abs(np.fft.irfft2(R))), (64, 64))
        dy, dx = [(p - 64 if p > 32 else p) for p in peak]
        shifts.append(float(np.hypot(dx, dy)))
    return {"laplacian_var": round(lap_var, 1), "clipped_pct": round(clipped, 2),
            "translate_px": round(float(np.median(shifts)) if shifts else 0.0, 2)}


def keyframe_times(start_s: float, end_s: float, n: int = KEYFRAMES) -> list[float]:
    """n times inside the usable range, away from its very edges (slate, cut)."""
    span = max(end_s - start_s, 0.0)
    if n <= 0:
        return []
    if n == 1:
        return [round(start_s + span / 2, 3)]
    return [round(start_s + span * (0.1 + 0.8 * i / (n - 1)), 3) for i in range(n)]


def keyframes(path: str, times: list[float], out_dir: str, width: int = KEYFRAME_W) -> list[str]:
    """One small JPEG per time (cached: an existing frame is not decoded again). Paths in time order."""
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    out = []
    for t in times:
        p = os.path.join(out_dir, f"{stem}_t{t:09.3f}.jpg")
        if not os.path.exists(p):
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                            "-vf", f"scale={width}:-2", "-q:v", "4", p], check=True)
        out.append(p)
    return out


def image_part(path: str) -> dict:
    with open(path, "rb") as f:
        return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()}


def check_picture(raw, n_frames: int) -> tuple[dict | None, list[str]]:
    """Spark's picture labels, checked: known values only, people as person_N, evidence tied to frames
    that were sent. -> (picture or None, notes for the decision log)."""
    if not isinstance(raw, dict) or n_frames <= 0:
        return None, []
    notes = []

    def pick(field, allowed):
        v = raw.get(field)
        if v in allowed:
            return v
        notes.append(f"{field}: {v!r} is not an allowed value, left empty")
        return None

    subjects, people = [], 0
    for x in raw.get("subjects") or []:
        x = str(x)[:80]
        if NAME_LIKE.match(x):
            people += 1
            notes.append("a subject looked like a real name and was replaced")
            x = f"person_{people}"
        elif re.fullmatch(r"person_\d+", x):
            people = max(people, int(x.split("_")[1]))
        subjects.append(x)
    issues = [i for i in raw.get("issues") or [] if i in ISSUES]
    dropped = [i for i in raw.get("issues") or [] if i not in ISSUES and i != "none"]
    if dropped:
        notes.append(f"dropped unknown issues {dropped}")
    evidence = {}
    for claim, idx in (raw.get("evidence") or {}).items():
        ok = [int(i) for i in (idx if isinstance(idx, list) else [idx])
              if isinstance(i, (int, float)) and not isinstance(i, bool) and 1 <= int(i) <= n_frames]
        if ok:
            evidence[str(claim)[:120]] = ok
    usable = raw.get("usable")
    if not isinstance(usable, bool):
        notes.append(f"usable: {usable!r} is not true/false, set to false")
        usable = False
    pic = {"shot_type": pick("shot_type", SHOT_TYPES), "camera_move": pick("camera_move", CAMERA),
           "subjects": subjects, "action": str(raw.get("action") or "")[:300],
           "location": str(raw.get("location") or "")[:150], "lighting": str(raw.get("lighting") or "")[:150],
           "issues": issues, "usable": usable, "why": str(raw.get("why") or "")[:300],
           "evidence": evidence, "frames": n_frames}
    for field in ("action", "why"):
        if NAME_IN_TEXT.search(pic[field]):
            notes.append(f"picture.{field} may contain a name: check before approving")
    return pic, notes


def _load_answers(d: str) -> dict:
    p = os.path.join(C.cache_dir(d, ""), "spark_labels.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _save_answers(d: str, answers: dict) -> None:
    with open(os.path.join(C.cache_dir(d, ""), "spark_labels.json"), "w") as f:
        json.dump(answers, f, indent=1)


def parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    j = json.loads(m.group(0))
    for k in ("scene", "shot", "take", "dialogue", "best_take_guess", "flags", "reason"):
        if k not in j:
            raise ValueError(f"missing key {k}")
    j["flags"] = [f for f in j["flags"] if f in ("focus", "clipping", "shake")]
    j["best_take_guess"] = bool(j["best_take_guess"])
    return j


def fallback_labels(t: dict, order: int, meas: dict, m: dict, all_meas: list[dict]) -> dict:
    laps = [x["laplacian_var"] for x in all_meas if x["laplacian_var"] is not None]
    soft = meas["laplacian_var"] is not None and laps and meas["laplacian_var"] < 0.4 * float(np.median(laps))
    flags = []
    if soft:
        flags.append("focus")
    if (meas["clipped_pct"] or 0) > 1.0:
        flags.append("clipping")
    if (meas["translate_px"] or 0) > 1.5:
        flags.append("shake")
    lab = {"scene": t.get("scene") or "1", "shot": t.get("shot") or f"{order}A", "take": t.get("take") or str(order),
           "dialogue": None, "best_take_guess": bool(t["events"]["circle_that"]), "flags": flags,
           "reason": "fallback heuristics (Spark not called); slate from the transcript parser or file order"}
    C.decide(m, t["id"], "label", "fallback labels",
             f"heuristics: flags {flags or 'none'} from the measurements; circle_that from the transcript")
    return lab


def run(d: str, use_spark: bool = True, require_spark: bool = False, n_frames: int = KEYFRAMES,
        replay: bool = False) -> dict:
    m = C.load_manifest(d)
    C.clear_decisions(m, "label")
    if not m.get("takes"):
        raise SystemExit("no takes yet - run takes.py first")
    api = MetaApi(shoot=d) if use_spark and not replay else None
    answers = _load_answers(d) if (use_spark or replay) else {}
    if require_spark and api is None and not replay:
        raise SystemExit("--require-spark with --no-spark makes no sense")
    files = {f["name"]: f for f in m.get("files", [])}
    meas_by_take, path_by_take = {}, {}
    for t in m["takes"]:
        f = files.get(t["file"], {})
        path = os.path.join(d, f.get("rel_path", os.path.join("originals", t["file"])))
        path_by_take[t["id"]] = path
        meas_by_take[t["id"]] = measure(path, t["usable"]["start_s"], t["usable"]["end_s"])
        t["measures"] = meas_by_take[t["id"]]
    for i, t in enumerate(m["takes"], 1):
        meas = meas_by_take[t["id"]]
        lab = None
        times = keyframe_times(t["usable"]["start_s"], t["usable"]["end_s"], n_frames) if (api or replay) else []
        sha = (files.get(t["file"], {}).get("sha256") or "")[:12]
        key = f"{t['id']}:{sha}:{t['usable']['start_s']:.3f}-{t['usable']['end_s']:.3f}:{len(times)}"
        if replay and key in answers:
            try:
                lab = parse_json(answers[key])
                C.decide(m, t["id"], "label", "replayed saved Spark answer", "--replay: no API call")
            except ValueError as e:
                C.decide(m, t["id"], "label", "saved Spark answer unreadable", str(e))
        if api:
            evidence = {"file": t["file"], "file_order": i, "slate": {"scene": t.get("scene"),
                        "shot": t.get("shot"), "take": t.get("take")}, "events": t["events"],
                        "circle_that_corroborated": bool(t["events"]["circle_that"]),
                        "usable_s": [t["usable"]["start_s"], t["usable"]["end_s"]],
                        "measurements": meas, "transcript": t["transcript"],
                        "dialogue_lines": [{k: ln[k] for k in ("start_s", "speaker", "text")}
                                           for ln in (t.get("dialogue") or {}).get("lines") or []]}
            content = [{"type": "input_text", "text": PROMPT + json.dumps(evidence)}]
            if times:
                frames = keyframes(path_by_take[t["id"]], times, C.cache_dir(d, os.path.join("keyframes", t["id"])))
                content += [{"type": "input_text", "text": f"Key frames 1 to {len(frames)} of the usable range, "
                                                           f"at {times} s:"}] + [image_part(p) for p in frames]
            for attempt in (1, 2):
                try:
                    # no max_output_tokens: Spark reasons first and would spend a small cap before answering
                    text, _ = api.respond(SPARK, content, step="label", take=t["id"])
                    lab = parse_json(text)
                    answers[key] = text
                    break
                except Exception as e:
                    C.decide(m, t["id"], "label", f"Spark attempt {attempt} failed", f"{type(e).__name__}: {e}")
        if lab is None and replay and require_spark:
            C.save_manifest(d, m)
            raise SystemExit(f"{t['id']}: no saved Spark answer to replay and --require-spark is set")
        if lab is None and require_spark:
            C.save_manifest(d, m)
            raise SystemExit(f"{t['id']}: Muse Spark failed twice and --require-spark is set: "
                             "for the competition, judgments come from Meta models or not at all. "
                             "Fix the API path and re-run; nothing was labelled by fallback code.")
        if lab is None:
            lab = fallback_labels(t, i, meas, m, list(meas_by_take.values()))
        # the transcript's "circle that" always wins the label; Spark's guess is advisory
        circled = bool(t["events"]["circle_that"]) or lab["best_take_guess"]
        by_spark = "fallback" not in lab["reason"] and (api is not None or replay)
        picture, pic_notes = check_picture(lab.get("picture"), len(times)) if by_spark else (None, [])
        # the transcription step's words win the dialogue label; Spark's line is the fallback
        heard = (t.get("dialogue") or {}).get("text")
        dlg_by = (t.get("dialogue") or {}).get("by") if heard else (
            "muse-spark-1.3" if by_spark and lab["dialogue"] else None)
        t["labels"] = {"scene": str(lab["scene"]), "shot": str(lab["shot"]), "take": str(lab["take"]),
                       "dialogue": heard or lab["dialogue"], "dialogue_by": dlg_by,
                       "best_take_guess": circled, "flags": lab["flags"],
                       "picture": picture,
                       "by": "muse-spark-1.3" if by_spark else "fallback"}
        for note in pic_notes:
            C.decide(m, t["id"], "label", "picture label corrected", note)
        if picture:
            C.decide(m, t["id"], "label",
                     f"picture: {picture['shot_type'] or '?'}, {picture['camera_move'] or '?'}"
                     f"{' · issues ' + ','.join(picture['issues']) if picture['issues'] else ''}"
                     f" · {'usable' if picture['usable'] else 'NOT usable'} (a guess, Caio decides)",
                     f"Muse Spark on {picture['frames']} key frames: {picture['why']}")
        if by_spark:
            C.decide(m, t["id"], "label",
                     f"scene {lab['scene']} shot {lab['shot']} take {lab['take']}"
                     f"{' · circle that?' if circled else ''}{' · flags ' + ','.join(lab['flags']) if lab['flags'] else ''}",
                     f"Muse Spark: {lab['reason']}")
        t["scene"], t["shot"], t["take"] = t["labels"]["scene"], t["labels"]["shot"], t["labels"]["take"]
        print(f"{t['id']}: {t['labels']['scene']}/{t['labels']['shot']}/{t['labels']['take']} "
              f"flags {lab['flags'] or '-'} circle={'yes' if circled else 'no'} ({t['labels']['by']})")
    # competition rule, surfaced: judgments are Meta models' (or the run is not competition-ready)
    judged_by = {t["id"]: t["labels"]["by"] for t in m["takes"]}
    fallbacks = [tid for tid, by in judged_by.items() if by == "fallback"]
    m["judgment"] = {
        "partner": "Meta API (muse-spark-1.3 labels; transcription by the model named per take)",
        "code_measures": "focus / clipping / shake are measured by code and decided by Muse Spark",
        "fallback_takes": fallbacks,
        "competition_ready": not fallbacks,
    }
    if fallbacks:
        C.decide(m, None, "label", f"{len(fallbacks)} take(s) labelled by FALLBACK heuristics",
                 f"{fallbacks}: not competition-ready; re-run with the Meta API up (or --require-spark "
                 "to fail instead)")
    if answers and api:
        _save_answers(d, answers)
    C.save_manifest(d, m)
    return m


def main(argv: list[str]) -> dict:
    n_frames, rest = KEYFRAMES, []
    it = iter(argv)
    for a in it:
        if a == "--frames":
            n_frames = int(next(it, "3"))
        elif a.startswith("--frames="):
            n_frames = int(a.split("=", 1)[1])
        else:
            rest.append(a)
    args = [a for a in rest if not a.startswith("--")]
    if len(args) != 1:
        raise SystemExit("usage: python -m precut.label shoot_<date> [--no-spark] [--require-spark] "
                         "[--frames N] [--replay]")
    return run(C.shoot_dir(args[0]), use_spark="--no-spark" not in rest,
               require_spark="--require-spark" in rest, n_frames=n_frames, replay="--replay" in rest)


if __name__ == "__main__":
    main(sys.argv[1:])
