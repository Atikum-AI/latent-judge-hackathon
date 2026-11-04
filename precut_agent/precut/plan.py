"""Pre-cut agent step 7 - the encode plan. NO ENCODE runs here or anywhere before approval.

  python -m precut.plan shoot_<date> [--recipe-id R]

Per kept range: 241-frame windows (10 s; LTX needs 8k+1 frames) with a stride of 192 frames, starts
aligned to 8-frame boundaries, at the 1280x704 working size (8 px cropped top and bottom; the original
stays 720). A range shorter than one window still gets one entry, marked padded: the encode pads the
tail, and the decision is logged.
"""
from __future__ import annotations

import sys

from . import common as C

DEFAULT_RECIPE = "ltx-develop-r1"   # one recipe for every window; a fixed seed and LoRA strengths live with it


def windows_for(start_f: int, end_f: int) -> list[dict]:
    n = end_f - start_f
    if n <= 0:
        return []
    if n < C.WINDOW_FRAMES:
        return [{"index": 1, "start_frame": start_f, "end_frame": end_f, "frames": n, "padded": True}]
    if n == C.WINDOW_FRAMES:
        return [{"index": 1, "start_frame": start_f, "end_frame": end_f, "frames": n, "padded": False}]
    out = []
    s = start_f - (start_f % 8)          # align the first window to an 8-frame boundary (spec)
    i = 0
    while True:
        e = min(s + C.WINDOW_FRAMES, end_f)
        out.append({"index": i + 1, "start_frame": s, "end_frame": e, "frames": e - s, "padded": False})
        if e >= end_f:
            break
        s += C.STRIDE_FRAMES
        if s + C.WINDOW_FRAMES > end_f and end_f - s < C.WINDOW_FRAMES:
            s = end_f - C.WINDOW_FRAMES  # the last window ends exactly at the range end
        i += 1
        if i > 10000:
            raise RuntimeError("window loop runaway")
    return out


def run(d: str, recipe_id: str = DEFAULT_RECIPE) -> dict:
    m = C.load_manifest(d)
    C.clear_decisions(m, "plan")
    if not m.get("takes"):
        raise SystemExit("no takes yet - run takes.py first")
    if m.get("approval", {}).get("approved"):
        raise SystemExit("the take list is already approved; the plan is frozen with it")
    plan_takes = []
    total = 0
    files = {f["name"]: f for f in m.get("files", [])}
    for t in m["takes"]:
        if not t.get("included", True):
            continue
        src = files.get(t["file"], {})
        source = src.get("source", "original")
        if source != "original" or src.get("superseded_by"):
            why = "nothing is encoded from a proxy; the plan fills in when the original lands" \
                if source == "proxy" else \
                "this take is a CONFORMED DEMO transcode - it demonstrates the chain and is never encoded"
            C.decide(m, t["id"], "plan", f"no windows: this take is a {source.upper()}", why)
            plan_takes.append({"take_id": t["id"], "range_frames": None, "windows": [],
                               "waiting_for": "original"})
            continue
        f0 = int(round(t["usable"]["start_s"] * C.FPS))
        f1 = int(round(t["usable"]["end_s"] * C.FPS))
        wins = windows_for(f0, f1)
        if wins and wins[0].get("padded"):
            C.decide(m, t["id"], "plan", "range shorter than one 241-frame window",
                     f"{wins[0]['frames']} frames usable; the encode pads the tail to 241 (no second window)")
        plan_takes.append({"take_id": t["id"], "range_frames": [f0, f1], "windows": wins})
        total += len(wins)
    m["plan"] = {"recipe_id": recipe_id, "width": C.WORK_W, "height": C.WORK_H, "crop_px": 8,
                 "window_frames": C.WINDOW_FRAMES, "stride_frames": C.STRIDE_FRAMES,
                 "overlap_frames": C.WINDOW_FRAMES - C.STRIDE_FRAMES,
                 "note": "PLAN ONLY - nothing is encoded before the take list is approved in the app",
                 "takes": plan_takes, "window_count": total}
    C.decide(m, None, "plan", f"{total} windows over {len(plan_takes)} takes, recipe {recipe_id}",
             "241-frame windows, stride 192, 8-frame aligned, 1280x704 working size; no encode has run")
    C.save_manifest(d, m)
    print(f"plan: {total} windows over {len(plan_takes)} takes at {C.WORK_W}x{C.WORK_H}, recipe {recipe_id} (no encode)")
    return m


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    if len(args) != 1:
        raise SystemExit("usage: python -m precut.plan shoot_<date> [--recipe-id=R]")
    run(C.shoot_dir(args[0]), opts.get("recipe-id", DEFAULT_RECIPE))
