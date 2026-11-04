"""Pre-cut agent step 8 - the take list for Caio's approval (markdown + JSON, same content).

  python -m precut.report shoot_<date>

Writes shoot_<date>/TAKES.md and shoot_<date>/takes_report.json. Approving happens in the Latent Raw
app (Ingest screen); this report is the same list on paper. Encoding is step 9 and runs only after
that approval.
"""
from __future__ import annotations

import json
import os
import sys

from . import common as C
from .dialogue import srt
from .metaapi import call_report


def picture_cell(pic: dict | None) -> str:
    """Spark's picture labels in one table cell: shot, move, subjects, issues, usable guess."""
    if not pic:
        return "—"
    bits = [x for x in (pic.get("shot_type"), pic.get("camera_move")) if x]
    if pic.get("subjects"):
        bits.append(", ".join(pic["subjects"][:3]))
    if pic.get("issues"):
        bits.append("issues: " + ", ".join(pic["issues"]))
    bits.append("usable?" if pic.get("usable") else "not usable?")
    return "; ".join(bits).replace("|", "/")


def dialogue_cell(t: dict, lab: dict) -> str:
    """The take's dialogue in one cell: 'person_1: words' per line when transcribed, else the label."""
    lines = (t.get("dialogue") or {}).get("lines") or []
    if lines:
        cell = " / ".join(f"{ln['speaker']}: {ln['text']}{'' if ln.get('corroborated', True) else ' (?)'}"
                          for ln in lines)
        cell = cell if len(cell) <= 160 else cell[:157] + "..."
    else:
        cell = lab.get("dialogue") or "—"
    return cell.replace("|", "/")


def dialogue_section(takes: list[dict]) -> list[str]:
    out = ["## Dialogue", ""]
    any_lines = False
    for t in takes:
        dg = t.get("dialogue") or {}
        if not dg:
            continue
        head = f"- {t['id']} ({t['file']}): {dg.get('status')}"
        if dg.get("by"):
            head += f"; by {dg['by']} from {dg.get('source')} audio"
        out.append(head)
        for ln in dg.get("lines") or []:
            any_lines = True
            mark = "" if ln.get("corroborated", True) else " *(no speech onset near this line: check)*"
            out.append(f"  - {ln['start_s']:.2f} s, {ln['speaker']}: {ln['text']}{mark}".replace("|", "/"))
    if len(out) == 2:
        return []
    if any_lines:
        out += ["", "Speakers are person_1, person_2 ... per take, never names. Subtitles per take: "
                    "`dialogue/<id>.srt`."]
    return out + [""]


def build(m: dict) -> tuple[str, dict]:
    plan = m.get("plan") or {}
    win_by_take = {p["take_id"]: p["windows"] for p in plan.get("takes", [])}
    rows, takes = [], []
    for t in m.get("takes", []):
        lab = t.get("labels") or {}
        sync = t.get("sync") or {}
        wins = win_by_take.get(t["id"], [])
        row = {
            "id": t["id"],
            "label": f"{lab.get('scene', '?')} / {lab.get('shot', '?')} / {lab.get('take', '?')}",
            "file": t["file"],
            "usable_tc": f"{C.s_to_tc(t['usable']['start_s'])} – {C.s_to_tc(t['usable']['end_s'])}",
            "usable_s": [t["usable"]["start_s"], t["usable"]["end_s"]],
            "sync_offset_ms": sync.get("offset_ms"),
            "drift_ms": sync.get("drift_ms"),
            "flags": lab.get("flags", []),
            "picture": lab.get("picture"),
            "dialogue": lab.get("dialogue"),
            "dialogue_lines": (t.get("dialogue") or {}).get("lines") or [],
            "dialogue_by": lab.get("dialogue_by") or (t.get("dialogue") or {}).get("by"),
            "circle_that_label": bool(lab.get("best_take_guess")),
            "included": t.get("included", True),
            "windows": len(wins),
            "how_ranged": t["usable"].get("how"),
        }
        takes.append(row)
        rows.append("| {id} | {label} | {file} | {usable_tc} | {sync} | {drift} | {flags} | {pic} | {dlg} | {circ} | {win} |"
                    .format(id=t["id"], label=row["label"], file=t["file"], usable_tc=row["usable_tc"],
                            pic=picture_cell(row["picture"]),
                            sync="—" if sync.get("offset_ms") is None else f"{sync['offset_ms']:+.1f} ms",
                            drift="—" if sync.get("drift_ms") is None else f"{sync['drift_ms']:+.1f} ms",
                            flags=", ".join(row["flags"]) or "—", dlg=dialogue_cell(t, lab),
                            circ="circle that?" if row["circle_that_label"] else "", win=row["windows"]))
    ap = m.get("approval", {})
    j = m.get("judgment") or {}
    judge_line = ("Judgments: **Meta API** (Muse Spark on code-measured evidence; transcription by the model named in Dialogue) - "
                  + ("competition-ready." if j.get("competition_ready")
                     else f"NOT competition-ready: fallback labels on {j.get('fallback_takes')}.")) if j else \
                 "Judgments: not run yet."
    md = [
        f"# Take list - shoot {m.get('shoot_date')}",
        "",
        judge_line,
        "",
        f"Originals are untouched (SHA-256 per file in manifest.json). "
        f"Approval: **{'approved by ' + str(ap.get('approved_by')) + ' at ' + str(ap.get('approved_at')) if ap.get('approved') else 'not approved - nothing is encoded'}**.",
        "",
        "| id | scene / shot / take | file | usable (tc) | sync | drift | flags | picture (Spark) | dialogue | agent guess | windows |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        f"Encode plan: {plan.get('window_count', 0)} windows of {plan.get('window_frames', 241)} frames "
        f"(stride {plan.get('stride_frames', 192)}) at {plan.get('width', 1280)}x{plan.get('height', 704)}, "
        f"recipe `{plan.get('recipe_id', '—')}`. Runs only after approval.",
        "",
        *dialogue_section(m.get("takes", [])),
        "## How each range was found",
        *[f"- {t['id']} ({t['file']}): {t['usable'].get('how')}; transcript: {t.get('transcript_by')}"
          for t in m.get("takes", [])],
        "",
        "## Meta API",
        "```", "```",
    ]
    return "\n".join(md), {"schema": "shoot/take-list@1", "shoot_date": m.get("shoot_date"),
                           "approval": ap, "takes": takes, "plan": plan}


def run(d: str) -> dict:
    m = C.load_manifest(d)
    md, rep = build(m)
    calls = call_report(d)
    md = md.replace("```\n```", "```\n" + calls + "\n```")
    with open(os.path.join(d, "TAKES.md"), "w") as f:
        f.write(md + "\n")
    with open(os.path.join(d, "takes_report.json"), "w") as f:
        json.dump(rep, f, indent=2)
    subs = [t for t in m.get("takes", []) if (t.get("dialogue") or {}).get("lines")]
    if subs:
        os.makedirs(os.path.join(d, "dialogue"), exist_ok=True)
        for t in subs:
            with open(os.path.join(d, "dialogue", f"{t['id']}.srt"), "w") as f:
                f.write(srt(t))
    print(f"wrote {os.path.join(d, 'TAKES.md')} and takes_report.json ({len(rep['takes'])} takes)")
    print(calls)
    return rep


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m precut.report shoot_<date>")
    run(C.shoot_dir(sys.argv[1]))
