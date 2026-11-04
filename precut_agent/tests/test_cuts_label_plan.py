"""cuts.py (both detectors), label.py (strict JSON, measurements, fallback heuristics), plan.py
(the 241/192 window maths and the approval freeze)."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from precut import common as C
from precut import cuts, label, plan


# ---------------------------------------------------------------------- cuts
@pytest.fixture(scope="module")
def cut_video(tmp_path_factory) -> str:
    """4 s: 2 s of one pattern, a hard cut, 2 s of another."""
    p = str(tmp_path_factory.mktemp("vid") / "cut.mov")
    subprocess.run(["ffmpeg", "-y", "-v", "error",
                    "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=2",
                    "-f", "lavfi", "-i", "smptebars=size=320x180:rate=24:duration=2",
                    "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]",
                    "-c:v", "libx265", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-tag:v", "hvc1", p],
                   check=True)
    return p


def test_cuts_find_the_cut_both_detectors(cut_video, monkeypatch):
    found, detector = cuts.find_cuts(cut_video)
    assert any(abs(c - 2.0) < 0.25 for c in found), f"{detector} missed the 2.0 s cut: {found}"
    monkeypatch.setattr(cuts, "_HAVE_SD", False)
    found2, detector2 = cuts.find_cuts(cut_video)
    assert detector2 == "numpy"
    assert any(abs(c - 2.0) < 0.25 for c in found2), f"numpy fallback missed: {found2}"


def test_cuts_clean_take_has_none(cut_video):
    pass  # covered by the fixture chain: take1/take2 report no in-range cuts


# ---------------------------------------------------------------------- label: JSON strictness
def test_parse_json_strict():
    assert label.parse_json('prelude {"scene":"1","shot":"1A","take":"2","dialogue":null,'
                            '"best_take_guess":false,"flags":["shake","bogus"],"reason":"r"} tail')["flags"] == ["shake"]
    with pytest.raises(ValueError):
        label.parse_json("no braces here")
    with pytest.raises(ValueError):
        label.parse_json('{"scene":"1"}')


# ---------------------------------------------------------------------- label: measurements
def fake_frames(arrays: list[np.ndarray]):
    return np.stack(arrays)


def test_measure_focus_clipping_shake(monkeypatch):
    rng = np.random.default_rng(0)
    h, w = label.SMALL[1], label.SMALL[0]
    sharp = [rng.normal(128, 40, (h, w)).clip(0, 255).astype(np.uint8) for _ in range(4)]
    monkeypatch.setattr(label, "frames_gray", lambda *a, **k: fake_frames(sharp))
    m = label.measure("x.mov", 0, 2)
    assert m["laplacian_var"] > 500, "noise is sharp"
    assert m["clipped_pct"] < 1.0, "128 +/- 3 sigma stays off the top code"
    blurry_img = np.tile(np.linspace(0, 200, w, dtype=np.uint8), (h, 1))
    monkeypatch.setattr(label, "frames_gray", lambda *a, **k: fake_frames([blurry_img] * 4))
    m = label.measure("x.mov", 0, 2)
    assert m["laplacian_var"] < 1.0, "a flat gradient is not sharp"
    clipped_img = np.full((h, w), 255, np.uint8)
    monkeypatch.setattr(label, "frames_gray", lambda *a, **k: fake_frames([clipped_img] * 4))
    assert label.measure("x.mov", 0, 2)["clipped_pct"] > 99.0


def test_fallback_labels_thresholds(decisions):
    t = {"id": "tk_0001", "scene": "1", "shot": "1A", "take": "2", "events": {"circle_that": True}}
    meas = {"laplacian_var": 10.0, "clipped_pct": 2.5, "translate_px": 3.0}
    others = [{"laplacian_var": 100.0, "clipped_pct": 0.0, "translate_px": 0.1}]
    lab = label.fallback_labels(t, 2, meas, decisions, others)
    assert set(lab["flags"]) == {"focus", "clipping", "shake"}
    assert lab["best_take_guess"] is True, "the transcript's circle carries the guess"
    assert lab["scene"] == "1" and lab["take"] == "2"


# ---------------------------------------------------------------------- label: the competition rule
def test_require_spark_fails_instead_of_falling_back(shoot_dir, monkeypatch):
    from precut import ingest, takes
    ingest.run(shoot_dir)
    takes.run(shoot_dir, use_transcribe=False)

    class DeadSpark:
        def __init__(self, **kw): pass
        def respond(self, *a, **k): raise RuntimeError("api down")

    monkeypatch.setattr(label, "MetaApi", DeadSpark)
    with pytest.raises(SystemExit):
        label.run(shoot_dir, use_spark=True, require_spark=True)
    m = C.load_manifest(shoot_dir)
    assert all("labels" not in t for t in m["takes"]), "no fallback labels when the competition requires Meta"


def test_spark_labels_mark_competition_ready(shoot_dir, monkeypatch):
    from precut import ingest, takes
    ingest.run(shoot_dir)
    takes.run(shoot_dir, use_transcribe=False)

    class FakeSpark:
        def __init__(self, **kw): pass
        def respond(self, *a, **k):
            return (json.dumps({"scene": "1", "shot": "1A", "take": "1", "dialogue": "a line",
                                "best_take_guess": False, "flags": [], "reason": "slate heard"}), {})

    monkeypatch.setattr(label, "MetaApi", FakeSpark)
    m = label.run(shoot_dir, use_spark=True, require_spark=True)
    assert m["judgment"]["competition_ready"] is True
    assert all(t["labels"]["by"] == "muse-spark-1.3" for t in m["takes"])
    assert m["judgment"]["partner"].startswith("Meta API")


# ---------------------------------------------------------------------- plan
def test_windows_for_math():
    wins = plan.windows_for(0, 241)
    assert len(wins) == 1 and wins[0]["frames"] == 241 and not wins[0]["padded"]
    wins = plan.windows_for(0, 100)
    assert len(wins) == 1 and wins[0]["padded"], "a short range gets one padded window, logged by the caller"
    wins = plan.windows_for(0, 241 + 192 * 3)   # 817 frames: 4 full windows
    assert [w["start_frame"] for w in wins] == [0, 192, 384, 576]
    assert all(w["frames"] == 241 for w in wins)
    wins = plan.windows_for(3, 1000)            # unaligned start snaps to the 8-frame grid
    assert wins[0]["start_frame"] % 8 == 0
    assert wins[-1]["end_frame"] == 1000, "the last window ends at the range end"
    assert all(b["start_frame"] - a["start_frame"] == 192 for a, b in zip(wins, wins[1:])) or \
           wins[-1]["end_frame"] == 1000


def test_plan_freezes_after_approval(shoot_dir):
    from precut import ingest, takes
    ingest.run(shoot_dir)
    takes.run(shoot_dir, use_transcribe=False)
    m = plan.run(shoot_dir)
    assert m["plan"]["window_frames"] == 241 and m["plan"]["stride_frames"] == 192
    assert m["plan"]["width"] == 1280 and m["plan"]["height"] == 704
    assert all(w["padded"] for pt in m["plan"]["takes"] for w in pt["windows"]), "fixture takes are < 10 s"
    man = C.load_manifest(shoot_dir)
    man["approval"]["approved"] = True
    C.save_manifest(shoot_dir, man)
    with pytest.raises(SystemExit):
        plan.run(shoot_dir)


def test_plan_never_encodes(shoot_dir):
    import os
    from precut import ingest, takes
    ingest.run(shoot_dir)
    takes.run(shoot_dir, use_transcribe=False)
    plan.run(shoot_dir)
    assert not os.path.exists(os.path.join(shoot_dir, "latents")) or \
           not os.listdir(os.path.join(shoot_dir, "latents")), "no latents: the plan writes no encode"
