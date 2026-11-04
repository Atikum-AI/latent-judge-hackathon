"""label.py picture labels (merged from the old top-level precut/ labeller): key frames sent with the
evidence, the checks on Spark's picture answer (vocabulary, person_N, evidence frames), --frames 0,
--replay from saved answers, and the picture column in the take list. No network: MetaApi is faked."""
from __future__ import annotations

import json
import os

import pytest

from precut import common as C
from precut import ingest, label, report, takes

PICTURE = {"shot_type": "medium", "camera_move": "static", "subjects": ["person_1", "Jane Doe", "a test card"],
           "action": "A test pattern holds still.", "location": "interior, studio", "lighting": "flat",
           "issues": ["none", "sparkles", "camera_shake"], "usable": True, "why": "Sharp and steady.",
           "evidence": {"static": [1, 2, 3], "bogus": [9], "flag": [True]}}


def answer(picture=PICTURE) -> str:
    return json.dumps({"scene": "1", "shot": "1A", "take": "1", "dialogue": "a line", "best_take_guess": False,
                       "flags": [], "reason": "slate heard", "picture": picture})


def fake_spark(sent: list, text: str = None):
    class FakeSpark:
        def __init__(self, **kw): pass

        def respond(self, model, content, **k):
            sent.append(sum(1 for c in content if c["type"] == "input_image"))
            assert all(c["image_url"].startswith("data:image/jpeg;base64,")
                       for c in content if c["type"] == "input_image")
            return (text or answer()), {}
    return FakeSpark


@pytest.fixture()
def ready(shoot_dir):
    ingest.run(shoot_dir)
    takes.run(shoot_dir, use_transcribe=False)
    return shoot_dir


def test_check_picture_vocabulary_and_anonymity():
    pic, notes = label.check_picture(PICTURE, 3)
    assert pic["subjects"] == ["person_1", "person_2", "a test card"], "a real-looking name is replaced"
    assert pic["issues"] == ["camera_shake"]
    assert pic["evidence"] == {"static": [1, 2, 3]}, "frames that were not sent (9, True) are dropped"
    assert any("real name" in n for n in notes) and any("sparkles" in n for n in notes)
    bad, notes = label.check_picture({"shot_type": "cinematic", "usable": "maybe"}, 3)
    assert bad["shot_type"] is None and bad["usable"] is False and len(notes) >= 2
    assert label.check_picture(None, 3) == (None, [])
    assert label.check_picture(PICTURE, 0) == (None, []), "no frames sent, no picture labels kept"
    _, notes = label.check_picture({**PICTURE, "action": "Jane Doe walks in."}, 3)
    assert any("may contain a name" in n for n in notes)


def test_keyframe_times_stay_inside():
    assert label.keyframe_times(10.0, 20.0, 3) == [11.0, 15.0, 19.0]
    assert label.keyframe_times(0.0, 4.0, 1) == [2.0]
    assert label.keyframe_times(0.0, 4.0, 0) == []


def test_spark_sees_key_frames_and_picture_lands_in_manifest(ready, monkeypatch):
    sent = []
    monkeypatch.setattr(label, "MetaApi", fake_spark(sent))
    m = label.run(ready, use_spark=True, require_spark=True)
    assert sent == [3] * len(m["takes"]), "three key frames per take go with the evidence"
    for t in m["takes"]:
        pic = t["labels"]["picture"]
        assert pic["shot_type"] == "medium" and pic["subjects"][1] == "person_2" and pic["frames"] == 3
        assert t["labels"]["by"] == "muse-spark-1.3"
        kf = os.path.join(ready, "precut_cache", "keyframes", t["id"])
        assert len(os.listdir(kf)) == 3, "key frames are cached in precut_cache, not in the repo"
    assert m["schema"] == C.SCHEMA == "shoot/manifest@1"
    steps = [x["decision"] for x in m["decisions"] if x["step"] == "label"]
    assert any(s.startswith("picture: medium, static") for s in steps)
    assert any(x["decision"] == "picture label corrected" for x in m["decisions"])


def test_frames_zero_is_text_only(ready, monkeypatch):
    sent = []
    monkeypatch.setattr(label, "MetaApi", fake_spark(sent))
    m = label.main([ready, "--frames", "0"])
    assert set(sent) == {0}
    assert all(t["labels"]["picture"] is None for t in m["takes"])


def test_replay_uses_saved_answers_without_calls(ready, monkeypatch):
    sent = []
    monkeypatch.setattr(label, "MetaApi", fake_spark(sent))
    live = label.run(ready, use_spark=True)
    assert os.path.exists(os.path.join(ready, "precut_cache", "spark_labels.json"))

    class NoCalls:
        def __init__(self, **kw): pytest.fail("--replay must not build an API client")

    monkeypatch.setattr(label, "MetaApi", NoCalls)
    again = label.run(ready, use_spark=True, require_spark=True, replay=True)
    assert [t["labels"] for t in again["takes"]] == [t["labels"] for t in live["takes"]]
    assert again["judgment"]["competition_ready"] is True


def test_replay_without_saved_answer_fails_under_require_spark(ready):
    with pytest.raises(SystemExit, match="no saved Spark answer"):
        label.run(ready, use_spark=True, require_spark=True, replay=True)


def test_fallback_has_no_picture(ready):
    m = label.run(ready, use_spark=False)
    assert all(t["labels"]["picture"] is None and t["labels"]["by"] == "fallback" for t in m["takes"])


def test_take_list_shows_picture(ready, monkeypatch):
    monkeypatch.setattr(label, "MetaApi", fake_spark([]))
    m = label.run(ready, use_spark=True)
    md, rep = report.build(m)
    assert "picture (Spark)" in md and "medium; static; person_1, person_2, a test card" in md
    assert rep["takes"][0]["picture"]["usable"] is True
    assert report.picture_cell(None) == "—"
