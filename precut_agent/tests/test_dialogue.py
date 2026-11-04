"""dialogue.py: Muse Voice Transcribe for the dialogue between action and cut (Wonder while Voice
Transcribe is not served), the code's checks on the lines (person_N speakers, crew calls, times,
corroboration, hallucinated lines), the recorder-vs-scratch choice, --replay, --no-transcribe, and the
words reaching label and the take list. No network: MetaApi is faked, nothing is spent."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from precut import common as C
from precut import dialogue, ingest, label, plan, report, sync, takes

LINES = [{"t": 1.4, "speaker": "Speaker A", "text": "You said you'd be here by six."},
         {"t": 0.2, "speaker": "person_1", "text": "Action!"},
         {"t": 3.9, "speaker": "Speaker B", "text": "I know."}]


def fake_api(calls: list, text: str | None = None, served=("muse-wonder-1.0",)):
    """404 for models not in `served` (as Voice Transcribe is today), JSON lines from the rest."""
    class FakeApi:
        def __init__(self, **kw): pass

        def respond(self, model, content, **k):
            calls.append(model)
            assert content[1]["audio_url"].startswith("data:audio/wav;base64,")
            if model not in served:
                raise RuntimeError("404 Client Error: model_not_found")
            return (text if text is not None else "\n".join(json.dumps(x) for x in LINES)), {}
    return FakeApi


def no_api(**kw):
    raise AssertionError("--replay must not build a Meta API client")


@pytest.fixture()
def synced(shoot_dir):
    ingest.run(shoot_dir)
    takes.run(shoot_dir, use_transcribe=False)
    sync.run(shoot_dir)
    return shoot_dir


# ---------------------------------------------------------------------- parsing and checks
def test_parse_lines_accepts_the_shapes_models_send():
    jl = "\n".join(json.dumps(x) for x in LINES)
    assert dialogue.parse_lines(jl) == LINES
    assert dialogue.parse_lines("```json\n" + json.dumps(LINES) + "\n```") == LINES
    ts = dialogue.parse_lines("[00:01.400] person_2: Hello there.\n[00:03] Back again.")
    assert ts == [{"t": 1.4, "speaker": "person_2", "text": "Hello there."},
                  {"t": 3.0, "speaker": None, "text": "Back again."}]
    assert dialogue.parse_lines("") == []


def test_check_lines_anonymises_drops_crew_calls_and_corroborates():
    lines, notes = dialogue.check_lines(LINES, clip_start=3.6, clip_dur=4.2, segs=[(1.3, 2.2)])
    assert [ln["speaker"] for ln in lines] == ["person_1", "person_2"], "model names become person_N"
    assert [ln["text"] for ln in lines] == ["You said you'd be here by six.", "I know."], "'Action!' dropped"
    assert lines[0]["start_s"] == pytest.approx(5.0), "times come back in take seconds"
    assert lines[0]["corroborated"] and not lines[1]["corroborated"]
    assert any("crew call" in n for n in notes) and any("replaced" in n for n in notes)
    assert any("not corroborated" in n for n in notes)


def test_check_lines_clamps_times_flags_names_and_drops_hallucinations():
    raw = [{"t": 0.5, "speaker": "person_1", "text": "Hi"},
           {"t": 99, "speaker": "person_1", "text": "Jane Doe is late."},
           {"t": "soon", "speaker": None, "text": "Scene 1, shot 1A, take 2."}]
    lines, notes = dialogue.check_lines(raw, 0.0, 4.0, segs=[(0.4, 1.0)])
    assert len(lines) == 2 and lines[1]["start_s"] == 0.5, "an out-of-clip time takes the previous line's"
    assert any("outside the clip" in n for n in notes)
    assert any("may contain a name" in n for n in notes)
    assert any("crew call" in n for n in notes), "the slate belongs to the takes step"
    lines, notes = dialogue.check_lines(raw[:1], 0.0, 4.0, segs=[])
    assert lines == [] and any("no speech" in n for n in notes), "words over silence are invented"


def test_dialogue_range_and_recorder_check():
    t = {"events": {"action_s": 3.6, "cut_s": 7.8}, "usable": {"start_s": 2.6, "end_s": 8.8, "handles_s": 1.0}}
    assert dialogue.dialogue_range(t)[:2] == (3.6, 7.8)
    t["events"] = {"action_s": None, "cut_s": None}
    assert dialogue.dialogue_range(t)[:2] == (2.6, 8.8)
    rng = np.random.default_rng(0)
    gate = lambda: np.repeat(rng.random(20) > 0.4, 2400).astype(np.float32)
    a = (rng.normal(0, 0.2, 16000 * 3) * gate()).astype(np.float32)
    b = (rng.normal(0, 0.2, 16000 * 3) * gate()).astype(np.float32)
    assert dialogue.recorder_matches(a, 0.5 * a)[0], "the same moment through another mic lines up"
    ok, why = dialogue.recorder_matches(a, b)
    assert not ok and "do not line up" in why, "different words do not"
    ok, why = dialogue.recorder_matches(a, a[:16000])
    assert not ok and "covers" in why, "a recorder that stopped early does not cover the range"
    assert not dialogue.recorder_matches(a * 0, a * 0)[0], "silence proves nothing"


def test_srt_times_and_speakers():
    t = {"dialogue": {"range_s": [3.6, 7.8], "lines": [
        {"start_s": 4.2, "speaker": "person_1", "text": "Hello."},
        {"start_s": 4.5, "speaker": "person_2", "text": "Hi."}]}}
    out = dialogue.srt(t)
    assert "00:00:04,200 --> 00:00:05,000\nperson_1: Hello." in out, "a line lasts at least 0.8 s"
    assert "00:00:04,500 --> 00:00:07,800\nperson_2: Hi." in out


# ---------------------------------------------------------------------- the step on the fixture shoot
def test_run_tries_voice_transcribe_first_then_wonder(synced, monkeypatch):
    calls = []
    monkeypatch.setattr(dialogue, "MetaApi", fake_api(calls))
    m = dialogue.run(synced)
    assert calls[:2] == ["muse-voice-transcribe-1.0", "muse-wonder-1.0"], "the named model first"
    for t in m["takes"]:
        dg = t["dialogue"]
        assert dg["by"] == "muse-wonder-1.0", "the take says which model really ran"
        assert dg["status"] == "ok" and dg["speakers"] == 2
        assert {ln["speaker"] for ln in dg["lines"]} == {"person_1", "person_2"}
        assert dg["source"] == "scratch" or dg["source"].startswith("external:take2_ext.wav")
    by_id = {t["id"]: t for t in m["takes"]}
    assert by_id["tk_0002"]["dialogue"]["source"].startswith("external:"), "take 2's recorder file is used"
    assert by_id["tk_0003"]["dialogue"]["source"] == "scratch", "take 3 does not match take 2's recorder file"
    assert any(d["decision"] == "used the scratch audio, not the recorder" and d["take_id"] == "tk_0003"
               for d in m["decisions"])
    assert os.path.exists(os.path.join(synced, "precut_cache", "dialogue.json"))
    assert C.load_manifest(synced)["schema"] == "shoot/manifest@1"

    # a re-run spends nothing (cached), and --replay needs no API client at all
    n = len(calls)
    dialogue.run(synced)
    assert len(calls) == n, "cached answers are reused"
    monkeypatch.setattr(dialogue, "MetaApi", no_api)
    m2 = dialogue.run(synced, replay=True)
    assert [t["dialogue"]["lines"] for t in m2["takes"]] == [t["dialogue"]["lines"] for t in m["takes"]]


def test_recorder_check_sees_a_different_line_next_to_the_same_crew_call(synced, monkeypatch):
    """Live run 3 Nov: with action..cut from the transcript (3.60-7.86 s), take 3 was given take 2's line,
    because a 3 s window held the identical 'Action!' of both takes. 1 s windows catch the different line."""
    m = C.load_manifest(synced)
    for t in m["takes"]:
        t["events"].update({"action_s": 3.6, "cut_s": 7.86})
    C.save_manifest(synced, m)
    calls = []
    monkeypatch.setattr(dialogue, "MetaApi", fake_api(calls))
    by_id = {t["id"]: t for t in dialogue.run(synced)["takes"]}
    assert by_id["tk_0002"]["dialogue"]["source"].startswith("external:take2_ext.wav")
    assert by_id["tk_0003"]["dialogue"]["source"] == "scratch"


def test_replay_without_saved_answers_leaves_takes_empty(synced, monkeypatch):
    monkeypatch.setattr(dialogue, "MetaApi", no_api)
    m = dialogue.run(synced, replay=True)
    assert all(t["dialogue"]["status"] == "no saved transcript to replay" for t in m["takes"])


def test_no_transcribe_and_refusal(synced, monkeypatch):
    m = dialogue.run(synced, use_transcribe=False)
    assert all(t["dialogue"]["status"].startswith("skipped") and t["dialogue"]["lines"] == [] for t in m["takes"])
    calls = []
    monkeypatch.setattr(dialogue, "MetaApi", fake_api(calls, text="Sorry, I can't transcribe this audio."))
    m = dialogue.run(synced)
    assert all(t["dialogue"]["status"].startswith("transcription failed") for t in m["takes"])
    assert not os.path.exists(os.path.join(synced, "precut_cache", "dialogue.json")), "a refusal is not cached"


def test_scratch_flag_and_no_model_served(synced, monkeypatch):
    calls = []
    monkeypatch.setattr(dialogue, "MetaApi", fake_api(calls, served=()))
    m = dialogue.run(synced, force_scratch=True)
    assert all(t["dialogue"]["status"].startswith("transcription failed") for t in m["takes"])
    assert all(t["dialogue"]["source"] == "scratch" for t in m["takes"])


def test_words_reach_label_and_take_list(synced, monkeypatch):
    calls = []
    monkeypatch.setattr(dialogue, "MetaApi", fake_api(calls))
    dialogue.run(synced)
    m = label.run(synced, use_spark=False)
    for t in m["takes"]:
        assert t["labels"]["dialogue"] == "You said you'd be here by six. I know."
        assert t["labels"]["dialogue_by"] == "muse-wonder-1.0"
    plan.run(synced)
    rep = report.run(synced)
    md = open(os.path.join(synced, "TAKES.md")).read()
    assert "## Dialogue" in md and "person_1: You said you'd be here by six." in md
    assert "by muse-wonder-1.0" in md, "the report names the model that ran, not the one we wish ran"
    assert rep["takes"][0]["dialogue_lines"] and rep["takes"][0]["dialogue_by"] == "muse-wonder-1.0"
    srt = open(os.path.join(synced, "dialogue", "tk_0001.srt")).read()
    assert "person_1: You said you'd be here by six." in srt
