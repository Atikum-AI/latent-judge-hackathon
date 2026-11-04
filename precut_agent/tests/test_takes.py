"""takes.py: slate parsing, phrase markers, the speech-onset event alignment (the part that places
action/cut and guards against hallucinated 'circle that'), usable ranges, and the offline run."""
from __future__ import annotations

import numpy as np
import pytest

from precut import takes as T
from precut.sync import SR


# ---------------------------------------------------------------------- parsing
@pytest.mark.parametrize("line,expect", [
    ("Scene 1, shot 1A, take 2.", ("1", "1A", "2")),
    ("scene one, shot one A, take two", ("1", "1A", "2")),
    ("[00:01.000] Scene two, shot 3, take twelve", ("2", "3", "12")),
])
def test_slate_regex(line, expect):
    m = T.SLATE_RE.search(line)
    assert m, line
    shot = T.norm_num(m.group(2)) + (m.group(3).upper() if m.group(3) else "")
    assert (T.norm_num(m.group(1)), shot, T.norm_num(m.group(4))) == expect


def test_parse_phrases_with_markers():
    ph = T.parse_phrases("[00:00.000] Scene 1, shot 1A, take 1.\n[00:02.800] action\nno marker line")
    assert ph[0] == {"model_t": 0.0, "text": "Scene 1, shot 1A, take 1."}
    assert ph[1]["model_t"] == 2.8
    assert ph[2]["model_t"] is None


# ---------------------------------------------------------------------- synthetic audio
def burst(dur_s: float, amp: float = 0.1, seed: int = 0) -> np.ndarray:
    """Speech-like noise burst (RMS well below the clap, as real speech is)."""
    return (np.random.default_rng(seed).normal(0, amp, int(dur_s * SR))).astype(np.float32)


def clap() -> np.ndarray:
    t = np.arange(int(0.08 * SR)) / SR
    return (0.9 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)


def build_audio(with_circle: bool) -> np.ndarray:
    """The protocol timeline: slate 0.3-1.8, clap 2.8, action 3.6, dialogue 4.2-5.5, cut 7.8,
    circle 8.6 (optional)."""
    a = np.zeros(int(10.0 * SR), np.float32)
    a += np.random.default_rng(9).normal(0, 0.001, len(a)).astype(np.float32)
    def put(seg, at):
        i = int(at * SR)
        a[i:i + len(seg)] += seg[: len(a) - i]
    put(burst(1.5, seed=1), 0.3)          # slate
    put(clap(), 2.8)                      # clap
    put(burst(0.4, seed=2), 3.6)          # action
    put(burst(1.3, seed=3), 4.2)          # dialogue
    put(burst(0.3, seed=4), 7.8)          # cut
    if with_circle:
        put(burst(1.0, seed=5), 8.6)      # circle that
    return a


def test_speech_segments_ignore_the_clap():
    segs = T.speech_segments(build_audio(True))
    starts = [round(s, 1) for s, _ in segs]
    assert 2.8 not in starts, "the 80 ms beep is not speech"
    assert min(starts, key=lambda s: abs(s - 3.6)) < 3.8
    assert any(abs(s - 8.6) < 0.3 for s in starts)


def test_align_events_lands_on_onsets_and_keeps_a_real_circle(decisions):
    audio = build_audio(True)
    claps = T.clap_times(audio)
    assert len(claps) == 1 and abs(claps[0] - 2.8) < 0.05
    phrases = [{"model_t": 0.0, "text": "Scene 1, shot 1A, take 1."},
               {"model_t": 2.8, "text": "action"},          # model clock ~0.8 s early, as measured live
               {"model_t": 4.1, "text": "you said it"},
               {"model_t": 6.9, "text": "cut"},
               {"model_t": 8.0, "text": "circle that one."}]
    ev = T.align_events(phrases, audio, claps, decisions, "tk_x")
    assert abs(ev["action_s"] - 3.6) < 0.1
    assert abs(ev["cut_s"] - 7.8) < 0.1
    assert ev["circle_that_s"] is not None and abs(ev["circle_that_s"] - 8.6) < 0.3
    assert ev["slates"][0]["scene"] == "1" and ev["slates"][0]["shot"] == "1A"


def test_align_events_drops_a_hallucinated_circle(decisions):
    """The transcribe model invents 'circle that' sometimes (seen live on 29 Oct): with no circle in
    the audio, the phrase must be dropped and the drop logged with its reason."""
    audio = build_audio(False)
    claps = T.clap_times(audio)
    phrases = [{"model_t": 0.0, "text": "Scene 1, shot 1A, take 1."},
               {"model_t": 2.8, "text": "action"},
               {"model_t": 4.1, "text": "you said it"},
               {"model_t": 7.0, "text": "cut"},
               {"model_t": 8.0, "text": "circle that"}]     # phantom: nothing after cut in the audio
    ev = T.align_events(phrases, audio, claps, decisions, "tk_x")
    assert ev["circle_that_s"] is None
    drops = [d for d in decisions["decisions"] if "IGNORED" in d["decision"]]
    assert drops and "hallucination" in drops[0]["reason"]


def test_usable_range_and_fallbacks(decisions):
    dur = 10.0
    ev = {"action_s": 3.6, "cut_s": 7.8}
    s, e, how = T.usable_range(dur, ev, [], "tk_x", decisions)
    assert (s, e) == (pytest.approx(2.6, abs=0.05), pytest.approx(8.8, abs=0.05)) and "action..cut" in how
    s, e, how = T.usable_range(dur, {"action_s": None, "cut_s": None}, [2.8], "tk_x", decisions)
    assert "estimated" in how and s == pytest.approx(2.8, abs=0.05) and e == dur
    s, e, how = T.usable_range(dur, {"action_s": None, "cut_s": None}, [], "tk_x", decisions)
    assert (s, e) == (0.0, dur) and "whole file" in how
    assert any("fallback" in d["decision"] for d in decisions["decisions"])


# ---------------------------------------------------------------------- transcribe fallback (stubbed API)
def test_transcribe_falls_back_to_wonder_and_logs(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)   # content is only read into base64 for the stub

    class StubApi:
        def __init__(self): self.models = []
        def respond(self, model, content, **kw):
            self.models.append(model)
            if model == T.TRANSCRIBE[0]:
                raise Exception("404 model_not_found")
            return ("[00:00.000] scene 1, shot 1A, take 1.", {})

    api = StubApi()
    text, model = T.transcribe(api, str(wav), "tk_x")
    assert api.models == ["muse-voice-transcribe-1.0", "muse-wonder-1.0"], "named model first, wonder fallback"
    assert "scene 1" in text and model == "muse-wonder-1.0"


def test_transcribe_raises_when_no_model_serves(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF" + b"\x00" * 100)

    class StubApi:
        def respond(self, model, content, **kw):
            raise Exception("404 model_not_found")
    with pytest.raises(RuntimeError):
        T.transcribe(StubApi(), str(wav), "tk_x")


# ---------------------------------------------------------------------- offline run on the fixture media
def test_takes_offline_finds_three_takes(shoot_dir):
    from precut import ingest
    ingest.run(shoot_dir)
    m = T.run(shoot_dir, use_transcribe=False)
    assert len(m["takes"]) == 3
    for t in m["takes"]:
        assert t["events"]["claps_s"], "the clap is found in every take"
        assert "estimated" in t["usable"]["how"], "offline ranges are marked estimated"
    take3 = next(t for t in m["takes"] if t["file"] == "take3.mov")
    assert any(abs(c - 6.0) < 0.2 for c in take3["cuts_in_range_s"]), "the planned camera stop at 6.0 s"
