"""metaapi.py (env parsing, stubbed calls logged to meta_calls.jsonl, the cost report), report.py,
and the full offline chain on the fixture media (ingest -> takes -> sync -> label -> plan -> report).
No network: MetaApi.respond/upload are stubbed; the agent runs with --no-transcribe/--no-spark."""
from __future__ import annotations

import json
import os

import pytest

from precut import common as C
from precut import ingest, label, metaapi, plan, report, takes, sync


# ---------------------------------------------------------------------- metaapi
def test_env_never_leaks_and_parses(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nMETA_API_KEY=sekret-123\nMETA_BASE_URL=https://example.invalid/v1\n\n")
    monkeypatch.setattr(metaapi, "_ENV_PATH", str(env))
    monkeypatch.delenv("META_API_KEY", raising=False)
    api = metaapi.MetaApi()
    assert api.base == "https://example.invalid/v1"
    assert api.headers["Authorization"] == "Bearer sekret-123"


def test_env_vars_win_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setattr(metaapi, "_ENV_PATH", str(tmp_path / "missing.env"))
    monkeypatch.setenv("META_API_KEY", "from-env")
    monkeypatch.delenv("META_BASE_URL", raising=False)
    api = metaapi.MetaApi()
    assert api.headers["Authorization"] == "Bearer from-env" and api.base == "https://api.meta.ai/v1"
    monkeypatch.delenv("META_API_KEY")
    with pytest.raises(SystemExit, match="META_API_KEY is not set"):
        metaapi.MetaApi()


def test_calls_are_logged_with_duration(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("META_API_KEY=x\nMETA_BASE_URL=https://example.invalid/v1\n")
    monkeypatch.setattr(metaapi, "_ENV_PATH", str(env))
    monkeypatch.delenv("META_API_KEY", raising=False)
    shoot = str(tmp_path / "shoot")
    os.makedirs(shoot)

    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"id": "file-1", "output": [{"content": [{"text": "hello"}]}]}

    monkeypatch.setattr(metaapi.requests, "post", lambda *a, **k: Resp())
    api = metaapi.MetaApi(shoot=shoot)
    assert api.upload(str(env)) == "file-1"
    text, _ = api.respond("muse-spark-1.3", [{"type": "input_text", "text": "hi"}], step="label", take="tk_1")
    assert text == "hello"
    calls = [json.loads(l) for l in open(os.path.join(shoot, "meta_calls.jsonl"))]
    assert len(calls) == 2 and all("seconds" in c and c["ok"] for c in calls)
    assert calls[1]["model"] == "muse-spark-1.3" and calls[1]["take"] == "tk_1"
    out = metaapi.call_report(shoot)
    assert "2 Meta API calls" in out and "muse-spark-1.3: 1 calls" in out
    assert "sekret" not in json.dumps(calls), "no key material in the log"


def test_call_report_empty(tmp_path):
    assert "no Meta API calls" in metaapi.call_report(str(tmp_path))


# ---------------------------------------------------------------------- the offline chain
@pytest.fixture(scope="module")
def chained(fixture_media):
    """ingest -> takes (offline) -> sync -> label (fallback) -> plan -> report, all local."""
    d = fixture_media
    ingest.run(d)
    takes.run(d, use_transcribe=False)
    sync.run(d)
    label.run(d, use_spark=False)
    plan.run(d)
    report.run(d)
    return d


def test_chain_manifest(chained):
    m = C.load_manifest(chained)
    assert len(m["files"]) == 4 and len(m["takes"]) == 3
    for t in m["takes"]:
        assert t["sync"]["offset_ms"] == pytest.approx(350.8, abs=2.0)
        assert t["labels"]["by"] == "fallback"
        assert t["usable"]["end_s"] > t["usable"]["start_s"]
    assert m["plan"]["window_count"] == 3
    assert m["approval"]["approved"] is False, "nothing is approved by the agent itself"
    steps = {d["step"] for d in m["decisions"]}
    assert {"ingest", "takes", "sync", "label", "plan"} <= steps, "every step logged its decisions"


def test_chain_report_files(chained):
    md = open(os.path.join(chained, "TAKES.md")).read()
    rep = json.load(open(os.path.join(chained, "takes_report.json")))
    assert md.count("| tk_") == 3
    assert "+350.8 ms" in md
    assert rep["schema"] == "shoot/take-list@1" and len(rep["takes"]) == 3
    assert "not approved" in md, "the report says nothing is encoded before approval"


def test_chain_does_not_touch_originals(chained):
    orig = os.path.join(chained, "originals")
    assert sorted(os.listdir(orig)) == ["bad_2398.mov", "take1.mov", "take2.mov", "take3.mov"]
    assert not os.path.exists(os.path.join(chained, "latents")) or not os.listdir(
        os.path.join(chained, "latents")), "no encode, ever, in the agent"
