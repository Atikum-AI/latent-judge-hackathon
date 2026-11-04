"""common.py helpers and ingest.py: probing, hashing, refusal-with-warning, the manifest round trip."""
from __future__ import annotations

import json
import os

import pytest

from precut import common as C
from precut import ingest


def test_tc_round_trip():
    assert C.s_to_tc(0) == "00:00:00"
    assert C.s_to_tc(3725.4) == "01:02:05"
    assert C.tc_to_s("01:02:05") == 3725.0
    assert C.tc_to_s("01:02:05:12") == 3725.0          # frames dropped by design (clap refines)


def test_fps_of_exact_and_fractional():
    assert C.fps_of({"avg_frame_rate": "24/1"}) == (24.0, "24/1")
    fps, raw = C.fps_of({"avg_frame_rate": "24000/1001"})
    assert abs(fps - 23.976) < 0.001 and raw == "24000/1001"
    assert C.fps_of({"r_frame_rate": "0/0"}) == (0.0, "0/0")


def test_manifest_round_trip_and_schema_guard(tmp_path):
    d = str(tmp_path / "shoot_2099-01-15")
    os.makedirs(d)
    m = {"schema": C.SCHEMA, "files": [], "takes": []}
    C.save_manifest(d, m)
    assert C.load_manifest(d)["schema"] == C.SCHEMA
    bad = {"schema": "something/else"}
    C.save_manifest(d, {**m, **bad})
    with pytest.raises(SystemExit):
        C.load_manifest(d)


def test_decide_and_clear(tmp_path):
    m = {"decisions": []}
    C.decide(m, "tk_0001", "takes", "a", "because")
    C.decide(m, None, "sync", "b", "because")
    C.decide(m, None, "takes", "c", "because")
    C.clear_decisions(m, "takes")
    assert [d["decision"] for d in m["decisions"]] == ["b"]
    assert m["decisions"][0]["step"] == "sync"


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"shootday")
    import hashlib
    assert C.sha256_file(str(p)) == hashlib.sha256(b"shootday").hexdigest()


# ---------------------------------------------------------------------- ingest on the fixture media
def test_ingest_probes_hashes_and_refuses_off_spec(shoot_dir):
    m = ingest.run(shoot_dir)
    assert m["schema"] == "shoot/manifest@1"
    files = {f["name"]: f for f in m["files"]}
    assert len(files) == 4, "3 good takes + the refused one, all listed"
    bad = files["bad_2398.mov"]
    assert bad["checks"]["ok"] is False and bad["checks"]["fps_ok"] is False
    assert any("23.976" in w for w in bad["warnings"]), "the refusal names the real fps"
    for name in ("take1.mov", "take2.mov", "take3.mov"):
        f = files[name]
        assert f["checks"]["ok"], name
        assert f["probe"]["fps_raw"] == "24/1", "exactly 24 fps, never 23.976"
        assert (f["probe"]["width"], f["probe"]["height"]) == (1280, 720)
        assert f["probe"]["codec"] == "hevc" and f["probe"]["bit_depth"] == 10
        assert f["probe"]["apple_log"].startswith("likely")
        assert len(f["sha256"]) == 64
    assert any(d["step"] == "ingest" for d in m["decisions"])
    # re-run is clean: same list, same hashes, decisions replaced not duplicated
    m2 = ingest.run(shoot_dir)
    assert [f["sha256"] for f in m2["files"]] == [f["sha256"] for f in m["files"]]
    assert sum(1 for d in m2["decisions"] if d["step"] == "ingest") == 1


def test_ingest_originals_untouched(shoot_dir):
    before = {n: os.path.getmtime(os.path.join(shoot_dir, "originals", n))
              for n in os.listdir(os.path.join(shoot_dir, "originals"))}
    ingest.run(shoot_dir)
    after = {n: os.path.getmtime(os.path.join(shoot_dir, "originals", n))
             for n in os.listdir(os.path.join(shoot_dir, "originals"))}
    assert before == after, "originals are never moved, renamed or trimmed"
