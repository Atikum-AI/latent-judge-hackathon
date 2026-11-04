"""sync.py: GCC-PHAT clap sync and the drift check, on signals with a known offset and clock drift,
then on the fixture media (350 ms late, 500 ppm slow external)."""
from __future__ import annotations

import numpy as np
import pytest

from precut.sync import SR, correlate, offset_at, run


def synth_pair(offset_s=0.350, stretch=1.0005, dur_s=12.0, seed=1):
    """scratch = noise; ext = scratch, offset_s late and (stretch-1) ppm slow."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, int(dur_s * SR)).astype(np.float32)
    n = int(len(x) * stretch)
    xi = np.interp(np.arange(n) / stretch, np.arange(len(x)), x).astype(np.float32)
    ext = np.zeros(int((dur_s + offset_s + 1) * SR), np.float32)
    ext[int(offset_s * SR):int(offset_s * SR) + len(xi)] = xi
    return x, ext


def test_correlate_finds_the_offset():
    x, ext = synth_pair(stretch=1.0)   # no drift here: the whole-signal offset is exact
    off, w = correlate(x, ext, 5)
    assert off == pytest.approx(0.350, abs=0.002), "ext content 350 ms late reads +350 ms"
    assert w > 20, "a strong PHAT peak"


def test_correlate_on_a_drifting_pair_reads_the_average():
    """With clock drift the whole-signal offset is the take's average; the start/end split is the
    drift check's job (offset_at), tested separately."""
    x, ext = synth_pair(offset_s=0.350, stretch=1.0005, dur_s=12.0)
    off, _ = correlate(x, ext, 5)
    mid = 0.350 + 6.0 * 0.0005
    assert off == pytest.approx(mid, abs=0.004)


def test_offset_at_tracks_the_drift():
    x, ext = synth_pair()
    off0, w0 = offset_at(x, ext, 2.0, 2.0, 0.350)
    off1, w1 = offset_at(x, ext, 10.0, 2.0, 0.350)
    assert off0 == pytest.approx(0.350 + 2.0 * 0.0005, abs=0.002)
    assert off1 == pytest.approx(0.350 + 10.0 * 0.0005, abs=0.002)
    assert (off1 - off0) == pytest.approx(8.0 * 0.0005, abs=0.001), "500 ppm over 8 s ≈ 4 ms"


def test_sync_on_the_fixture(shoot_dir):
    """take2_ext.wav is take 2's audio 350 ms late and 500 ppm slow (made by make_test_fixtures)."""
    from precut import ingest, takes
    ingest.run(shoot_dir)
    takes.run(shoot_dir, use_transcribe=False)
    m = run(shoot_dir)
    by_file = {t["file"]: t["sync"] for t in m["takes"]}
    for name in ("take1.mov", "take2.mov"):
        s = by_file[name]
        assert s["external_file"] == "take2_ext.wav"
        assert s["offset_ms"] == pytest.approx(350.8, abs=2.0)
        assert abs(s["drift_frames"]) < 1.0, "well inside the 1-frame pass bar"
        assert s["resampled"] is False, "0.08 f of drift does not trigger a resample"
    s3 = by_file["take3.mov"]
    assert s3["drift_ms"] is None, "the external file ends before take 3: drift is honestly unknown"
    assert any("ends at" in d["reason"] for d in m["decisions"]
               if d["step"] == "sync" and d["take_id"] == "tk_0003")


def test_sync_without_sound_folder(tmp_path):
    import os
    from precut import common as C
    d = str(tmp_path / "shoot_x")
    os.makedirs(os.path.join(d, "originals"))
    C.save_manifest(d, {"schema": C.SCHEMA, "files": [],
                        "takes": [{"id": "tk_0001", "file": "x.mov"}]})
    m = run(d)
    assert m["takes"][0]["sync"]["source"] == "none"
    assert m["takes"][0]["sync"]["offset_ms"] is None
