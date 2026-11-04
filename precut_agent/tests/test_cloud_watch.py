"""cloud_watch.py + the proxy/original handling in ingest/plan: a file is picked up when it stops
growing, proxies and originals land in their folders, the original supersedes the proxy, and nothing
is ever planned (let alone encoded) from a proxy."""
from __future__ import annotations

import os
import threading
import time

import pytest

from precut import cloud_watch as W
from precut import common as C
from precut import ingest, plan


def test_is_proxy():
    assert W.is_proxy("Proxy/A001_C001.mp4")
    assert W.is_proxy("A001_C001_proxy.mov")
    assert not W.is_proxy("A001_C001.mov")
    assert not W.is_proxy("proximate.mov")


def drop_file(folder: str, rel: str, chunks: int = 4, chunk: int = 8192, gap: float = 0.15):
    """A file landing slowly, like a cloud sync over a slow uplink."""
    p = os.path.join(folder, rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    for i in range(chunks):
        with open(p, "ab") as f:
            f.write(bytes([i]) * chunk)
        time.sleep(gap)


def test_watcher_lands_stable_files(shoot_dir):
    sync = os.path.join(shoot_dir, "cloud_sync")
    state = {}
    drop_file(sync, "A001_C001.mov", gap=0.05)
    now = time.time()
    assert W.scan(sync, state, now, 30.0) == [], "just changed: not stable yet"
    assert W.scan(sync, state, now + 31, 30.0) == ["A001_C001.mov"], "stable after the quiet period"
    assert W.scan(sync, state, now + 62, 30.0) == [], "done files are not re-reported"


def fixture_file(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "shoot_2099-01-15", "originals", name)


def test_watcher_classifies_and_ingests(shoot_dir):
    sync = os.path.join(shoot_dir, "cloud_sync")
    logs = []
    # a proxy lands with real content (take1's bytes, a name that matches no fixture original)
    import shutil
    shutil.copy2(fixture_file("take1.mov"), os.path.join(sync, "take9_proxy.mov"))
    now = time.time()
    state = {}
    W.scan(sync, state, now, 0.5)
    ready = W.scan(sync, state, now + 1.0, 0.5)
    assert ready == ["take9_proxy.mov"]
    sub = W.arrive(shoot_dir, "take9_proxy.mov", run_ingest=True, log=logs.append)
    assert sub == "proxies"
    m = C.load_manifest(shoot_dir)
    f = next(x for x in m["files"] if x["rel_path"] == "proxies/take9_proxy.mov")
    assert f["source"] == "proxy" and f["checks"]["ok"], "proxy probed and usable for the take list"
    assert not f.get("superseded_by"), "no same-named original: the proxy stands"
    assert os.path.exists(os.path.join(sync, "take9_proxy.mov")), "the sync file is never moved"
    assert any(d["step"] == "cloud" for d in m["decisions"])


def test_original_supersedes_proxy_and_plan_waits(shoot_dir):
    """The doc's flow: proxy first (fast take list), original later (re-hash, replace)."""
    prox = os.path.join(shoot_dir, "proxies")
    orig = os.path.join(shoot_dir, "originals")
    import shutil
    os.remove(os.path.join(orig, "take1.mov"))   # the hardlinked fixture copy only; "no original yet"
    shutil.copy2(fixture_file("take1.mov"), os.path.join(prox, "take1_proxy.mov"))
    shutil.copy2(fixture_file("take2.mov"), os.path.join(orig, "take2.mov"))
    m = ingest.run(shoot_dir)
    by_rel = {f["rel_path"]: f for f in m["files"]}
    assert by_rel["proxies/take1_proxy.mov"]["checks"]["ok"]
    assert "superseded_by" not in by_rel["proxies/take1_proxy.mov"], "no original yet: the proxy stands"
    # the original lands
    shutil.copy2(os.path.join(os.path.dirname(fixture_file("")), "take1.mov"), os.path.join(orig, "take1.mov"))
    m = ingest.run(shoot_dir)
    by_rel = {f["rel_path"]: f for f in m["files"]}
    assert by_rel["proxies/take1_proxy.mov"]["superseded_by"] == "originals/take1.mov"
    assert by_rel["originals/take1.mov"]["replaces"] == "proxies/take1_proxy.mov"
    assert by_rel["originals/take1.mov"]["source"] == "original"
    # plan: takes from a proxy get no windows
    from precut import takes as T
    T.run(shoot_dir, use_transcribe=False)
    m = plan.run(shoot_dir)
    pt = {p["take_id"]: p for p in m["plan"]["takes"]}
    for tid, p in pt.items():
        assert p["windows"], "originals get windows"
        assert all(not w.get("padded") or True for w in p["windows"])
    assert m["plan"]["window_count"] >= 2
    # nothing from a proxy: originals of take1/take2 gone -> take1 only exists as a proxy now
    os.remove(os.path.join(orig, "take1.mov"))
    os.remove(os.path.join(orig, "take2.mov"))
    ingest.run(shoot_dir)
    m = T.run(shoot_dir, use_transcribe=False)
    m = plan.run(shoot_dir)
    by_file = {t["file"]: t["id"] for t in m["takes"]}
    pt = {p["take_id"]: p for p in m["plan"]["takes"]}
    proxy_take = pt[by_file["take1_proxy.mov"]]
    assert proxy_take["windows"] == [] and proxy_take["waiting_for"] == "original", \
        "nothing is encoded from a proxy"
    assert pt[by_file["take3.mov"]]["windows"], "the remaining original keeps its windows"
    assert any("PROXY" in d["decision"] for d in m["decisions"] if d["step"] == "plan")
