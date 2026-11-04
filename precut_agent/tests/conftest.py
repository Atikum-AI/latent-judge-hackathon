"""Shared fixtures. The media comes from the committed synthetic shoot (shoot_2099-01-15/: 3 takes with
spoken slates + clap, a mid-take picture cut, a refused 23.976 file, a drifted external sound file).
Tests copy only originals/ + sound/ into a tmp shoot folder with a fresh manifest: the committed fixture
is never mutated, and no test touches the network (Meta calls are stubbed or run with --no-*)."""
from __future__ import annotations

import json
import os
import shutil

import pytest

from precut import common as C

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shoot_2099-01-15")


def fresh_manifest(date: str = "2099-01-15") -> dict:
    return {"schema": C.SCHEMA, "shoot_date": date, "created_at": C.utcnow(), "files": [], "takes": [],
            "plan": None,
            "approval": {"approved": False, "approved_by": None, "approved_at": None,
                         "approved_take_ids": [], "encoding": "not_started"},
            "meta_calls_file": "meta_calls.jsonl"}


@pytest.fixture(scope="session")
def fixture_media(tmp_path_factory) -> str:
    """A writable copy of the fixture media (originals + sound) with an empty manifest."""
    d = str(tmp_path_factory.mktemp("shoot") / "shoot_2099-01-15")
    shutil.copytree(os.path.join(FIXTURE, "originals"), os.path.join(d, "originals"))
    shutil.copytree(os.path.join(FIXTURE, "sound"), os.path.join(d, "sound"))
    C.save_manifest(d, fresh_manifest())
    return d


LAYOUT = ("originals", "sound", "stream", "latents", "developed", "cloud_sync", "proxies")


@pytest.fixture()
def shoot_dir(fixture_media, tmp_path) -> str:
    """Per-test fresh copy of the fixture media manifest state (media shared by hardlink via copy)."""
    d = str(tmp_path / "shoot_2099-01-15")
    for sub in LAYOUT:
        os.makedirs(os.path.join(d, sub))
    for sub in ("originals", "sound"):
        for n in os.listdir(os.path.join(fixture_media, sub)):
            os.link(os.path.join(fixture_media, sub, n), os.path.join(d, sub, n))
    C.save_manifest(d, fresh_manifest())
    return d


@pytest.fixture()
def decisions() -> dict:
    """A bare manifest-like dict that collects decision log entries."""
    return {"decisions": []}
