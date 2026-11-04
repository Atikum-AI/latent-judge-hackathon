"""Step 2 (next-steps 29 Oct): watch the Blackmagic Cloud sync folder for takes landing from the phone.

  python -m precut.cloud_watch shoot_<date> [--poll-s 5] [--stable-s 30]

Blackmagic Camera uploads proxy first, then the camera original; both sync to the Mac through Resolve's
cloud project media folder (Caio points Resolve's sync at shoot_<date>/cloud_sync/ - his clicks, not ours).
A file that has not grown for --stable-s seconds is complete:

  proxy     (name contains "_proxy" or sits under a Proxy/ folder) -> linked into proxies/
  original  (anything else)                                        -> linked into originals/

Linking is a hardlink (copy as fallback across volumes): nothing is moved, renamed or trimmed, and the
sync folder's files are never written to. After each arrival, ingest re-runs so the manifest is current.
The manifest records source: proxy|original per file; NOTHING is encoded from a proxy (plan.py refuses
windows for it), and when the original lands it supersedes the proxy's entry (both stay listed).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

from . import common as C
from . import ingest


def is_proxy(relpath: str) -> bool:
    parts = relpath.lower().split("/")
    return any(p == "proxy" for p in parts[:-1]) or "_proxy" in parts[-1]


def link_in(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return "already"
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def scan(sync_dir: str, state: dict, now: float, stable_s: float) -> list[str]:
    """One pass over cloud_sync/. Returns the relative paths that just became stable."""
    ready = []
    for root, _, names in os.walk(sync_dir):
        for n in names:
            if os.path.splitext(n)[1].lower() not in C.VIDEO_EXT:
                continue
            full = os.path.join(root, n)
            rel = os.path.relpath(full, sync_dir)
            size = os.path.getsize(full)
            ent = state.get(rel)
            if ent is None:
                state[rel] = {"size": size, "since": now, "done": False}
            elif ent["done"]:
                continue
            elif size != ent["size"]:
                ent.update({"size": size, "since": now})
            elif now - ent["since"] >= stable_s and size > 0:
                ent["done"] = True
                ready.append(rel)
    return ready


def arrive(d: str, rel: str, run_ingest: bool, log) -> str:
    sync_dir = os.path.join(d, "cloud_sync")
    src = os.path.join(sync_dir, rel)
    proxy = is_proxy(rel)
    sub = "proxies" if proxy else "originals"
    dst = os.path.join(d, sub, os.path.basename(rel))
    how = link_in(src, dst)
    m = C.load_manifest(d)
    C.decide(m, None, "cloud", f"{os.path.basename(rel)} landed as {sub}/ ({how})",
             f"Blackmagic Cloud sync; classified {'proxy' if proxy else 'original'} by its path; "
             f"the sync file was not moved or renamed")
    C.save_manifest(d, m)
    log(f"{rel} -> {sub}/ ({how})")
    if run_ingest:
        ingest.run(d)
        log("ingest re-run (manifest current)")
    return sub


def watch(d: str, poll_s: float = 5.0, stable_s: float = 30.0, run_ingest: bool = True,
          max_idle: int | None = None, log=print) -> None:
    sync_dir = os.path.join(d, "cloud_sync")
    os.makedirs(sync_dir, exist_ok=True)
    state: dict = {}
    idle = 0
    log(f"watching {sync_dir} (stable after {stable_s:g} s, poll {poll_s:g} s) - Ctrl-C stops")
    while True:
        ready = scan(sync_dir, state, time.time(), stable_s)
        if ready:
            idle = 0
            for rel in ready:
                try:
                    arrive(d, rel, run_ingest, log)
                except Exception as e:
                    log(f"!! {rel}: {type(e).__name__}: {e}")   # keep watching; the failure is in STATUS.md's remit
        else:
            idle += 1
            if max_idle is not None and idle >= max_idle:
                log(f"idle for {idle} polls - exiting (test mode)")
                return
        time.sleep(poll_s)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    opts = {a.split("=")[0][2:]: a.split("=")[1] for a in sys.argv[1:] if a.startswith("--") and "=" in a}
    if len(args) != 1:
        raise SystemExit("usage: python -m precut.cloud_watch shoot_<date> [--poll-s=5] [--stable-s=30] "
                         "[--max-idle=N] [--no-ingest]")
    watch(C.shoot_dir(args[0]), float(opts.get("poll-s", 5)), float(opts.get("stable-s", 30)),
          run_ingest="--no-ingest" not in sys.argv,
          max_idle=int(opts["max-idle"]) if "max-idle" in opts else None)
