"""Meta Model API client for the pre-cut agent: upload once, then /responses.

Follows the proven pattern of latent-judge/sam/sam_track.py (POST {BASE}/files, then {BASE}/responses).
Keys come from the environment (META_API_KEY, META_BASE_URL) or else ~/Documents/hackathon/meta/.env, and are
never printed, logged or written anywhere.

Every call is logged to shoot_<date>/meta_calls.jsonl with its model, duration and size, so the cost
review (handoff test 6) is a sum over that file.
"""
from __future__ import annotations

import json
import os
import time

import requests

_ENV_PATH = os.path.expanduser("~/Documents/hackathon/meta/.env")

SPARK = "muse-spark-1.3"
# Tried in order. muse-voice-transcribe-1.0 is listed in /models but not served on /responses on this
# account (404 model_not_found, verified 29 Oct); muse-wonder-1.0 transcribes audio_url input well.
TRANSCRIBE = ("muse-voice-transcribe-1.0", "muse-wonder-1.0")


def _env() -> dict:
    """META_API_KEY / META_BASE_URL from the environment when set (as the README says), else from
    ~/Documents/hackathon/meta/.env. Neither value is ever printed."""
    if os.environ.get("META_API_KEY"):
        return {"META_API_KEY": os.environ["META_API_KEY"],
                "META_BASE_URL": os.environ.get("META_BASE_URL") or "https://api.meta.ai/v1"}
    if not os.path.exists(_ENV_PATH):
        raise SystemExit("META_API_KEY is not set and ~/Documents/hackathon/meta/.env does not exist: "
                         "export META_API_KEY (and META_BASE_URL) or create that file")
    env = {}
    for line in open(_ENV_PATH):
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k] = v
    return env


class MetaApi:
    def __init__(self, shoot: str | None = None, timeout: int = 240):  # a hung model must not hang the pipeline
        env = _env()
        self.base = env["META_BASE_URL"].rstrip("/")
        self.headers = {"Authorization": f"Bearer {env['META_API_KEY']}"}
        self.timeout = timeout
        self.log_path = os.path.join(shoot, "meta_calls.jsonl") if shoot else None

    def _log(self, entry: dict) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry}
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def upload(self, path: str) -> str:
        """POST /files (purpose user_data), returns the file id. One upload per file per run."""
        t0 = time.time()
        size = os.path.getsize(path)
        err = None
        try:
            with open(path, "rb") as fh:
                r = requests.post(f"{self.base}/files", headers=self.headers,
                                  files={"file": fh}, data={"purpose": "user_data"}, timeout=self.timeout)
            r.raise_for_status()
            return r.json()["id"]
        except Exception as e:  # logged, then raised: the caller decides whether the step can fall back
            err = f"{type(e).__name__}: {e}"[:300]
            raise
        finally:
            self._log({"step": "upload", "endpoint": "files", "file_bytes": size,
                       "seconds": round(time.time() - t0, 2), "ok": err is None, **({"error": err} if err else {})})

    def respond(self, model: str, content: list[dict], step: str, take: str | None = None,
                max_output_tokens: int | None = None) -> tuple[str, dict]:
        """POST /responses with one user message; returns (text, raw json). Never logs key material."""
        body = {"model": model, "input": [{"role": "user", "content": content}]}
        if max_output_tokens:
            body["max_output_tokens"] = max_output_tokens
        t0 = time.time()
        err = None
        try:
            r = requests.post(f"{self.base}/responses", headers={**self.headers, "Content-Type": "application/json"},
                              json=body, timeout=self.timeout)
            r.raise_for_status()
            j = r.json()
            text = "".join(c.get("text", "") for o in j.get("output", []) for c in o.get("content", []))
            return text, j
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:300]
            raise
        finally:
            self._log({"step": step, "model": model, "endpoint": "responses", "take": take,
                       "seconds": round(time.time() - t0, 2), "ok": err is None,
                       **({"error": err} if err else {})})


def call_report(shoot: str) -> str:
    """Handoff test 6: count and duration of the Meta API calls made for a shoot."""
    p = os.path.join(shoot, "meta_calls.jsonl")
    if not os.path.exists(p):
        return "no Meta API calls logged"
    calls = [json.loads(l) for l in open(p)]
    by_model: dict[str, dict] = {}
    for c in calls:
        m = c.get("model") or "(upload)"
        e = by_model.setdefault(m, {"n": 0, "seconds": 0.0, "failed": 0})
        e["n"] += 1
        e["seconds"] += c.get("seconds", 0.0)
        e["failed"] += 0 if c.get("ok") else 1
    lines = [f"{len(calls)} Meta API calls for {os.path.basename(shoot)}:"]
    for m, e in sorted(by_model.items()):
        bad = f", {e['failed']} failed" if e["failed"] else ""
        lines.append(f"  {m}: {e['n']} calls, {e['seconds']:.1f} s total{bad}")
    return "\n".join(lines)
