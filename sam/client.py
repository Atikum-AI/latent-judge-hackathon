"""SAM 3.1 through the Meta Model API, for whole clips.

- upload(): POST /v1/files, so a clip goes up once and every concept reuses its file id
- segment(): one streamed POST /v1/responses (model sam-3.1) for one concept on one clip.
             Returns the raw output text, one line per frame, that tracks.parse() reads.
- every live call appends one JSON line to logs/sam.jsonl (no mask payloads, only sizes)

SAMPLE_MODE=1 replays the saved output text from a JSON cache, so judges need no API key.

Env: META_API_KEY, META_BASE_URL (default https://api.meta.ai/v1), SAM_LOG.
The docs' names MODEL_API_KEY and META_API_BASE_URL work too.
"""
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

MODEL = "sam-3.1"
DEFAULT_BASE = "https://api.meta.ai/v1"
DEFAULT_CACHE = "examples/sam_cache.json"


def _base():
    return (os.environ.get("META_BASE_URL") or os.environ.get("META_API_BASE_URL") or DEFAULT_BASE).rstrip("/")


def _key():
    key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY", "")
    if not key:
        raise SystemExit("META_API_KEY is not set (add it to .env, or run with SAMPLE_MODE=1)")
    return key


def _sample_mode():
    return os.environ.get("SAMPLE_MODE", "0") == "1"


def _log(kind, request, response):
    path = os.environ.get("SAM_LOG", "logs/sam.jsonl")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = {"ts": datetime.now(timezone.utc).isoformat(), "kind": kind,
            "request": request, "response": response}
    with open(path, "a") as f:
        f.write(json.dumps(line, default=str) + "\n")


def upload(video_path):
    """Upload an MP4/MOV (H.264, H.265 or MPEG-4) and return its file id."""
    with open(video_path, "rb") as f:
        r = requests.post(f"{_base()}/files", headers={"Authorization": f"Bearer {_key()}"},
                          data={"purpose": "user_data"},
                          files={"file": (Path(video_path).name, f)}, timeout=600)
    r.raise_for_status()
    body = r.json()
    _log("upload", {"file": str(video_path)}, {"id": body.get("id"), "status": body.get("status")})
    return body["id"]


def request_body(concept, file_id=None, video_url=None, image_url=None,
                 one_bit=False, confidence=True):
    """The Responses API body for one noun-phrase concept on one clip (or one image)."""
    if image_url:
        media = {"type": "input_image", "image_url": image_url}
    elif file_id:
        media = {"type": "input_video", "file_id": file_id}
    elif video_url:
        media = {"type": "input_video", "video_url": video_url}
    else:
        raise ValueError("give file_id, video_url or image_url")
    return {
        "model": MODEL,
        "stream": True,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": concept}, media]}],
        "metadata": {"include_confidence": "true" if confidence else "false",
                     "mask_encoding": "one_bit" if one_bit else "lossless"},
    }


def read_stream(lines):
    """Collect output text from Server-Sent Events lines. Raises on a failed response."""
    text = []
    completed = False
    for line in lines:
        if isinstance(line, bytes):
            line = line.decode()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        kind = event.get("type")
        if kind == "response.output_text.delta":
            text.append(event.get("delta") or "")
        elif kind == "response.completed":
            completed = True
        elif kind in ("response.failed", "response.incomplete", "error"):
            raise RuntimeError(f"SAM 3.1 {kind}: {json.dumps(event)[:500]}")
    if not completed:
        raise RuntimeError("SAM 3.1 stream ended before response.completed")
    # each delta is one frame line; make sure every line ends with a newline
    return "".join(t if t.endswith("\n") else t + "\n" for t in text)


def segment(concept, file_id=None, video_url=None, image_url=None, one_bit=False,
            cache_key=None, cache_path=DEFAULT_CACHE):
    """Segment and track every instance of one concept. Returns the raw output text.

    Live mode calls the API and saves the text under cache_key.
    Sample mode reads the cache and never calls the API.
    """
    cache_file = Path(cache_path)
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    if _sample_mode():
        if cache_key not in cache:
            raise KeyError(f"sample mode: no saved SAM output for {cache_key} in {cache_path}")
        return cache[cache_key]["output"]

    body = request_body(concept, file_id, video_url, image_url, one_bit)
    t0 = time.time()
    with requests.post(f"{_base()}/responses", json=body, stream=True, timeout=900,
                       headers={"Authorization": f"Bearer {_key()}",
                                "Accept": "text/event-stream"}) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"SAM 3.1 HTTP {r.status_code}: {r.text[:500]}")
        output = read_stream(r.iter_lines())
    elapsed = round(time.time() - t0, 2)
    _log("segment", {"concept": concept, "file_id": file_id, "video_url": video_url,
                     "image_url": None if image_url is None else image_url[:80],
                     "one_bit": one_bit},
         {"frames": output.count("\n"), "chars": len(output), "elapsed_s": elapsed})
    if cache_key:
        cache[cache_key] = {"concept": concept, "elapsed_s": elapsed, "output": output}
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=1))
    return output
