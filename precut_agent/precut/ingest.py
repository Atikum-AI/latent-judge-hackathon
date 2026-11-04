"""Pre-cut agent step 1 - ingest: probe every original, hash it, write the manifest.

  python -m precut.ingest shoot_<date>

Non-destructive: originals are never moved, renamed or trimmed. A file that is not 24.000 fps or
1280x720 is REFUSED for processing with a warning, but it stays in the manifest's files list (it is
never dropped silently).
"""
from __future__ import annotations

import os
import sys

from . import common as C

# Apple Log on iPhone records HEVC/ProRes 10-bit tagged BT.2020. ffprobe does not name Apple Log as a
# transfer characteristic, so this is a documented best-effort: 10-bit + a BT.2020 colour tag reads as
# "likely", an explicit BT.709 tag reads as "no" (the phone recorded Rec.709), untagged reads "unknown".
def bit_depth_of(video: dict) -> int | None:
    raw = video.get("bits_per_raw_sample")
    if raw not in (None, "N/A"):
        return int(raw)
    import re
    m = re.search(r"p(\d+)(le|be)?$", video.get("pix_fmt") or "")
    return int(m.group(1)) if m else None


def apple_log(video: dict, bits: int | None) -> str:
    codec = video.get("codec_name")
    transfer = str(video.get("color_transfer") or "").lower()
    tags = " ".join(str(video.get(k) or "") for k in ("color_primaries", "color_transfer", "color_space")).lower()
    if "arib" in transfer or "hlg" in transfer:
        return "no (HLG)"                       # HDR, not log
    if "smpte2084" in transfer:
        return "no (PQ)"                        # HDR PQ, not log
    if codec in ("hevc", "prores") and bits == 10 and "2020" in tags:
        return f"likely ({codec} 10-bit bt2020)"
    if "bt709" in tags or "bt470" in tags or "smpte170" in tags:
        return f"no (tagged {video.get('color_space') or 'bt709'})"
    return "unknown"


# The working size is 1280x720 (crop to 704); 4K originals are accepted and downscaled for the
# working copy (the plan, 28-29 Oct). Anything else is refused.
SIZES_OK = {(C.WIDTH, C.HEIGHT): "working size", (3840, 2160): "4K, downscaled to 1280x704 for the working copy"}


def base_of(name: str) -> str:
    """A proxy and its original share the base name once the _proxy marker is stripped."""
    b = os.path.splitext(os.path.basename(name))[0]
    for marker in ("_proxy", "_Proxy", "_PROXY"):
        if b.endswith(marker):
            b = b[: -len(marker)]
    return b


def probe_file(d: str, sub: str, name: str) -> dict:
    path = os.path.join(d, sub, name)
    source = {"proxies": "proxy", "compat": "conformed_demo"}.get(sub, "original")
    j = C.ffprobe(path)
    v = next((s for s in j["streams"] if s.get("codec_type") == "video"), None)
    if v is None:
        return {"name": name, "error": "no video stream", "checks": {"ok": False}, "warnings": ["no video stream"]}
    fps, fps_raw = C.fps_of(v)
    w, h = v.get("width"), v.get("height")
    tags = v.get("tags") or j.get("format", {}).get("tags") or {}
    warnings = []
    fps_ok = fps_raw == "24/1"
    if not fps_ok:
        warnings.append(f"codec frame rate is {fps_raw} ({fps:.3f}), the protocol is 24/1 exactly - "
                        "refused for processing")
    size_ok = (w, h) in SIZES_OK
    if not size_ok:
        if source == "proxy":
            # a proxy previews the take list only; the working-size rule binds the ORIGINAL
            warnings.append(f"size is {w}x{h} (proxy; the size rule binds the original)")
        else:
            warnings.append(f"size is {w}x{h}, the protocol takes 1280x720 or 3840x2160 - refused for processing")
    ok = fps_ok and (size_ok or source in ("proxy", "conformed_demo"))
    dur_frames = float(v.get("duration") or 0) * fps
    nb = int(v.get("nb_frames") or 0)
    if fps_ok and nb and abs(dur_frames - nb) > 1:
        warnings.append(f"container duration is off by {abs(dur_frames - nb):.1f} frames vs the frame count "
                        f"(coarse timebase at stop); the codec rate is exactly 24 - no drift")
    dur = float(j["format"].get("duration") or v.get("duration") or 0)
    bits = bit_depth_of(v)
    return {
        "name": name,
        "rel_path": f"{sub}/{name}",
        "source": source,
        "size_bytes": os.path.getsize(path),
        "sha256": C.sha256_file(path),
        "probe": {
            "codec": v.get("codec_name"),
            "profile": v.get("profile"),
            "width": w, "height": h,
            "fps": round(fps, 6), "fps_raw": fps_raw, "avg_fps_raw": C.avg_fps_of(v),
            "duration_s": round(dur, 3),
            "frames": int(v.get("nb_frames") or 0) or None,
            "bit_depth": bits,
            "pix_fmt": v.get("pix_fmt"),
            "timecode": tags.get("timecode"),
            "color": {"primaries": v.get("color_primaries"), "transfer": v.get("color_transfer"),
                      "space": v.get("color_space"), "range": v.get("color_range")},
            "apple_log": "no (conformed demo; not shot log)" if source == "conformed_demo"
                         else apple_log(v, bits),
            "audio": [{"codec": a.get("codec_name"), "sample_rate": a.get("sample_rate"),
                       "channels": a.get("channels")} for a in j["streams"] if a.get("codec_type") == "audio"],
        },
        "checks": {"fps_ok": fps_ok, "size_ok": size_ok, "ok": ok},
        "warnings": warnings,
    }


def run(d: str) -> dict:
    m = C.load_manifest(d)
    C.clear_decisions(m, "ingest")
    found = []                                   # (sub, name)
    for sub in ("originals", "proxies", "compat"):   # compat: clearly-labelled test/demo transcodes
        p = os.path.join(d, sub)
        if os.path.isdir(p):
            found += [(sub, n) for n in sorted(os.listdir(p))
                      if os.path.splitext(n)[1].lower() in C.VIDEO_EXT]
    if not found:
        print(f"no camera files in {d}/originals/ - copy the phone files there first "
              f"(or let the cloud watcher land them)")
        return m
    files = []
    for sub, name in found:
        f = probe_file(d, sub, name)
        files.append(f)
        ok = f["checks"]["ok"]
        p = f.get("probe", {})
        print(f"{'ok ' if ok else 'REFUSED'} {f['rel_path']}: {p.get('width')}x{p.get('height')} {p.get('fps_raw')} fps "
              f"({p.get('fps')}), {p.get('codec')} {p.get('bit_depth') or '?'}bit, {p.get('duration_s')} s, "
              f"tc {p.get('timecode') or '-'}, apple log {p.get('apple_log')}, sha256 {f['sha256'][:12]}…")
        for w in f["warnings"]:
            print(f"     warning: {w}")
    # an original supersedes its proxy (proxy first, original later - the Blackmagic Cloud order)
    orig_bases = {base_of(f["name"]): f for f in files if f["source"] == "original"}
    superseded = 0
    for f in files:
        if f["source"] != "proxy":
            continue
        o = orig_bases.get(base_of(f["name"]))
        if o is not None:
            f["superseded_by"] = o["rel_path"]
            o["replaces"] = f["rel_path"]
            superseded += 1
            C.decide(m, None, "ingest", f"{o['name']} supersedes its proxy",
                     f"the original landed ({o['sha256'][:12]}…); {f['rel_path']} stays in the list, "
                     "takes and the encode plan use the original")
    m["files"] = files
    m["ingested_at"] = C.utcnow()
    C.decide(m, None, "ingest", f"probed and hashed {len(files)} files",
             f"{sum(1 for f in files if f['checks']['ok'])} usable, "
             f"{sum(1 for f in files if not f['checks']['ok'])} refused (kept in the list with a warning), "
             f"{superseded} proxies superseded by their originals")
    C.save_manifest(d, m)
    print(f"\nwrote {C.manifest_path(d)} ({len(files)} files, {superseded} superseded)")
    return m


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python -m precut.ingest shoot_<date>")
    run(C.shoot_dir(sys.argv[1]))
