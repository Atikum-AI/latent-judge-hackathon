"""Turn SAM 3.1 output text into per-item tracks of full-frame masks.

SAM 3.1 sends one line per frame: <Nf>, then for each object its id, a box and a
box-local mask. Ids stay stable while an object is in view, so one id is one track.
Decoding uses Meta's own parser (meta-sam-parser), so the mask codec is theirs, not ours.

A track is named <concept>_<id>, e.g. car_3, and that name is the item label in measure/.
"""
import re
from dataclasses import dataclass, field

import numpy as np
from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format, \
    video_segmentation_format
from PIL import Image


@dataclass
class Track:
    item: str                 # car_3
    concept: str              # car
    object_id: str            # 3
    boxes: dict = field(default_factory=dict)        # frame -> (left, top, right, bottom), half-open
    confidence: dict = field(default_factory=dict)   # frame -> 0..1 or None
    masks: dict = field(default_factory=dict)        # frame -> bool array, frame-sized

    @property
    def frames(self):
        return sorted(self.masks)


def slug(concept):
    return re.sub(r"[^a-z0-9]+", "_", concept.lower()).strip("_") or "item"


def parse(output, media="video"):
    """Parse the full output text with Meta's parser. Returns its result (records, diagnostics)."""
    fmt = video_segmentation_format() if media == "video" else image_segmentation_format()
    parser = fmt.create_parser()
    parser.push(output if output.endswith("\n") else output + "\n", emit=False)
    return parser.finish(CompletedOutcome()).result


def paste(record, width, height):
    """A box-local mask record -> a frame-sized boolean mask."""
    m = record.mask
    raster = np.frombuffer(decode_mask_to_raster(m), np.uint8).reshape(m.height, m.width)
    b = record.bounds
    left, top = int(round(b.left)), int(round(b.top))
    right, bottom = int(round(b.right)), int(round(b.bottom))
    bw, bh = max(right - left, 1), max(bottom - top, 1)
    if (m.width, m.height) != (bw, bh):
        raster = np.asarray(Image.fromarray(raster * 255).resize((bw, bh), Image.NEAREST)) > 127
    out = np.zeros((height, width), bool)
    x0, y0, x1, y1 = max(left, 0), max(top, 0), min(left + bw, width), min(top + bh, height)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = raster[y0 - top:y1 - top, x0 - left:x1 - left].astype(bool)
    return out


def tracks_from_output(output, concept, width, height, media="video"):
    """{item: Track} for one concept. The latest revision of a mask wins."""
    result = parse(output, media)
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise ValueError(f"SAM output for '{concept}' has {len(errors)} bad lines, "
                         f"first: {errors[0].code}: {errors[0].message}")
    tracks = {}
    for rec in result.records:
        if rec.kind == "text":
            continue
        frame = rec.frame.frame_index if rec.frame is not None else 0
        item = f"{slug(concept)}_{rec.object_id}"
        t = tracks.setdefault(item, Track(item, concept, rec.object_id))
        if rec.kind == "box":
            t.boxes[frame] = (rec.left, rec.top, rec.right, rec.bottom)
            t.confidence[frame] = rec.confidence
        elif rec.kind == "mask":
            t.masks[frame] = paste(rec, width, height)
            if rec.confidence is not None:
                t.confidence[frame] = rec.confidence
    return tracks


def by_frame(tracks):
    """{frame: {item: mask}} across all tracks, the shape measure.metrics.item_rows wants."""
    out = {}
    for t in tracks.values():
        for f, m in t.masks.items():
            out.setdefault(f, {})[t.item] = m
    return dict(sorted(out.items()))
