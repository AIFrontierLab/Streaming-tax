"""Frame digest: a short fingerprint of the frames a model was given for one video.

Every offline video row (conditions P and C) stores it as `info.frame_indices`, so it can be checked that all members
of a pair saw exactly the same frames, and that a re-run (or this code) samples the frames the paper's runs did.
The format is `"<number of frames>:<first 12 hex digits of a sha256>"`, e.g. "148:52bbfceee0cb".
"""
from __future__ import annotations

import hashlib


def frame_digest(meta: dict) -> str:
    """Fingerprint of a frame selection `meta` = {"indices": [...]} (+ "files" for a frame list).

    The hash covers the comma-joined frame indices, or for a frame list (a `.frames.json` video, whose indices are
    always 0..n-1) the frame files as stored in the list. It is computed on the sampled frames, before any
    model-specific padding (Qwen repeats the last frame of an odd count), so it is the same for every model.
    """
    keys = meta["files"] if "files" in meta else map(str, meta["indices"])
    return f"{len(meta['indices'])}:" + hashlib.sha256(",".join(keys).encode()).hexdigest()[:12]
