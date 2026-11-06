"""Deterministic smoke / --limit order shared by the benchmarks (ported unchanged).

Groups (or single ids) are sorted by the sha256 of their key and appended whole until about n ids are chosen, so a
limited run keeps complete groups (an MMBench question with all its rotations, a TimeBlind instance, a video)."""
from __future__ import annotations

from collections import defaultdict
import hashlib


def smoke(samples, group: str | None = None, n: int = 50) -> list[str]:
    """Deterministic subset of about n samples; whole groups are kept together."""
    groups = defaultdict(list)
    for sample in samples:
        groups[str(sample.metadata[group]) if group else sample.id].append(sample.id)
    order = sorted(groups, key=lambda k: hashlib.sha256(k.encode()).hexdigest())
    chosen: list[str] = []
    for key in order:
        if len(chosen) >= n:
            break
        chosen.extend(groups[key])
    return chosen
