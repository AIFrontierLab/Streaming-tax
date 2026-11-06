"""Metric helpers shared by the benchmarks (values in percent, rounded to 2 decimals; ported unchanged).

A benchmark's metrics(rows) returns an ordered dict whose FIRST key is the headline metric reported in the paper.
rows are counted samples: {"id", "correct", "kind", "metadata", ...} (and "video" for by_length)."""
from __future__ import annotations

from collections import defaultdict


def pct(values) -> float | None:
    values = list(values)
    return round(100 * sum(bool(v) for v in values) / len(values), 2) if values else None


def by(rows, key: str, prefix: str = "") -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["metadata"].get(key, "unknown"))].append(row["correct"])
    return {f"{prefix}{name}": pct(values) for name, values in sorted(groups.items())}


def by_length(rows, cut: float = 80.0) -> dict:
    """Accuracy for videos up to / over `cut` seconds (p1 offline caps videos over 80 s at 160 uniform frames)."""
    from ...media.video_meta import video_info
    groups: dict[str, list] = {f"len/<={cut:g}s": [], f"len/>{cut:g}s": []}
    durations: dict[str, float] = {}
    for row in rows:
        if row.get("video"):
            if row["video"] not in durations:
                durations[row["video"]] = video_info(row["video"])["duration"]
            groups[f"len/<={cut:g}s" if durations[row["video"]] <= cut else f"len/>{cut:g}s"].append(row["correct"])
    return {name: pct(values) for name, values in groups.items() if values}


def group_all(rows, key: str, size: int | None = None) -> tuple[float | None, int]:
    """Share of groups whose members are all correct; groups missing members are skipped."""
    groups = defaultdict(list)
    for row in rows:
        groups[str(row["metadata"][key])].append(row["correct"])
    complete = [v for v in groups.values() if size is None or len(v) == size]
    return pct(all(v) for v in complete), len(complete)


def metrics(rows) -> dict:
    result = {"acc": pct(r["correct"] for r in rows)}
    if any("category" in r["metadata"] for r in rows):
        result.update(by(rows, "category", "cat/"))
    return result
