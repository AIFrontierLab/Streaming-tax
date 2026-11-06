"""Judge caches: one append-only JSONL file per judge configuration, `<outputs>/judges/<judge>.<config_hash>.jsonl`.

Each line is {"key": <cache key>, ...result fields}; the result fields are the judge's own (extraction judge:
"extracted" | "reply" | "verdict"; HarmBench classifiers: "label", "how"; RJScore: "probs", "rjscore", "backend",
"missing"). Keys are computed from exactly the fields the paper's runs hashed (judges/base.py), so result files written
by those runs can be appended unchanged and resolve every reported row.

- The file name carries the judge's config hash (checkpoint and revision, backend, decoding, prompt version), so
  changing the judge never reuses another judge's answers.
- Reading: the last line per key wins. Appends are whole lines under an exclusive file lock, so judge processes on
  several machines may share one file on a shared disk; no host names are involved.
- A judge process appends one line per answered request (flushed), so an interrupted run resumes where it stopped.
  A request that failed is never written and is asked again on the next pass."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
from typing import Iterable, Iterator

from ..core.io import read_jsonl

CACHE_DIR = "judges"


def cache_path(outputs: Path, judge: str, config_hash: str) -> Path:
    return Path(outputs) / CACHE_DIR / f"{judge}.{config_hash}.jsonl"


class JudgeCache:
    """The results of one judge configuration (see the module docstring). Loaded lazily; refresh() re-reads."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._rows: dict[str, dict] | None = None
        self._lock = threading.Lock()

    @classmethod
    def at(cls, outputs: Path, judge: str, config_hash: str) -> JudgeCache:
        return cls(cache_path(outputs, judge, config_hash))

    def __repr__(self) -> str:
        return f"JudgeCache({str(self.path)!r})"

    @property
    def rows(self) -> dict[str, dict]:
        if self._rows is None:
            self._rows = {r["key"]: r for r in read_jsonl(self.path)}
        return self._rows

    def refresh(self) -> JudgeCache:
        self._rows = None
        return self

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[str]:
        return iter(self.rows)

    def missing(self, keys: Iterable[str]) -> list[str]:
        """The keys without a result, in first-appearance order, without duplicates."""
        return [k for k in dict.fromkeys(keys) if k not in self.rows]

    def add(self, results: Iterable[dict]) -> int:
        """Appends results ({"key", ...}) in one locked write; returns how many lines were written."""
        import fcntl
        entries, lines = [], []
        for result in results:
            if not isinstance(result, dict) or not isinstance(result.get("key"), str):
                raise ValueError(f"{self.path}: a cache entry needs a string 'key': {str(result)[:200]}")
            line = json.dumps(result, ensure_ascii=False)
            entries.append(json.loads(line))
            lines.append(line + "\n")
        if not lines:
            return 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                try:
                    handle.write("".join(lines))
                    handle.flush()
                    os.fsync(handle.fileno())
                finally:
                    fcntl.flock(handle, fcntl.LOCK_UN)
            if self._rows is not None:
                self._rows.update((entry["key"], entry) for entry in entries)
        return len(lines)

    def append(self, result: dict) -> None:
        """One result, written and flushed at once (judge processes call this as answers arrive; thread-safe)."""
        self.add([result])
