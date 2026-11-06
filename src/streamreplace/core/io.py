"""File primitives for the outputs folder: fingerprints, atomic writes, locked appends, tolerant JSONL reading.

Generation rows are appended one whole line at a time under an exclusive file lock with fsync, so several shard
workers (and a stream adapter's request threads) can share one output file, also on NFS. Score files, metas and
caches are rewritten atomically (temporary file + fsync + rename). Readers tolerate a truncated final line (a killed
writer) and let the last row per id win."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Iterable, Iterator


def fingerprint(value) -> str:
    """sha256 of the canonical JSON of `value` (sorted keys, unescaped unicode, default separators), first 16 hex
    digits. The paper's runs used exactly this for run identities and judge cache keys, so it must not change."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def file_digest(path: Path) -> str:
    """sha256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    """Reads a JSONL file; a truncated final line (killed writer) is ignored, any other invalid line is an error.
    A missing file reads as no rows."""
    path = Path(path)
    if not path.exists():
        return []
    rows, lines = [], path.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except ValueError:
            if number == len(lines):
                break
            raise ValueError(f"Invalid JSONL: {path}:{number}")
    return rows


def latest(rows: Iterable[dict], key: str = "id") -> dict[str, dict]:
    """Retried samples appear more than once: the last row per id wins (dict order = first appearance)."""
    return {r[key]: r for r in rows}


def read_latest(path: Path, key: str = "id") -> dict[str, dict]:
    return latest(read_jsonl(path), key)


def atomic_text(path: Path, text: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(handle.name, 0o644)
    os.replace(handle.name, path)


def write_json(path: Path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows):
    atomic_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


_APPEND = threading.Lock()


def append_jsonl(path: Path, row: dict):
    """One whole line per call under an exclusive lock: shard workers (and a stream adapter's request threads)
    append to the same file. On NFS flock becomes a POSIX record lock, which does not exclude threads of one process,
    hence the thread lock as well."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _APPEND, path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def file_lock(path: Path, shared: bool = False, blocking: bool = True) -> Iterator:
    """Holds an flock on `path` (created if missing) for the duration of the block. With blocking=False a busy lock
    raises BlockingIOError immediately: the scorer uses this to skip output files a worker is still writing."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        flags = (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(handle, flags)
        try:
            yield handle
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
