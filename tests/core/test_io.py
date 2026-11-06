"""Output-file primitives (core/io.py): tolerant JSONL reading with last-row-wins, atomic rewrites, whole-line
appends shared by concurrent writers, and the non-blocking lock the scorer uses to skip busy files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from streamreplace.core.io import (append_jsonl, atomic_text, file_digest, file_lock, latest, read_json, read_jsonl,
                                   read_latest, write_json, write_jsonl)

SRC = Path(__file__).resolve().parents[2] / "src"


def test_read_jsonl_tolerates_only_a_truncated_last_line(tmp_path):
    path = tmp_path / "rows.jsonl"
    assert read_jsonl(path) == []                                     # a missing file has no rows
    path.write_text('{"id": "a", "raw": "x"}\n\n{"id": "b", "raw": "y"}\n{"id": "c", "ra')
    assert [r["id"] for r in read_jsonl(path)] == ["a", "b"]
    path.write_text('{"id": "a"}\n{"id": \n{"id": "c"}\n')
    with pytest.raises(ValueError, match="rows.jsonl:2"):
        read_jsonl(path)


def test_the_last_row_per_id_wins_in_first_appearance_order(tmp_path):
    rows = [{"id": "b", "v": 1}, {"id": "a", "v": 1}, {"id": "b", "v": 2}]
    assert latest(rows) == {"b": {"id": "b", "v": 2}, "a": {"id": "a", "v": 1}} and list(latest(rows)) == ["b", "a"]
    write_jsonl(tmp_path / "r.jsonl", rows)
    assert read_latest(tmp_path / "r.jsonl")["b"]["v"] == 2
    assert latest([{"key": "k", "x": 1}, {"key": "k", "x": 2}], key="key") == {"k": {"key": "k", "x": 2}}


def test_atomic_writes_leave_no_temporary_files(tmp_path):
    write_json(tmp_path / "deep" / "meta.json", {"identity": "abc", "text": "é"})
    assert read_json(tmp_path / "deep" / "meta.json") == {"identity": "abc", "text": "é"}
    assert "é" in (tmp_path / "deep" / "meta.json").read_text(encoding="utf-8")       # unescaped unicode
    write_jsonl(tmp_path / "deep" / "scores.jsonl", [{"id": "1"}, {"id": "2"}])
    atomic_text(tmp_path / "deep" / "scores.jsonl", '{"id": "3"}\n')
    assert read_jsonl(tmp_path / "deep" / "scores.jsonl") == [{"id": "3"}]
    assert sorted(p.name for p in (tmp_path / "deep").iterdir()) == ["meta.json", "scores.jsonl"]
    assert (tmp_path / "deep" / "meta.json").stat().st_mode & 0o777 == 0o644
    assert file_digest(tmp_path / "deep" / "scores.jsonl") == hashlib.sha256(b'{"id": "3"}\n').hexdigest()


APPEND = """
import sys
sys.path.insert(0, sys.argv[1])
from streamreplace.core.io import append_jsonl
for i in range(int(sys.argv[3])):
    append_jsonl(sys.argv[2], {"id": f"{sys.argv[4]}-{i}", "raw": "x" * 5000})
"""


def test_concurrent_appends_keep_every_line_whole(tmp_path):
    path = tmp_path / "job.jsonl"
    procs = [subprocess.Popen([sys.executable, "-c", APPEND, str(SRC), str(path), "60", f"p{k}"]) for k in range(4)]
    threads = [threading.Thread(target=lambda k=k: [append_jsonl(path, {"id": f"t{k}-{i}", "raw": "y" * 5000})
                                                     for i in range(60)]) for k in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(proc.wait(timeout=120) == 0 for proc in procs)
    lines = path.read_text().splitlines()
    rows = [json.loads(line) for line in lines]                       # every line parses: no interleaving
    assert len(rows) == 7 * 60 and len({r["id"] for r in rows}) == 7 * 60


HOLD = """
import sys, time
sys.path.insert(0, sys.argv[1])
from streamreplace.core.io import file_lock
with file_lock(sys.argv[2]):
    print("locked", flush=True)
    time.sleep(30)
"""


def test_a_busy_file_lock_is_reported_without_waiting(tmp_path):
    lock = tmp_path / ".job.lock"
    holder = subprocess.Popen([sys.executable, "-c", HOLD, str(SRC), str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        with pytest.raises(BlockingIOError):
            with file_lock(lock, blocking=False):
                pass
        with pytest.raises(BlockingIOError):
            with file_lock(lock, shared=True, blocking=False):
                pass
    finally:
        holder.kill()
        holder.wait()
    with file_lock(lock, blocking=False):                             # free again once the holder is gone
        pass
    with file_lock(lock, shared=True), file_lock(lock, shared=True, blocking=False):
        pass                                                          # shared locks coexist
    assert os.path.exists(lock)
