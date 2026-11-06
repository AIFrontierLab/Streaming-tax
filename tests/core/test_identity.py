"""Run identity and meta.json (core/identity.py), and the fingerprint shared with the original runs' identities and
judge cache keys (core/io.py, judges/base.py)."""
from __future__ import annotations

import hashlib
import json
import socket

import pytest

from streamreplace.core.identity import (IdentityMismatch, add_segment, canonical_json, check_identity, code_version,
                                         new_meta, read_meta, run_identity, run_payload, run_segment, runtime_info,
                                         write_meta)
from streamreplace.core.io import fingerprint
from streamreplace.core.protocol import Arm, Mode
from streamreplace.judges.base import cache_key

SETTINGS = {"generation": {"max_new_tokens": 512}, "video": {"fps": 2, "sampler": "qwen_vl_utils"}}
MODEL = {"checkpoint": {"repo": "aurateam/AURA", "revision": "88ee550629fcb4d84428cdbfb346d07f01ea6e03"},
         "adapter": "aura_stream", "adapter_protocol": 1, "decoding": {"repetition_penalty": 1.0}, "plugin": None}


def test_fingerprint_is_the_original_formula():
    value = {"question": "Qu'est-ce que c'est ?", "choices": {"B": "deux", "A": "un"}, "kind": "extract"}
    text = json.dumps(value, sort_keys=True, ensure_ascii=False)
    assert canonical_json(value) == text and '"choices": {"A": "un", "B": "deux"}' in text
    assert fingerprint(value) == cache_key(value) == hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    assert fingerprint({"a": 1}) == hashlib.sha256(b'{"a": 1}').hexdigest()[:16]
    assert fingerprint({"q": "é"}) == hashlib.sha256('{"q": "é"}'.encode()).hexdigest()[:16]
    assert fingerprint({"q": "é"}) != hashlib.sha256(b'{"q": "\\u00e9"}').hexdigest()[:16]


def test_payload_and_identity():
    payload = run_payload(Arm("aura_8b", Mode.STREAM), MODEL, "videomme", SETTINGS)
    assert payload == {"schema": 1, "job": {"model": "aura_8b", "mode": "stream", "variant": None}, "model": MODEL,
                       "dataset": "videomme", "settings": SETTINGS, "limit": None, "subset": None,
                       "prompt_overrides": None}
    identity = run_identity(payload)
    assert identity == hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16] and len(identity) == 16
    reordered = json.loads(json.dumps(payload))
    reordered["settings"] = {"video": {"sampler": "qwen_vl_utils", "fps": 2}, "generation": {"max_new_tokens": 512}}
    assert run_identity(reordered) == identity                        # key order never matters
    subset = run_payload(Arm("aura_8b", Mode.STREAM), MODEL, "videomme", SETTINGS, subset=["b", "a"])
    assert subset["subset"] == fingerprint(["a", "b"]) and run_identity(subset) != identity
    edits = run_payload(Arm("aura_8b", Mode.STREAM), MODEL, "videomme", SETTINGS, prompt_overrides={"1": [["a", "b"]]})
    assert edits["prompt_overrides"] == fingerprint({"1": [["a", "b"]]})
    variant = run_payload(Arm("aura_8b", Mode.STREAM, "sfix"), MODEL, "videomme", SETTINGS)
    assert variant["job"]["variant"] == "sfix" and run_identity(variant) != identity
    limited = run_payload(Arm("aura_8b", Mode.STREAM), MODEL, "videomme", SETTINGS, limit=5)
    assert run_identity(limited) != identity


def test_meta_segments_and_identity_check(tmp_path):
    payload = run_payload(Arm("aura_8b", Mode.STREAM), MODEL, "videomme", SETTINGS)
    path = tmp_path / "videomme" / "aura_8b.stream.meta.json"
    assert read_meta(path) is None
    check_identity(None, payload)                                     # nothing on disk: nothing to check
    meta = new_meta(payload)
    assert meta["identity"] == run_identity(payload) and meta["segments"] == [] and meta["created"].endswith("Z")
    write_meta(path, meta)
    assert read_meta(path) == meta
    check_identity(read_meta(path), payload)
    changed = run_payload(Arm("aura_8b", Mode.STREAM), MODEL, "videomme", {**SETTINGS, "stream": {"fps": 2}})
    with pytest.raises(IdentityMismatch, match="--fresh"):
        check_identity(read_meta(path), changed, "videomme/aura_8b.stream")
    add_segment(path, {"load_seconds": 1.0})
    add_segment(path, {"load_seconds": 2.0})
    assert [s["load_seconds"] for s in read_meta(path)["segments"]] == [1.0, 2.0]
    with pytest.raises(FileNotFoundError):
        add_segment(tmp_path / "missing.meta.json", {})


def test_run_segments_never_record_the_host():
    segment = run_segment(load_seconds=3.5, samples_sha256="0" * 64)
    assert segment["load_seconds"] == 3.5 and "packages" in segment and "code" in segment
    assert segment["streamreplace"] and segment["python"]

    def values(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from values(item)
        elif isinstance(value, list):
            for item in value:
                yield from values(item)
        else:
            yield value

    seen = {str(v) for v in values(segment)}
    assert not seen & {"host", "hostname", "node", socket.gethostname(), socket.getfqdn()}
    assert set(runtime_info(["no-such-package-xyz"])["packages"]) == set()


def test_code_version_reads_git_without_the_binary(tmp_path, monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: None)
    sha, other = "a" * 40, "b" * 40
    main = tmp_path / "main"
    (main / ".git/refs/heads").mkdir(parents=True)
    (main / ".git/HEAD").write_text("ref: refs/heads/main\n")
    (main / ".git/refs/heads/main").write_text(sha + "\n")
    assert code_version(main) == {"commit": sha, "branch": "main", "dirty": None}
    (main / ".git/refs/heads/main").unlink()                          # after `git gc`: packed refs only
    (main / ".git/packed-refs").write_text(f"# pack-refs with: peeled\n{sha} refs/heads/main\n")
    assert code_version(main)["commit"] == sha
    worktree, gitdir = tmp_path / "dev", main / ".git/worktrees/dev"  # layout of `git worktree add ../dev -b dev`
    gitdir.mkdir(parents=True)
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n")
    (gitdir / "commondir").write_text("../..\n")
    (gitdir / "HEAD").write_text("ref: refs/heads/dev\n")
    (main / ".git/refs/heads/dev").write_text(other + "\n")
    assert code_version(worktree) == {"commit": other, "branch": "dev", "dirty": None}
    (gitdir / "HEAD").write_text(sha + "\n")                          # detached HEAD
    assert code_version(worktree) == {"commit": sha, "branch": None, "dirty": None}
    assert code_version(tmp_path / "no-repo") == {"commit": None, "branch": None, "dirty": None}
