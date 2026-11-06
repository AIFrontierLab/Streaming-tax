"""Run identity and the per-job meta.json.

A job's identity is a hash of everything that can change what the model generates: the arm (model, mode, plugin
variant), the checkpoint(s) at their pinned revisions, the adapter and its protocol version, the decoding block,
the plugin recipe and artefact hashes, the dataset, the resolved settings, and any sample subset or prompt
overrides. `run` refuses to append to an output file whose meta.json identity differs (use --fresh to archive it).

Deliberately NOT part of the identity: the python env, GPU ids, shards, threads, judge settings, paths and the code
version. Those are recorded per run segment in meta.json instead (package version, git commit and dirty flag when
available, python and key package versions, GPU names, load time, prepared-samples and upstream-manifest hashes).
Host names are never recorded.

Image benchmarks never read the video/stream settings, so their identities use the settings with video/stream
pinned to the defaults (config.settings.resolve_settings does this): a pair's video protocol or a video override
leaves image identities unchanged, as in the original runs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
from typing import Any, Iterable

from .io import fingerprint, read_json, write_json
from .protocol import Arm
from .schema import utc_now

SCHEMA = 1
PACKAGES = ("torch", "transformers", "vllm", "flash-attn", "flashinfer-python", "qwen-vl-utils", "torchcodec",
            "decord", "av", "numpy", "accelerate", "peft")


def canonical_json(value) -> str:
    """The serialisation every identity hash is taken over (sorted keys, unescaped unicode, default separators)."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def run_payload(arm: Arm, model: dict, dataset: str, settings: dict, limit: int | None = None,
                subset: Iterable[str] | None = None, prompt_overrides: Any = None) -> dict:
    """The identity payload of one (arm, dataset) job.

    model: the model block, as returned by config.schema.ModelSpec.identity_block() (checkpoint refs, adapter,
    adapter protocol version, decoding, plugin). settings: config.settings.resolve_settings() for this job.
    subset: the sample ids the folder is restricted to (None = all). prompt_overrides: the prompt edits applied in
    this folder (None = none)."""
    return {"schema": SCHEMA,
            "job": {"model": arm.model, "mode": arm.mode.value, "variant": arm.variant},
            "model": model,
            "dataset": dataset,
            "settings": settings,
            "limit": limit,
            "subset": fingerprint(sorted(subset)) if subset is not None else None,
            "prompt_overrides": fingerprint(prompt_overrides) if prompt_overrides is not None else None}


def run_identity(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()[:16]


class IdentityMismatch(ValueError):
    """The outputs on disk were generated under another identity (settings, checkpoint, adapter, ... changed)."""


def new_meta(payload: dict) -> dict:
    return {"schema": SCHEMA, "identity": run_identity(payload), "payload": payload, "created": utc_now(),
            "segments": []}


def read_meta(path: Path) -> dict | None:
    path = Path(path)
    return read_json(path) if path.exists() else None


def write_meta(path: Path, meta: dict) -> None:
    write_json(path, meta)


def check_identity(meta: dict | None, payload: dict, where: str = "") -> None:
    """Raises IdentityMismatch when an existing meta.json was written for a different payload."""
    if meta is None:
        return
    want = run_identity(payload)
    if meta.get("identity") != want:
        raise IdentityMismatch(f"{where or 'job'}: stored identity {meta.get('identity')} != {want}; the settings, "
                               "checkpoint, adapter or plugin changed (use --fresh to archive the old outputs)")


def add_segment(path: Path, segment: dict) -> dict:
    """Appends a run segment to meta.json (callers serialise concurrent shards with core.io.file_lock)."""
    meta = read_meta(path)
    if meta is None:
        raise FileNotFoundError(f"{path}: no meta.json to add a run segment to")
    meta.setdefault("segments", []).append(segment)
    write_meta(path, meta)
    return meta


def runtime_info(packages: Iterable[str] = PACKAGES) -> dict:
    """Python and key package versions plus GPU names (when torch is importable). No host names."""
    from importlib import metadata
    from .. import __version__
    info: dict[str, Any] = {"streamreplace": __version__, "python": platform.python_version(), "packages": {}}
    for name in packages:
        try:
            info["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch
        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return info


def code_version(repo: Path | None = None) -> dict:
    """Git commit/branch of the checkout this code runs from, read straight from .git (git is often not on PATH;
    works for worktrees and packed refs). `dirty` (uncommitted tracked changes) needs the git binary, else None."""
    import shutil
    import subprocess
    repo = Path(repo or Path(__file__).resolve().parents[3])
    dot = repo / ".git"
    if dot.is_file():                                     # worktree: "gitdir: <main>/.git/worktrees/<name>"
        gitdir = Path(dot.read_text().split("gitdir:", 1)[1].strip())
        gitdir = gitdir if gitdir.is_absolute() else repo / gitdir
    elif dot.is_dir():
        gitdir = dot
    else:
        return {"commit": None, "branch": None, "dirty": None}
    common = gitdir / (gitdir / "commondir").read_text().strip() if (gitdir / "commondir").exists() else gitdir
    head, commit, branch = (gitdir / "HEAD").read_text().strip(), None, None
    if head.startswith("ref:"):
        ref = head[4:].strip()
        branch = ref.removeprefix("refs/heads/")
        for base in (gitdir, common):
            if (base / ref).is_file():
                commit = (base / ref).read_text().strip()
                break
        else:
            packed = common / "packed-refs"
            for line in packed.read_text().splitlines() if packed.exists() else []:
                if line.endswith(" " + ref):
                    commit = line.split()[0]
    else:
        commit = head                                     # detached HEAD
    dirty, git = None, shutil.which("git")
    if git:
        try:
            status = subprocess.run([git, "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, timeout=30)
            dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            pass
    return {"commit": commit, "branch": branch, "dirty": dirty}


def run_segment(**extra) -> dict:
    """A meta.json run segment: start time, runtime_info(), code_version(), plus caller fields such as
    load_seconds, samples_sha256 (prepared samples.jsonl) and upstream_manifest_sha256."""
    return {"started": utc_now(), **runtime_info(), "code": code_version(), **extra}
