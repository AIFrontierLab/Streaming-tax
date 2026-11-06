"""The interpreter and environment variables of a model worker process (conditions P, C and S alike).

Each model runs in its own Python environment, `<envs>/<env>/bin/python` (env "local" = the interpreter running this
code: the tests' mock model). A worker gets the variables the paper's runs used:

    CUDA_VISIBLE_DEVICES      its one GPU slot
    PYTHONPATH                this checkout's src/ only, so the model env imports exactly this package
    PATH                      the env's bin/ first (tools some stacks call: ninja, vllm)
    HF_HUB_OFFLINE=1          checkpoints are read from local folders, never fetched mid-run
    TOKENIZERS_PARALLELISM    false
    OMP / MKL / OPENBLAS_NUM_THREADS   CPU threads per worker = cores // GPU slots: torch and OpenMP otherwise take
                              every core, and several workers oversubscribing the cores slowed decoding a lot
    FLASHINFER_WORKSPACE_BASE <cache>/flashinfer (kernels JIT-compiled for the env)
    LD_LIBRARY_PATH           <envs>/ffmpeg/lib first, for models marked `ffmpeg: true` (torchcodec and qwen-vl-utils
                              need the FFmpeg 6 shared libraries)
    STREAMREPLACE_*           the resolved roots (paths.py), so the worker reads the same data, cache and upstream

Models with `options.deterministic: true` (the fine-tuning experiment's evaluations) also get
CUBLAS_WORKSPACE_CONFIG=:4096:8 and FLASH_ATTENTION_DETERMINISTIC=1, and the worker turns on torch's deterministic
algorithms before loading (some vision towers otherwise differed from run to run through cuBLAS)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from ..paths import Paths

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
DETERMINISTIC = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "FLASH_ATTENTION_DETERMINISTIC": "1"}


def src_dir() -> Path:
    """The `src` folder this package is imported from (what a worker's PYTHONPATH points at)."""
    return Path(__file__).resolve().parents[2]


def python_for(paths: Paths, env: str) -> Path:
    return Path(paths.env_python(env))


def ffmpeg_lib(paths: Paths) -> Path:
    return Path(paths.envs) / "ffmpeg" / "lib"


def default_threads(slots: int) -> int:
    """CPU threads per worker: cores // GPU slots (at least 1)."""
    return max(1, (os.cpu_count() or 1) // max(1, int(slots)))


def worker_vars(paths: Paths, env: str, gpu, threads: int, ffmpeg: bool = False, deterministic: bool = False,
                base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The variables a worker process gets on top of the inherited environment `base` (default os.environ)."""
    base = os.environ if base is None else base
    python = python_for(paths, env)
    out = dict(paths.environ())
    out.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(src_dir()),
        "PATH": f"{python.parent}{os.pathsep}{base.get('PATH', '')}",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_OFFLINE": "1",
        "FLASHINFER_WORKSPACE_BASE": str(Path(paths.cache) / "flashinfer"),
        **{name: str(threads) for name in THREAD_VARS},
    })
    lib = ffmpeg_lib(paths)
    if ffmpeg and lib.is_dir():
        current = base.get("LD_LIBRARY_PATH", "")
        out["LD_LIBRARY_PATH"] = f"{lib}{os.pathsep}{current}" if current else str(lib)
    if deterministic:
        out.update(DETERMINISTIC)
    return out


def worker_env(paths: Paths, env: str, gpu, threads: int, ffmpeg: bool = False, deterministic: bool = False,
               base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The complete environment of a worker process: `base` (default os.environ) plus worker_vars()."""
    base = os.environ if base is None else base
    return {**base, **worker_vars(paths, env, gpu, threads, ffmpeg, deterministic, base)}
