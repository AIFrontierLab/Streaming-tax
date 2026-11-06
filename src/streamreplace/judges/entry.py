"""The judge process: `python -m streamreplace.judges.entry <job.json>`, run inside the judge's own Python
environment (configs/judges.yaml `env`; the judge env holds vLLM, transformers, spaCy and datasketch).

`launch()` (scoring side) writes the pending requests and a job file under `<outputs>/logs/judges/`, then starts this
module with the judge env's interpreter, the GPUs in CUDA_VISIBLE_DEVICES, the streamreplace roots in the environment
(paths.Paths.environ) and offline Hugging Face access. `main()` (judge side) skips requests already in the cache and
hands the rest to the judge module's `serve(requests, cache, spec)`, which appends one cache line per answer.

Job file: {"judge": name, "requests": <requests.jsonl>, "cache": <cache.jsonl>, "spec": {...}} where spec holds what
the judge process needs (resolved checkpoint folders and the judge's options; for RJScore also its self-test, which
runs before any request is judged)."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping

# judge name -> module (under streamreplace.judges) defining serve(requests, cache, spec)
SERVERS = {"extract": "extraction", "harmbench_cls": "harmbench_cls", "rjscore": "rjscore", "lingo": "lingo"}
SRC = Path(__file__).resolve().parents[2]          # the folder holding the streamreplace package


def judge_environ(paths, gpus: list[int], python: Path, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment of a judge process: the GPUs, offline Hugging Face, this checkout on PYTHONPATH, the env's bin
    first on PATH, the streamreplace roots, and vLLM's greedy sampler without FlashInfer kernels (the paper's runs
    set VLLM_USE_FLASHINFER_SAMPLER=0 for every judge)."""
    env = {**os.environ, **paths.environ(),
           "CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "HF_HUB_OFFLINE": "1",
           "PYTHONPATH": os.pathsep.join([str(SRC), *filter(None, [os.environ.get("PYTHONPATH")])]),
           "PATH": f"{python.parent}{os.pathsep}{os.environ.get('PATH', '')}",
           "FLASHINFER_WORKSPACE_BASE": str(paths.cache / "flashinfer"),
           "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def launch(judge: str, requests: list[dict], cache_path: Path, spec: dict, gpus: list[int], *, env: str, paths,
           extra_env: Mapping[str, str] | None = None, tag: str = "") -> None:
    """Writes the job and runs the judge process to completion (raises CalledProcessError when it fails; answers
    written before a failure stay in the cache)."""
    from ..core.io import write_json, write_jsonl
    python = paths.env_python(env)
    if not python.exists():
        raise ValueError(f"judge {judge!r}: environment {env!r} is missing ({python}); build it first "
                         f"(see envs/README.md)")
    if not gpus:
        raise ValueError(f"judge {judge!r}: no GPUs given (score --judge-gpus)")
    folder = paths.outputs / "logs" / "judges"
    stem = f"{judge}.{tag}" if tag else judge
    requests_path, job_path, log_path = (folder / f"{stem}.requests.jsonl", folder / f"{stem}.job.json",
                                         folder / f"{stem}.log")
    write_jsonl(requests_path, requests)
    write_json(job_path, {"judge": judge, "requests": str(requests_path), "cache": str(Path(cache_path).resolve()),
                          "spec": spec})
    print(f"judge {judge}: {len(requests)} requests on GPUs {gpus} (log: {log_path})", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        subprocess.run([str(python), "-m", "streamreplace.judges.entry", str(job_path)], stdout=log,
                       stderr=subprocess.STDOUT, env=judge_environ(paths, gpus, python, extra_env), check=True)


def main(argv: list[str] | None = None) -> int:
    from ..core.io import read_json, read_jsonl
    from .cache import JudgeCache
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m streamreplace.judges.entry <job.json>", file=sys.stderr)
        return 2
    job = read_json(Path(argv[0]))
    spec = job.get("spec") or {}
    cache = JudgeCache(Path(job["cache"]))
    pending = list({r["key"]: r for r in read_jsonl(Path(job["requests"])) if r["key"] not in cache}.values())
    print(f"judge {job['judge']}: {len(pending)} new requests", flush=True)
    if not pending and not spec.get("selftest"):          # a self-test runs even without new requests
        return 0
    module = importlib.import_module(f"{__package__}.{SERVERS[job['judge']]}")
    module.serve(pending, cache, spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
