"""Prepared samples of a benchmark, read the same way by every stage (generation for P, C and S, scoring, analysis).

`<data>/<benchmark>/prepared/samples.jsonl` (written by data/prepare.py) holds one Sample per line, in prepared
order, with media paths relative to the benchmark's dataset directory `<data>/<benchmark>` (paths.py can map a
benchmark to another folder). load_samples() returns them with absolute media paths, optionally

- restricted to a subset of ids (an experiment folder's subset.json, below), and then
- limited to the first N ids of the benchmark's smoke order (`run --limit N`). The smoke order is deterministic and
  keeps whole groups together (an MMBench question with all its rotations, a TimeBlind instance, a video), so a
  limited run still has complete units; past the end of that order the remaining samples follow in prepared order.

The prepared order matters: shard k of n generates samples[k::n] (run/worker.py), in that order.

Experiment folders. An outputs folder may hold `subset.json` = {benchmark: [ids]}: every run on that folder sees only
those ids (benchmarks missing from the file are not restricted) and the subset enters the run identity. The main
outputs folder of the paper has none; side experiments (the plugin's item sets, the truncation reruns, the prompt
robustness conditions) use one."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from ..core.io import read_jsonl
from ..core.schema import Sample

if TYPE_CHECKING:
    from ..paths import Paths

FILES = {"test": "samples.jsonl", "train": "train.jsonl"}
SUBSET_FILE = "subset.json"


def _paths(paths: Paths | None) -> Paths:
    if paths is not None:
        return paths
    from .. import paths as paths_module
    return paths_module.current()


def dataset_dir(name: str, paths: Paths | None = None) -> Path:
    """`<data>/<name>` (or the folder the paths configuration maps the benchmark to): raw/, prepared/, media/."""
    return Path(_paths(paths).dataset_dir(name))


def prepared_dir(name: str, paths: Paths | None = None) -> Path:
    return dataset_dir(name, paths) / "prepared"


def samples_file(name: str, split: str = "test", paths: Paths | None = None) -> Path:
    if split not in FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {', '.join(FILES)}")
    return prepared_dir(name, paths) / FILES[split]


def read_subset(outputs: str | Path) -> dict[str, list[str]] | None:
    """The outputs folder's subset.json ({benchmark: [ids]}), or None when the folder has none."""
    path = Path(outputs) / SUBSET_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(isinstance(ids, list) for ids in data.values()):
        raise ValueError(f"{path}: expected a JSON object {{benchmark: [ids]}}")
    return {str(name): [str(i) for i in ids] for name, ids in data.items()}


def smoke_order(name: str, samples: list[Sample], bench=None) -> list[str]:
    """Every sample id in `--limit` order: the benchmark's smoke order first, then the remaining ids in prepared
    order."""
    if bench is None:
        from ..benchmarks import registry
        bench = registry.get(name)
    order = list(dict.fromkeys(bench.smoke(samples)))
    known = {s.id for s in samples}
    order = [i for i in order if i in known]
    chosen = set(order)
    return order + [s.id for s in samples if s.id not in chosen]


def load_samples(name: str, split: str = "test", subset: Iterable[str] | None = None, limit: int | None = None,
                 paths: Paths | None = None, bench=None) -> list[Sample]:
    """The prepared samples of benchmark `name` in prepared order, media paths absolute.

    subset: keep only these ids (unknown ids are ignored). limit: then keep the first `limit` ids of smoke_order().
    Both apply to the test split only. bench: the benchmark module (default: registry.get(name)), needed for limit."""
    path = samples_file(name, split, paths)
    if not path.is_file():
        raise ValueError(f"{name} ({split}) is not prepared: {path} is missing; run: streamreplace prepare {name}")
    base = path.parent.parent                         # media paths are relative to the dataset directory
    samples = [Sample.from_dict(row) for row in read_jsonl(path)]
    if subset is not None and split == "test":
        keep = set(subset)
        samples = [s for s in samples if s.id in keep]
    if limit is not None and split == "test":
        if int(limit) < 1:
            raise ValueError(f"limit must be a positive number of samples, got {limit}")
        keep = set(smoke_order(name, samples, bench)[:int(limit)])
        samples = [s for s in samples if s.id in keep]
    return [replace(s, images=tuple(str(base / p) for p in s.images),
                    video=str(base / s.video) if s.video else None) for s in samples]
