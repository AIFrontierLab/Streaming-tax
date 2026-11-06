"""A temporary streamreplace project for the end-to-end tests: configs, a workspace (STREAMREPLACE_HOME), the demo
benchmark prepared, and helpers to plan, run and inspect jobs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import yaml

from streamreplace import paths as paths_module
from streamreplace.config.load import load_config
from streamreplace.core.io import read_jsonl, read_latest, write_jsonl
from streamreplace.core.schema import score_row

DEFAULTS = {"generation": {"max_new_tokens": 16}, "image": {"max_pixels": 1003520},
            "video": {"nframes": 8, "max_pixels": 200704}, "stream": {"fps": 1}}
MODELS = {
    "parent": {"role": "parent", "checkpoint": "test/parent@0", "offline": {"adapter": "mock", "env": "local"},
               "options": {"accuracy": 1.0}},
    "cand": {"role": "child", "checkpoint": "test/cand@0", "offline": {"adapter": "mock", "env": "local"},
             "stream": {"adapter": "mock", "env": "local"},
             "options": {"accuracy": 0.5, "seed": 3, "fail": ["yn-2"]}},
}
PAIRS = {"p": {"parent": "parent", "children": ["cand"]}}
BENCHMARKS = {"demo": {"media": "video"}}
JOBS = ["demo/parent.offline", "demo/cand.offline", "demo/cand.stream"]


def dump(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def write_configs(folder: Path, models=None, pairs=None, benchmarks=None, defaults=None, sections=None) -> Path:
    dump(folder / "defaults.yaml", {"defaults": defaults or DEFAULTS, "run": {"gpus": [0, 1]}})
    dump(folder / "models.yaml", {"models": models or MODELS})
    dump(folder / "pairs.yaml", {"pairs": pairs or PAIRS})
    dump(folder / "benchmarks.yaml", {"benchmarks": benchmarks or BENCHMARKS, "suites": {}})
    for name, data in (sections or {}).items():
        dump(folder / f"{name}.yaml", data)
    return folder


@dataclass
class Project:
    root: Path
    configs: Path
    home: Path

    @property
    def paths(self):
        return paths_module.resolve(home=self.home)

    @property
    def outputs(self) -> Path:
        return self.paths.outputs

    def config(self, extra=()):
        return load_config(self.configs, extra)

    def plan(self, extra=(), **kwargs):
        from streamreplace.run.plan import plan
        kwargs.setdefault("pairs", "p")
        kwargs.setdefault("benchmarks", "demo")
        return plan(self.config(extra), self.paths, **kwargs)

    def run(self, jobs, gpus=(0, 1), **kwargs):
        from streamreplace.run import dispatch
        return dispatch.run(jobs, list(gpus), paths=self.paths, log=lambda line: None, **kwargs)

    def cli(self, *argv) -> int:
        from streamreplace.cli.main import main
        try:
            return main(["--config-dir", str(self.configs), "--home", str(self.home), *argv])
        finally:
            paths_module.set_current(self.paths)

    def rows(self, job: str, outputs: Path | None = None) -> list[dict]:
        dataset, name = job.split("/")
        return read_jsonl((outputs or self.outputs) / dataset / f"{name}.jsonl")


def make_project(root: Path, models=None, pairs=None, benchmarks=None, defaults=None, sections=None) -> Project:
    """Configs under root/configs, workspace under root/home, every configured benchmark prepared."""
    from streamreplace.data.prepare import prepare_dataset
    project = Project(root, write_configs(root / "configs", deepcopy(models), pairs, benchmarks, defaults, sections),
                      root / "home")
    paths_module.set_current(project.paths)
    for name in benchmarks or BENCHMARKS:
        prepare_dataset(name, paths=project.paths)
    return project


def run_all(project: Project, fresh: bool = False):
    jobs = project.plan()
    return jobs, project.run(jobs, fresh=fresh)


def rows_of(project: Project) -> dict:
    """{file name: sorted (id, raw, error)} of every generation file of the demo benchmark."""
    folder = project.outputs / "demo"
    return {p.name: sorted((r["id"], r["raw"], r.get("error")) for r in read_jsonl(p))
            for p in sorted(folder.glob("*.jsonl")) if not p.name.endswith(".scores.jsonl")}


def score_rule_layer(project: Project, dataset: str, job: str) -> Path:
    """Minimal rule-layer scoring of one job with the benchmark's own parse/correct (no judge): enough for the
    analysis to count rows. Errored rows get no score row (the analysis counts them on their own)."""
    from streamreplace.benchmarks import registry
    from streamreplace.data.samples import load_samples
    bench = registry.get(dataset)
    samples = {s.id: s for s in load_samples(dataset, paths=project.paths)}
    scores = []
    for sample_id, row in read_latest(project.outputs / dataset / f"{job}.jsonl").items():
        if row.get("error") or sample_id not in samples:
            continue
        sample = samples[sample_id]
        pred, status = bench.parse(bench.view(sample), row["raw"])
        scores.append(score_row(sample_id, pred, status, bool(bench.correct(sample, pred))))
    path = project.outputs / dataset / f"{job}.scores.jsonl"
    write_jsonl(path, scores)
    return path
