"""The diagnostic and supplementary experiments of the paper (Sec. 5, Supp. C and D), declared in
configs/experiments.yaml and run with `streamreplace experiment list | run | report` (cli/cmd_experiment.py).

An experiment is a frozen item list and a set of *arms*; every arm is an ordinary generation job, so it is planned,
resumed, scored and counted exactly like a P/C/S job (run/plan.py, scoring/, analysis/records.py):

  items      {benchmark: [ids]} (a file under paper/frozen/experiments/), or whole benchmarks
  children   the children the experiment covers; {child} and {parent} in an arm's job name are filled in per child
  arms       label -> job. The reference arms P, C and S are always available and are read from the formal outputs
             folder restricted to the items (the paper reuses its formal rows as references). Every other arm runs
             into the experiment's own folder <outputs>/experiments/<name>/[<folder>/]:
               job     "<model>.<mode>" or "<child>@<plugin arm>.stream" (or a mapping child -> job)
               set     settings overrides of this arm (as --set), the same for every child or keyed by child
               folder  a sub-folder, needed when two arms run the same job with different settings
  contrasts  "B - A" pairs of arm labels: paired differences with the analysis engine of the main results
             (analysis/contrasts.py: cluster bootstrap over the benchmark's units, the same metric definitions)

Arms that are not one of the eleven paper models are *derived model ids* (the `models` block of the file):
  from        the configs/models.yaml id it is derived from (its pair's video protocol is carried in the settings)
  settings    an extra settings layer, e.g. {video: {last_seconds: 30}} (the window control C_W)
  options     adapter options
  checkpoint  another model id whose weights are loaded instead: the parent in the child's streaming interface (P_S)
  stream      another stream adapter (AURA's transformers rebuild for its memory controls)

Three kinds of report (experiment `kind`):
  qa      report.py: benchmark-scored arms, paired contrasts (history, response policy, question timing, parent
          controls, memory budgets, streaming benchmarks)
  forms   forms.py: replies labelled by the response-form judge, rates of refusal / help / non-answers
  probe   probe.py: hidden-state harm scores and steering (Sec. 5.3)"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from ..config.schema import Config
    from ..paths import Paths

SECTION = "experiments"
REFERENCES = {"P": "{parent}.offline", "C": "{child}.offline", "S": "{child}.stream"}
MODEL_KEYS = {"from", "settings", "options", "checkpoint", "stream"}
EXPERIMENT_KEYS = {"title", "paper", "kind", "items", "benchmarks", "children", "arms", "contrasts", "metrics",
                   "report", "options"}
ARM_KEYS = {"job", "set", "folder"}


@dataclass(frozen=True)
class Arm:
    """One arm of an experiment: `job` with {child} / {parent} placeholders; reference arms have formal=True."""
    label: str
    job: str | dict                 # a job name, or {child: job name} when the arm is another job for some children
    sets: dict
    folder: str = ""
    formal: bool = False

    def job_name(self, child: str, parent: str) -> str | None:
        """The arm's job for one child (None when a per-child mapping has no entry for it)."""
        template = self.job.get(child) if isinstance(self.job, dict) else self.job
        return None if template is None else template.format(child=child, parent=parent)

    def settings(self, child: str) -> dict:
        """The arm's --set overrides for one child: a mapping keyed by child ids, or one layer for every child."""
        if self.sets and all(isinstance(v, dict) and k not in ("generation", "image", "video", "stream", "system")
                             for k, v in self.sets.items()):
            return deepcopy(self.sets.get(child) or {})
        return deepcopy(self.sets)


@dataclass(frozen=True)
class Experiment:
    name: str
    spec: dict
    root: Path                      # <outputs>/experiments/<name>
    project: Path                   # the source checkout (item lists are relative to it)

    @property
    def kind(self) -> str:
        return self.spec.get("kind", "qa")

    @property
    def children(self) -> list[str]:
        return list(self.spec.get("children") or [])

    @property
    def contrasts(self) -> list[tuple[str, str]]:
        """(a, b) for every "B - A" entry: the reported difference is b - a."""
        out = []
        for text in self.spec.get("contrasts") or []:
            b, _, a = str(text).partition(" - ")
            if not a.strip():
                raise ValueError(f"experiments.{self.name}.contrasts: {text!r} is not 'B - A'")
            out.append((a.strip(), b.strip()))
        return out

    @property
    def arms(self) -> dict[str, Arm]:
        arms = {label: Arm(label, job, {}, formal=True) for label, job in REFERENCES.items()}
        for label, value in (self.spec.get("arms") or {}).items():
            value = {"job": value} if isinstance(value, str) else dict(value)
            unknown = sorted(set(value) - ARM_KEYS)
            if unknown or "job" not in value:
                raise ValueError(f"experiments.{self.name}.arms.{label}: needs `job`, allows {sorted(ARM_KEYS)}")
            arms[label] = Arm(label, value["job"], dict(value.get("set") or {}), str(value.get("folder") or ""))
        return arms

    def folder(self, arm: Arm) -> Path:
        return self.root / arm.folder if arm.folder else self.root

    def items(self, config: Config | None = None) -> dict[str, list[str]] | None:
        """{benchmark: [ids]} of the frozen item list; None when the experiment runs whole benchmarks."""
        ref = self.spec.get("items")
        if ref is None:
            return None
        path = Path(ref) if Path(ref).is_absolute() else self.project / ref
        data = json.loads(path.read_text(encoding="utf-8"))
        data = data.get("items", data) if isinstance(data.get("items"), dict) else data
        return {d: [str(i) for i in ids] for d, ids in data.items() if isinstance(ids, list)}

    def benchmarks(self) -> list[str]:
        items = self.items()
        return list(self.spec.get("benchmarks") or (items or {}))


def section(config: Config) -> dict:
    data = config.section(SECTION)
    if not data:
        raise ValueError("configs/experiments.yaml is missing or empty")
    return data


def names(config: Config) -> list[str]:
    return list(section(config).get("experiments") or {})


def load(config: Config, paths: Paths, name: str) -> Experiment:
    from ..paths import project_root
    specs = section(config).get("experiments") or {}
    if name not in specs:
        raise ValueError(f"unknown experiment {name!r}; configured: {', '.join(specs)}")
    spec = dict(specs[name] or {})
    unknown = sorted(set(spec) - EXPERIMENT_KEYS)
    if unknown:
        raise ValueError(f"experiments.{name}: unknown keys {unknown} (allowed: {', '.join(sorted(EXPERIMENT_KEYS))})")
    return Experiment(name, spec, Path(paths.outputs) / "experiments" / name, project_root())


# ---------- derived model ids ----------

def derived_entry(config: Config, model: str, derived: dict) -> dict:
    """The configs/models.yaml entry of a derived id (see the module docstring)."""
    from ..config.settings import merge
    from ..run.plan import spec_dict
    unknown = sorted(set(derived) - MODEL_KEYS)
    if unknown or "from" not in derived:
        raise ValueError(f"experiments models.{model}: needs `from`, allows {sorted(MODEL_KEYS)}")
    source = config.model(derived["from"])
    pair = config.pair_of(source.id)
    entry = spec_dict(source)
    video = {"video": deepcopy(pair.video)} if pair is not None else {}
    entry["settings"] = merge(video, entry.get("settings") or {}, derived.get("settings") or {})
    if derived.get("options"):
        entry["options"] = merge(entry.get("options") or {}, derived["options"])
    if derived.get("checkpoint"):                  # another model's weights in this model's adapters
        entry["checkpoint"] = str(config.model(derived["checkpoint"]).checkpoint)
    if derived.get("stream"):
        if "stream" not in entry:
            raise ValueError(f"experiments models.{model}: {source.id} has no stream adapter to replace")
        entry["stream"] = {**entry["stream"], "adapter": str(derived["stream"])}
    return entry


def with_models(config: Config) -> Config:
    """The loaded configuration with the derived model ids of configs/experiments.yaml added (no pair membership:
    each id carries its pair's video protocol in its own settings layer)."""
    from ..config.schema import ModelSpec
    added = {mid: ModelSpec.parse(mid, derived_entry(config, mid, dict(d or {})))
             for mid, d in (section(config).get("models") or {}).items()}
    clash = sorted(set(added) & set(config.models))
    if clash:
        raise ValueError(f"configs/experiments.yaml models: {clash[:3]} are already model ids")
    return replace(config, models={**config.models, **added})


def parent_of(config: Config, child: str) -> str:
    pair = config.pair_of(child)
    if pair is None or child not in pair.children:
        raise ValueError(f"{child} is not a child of any pair (configs/pairs.yaml)")
    return pair.parent


# ---------- planning ----------

def write_subset(folder: Path, items: dict[str, list[str]] | None) -> None:
    """<folder>/subset.json, the item list every job of the folder is restricted to (written once)."""
    if items is None:
        return
    path = Path(folder) / "subset.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != items:
            raise ValueError(f"{path} holds another item list: use another --outputs root for a changed experiment")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")


def write_edits(folder: Path, items: dict[str, list[str]] | None, edits: list) -> None:
    """<folder>/prompt_overrides.jsonl: the same [old, new] edits on every item's prompt (data/prompts.py), e.g. the
    framing experiment's bare request, whose prompt drops the sentence "These are frames of a video."."""
    from ..data.prompts import OVERRIDES_FILE
    if items is None:
        raise ValueError("prompt edits need an item list")
    path = Path(folder) / OVERRIDES_FILE
    text = "".join(json.dumps({"dataset": d, "id": i, "edits": edits}, ensure_ascii=False) + "\n"
                   for d, ids in items.items() for i in ids)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"{path} holds other prompt edits")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def plan(exp: Experiment, config: Config, paths: Paths, children: Iterable[str] = (), arms: Iterable[str] = (),
         benchmarks: Iterable[str] = ()) -> list:
    """The generation jobs of the experiment's own arms (reference arms are the formal jobs), for the chosen
    children, arm labels and benchmarks (default: all)."""
    from ..core.protocol import Arm as JobArm
    from ..core.protocol import Mode, Role
    from ..run.plan import plan as plan_jobs
    full = with_models(config)
    items = exp.items()
    chosen = [a for a in exp.arms.values() if not a.formal and (not list(arms) or a.label in set(arms))]
    datasets = [b for b in exp.benchmarks() if not list(benchmarks) or b in set(benchmarks)]
    jobs, seen = [], set()
    edits = (exp.spec.get("options") or {}).get("edits") or {}
    for arm in chosen:
        write_subset(exp.folder(arm), items)
        if arm.folder in edits:
            write_edits(exp.folder(arm), items, edits[arm.folder])
        for child in (list(children) or exp.children):
            name = arm.job_name(child, parent_of(config, child))
            key = (str(exp.folder(arm)), name)
            if name is None or key in seen:           # no such arm for this child; a parent arm shared by two children
                continue
            seen.add(key)
            job = JobArm.parse(name)
            model = f"{job.model}@{job.variant}" if job.variant else job.model
            role = full.model(job.model).role
            protocol = "S" if job.mode is Mode.STREAM else ("P" if role is Role.PARENT else "C")
            jobs += plan_jobs(full, paths, benchmarks=datasets, models=[model],
                              protocols=[] if job.variant else [protocol], sets=arm.settings(child) or None,
                              outputs=exp.folder(arm))
    return jobs
