"""What the fine-tuning experiment runs, and how one piece of it is launched: one training unit per process, the
evaluations through the ordinary run pipeline, and the whole experiment printed as shell commands for any scheduler.

Evaluation model ids. Every pair member gets, per dataset, a zero-shot id `<model>@<dataset>-0` and one fine-tuned id
per training unit `<model>@<dataset>-<label>` (model_entries()). Each is an ordinary configs/models.yaml entry derived
from the member's own entry (models outside configs/models.yaml are derived by the specification's `models` block,
e.g. MMDuet with its LoRA merged), offline only, with
  settings  the member's pair video protocol (configs/pairs.yaml) plus the specification's `overrides` settings: the
            ids are not pair members, so the protocol is carried in the model's own settings layer, which is where
            the original runs had it
  options   the `overrides` options (LLaVA-OneVision / MMDuet in bf16), `deterministic: true` (deterministic kernels
            for training and evaluation alike), and for a fine-tuned id `lora`: its adapter folder. The run identity
            hashes the adapter's weights, never the path
so zero-shot and fine-tuned ids differ only in the adapter. The run worker hands `options.lora` to the offline
adapter's load_lora(), which attaches it unmerged.

Training. `streamreplace finetune train` writes the unit's payload (the zero-shot id's spec and resolved settings, the
LoRA and optimiser configuration) and runs `python -m streamreplace.generalization.finetune.train` in the model's
environment on one GPU with the worker environment of the run pipeline plus the deterministic-kernel variables. A unit
resumes from its checkpoint when restarted; a second process on the same unit exits at once (lock).

There is no scheduler here: `streamreplace finetune commands` prints every step (training units longest first, then
evaluations, scoring and the report) for a shell loop or a batch system."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import os
from pathlib import Path
import shlex
import shutil
import subprocess
from typing import TYPE_CHECKING, Iterable

from ...config.settings import merge
from ...core.io import read_json, read_jsonl, write_json
from . import Experiment, find_unit, label, model_id, seed_units, unit_dir, units
from .lora import family_of

if TYPE_CHECKING:
    from ...config.schema import Config
    from ...paths import Paths

PROTOCOLS = ("P", "C")          # the experiment is offline only: parents run as P, children as C
TRAIN_MODULE = "streamreplace.generalization.finetune.train"


# ---------- model ids for the run pipeline ----------

def source_model(exp: Experiment, model: str) -> tuple[str, dict]:
    """(configs/models.yaml id the member is derived from, the specification's derivation block)."""
    derived = (exp.spec.get("models") or {}).get(model)
    if derived is None:
        return model, {}
    return derived["from"], derived


def member_entry(exp: Experiment, config: Config, model: str) -> dict:
    """The configs/models.yaml entry of a pair member as evaluated in the experiment (see the module docstring)."""
    from ...run.plan import spec_dict
    source, derived = source_model(exp, model)
    spec = config.model(source)
    pair = config.pair_of(source)
    if pair is None:
        raise ValueError(f"{model}: {source} is in no pair of configs/pairs.yaml, so it has no video protocol")
    entry = spec_dict(spec)
    entry.pop("stream", None)
    override = (exp.spec.get("overrides") or {}).get(model) or {}
    entry["settings"] = merge({"video": deepcopy(pair.video)}, entry.get("settings") or {}, override.get("settings") or {})
    entry["options"] = merge(entry.get("options") or {}, derived.get("options") or {}, override.get("options") or {},
                             {"deterministic": True})
    return entry


def model_entries(exp: Experiment, config: Config) -> dict[str, dict]:
    """configs/models.yaml entries of every evaluation id: zero-shot and fine-tuned (formal and seed units)."""
    members = {m: member_entry(exp, config, m) for m in exp.members}
    out = {}
    for model, entry in members.items():
        for dataset in exp.datasets:
            out[model_id(model, dataset, 0)] = deepcopy(entry)
    for unit in units(exp) + seed_units(exp):
        entry = deepcopy(members[unit["model"]])
        entry["options"]["lora"] = str(unit_dir(exp, unit) / "adapter")
        out[model_id(unit["model"], unit["dataset"], unit["budget"], unit["tag"])] = entry
    return out


def experiment_config(exp: Experiment, config: Config) -> Config:
    """The loaded configuration with the experiment's evaluation ids added (no pair membership: settings come from
    each id's own settings layer)."""
    from ...config.schema import ModelSpec
    added = {mid: ModelSpec.parse(mid, entry) for mid, entry in model_entries(exp, config).items()}
    clash = sorted(set(added) & set(config.models))
    if clash:
        raise ValueError(f"configs/models.yaml already defines {clash[:3]}")
    return replace(config, models={**config.models, **added})


def write_models_file(exp: Experiment, config: Config, path: Path) -> Path:
    """The evaluation ids as a models file for the core commands (`streamreplace --config FILE ...`)."""
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# Generated by `streamreplace finetune models`: the fine-tuning experiment's evaluation ids.\n"
                    + yaml.safe_dump({"models": model_entries(exp, config)}, sort_keys=False), encoding="utf-8")
    return path


# ---------- units ----------

def trained(exp: Experiment, unit: dict) -> bool:
    return (unit_dir(exp, unit) / "DONE.json").is_file()


def unit_state(exp: Experiment, unit: dict) -> str:
    folder = unit_dir(exp, unit)
    if (folder / "DONE.json").is_file():
        return f"trained (adapter sha256 {read_json(folder / 'DONE.json')['adapter_sha256'][:12]})"
    steps = read_jsonl(folder / "loss.jsonl")
    if steps:
        total = read_json(folder / "unit.json").get("steps") if (folder / "unit.json").is_file() else None
        return f"running or stopped at step {steps[-1]['step']}/{total or '?'}"
    return "not started"


def select_units(exp: Experiment, datasets: Iterable[str] = (), models: Iterable[str] = (),
                 seeds: bool = False) -> list[dict]:
    datasets, models = set(datasets), set(models)
    chosen = seed_units(exp) if seeds else units(exp)
    return [u for u in chosen if (not datasets or u["dataset"] in datasets) and (not models or u["model"] in models)]


def longest_first(exp: Experiment, chosen: list[dict]) -> list[dict]:
    """Training order for a GPU pool: most training samples first."""
    return sorted(chosen, key=lambda u: u["budget"] * u["epochs"], reverse=True)


# ---------- training ----------

def training_payload(exp: Experiment, config: Config, paths: Paths, unit: dict, directory: Path | None = None,
                     max_steps: int | None = None) -> dict:
    zs = model_id(unit["model"], unit["dataset"], 0)
    ft = experiment_config(exp, config)
    spec = ft.model(zs)
    from ...run.plan import spec_dict
    adapter = spec.adapter("offline").adapter
    return {"version": 1, "unit": unit, "model": {"id": zs, "spec": spec_dict(spec)}, "adapter": adapter,
            "family": family_of(adapter), "settings": ft.settings(unit["dataset"], zs, "offline"),
            "lora": exp.lora, "train": exp.train, "selection": str(exp.root),
            "dir": str(directory or unit_dir(exp, unit)), "max_steps": max_steps,
            "paths": {"checkpoint_dirs": {k: str(v) for k, v in paths.checkpoint_dirs.items()},
                      "dataset_dirs": {k: str(v) for k, v in paths.dataset_dirs.items()}}}


def archive(exp: Experiment, unit: dict) -> Path | None:
    """--fresh: moves the unit's folder to <adapters>/_old/<time>/<dataset>/<model>/<label>."""
    directory = unit_dir(exp, unit)
    if not directory.exists():
        return None
    target = directory.parents[2] / "_old" / datetime.now().strftime("%Y%m%d-%H%M%S") / directory.relative_to(directory.parents[2])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), target)
    return target


def train_command(exp: Experiment, config: Config, paths: Paths, unit: dict, gpu, threads: int,
                  directory: Path | None = None, max_steps: int | None = None) -> tuple[list[str], dict, Path]:
    """(command, environment variables, payload file) of one training process."""
    from ...run.envs import python_for, worker_vars
    payload = training_payload(exp, config, paths, unit, directory, max_steps)
    folder = Path(payload["dir"])
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "payload.json"
    write_json(path, payload)
    spec = experiment_config(exp, config).model(payload["model"]["id"]).adapter("offline")
    variables = worker_vars(paths, spec.env, gpu, threads, spec.ffmpeg, deterministic=True)
    return [str(python_for(paths, spec.env)), "-m", TRAIN_MODULE, str(path)], variables, folder


def run_training(exp: Experiment, config: Config, paths: Paths, model: str, dataset: str, budget: int,
                 tag: str | None = None, gpu=0, threads: int | None = None, fresh: bool = False,
                 directory: Path | None = None, max_steps: int | None = None, launcher: str = "local") -> int:
    """Trains one unit in its model's environment (resumes from its checkpoint); launcher "print" only prints the
    command."""
    from ...paths import project_root
    unit = find_unit(exp, model, dataset, budget, tag)
    if fresh and directory is None:
        moved = archive(exp, unit)
        if moved:
            print(f"archived {moved}")
    threads = threads or max(1, (os.cpu_count() or 8) // 8)
    command, variables, folder = train_command(exp, config, paths, unit, gpu, threads, directory, max_steps)
    if launcher == "print":
        print(" ".join([f"cd {shlex.quote(str(project_root()))} &&", "env",
                        *(f"{k}={shlex.quote(v)}" for k, v in sorted(variables.items())),
                        *(shlex.quote(part) for part in command)]))
        return 0
    python = Path(command[0])
    if not python.exists():
        raise ValueError(f"missing environment {python}; build it with streamreplace env build")
    with (folder / "train.log").open("a", encoding="utf-8") as log:
        log.write(f"\n==== {datetime.now():%Y-%m-%d %H:%M:%S} gpu={gpu}\n")
        log.flush()
        code = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **variables},
                                cwd=str(project_root())).wait()
    print(f"{model} {dataset} {label(budget, tag)}: exit {code} (log: {folder / 'train.log'})")
    return code


# ---------- evaluation ----------

def folder_subsets(exp: Experiment, control: str | None = None, seeds: bool = False,
                   datasets: Iterable[str] | None = None) -> dict[str, list[str]]:
    """The subset.json entries a control or seed folder needs: a control's own questions (data.control_subset) or
    the formal test ids (seed replication)."""
    from .data import control_subset, subset_ids
    if seeds:
        return {d: subset_ids(exp.root, d) for d in (datasets or exp.seeds["datasets"])}
    spec = exp.controls[control]
    return {d: control_subset(exp, d, spec.get("size")) for d in (datasets or spec.get("datasets") or exp.datasets)}


def evaluation_ids(exp: Experiment, dataset: str, models: Iterable[str] = (), budgets: Iterable[int] | None = None,
                   tags: Iterable[str | None] = (None,)) -> tuple[list[str], list[str]]:
    """(ids ready to evaluate, fine-tuned ids whose adapter is not trained yet) for one dataset."""
    ready, waiting = [], []
    budgets = [0, *exp.budgets] if budgets is None else [int(b) for b in budgets]
    for model in (list(models) or exp.members):
        for budget in budgets:
            for tag in tags:
                if budget == 0:
                    if tag is None:
                        ready.append(model_id(model, dataset, 0))
                    continue
                try:
                    unit = find_unit(exp, model, dataset, budget, tag)
                except ValueError:
                    continue
                (ready if trained(exp, unit) else waiting).append(model_id(model, dataset, budget, tag))
    return list(dict.fromkeys(ready)), waiting


def plan_evaluations(exp: Experiment, config: Config, paths: Paths, datasets: Iterable[str] = (),
                     models: Iterable[str] = (), budgets: Iterable[int] | None = None, control: str | None = None,
                     seeds: bool = False, tags: Iterable[str] = ()) -> tuple[list, list[str], Path]:
    """(jobs of the run pipeline, ids skipped because untrained, outputs folder). The formal folder evaluates every
    member zero-shot and after each budget (and extra units with `tags`); a control its own models, budgets and
    --set overrides; the seed folder the seed units (their zero-shot rows are the formal ones)."""
    from ...run.plan import plan
    sets, chosen_tags = None, list(tags) or [None]
    if seeds:
        spec = exp.seeds
        folder, default_models = exp.seeds_folder(), spec["models"]
        default_datasets, default_budgets = spec["datasets"], spec["budgets"]
        chosen_tags = [f"s{k}" for k in spec["seeds"]]
    elif control is not None:
        spec = exp.controls[control]
        folder, default_models = exp.folder(control), spec.get("models") or exp.members
        default_datasets, default_budgets = spec.get("datasets") or exp.datasets, spec.get("budgets")
        sets = spec.get("set")
    else:
        folder, default_models, default_datasets, default_budgets = exp.root, exp.members, exp.datasets, None
    datasets = list(datasets) or list(default_datasets)
    if seeds or control is not None:
        from .data import write_subset
        write_subset(folder, folder_subsets(exp, control, seeds, datasets))
    elif not (folder / "subset.json").is_file():
        raise ValueError(f"{folder}/subset.json is missing; run: streamreplace finetune prepare {' '.join(datasets)}")
    ft = experiment_config(exp, config)
    jobs, waiting = [], []
    for dataset in datasets:
        ready, untrained = evaluation_ids(exp, dataset, list(models) or default_models,
                                          budgets if budgets is not None else default_budgets, chosen_tags)
        waiting += untrained
        if ready:
            jobs += plan(ft, paths, benchmarks=[dataset], models=ready, protocols=PROTOCOLS, sets=sets, outputs=folder)
    return jobs, waiting, folder


# ---------- the whole experiment as commands ----------

def commands(exp: Experiment, prefix: str = "streamreplace", gpus: list[int] | None = None,
             finetune: str | None = None) -> list[str]:
    """Every step of the experiment as shell commands (comments start with #). Training units are spread
    round-robin over `gpus`; each line is independent, so a scheduler may run them in any order within a step.
    prefix: the command with its global options (`streamreplace --config ...`); finetune: the prefix of the
    fine-tuning steps (default: `<prefix> finetune --root <the experiment root>`), so every line works on this
    experiment's root wherever it runs."""
    gpus = list(gpus or [0])
    gpu_text = ",".join(map(str, gpus))
    datasets = " ".join(exp.datasets)
    ft = finetune or f"{prefix} finetune --root {shlex.quote(os.path.abspath(exp.root))}"
    lines = ["# 1. data: prepare the benchmarks, then fix the test subsets and training orders (write-once;",
             "#    --frozen copies the paper's lists from paper/frozen/finetune/ instead of recomputing them)",
             f"{prefix} prepare {' '.join(exp.datasets)}", f"{ft} prepare {datasets}",
             "# 2. zero-shot evaluations",
             *(f"{ft} eval --datasets {d} --budgets 0 --gpus {gpu_text}" for d in exp.datasets),
             "# 3. training units, one GPU each (longest first; a restarted unit resumes from its checkpoint)"]
    for number, unit in enumerate(longest_first(exp, units(exp))):
        tag = f" --tag {unit['tag']}" if unit["tag"] else ""
        lines.append(f"{ft} train --model {unit['model']} --dataset {unit['dataset']} "
                     f"--budget {unit['budget']}{tag} --gpus {gpus[number % len(gpus)]}")
    budgets = ",".join(str(b) for b in exp.budgets)
    lines += ["# 4. evaluations of the trained adapters",
              *(f"{ft} eval --datasets {d} --budgets {budgets} --gpus {gpu_text}" for d in exp.datasets)]
    for extra in exp.spec.get("extra_units") or []:
        lines.append(f"{ft} eval --datasets {extra['dataset']} --budgets {extra['budget']} "
                     f"--tags {extra['tag']} --models {','.join(extra['models'])} --gpus {gpu_text}")
    lines.append("# 5. controls (their own folders and subset.json)")
    lines += [f"{ft} eval --control {name} --gpus {gpu_text}" for name in exp.controls]
    lines += ["# 6. scoring (LingoQA needs the Lingo-Judge: --judge) and the report",
              f"{prefix} --outputs {shlex.quote(os.path.abspath(exp.root))} score --judge --judge-gpus {gpus[0]}",
              *(f"{prefix} --outputs {shlex.quote(os.path.abspath(exp.folder(name)))} score --judge --judge-gpus {gpus[0]}"
                for name in exp.controls),
              f"{ft} report"]
    if exp.seeds:
        lines.append("# 7. training-seed replication (its own folder; zero-shot rows are the formal ones)")
        for number, unit in enumerate(longest_first(exp, seed_units(exp))):
            lines.append(f"{ft} train --model {unit['model']} --dataset {unit['dataset']} "
                         f"--budget {unit['budget']} --tag {unit['tag']} --gpus {gpus[number % len(gpus)]}")
        lines += [f"{ft} eval --seeds --gpus {gpu_text}",
                  f"{prefix} --outputs {shlex.quote(os.path.abspath(exp.seeds_folder()))} score --judge --judge-gpus {gpus[0]}",
                  f"{ft} seeds"]
    return lines


def describe_units(exp: Experiment, chosen: list[dict]) -> list[str]:
    return [f"{u['model']:<18} {u['dataset']:<11} {label(u['budget'], u['tag']):<8} {unit_state(exp, u)}"
            for u in chosen]
