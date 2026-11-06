"""LoRA fine-tuning: do streaming children adapt to a new domain as well as their offline parents? (Supp. Tables
eval_finetuning and eval_finetuning_gains, the fine-tuning column of Table 2; docs/generalization/finetune.md.)

Every pair member (11 checkpoints; MMDuet as its LoRA-merged checkpoint) gets the same LoRA recipe on the same
nested training subsets (1K, 2K, 5K samples) of each dataset (LingoQA driving, EyePCR ophthalmic surgery; OpenUAV-QA
drones, reported but outside the primary endpoint), and is evaluated offline, zero-shot (budget 0) and after each
budget. The primary endpoint per child and dataset is the budget-averaged dGain = (C_b - C_0) - (P_b - P_0).

Modules:
  __init__  the experiment specification (configs/generalization/finetune.yaml) and the names of its units and models
  data      the test subset and the training order of each dataset (`finetune prepare`), written once
  lora      PEFT LoRA on the decoder's linear layers: inject, train mode, save (an evaluation attaches the saved
            adapter through the offline model adapter's load_lora)
  train     one training unit inside the model's environment (the hand-written AdamW loop, bit-exact resume)
  launch    the evaluation model ids, launching a unit's training or the evaluations, printed commands
  report    accuracy per budget, dAcc / dGain, the primary endpoint with Holm, the controls, the post-fine-tuning readout
  seeds     the training-seed replication report
Command line: `streamreplace finetune prepare|check|models|units|train|eval|commands|report|seeds`
(cli/cmd_finetune.py).

Names. A training unit is (model, dataset, budget[, epochs, tag]); its adapter lives in
`<root>/adapters/<dataset>/<model>/<label>/`, label = "1k" | "2k" | "5k" [+ "-<tag>"]. The model id evaluated is
`<model>@<dataset>-<label>` (label "0" = zero-shot, no adapter); zero-shot and fine-tuned ids of a model differ only in
the adapter. Seed-replication units carry the tag "s<seed>" and live in their own folder (`seeds.outputs`).
The experiment root is `<outputs root>/<outputs>` (default `<outputs root>/finetune`): an ordinary outputs folder with
subset.json, so `streamreplace --outputs <root> score` scores it like any other folder."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

SPEC_FILE = Path("generalization") / "finetune.yaml"
KEYS = {"outputs", "datasets", "budgets", "test_size", "pairs", "models", "overrides", "lora", "train",
        "primary_excludes", "controls", "extra_units", "seeds"}
REQUIRED = ("datasets", "budgets", "test_size", "pairs", "lora", "train")
TRAIN_KEYS = {"lr", "weight_decay", "warmup_ratio", "epochs", "grad_accum", "clip", "seed", "save_every_steps",
              "save_every_minutes"}


@dataclass(frozen=True)
class Experiment:
    """The fine-tuning specification (configs/generalization/finetune.yaml) bound to its root folder."""
    spec: dict
    root: Path

    @property
    def datasets(self) -> list[str]:
        return list(self.spec["datasets"])

    @property
    def budgets(self) -> list[int]:
        return [int(b) for b in self.spec["budgets"]]

    @property
    def top_budget(self) -> int:
        return max(self.budgets)

    @property
    def test_size(self) -> int:
        return int(self.spec["test_size"])

    @property
    def pairs(self) -> dict[str, list[str]]:
        """pair -> [parent, child, ...] (the parent first)."""
        return {pair: list(members) for pair, members in self.spec["pairs"].items()}

    @property
    def members(self) -> list[str]:
        return list(dict.fromkeys(m for ms in self.spec["pairs"].values() for m in ms))

    @property
    def seed(self) -> int:
        return int(self.spec["train"]["seed"])

    @property
    def lora(self) -> dict:
        return deepcopy(self.spec["lora"])

    @property
    def train(self) -> dict:
        return deepcopy(self.spec["train"])

    @property
    def primary_excludes(self) -> dict[str, str]:
        return dict(self.spec.get("primary_excludes") or {})

    @property
    def controls(self) -> dict[str, dict]:
        return deepcopy(self.spec.get("controls") or {})

    @property
    def seeds(self) -> dict | None:
        return deepcopy(self.spec.get("seeds")) or None

    def folder(self, control: str | None = None) -> Path:
        """The experiment root, or the outputs folder of control `control` (a subfolder of the root)."""
        if control is None:
            return self.root
        if control not in self.controls:
            raise ValueError(f"unknown control {control!r}; configured: {', '.join(self.controls) or 'none'}")
        return self.root / self.controls[control]["outputs"]

    def seeds_folder(self) -> Path:
        if not self.seeds:
            raise ValueError("the specification has no `seeds` block (training-seed replication)")
        return self.root / self.seeds["outputs"]


def validate_spec(spec: dict) -> None:
    problems = [f"missing key {k!r}" for k in REQUIRED if k not in spec]
    problems += [f"unknown keys {sorted(set(spec) - KEYS)}"] if set(spec) - KEYS else []
    if not problems:
        unknown = sorted(set(spec["train"]) - TRAIN_KEYS)
        if unknown or TRAIN_KEYS - set(spec["train"]):
            problems.append(f"train: needs exactly {sorted(TRAIN_KEYS)}")
        if any(int(b) <= 0 for b in spec["budgets"]):
            problems.append("budgets must be positive (budget 0 = zero-shot is always evaluated)")
        for pair, members in spec["pairs"].items():
            if not isinstance(members, list) or len(members) < 2:
                problems.append(f"pairs.{pair}: [parent, child, ...] with at least two models")
        for name, control in (spec.get("controls") or {}).items():
            if not isinstance(control, dict) or "outputs" not in control:
                problems.append(f"controls.{name}: needs an `outputs` folder")
        seeds = spec.get("seeds")
        if seeds and not {"outputs", "datasets", "models", "budgets", "seeds"} <= set(seeds):
            problems.append("seeds: needs outputs, datasets, models, budgets and seeds")
    if problems:
        raise ValueError("fine-tuning specification: " + "; ".join(problems))


def load_experiment(config_dir: str | Path | None = None, path: str | Path | None = None,
                    root: str | Path | None = None, outputs: str | Path | None = None,
                    overrides: dict | None = None) -> Experiment:
    """configs/generalization/finetune.yaml (or `path`), with `overrides` merged on top (e.g. the `finetune:` block of
    configs/local.yaml). root: the experiment root (default: <outputs root>/<spec outputs>); outputs: the outputs
    root (default: the resolved paths' outputs)."""
    import yaml
    from ...config.settings import merge
    from ...paths import default_config_dir
    path = Path(path) if path else Path(config_dir or default_config_dir()) / SPEC_FILE
    if not path.is_file():
        raise ValueError(f"fine-tuning specification not found: {path}")
    spec = merge(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, overrides or {})
    validate_spec(spec)
    if root is None:
        if outputs is None:
            from ... import paths
            outputs = paths.current().outputs
        root = Path(outputs) / spec.get("outputs", "finetune")
    return Experiment(spec, Path(root))


# ---------- names of units and models (ported unchanged) ----------

def label(budget: int, tag: str | None = None) -> str:
    return ("0" if not budget else f"{budget // 1000}k" if budget % 1000 == 0 else str(budget)) + (f"-{tag}" if tag else "")


def model_id(model: str, dataset: str, budget: int, tag: str | None = None) -> str:
    return f"{model}@{dataset}-{label(budget, tag)}"


def units(exp: Experiment) -> list[dict]:
    """Every training unit: each pair member x dataset x budget, plus `extra_units` (e.g. 1K x 3 epochs). The unit
    dict is part of the unit's identity (train.py), so its keys must stay as they are."""
    spec, members = exp.spec, exp.members
    epochs = int(spec["train"]["epochs"])
    out = [{"model": m, "dataset": d, "budget": int(b), "epochs": epochs, "tag": None}
           for d in spec["datasets"] for m in members for b in spec["budgets"] if int(b) > 0]
    for extra in spec.get("extra_units") or []:
        out += [{"model": m, "dataset": extra["dataset"], "budget": int(extra["budget"]),
                 "epochs": int(extra.get("epochs", epochs)), "tag": extra["tag"]} for m in extra["models"]]
    return out


def seed_units(exp: Experiment) -> list[dict]:
    """The training-seed replication's units: every model x dataset x budget x seed, tag "s<seed>". Each carries its
    seed, the epoch-0 shuffle flag and its folder (adapters in <root>/<folder>/adapters, evaluations in that folder)."""
    s = exp.seeds
    if not s:
        return []
    epochs = int(exp.spec["train"]["epochs"])
    return [{"model": m, "dataset": d, "budget": int(b), "epochs": epochs, "tag": f"s{k}", "seed": int(k),
             "shuffle": bool(s.get("shuffle")), "root": s["outputs"]}
            for d in s["datasets"] for m in s["models"] for b in s["budgets"] for k in s["seeds"]]


def unit_dir(exp: Experiment, unit: dict) -> Path:
    base = exp.root / unit["root"] / "adapters" if unit.get("root") else exp.root / "adapters"
    return base / unit["dataset"] / unit["model"] / label(unit["budget"], unit["tag"])


def find_unit(exp: Experiment, model: str, dataset: str, budget: int, tag: str | None = None) -> dict:
    for unit in units(exp) + seed_units(exp):
        if (unit["model"], unit["dataset"], unit["budget"], unit["tag"]) == (model, dataset, budget, tag):
            return unit
    raise ValueError(f"no fine-tuning unit {model} {dataset} {budget} {tag or ''}".rstrip())
