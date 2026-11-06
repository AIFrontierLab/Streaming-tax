"""Prompt-robustness experiment: its specification, the analysis units, the item selection and the condition
folders (Supp. Tables eval_robustness and eval_instruction, the prompt-robustness columns of Table 2).

The specification is configs/generalization/robustness.yaml (docs/generalization/robustness.md). Every condition is
one ordinary outputs folder under the experiment root (default `<outputs root>/robustness`):

    <root>/<condition>/subset.json              {benchmark: [sample ids]}: the condition's items. Every question-stem
                                                condition holds the same file; a condition with `datasets` / `units`
                                                holds a subset nested in it (the instruction-level conditions)
    <root>/<condition>/prompt_overrides.jsonl   every condition but clean (edits.py): the exact [old, new] edits the
                                                runner applies to the rendered prompts (data/prompts.py)
    <root>/<condition>/<benchmark>/<model>.offline.jsonl, .scores.jsonl   `streamreplace --outputs <folder> run|score`
    <root>/subset.info.json                     strata counts and the sha256 of each benchmark's id list
    <root>/paraphrase_cache.jsonl, audit/, report/

Units are the samples that share one perturbed stem and form one item of the analysis: a Video-MME question (one
question per video is kept), or an MMBench question with all its circular rotations (right only if every rotation is
right). Item selection, seed 0 (the paper's ids are in paper/frozen/robustness/items.json):
- Video-MME short: for each video, the question whose sha256("<seed>:<sample id>") is smallest: 300 units;
- MMBench DEV-EN: 300 questions, a largest-remainder allocation over the `l2` category, within a category in
  sha256("<seed>:<id of the question's first rotation>") order; all rotations of a chosen question are kept
  (1,118 rows).
Nothing written here is ever replaced by different content (write_once): a folder's outputs were generated from it."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from ...core.io import read_json, read_jsonl, write_json, write_jsonl

SPEC_FILE = Path("generalization") / "robustness.yaml"
SUBSET_FILE = "subset.json"
INFO_FILE = "subset.info.json"
EDIT_KINDS = ("typo", "para", "stress", "irrelevant", "choice")
CONDITION_KEYS = {"edit", "family", "target", "twin", "datasets", "units", "seed_as", "rate", "max_words", "pivot",
                  "texts"}


@dataclass(frozen=True)
class UnitSpec:
    """How a benchmark's samples form the units of the analysis."""
    stem: Callable[[Any], str]        # the exact question text a stem edit replaces (instruction, options, hint kept)
    unit: Callable[[Any], str]        # samples sharing one edit = one analysis item
    cluster: Callable[[Any], str]     # bootstrap resampling unit
    strata: tuple[str, ...]           # metadata keys of the largest-remainder stratification
    one_per: str | None = None        # keep one unit per value of this metadata key
    instruction: Callable[[], str] | None = None   # the single official answer instruction (instruction-level edits)


def _mmbench_instruction() -> str:
    """MMBench's answer instruction: its POST_PROMPT without the leading newline."""
    from ...benchmarks import registry
    return registry.get("mmbench").POST_PROMPT.strip()


# Adding a benchmark = one entry here and in the specification's `datasets` (its prompt must contain the stem once).
# Video-MME states its answer format twice (head and tail), so it has no single instruction sentence.
DATASETS: dict[str, UnitSpec] = {
    "videomme": UnitSpec(stem=lambda s: s.question, unit=lambda s: s.id, cluster=lambda s: s.metadata["video"],
                         strata=("task",), one_per="video"),
    "mmbench": UnitSpec(stem=lambda s: s.metadata["raw_question"], unit=lambda s: str(s.metadata["question"]),
                        cluster=lambda s: str(s.metadata["question"]), strata=("l2",),
                        instruction=_mmbench_instruction),
}


# ---------- specification ----------

def load_spec(config_dir: str | Path | None = None, path: str | Path | None = None,
              overrides: dict | None = None) -> dict:
    """configs/generalization/robustness.yaml (or `path`), with `overrides` merged on top (e.g. the `robustness:`
    block of configs/local.yaml), validated."""
    import yaml
    from ...config.settings import merge
    from ...paths import default_config_dir
    path = Path(path) if path else Path(config_dir or default_config_dir()) / SPEC_FILE
    if not path.is_file():
        raise ValueError(f"prompt-robustness specification not found: {path}")
    spec = merge(yaml.safe_load(path.read_text(encoding="utf-8")) or {}, overrides or {})
    validate_spec(spec)
    return spec


def validate_spec(spec: dict) -> None:
    """Raises one ValueError listing every problem of a specification."""
    problems = []
    for key in ("seed", "datasets", "pairs", "conditions"):
        if key not in spec:
            problems.append(f"missing key {key!r}")
    if problems:
        raise ValueError("robustness specification: " + "; ".join(problems))
    unknown = sorted(set(spec["datasets"]) - set(DATASETS))
    if unknown:
        problems.append(f"datasets {unknown} have no unit definition (make.DATASETS)")
    conditions = spec["conditions"]
    if "clean" not in conditions or conditions["clean"] not in ({}, None):
        problems.append("conditions.clean must exist and be empty (the unperturbed prompts)")
    for name, c in conditions.items():
        if name == "clean":
            continue
        c = c or {}
        if set(c) - CONDITION_KEYS:
            problems.append(f"conditions.{name}: unknown keys {sorted(set(c) - CONDITION_KEYS)}")
        if c.get("edit") not in EDIT_KINDS:
            problems.append(f"conditions.{name}: edit must be one of {', '.join(EDIT_KINDS)}")
        if c.get("target") not in (None, "instruction"):
            problems.append(f"conditions.{name}: target must be 'instruction' or absent")
        if c.get("target") is None and not c.get("family"):
            problems.append(f"conditions.{name}: a question-stem condition needs a `family`")
        if c.get("target") == "instruction" and c.get("twin") not in conditions:
            problems.append(f"conditions.{name}: an instruction-level condition needs its stem `twin` condition")
        if c.get("edit") == "choice" and not c.get("texts"):
            problems.append(f"conditions.{name}: edit 'choice' needs `texts`")
        if c.get("edit") == "para" and not c.get("pivot"):
            problems.append(f"conditions.{name}: edit 'para' needs a `pivot` language")
        if c.get("seed_as") is not None and c["seed_as"] not in conditions:
            problems.append(f"conditions.{name}: seed_as names an unknown condition")
        for dataset in c.get("datasets") or []:
            if dataset not in spec["datasets"]:
                problems.append(f"conditions.{name}: dataset {dataset!r} is not in `datasets`")
    if problems:
        raise ValueError("robustness specification: " + "; ".join(problems))


def stem_conditions(spec: dict) -> list[str]:
    """clean + the question-stem conditions (the primary analysis); instruction-level conditions are analysed by
    instruction.py."""
    return [c for c, v in spec["conditions"].items() if (v or {}).get("target") != "instruction"]


def instruction_conditions(spec: dict) -> list[str]:
    return [c for c, v in spec["conditions"].items() if (v or {}).get("target") == "instruction"]


def families(spec: dict) -> dict[str, list[str]]:
    """family -> its question-stem conditions, in specification order (distraction = stress and irrelevant)."""
    out: dict[str, list[str]] = {}
    for c in stem_conditions(spec):
        if c != "clean":
            out.setdefault(spec["conditions"][c]["family"], []).append(c)
    return out


# ---------- the experiment ----------

@dataclass
class Experiment:
    """The specification bound to an experiment root and to where prepared samples are read.

    pairs: pair id -> (parent, *children), for the pairs the specification names (from configs/pairs.yaml).
    paths: a streamreplace.paths.Paths whose data root holds the prepared benchmarks (None: paths.current())."""
    spec: dict
    root: Path
    pairs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    paths: Any = None
    _prepared: dict = field(default_factory=dict, repr=False)

    @property
    def seed(self) -> int:
        return int(self.spec["seed"])

    @property
    def conditions(self) -> dict:
        return {name: (c or {}) for name, c in self.spec["conditions"].items()}

    def condition(self, name: str) -> dict:
        try:
            return self.conditions[name]
        except KeyError:
            raise ValueError(f"unknown robustness condition {name!r}") from None

    def folder(self, condition: str) -> Path:
        return Path(self.root) / condition

    def datasets(self, condition: str | None = None) -> list[str]:
        """The benchmarks of a condition (its `datasets`, else the experiment's)."""
        c = self.condition(condition) if condition else {}
        return list(c.get("datasets") or self.spec["datasets"])

    @property
    def models(self) -> list[str]:
        return [m for members in self.pairs.values() for m in members]

    @property
    def children(self) -> list[tuple[str, str, str]]:
        """(pair, parent, child) for every child, in the specification's pair order."""
        return [(p, members[0], child) for p, members in self.pairs.items() for child in members[1:]]

    def subset(self, condition: str) -> dict[str, list[str]] | None:
        from ...data.samples import read_subset
        return read_subset(self.folder(condition))

    def prepared(self, dataset: str) -> list:
        """The benchmark's prepared test samples, prepared order (read once)."""
        if dataset not in self._prepared:
            from ...data.samples import load_samples
            self._prepared[dataset] = load_samples(dataset, paths=self.paths)
        return self._prepared[dataset]

    def samples(self, dataset: str, condition: str | None = None) -> list:
        """Prepared samples, restricted to the condition folder's subset.json (as the runner restricts them)."""
        samples = self.prepared(dataset)
        if condition is None:
            return list(samples)
        ids = (self.subset(condition) or {}).get(dataset)
        if ids is None:
            return list(samples)
        keep = set(ids)
        return [s for s in samples if s.id in keep]

    def units(self, dataset: str, condition: str | None = None) -> dict[str, list]:
        """unit -> its samples in id order, from all prepared samples (condition None) or a condition's subset."""
        out: dict[str, list] = defaultdict(list)
        for s in sorted(self.samples(dataset, condition), key=lambda s: s.id):
            out[DATASETS[dataset].unit(s)].append(s)
        return dict(out)


def bind(spec: dict, config, paths, root: str | Path | None = None) -> Experiment:
    """The Experiment of `spec` for the loaded configuration (pair members from configs/pairs.yaml) and paths: the
    experiment root is `root`, else `<outputs root>/<spec outputs>`."""
    folder = Path(root) if root else Path(paths.outputs) / str(spec.get("outputs") or "robustness")
    pairs = {p: tuple(config.pair(p).members) for p in spec["pairs"]}
    return Experiment(spec, folder, pairs, paths)


# ---------- item selection ----------

def key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stratified(samples, size: int, keys: tuple, seed: int) -> list[str]:
    """`size` ids, each stratum's share by largest remainder, within a stratum in seeded hash order (ties of the
    remainder broken by the stratum's string). The fine-tuning test subsets use the same allocation."""
    if len(samples) <= size:
        return sorted(s.id for s in samples)
    strata: dict[tuple, list] = {}
    for s in samples:
        strata.setdefault(tuple(s.metadata.get(k) for k in keys), []).append(s)
    quota = {k: size * len(v) / len(samples) for k, v in strata.items()}
    take = {k: int(q) for k, q in quota.items()}
    for k in sorted(quota, key=lambda k: (take[k] - quota[k], str(k)))[:size - sum(take.values())]:
        take[k] += 1
    chosen = [s.id for k, v in strata.items() for s in sorted(v, key=lambda s: key(f"{seed}:{s.id}"))[:take[k]]]
    return sorted(chosen)


def select(exp: Experiment, dataset: str, size: int) -> list[str]:
    """The sample ids of `size` units of `dataset` (all samples of each chosen unit), sorted."""
    d, groups, seed = DATASETS[dataset], exp.units(dataset), exp.seed
    candidates = list(groups)
    if d.one_per:                                       # e.g. one question per video: the unit with the lowest hash
        best: dict = {}
        for u in candidates:
            value = groups[u][0].metadata[d.one_per]
            if value not in best or key(f"{seed}:{u}") < key(f"{seed}:{best[value]}"):
                best[value] = u
        candidates = sorted(best.values())
    representatives = {groups[u][0].id: u for u in candidates}
    chosen = stratified([groups[u][0] for u in candidates], size, d.strata, seed)
    return sorted(s.id for rep in chosen for s in groups[representatives[rep]])


def nested(exp: Experiment, subset: dict, condition: str) -> dict:
    """A condition's own subset: its datasets only and, with `units: n`, n units of the shared subset per dataset
    (stratified as select(), seeded hash order), so its items are a subset of every other folder's."""
    c, out = exp.condition(condition), {}
    for dataset in exp.datasets(condition):
        ids = subset[dataset]
        if c.get("units"):
            d, chosen = DATASETS[dataset], set(ids)
            groups = [u for u in exp.units(dataset).values() if u[0].id in chosen]
            keep = set(stratified([u[0] for u in groups], c["units"], d.strata, exp.seed))
            ids = sorted(s.id for u in groups if u[0].id in keep for s in u)
        out[dataset] = ids
    return out


def is_nested(c: dict) -> bool:
    return bool(c.get("datasets") or c.get("units"))


def write_once(path: Path, value: list[dict] | dict) -> str:
    """Writes `value` (rows as JSONL for a .jsonl path, else JSON) unless the file exists; an existing file must hold
    the same content."""
    jsonl = path.suffix == ".jsonl"
    if path.exists():
        old = read_jsonl(path) if jsonl else read_json(path)
        if old != value:
            raise ValueError(f"{path} exists with different content; the outputs in its folder were generated from it. "
                             "Move the folder's outputs and this file away first if the change is intended.")
        return "unchanged"
    (write_jsonl if jsonl else write_json)(path, value)
    return "written"


def make_subset(exp: Experiment) -> dict:
    """Writes every condition folder's subset.json and the experiment's subset.info.json (once)."""
    spec = exp.spec
    subset = {dataset: select(exp, dataset, size) for dataset, size in spec["datasets"].items()}
    status = {c: write_once(exp.folder(c) / SUBSET_FILE, nested(exp, subset, c) if is_nested(v) else subset)
              for c, v in exp.conditions.items()}
    info = {}
    for dataset, ids in subset.items():
        d, chosen = DATASETS[dataset], set(ids)
        picked = [u for u in exp.units(dataset).values() if u[0].id in chosen]
        strata: dict = defaultdict(int)
        for unit in picked:
            strata["/".join(str(unit[0].metadata.get(k)) for k in d.strata)] += 1
        info[dataset] = {"samples": len(ids), "units": len(picked), "clusters": len({d.cluster(u[0]) for u in picked}),
                         "units_per_stratum": dict(sorted(strata.items())), "ids_sha256": key("\n".join(ids))}
    write_once(Path(exp.root) / INFO_FILE, {"seed": spec["seed"], "datasets": info})
    return {"folders": status, **{k: v["samples"] for k, v in info.items()}}


def item_lists(exp: Experiment) -> dict:
    """The condition folders' item lists (ids only), as paper/frozen/robustness/items.json stores them."""
    shared, nested_sets = None, {}
    for c, v in exp.conditions.items():
        subset = exp.subset(c)
        if subset is None:
            raise ValueError(f"{exp.folder(c)}: no {SUBSET_FILE}; run `streamreplace robustness subset` first")
        if is_nested(v):
            nested_sets[c] = subset
        elif shared is None:
            shared = subset
    return {"seed": exp.seed, "subset": shared, "nested": nested_sets,
            "ids_sha256": {d: key("\n".join(ids)) for d, ids in (shared or {}).items()}}


def compare_items(exp: Experiment, frozen: dict) -> list[str]:
    """Differences between the condition folders' subsets and a frozen item list."""
    mine, problems = item_lists(exp), []
    if mine["seed"] != frozen["seed"]:
        problems.append(f"seed {mine['seed']} differs from the frozen seed {frozen['seed']}")
    if json.dumps(mine["subset"], sort_keys=True) != json.dumps(frozen["subset"], sort_keys=True):
        problems.append("the shared subset differs from the frozen one")
    for c, ids in frozen.get("nested", {}).items():
        if c in mine["nested"] and json.dumps(mine["nested"][c], sort_keys=True) != json.dumps(ids, sort_keys=True):
            problems.append(f"{c}: the nested subset differs from the frozen one")
    return problems
