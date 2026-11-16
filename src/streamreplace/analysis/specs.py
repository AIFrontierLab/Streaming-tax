"""The analysis profile: which metrics are contrasted, on which units, in which Holm families, with which margins.

`load_profile()` reads configs/analysis.yaml (the "paper" profile) and configs/pairs.yaml (each child's parent) into
a `Profile`. The metric list reproduces, field for field and in the same order, the table behind the paper's 354
P/C/S contrasts (10 image metrics x 6 children, C-P only; 4 video metrics x 6 x 3 gaps; HarmBench x 6, C-P only;
4 VSB/MCV splits x 6 x 3; 8 VMDT metrics x 6 x 3). See docs/statistics.md."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Gap -> (arm a, arm b); delta = metric(b) - metric(a), oriented so that + = the later arm (child, or its stream
# run) is better. Arms: P = parent offline, C = child offline, S = child native streaming.
GAPS = {"C-P": ("P", "C"), "S-C": ("C", "S"), "S-P": ("P", "S")}


@dataclass(frozen=True)
class Metric:
    """One reported metric and the unit structure its statistics use (see configs/analysis.yaml `metrics`)."""
    dataset: str
    name: str                      # the metric key in the benchmark's metrics()
    kind: str                      # acc | group | mme | safe | vmdt
    cluster: str | None = None     # acc/safe: metadata key (or "video" = the sample's video file); None = the item
    group: str | None = None       # group: metadata key of the all-correct group
    size: int | str | None = None  # group size (int) or the metadata key holding it
    split: str | None = None       # only the items whose benchmark stratum equals this
    scale: float = 1.0             # delta units per percentage point
    lower_is_better: bool = False  # the reported metric (ASR, leakage); the delta uses its complement
    family: str = ""
    headline: bool = True
    modes: tuple = ("offline", "stream")
    coverage: str | None = None    # answered | responded (default: responded for kind safe, else answered)
    key: str | None = None         # vmdt: the value's name in the benchmark's scores() (default: name)
    official: str | None = None    # the value's name in the benchmark's metrics() (default: name)
    blocks: str | None = None      # vmdt: "scenario" = one resampling block per scenario, else one block

    @property
    def coverage_rule(self) -> str:
        return self.coverage or ("responded" if self.kind == "safe" else "answered")

    @property
    def score_key(self) -> str:
        return self.key or self.name

    @property
    def official_key(self) -> str:
        return self.official or self.name


@dataclass
class Profile:
    name: str
    children: list[str]
    parents: dict[str, str]                 # child -> parent
    pairs: dict[str, list[str]]             # pair -> [parent, *children]
    models: list[str]
    metrics: list[Metric]
    reps: int = 10_000
    seed: int = 0
    vmdt_draws: int = 20_000
    holm: dict = field(default_factory=dict)
    margins: dict = field(default_factory=dict)
    check_official: dict = field(default_factory=dict)
    duration_split: dict | None = None
    overlays: dict = field(default_factory=dict)
    flags: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    display: dict = field(default_factory=dict)
    root: Path | None = None                # repository root: relative paths in the profile resolve against it

    def path(self, relative: str | Path) -> Path:
        p = Path(relative)
        return p if p.is_absolute() or self.root is None else self.root / p

    def name_of(self, model: str, table: str | None = None) -> str:
        """Display name of a model: the short name, or a per-table override (`gap_table_names`)."""
        if table and model in self.display.get(table, {}):
            return self.display[table][model]
        return self.display.get("names", {}).get(model, model)

    def full_name(self, model: str) -> str:
        return self.display.get("full_names", {}).get(model, model)

    def pair_of(self, model: str) -> str | None:
        return next((p for p, members in self.pairs.items() if model in members), None)

    @property
    def tost_margins(self) -> list:
        return list(self.margins.get("tost", [3, 2]))

    @property
    def ni_margins(self) -> list:
        return list(self.margins.get("ni", [2, 3, 5]))


def repo_root() -> Path:
    """The source checkout holding configs/ and paper/ (paths.project_root(): the checkout this package lives in,
    else the nearest folder up from the current directory with configs/defaults.yaml)."""
    from ..paths import project_root
    return project_root()


def _metric(entry: dict) -> Metric:
    entry = dict(entry)
    if "modes" in entry:
        entry["modes"] = tuple(entry["modes"])
    if "scale" in entry:
        entry["scale"] = float(entry["scale"])
    return Metric(**entry)


def _pairs_from_file(path: Path) -> dict[str, list[str]] | None:
    """configs/pairs.yaml: pairs.<p>.parent and pairs.<p>.children (other keys are run settings)."""
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pairs = data.get("pairs", data)
    out = {}
    for name, spec in pairs.items():
        if isinstance(spec, dict) and "parent" in spec:
            out[name] = [spec["parent"], *spec.get("children", [])]
    return out or None


def profile_from_dict(data: dict, pairs: dict[str, list[str]] | None = None, root: Path | None = None) -> Profile:
    """Builds a Profile from the parsed analysis.yaml. `pairs` (pair -> [parent, *children]) overrides the
    profile's own roster.pairs."""
    roster = data.get("roster", {})
    if pairs is None:
        pairs = {p: [v["parent"], *v.get("children", [])] for p, v in roster.get("pairs", {}).items()}
    parents = {c: members[0] for members in pairs.values() for c in members[1:]}
    children = list(roster.get("children") or parents)
    missing = [c for c in children if c not in parents]
    if missing:
        raise ValueError(f"analysis profile: no parent for {missing} (configs/pairs.yaml or roster.pairs)")
    engine = data.get("engine", {})
    return Profile(
        name=data.get("profile", "custom"), children=children, parents=parents, pairs=pairs,
        models=list(roster.get("models") or [m for members in pairs.values() for m in members]),
        metrics=[_metric(m) for m in data.get("metrics", [])],
        reps=int(engine.get("reps", 10_000)), seed=int(engine.get("seed", 0)),
        vmdt_draws=int(engine.get("vmdt_draws", 20_000)),
        holm=data.get("holm", {}), margins=data.get("margins", {}), check_official=data.get("check_official", {}),
        duration_split=data.get("duration_split"), overlays=data.get("overlays", {}) or {},
        flags=data.get("flags", {}), report=data.get("report", {}), display=data.get("display", {}), root=root)


def load_profile(path: str | Path | None = None, pairs_path: str | Path | None = None) -> Profile:
    """Reads the analysis profile (default: <repo>/configs/analysis.yaml) and the pairs (default:
    configs/pairs.yaml next to it; the profile's roster.pairs when that file is absent)."""
    path = Path(path) if path else repo_root() / "configs" / "analysis.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    pairs = _pairs_from_file(Path(pairs_path) if pairs_path else path.parent / "pairs.yaml")
    root = path.resolve().parent.parent
    return profile_from_dict(data, pairs=pairs, root=root)


def profile_from_config(config, path: str | Path | None = None) -> Profile:
    """The analysis profile of a loaded configuration (config.load.load_config): its `analysis` section (configs/
    analysis.yaml with local / --config overrides), or the YAML file `path` instead; parents from its pairs.
    Relative paths in the profile (frozen id lists, flag tables) resolve against the configuration folder's parent."""
    data = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}) if path else config.section("analysis")
    if not data.get("metrics"):
        raise ValueError(f"analysis profile {path or 'configs/analysis.yaml'}: no metrics defined")
    pairs = {p.id: [p.parent, *p.children] for p in config.pairs.values()} or None
    root = Path(config.config_dir).resolve().parent if config.config_dir else repo_root()
    return profile_from_dict(data, pairs=pairs, root=root)


def find(profile: Profile, dataset: str, name: str) -> Metric:
    for m in profile.metrics:
        if (m.dataset, m.name) == (dataset, name):
            return m
    raise KeyError(f"{dataset} {name}: not in the analysis profile")
