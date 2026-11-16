"""Arms of a contrast: formal jobs, and jobs derived from them by overlays.

Overlays (applied with `Job.with_overlays`):
- Replace(ids, source): the source job's records replace the job's records on `ids` (generation-cap reruns; the
  plugin's fixed refusal on the items where the gate fired).
- ForceUnanswered(ids): the records on `ids` (all when None) become unanswered and wrong (the plugin's refusal on
  understanding sets, which a parser could otherwise read as an option).
- Restrict(ids): only the records on `ids` are kept (a plugin item set).

`Arms` resolves the arms of every contrast. With the truncation-rerun overlay (configs/analysis.yaml
overlays.truncation_rerun) the two OFFLINE arms of a parent-child pair on a rerun dataset are merged: on the ids that
either the parent's or the child's formal row capped, both arms take their rerun row (the rerun at 8,192 tokens
ran every member of the pair on the pair's ids), everywhere else the formal row. A pair with two children listed
under `shared` for a dataset uses one merge for the whole group instead (`rerun_ids`): the ids ANY member capped,
so both children are compared with the same parent arm (ThinkStream and MMDuet2 on MathVista)."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path

from .records import Catalog, Job, Jobs, load_job


@dataclass(frozen=True)
class Replace:
    ids: frozenset
    source: Job
    add: bool = False              # also add ids the job does not have (a rerun of an id the formal job lacks)
    label: str = "replace"

    def apply(self, rows: dict) -> dict:
        missing = sorted(i for i in self.ids if i not in self.source.rows and (self.add or i in rows))
        if missing:
            raise ValueError(f"{self.source.job} ({self.source.path}): no counted record for {len(missing)} id(s) to "
                             f"replace, e.g. {missing[:3]}")
        out = {i: (self.source.rows[i] if i in self.ids else r) for i, r in rows.items()}
        if self.add:
            out.update({i: r for i, r in self.source.rows.items() if i in self.ids and i not in out})
        return out


@dataclass(frozen=True)
class ForceUnanswered:
    ids: frozenset | None = None
    label: str = "unanswered"

    def apply(self, rows: dict) -> dict:
        # Only the final-scoring fields change; the rule / judge predictions stay (as the reported runs did).
        return {i: (replace(r, correct=False, answered=False, judged=False, pred=None)
                    if self.ids is None or i in self.ids else r) for i, r in rows.items()}


@dataclass(frozen=True)
class Restrict:
    ids: frozenset
    label: str = "restrict"

    def apply(self, rows: dict) -> dict:
        return {i: r for i, r in rows.items() if i in self.ids}


# ---------- generation-cap reruns ----------

def rerun_ids(per_model: dict, parent: str, child: str, shared: bool = False) -> frozenset:
    """The ids whose rerun rows replace the formal rows in the contrast of `parent` and `child`: the ids either of
    them capped or, with one shared merge for the pair's whole group, the ids any member capped. per_model: model ->
    {"ids": [...]} of one pair and dataset (capped.json)."""
    members = list(per_model) if shared else [parent, child]
    return frozenset(i for model in members for i in per_model.get(model, {}).get("ids", []))


@dataclass
class TruncationRerun:
    """Capped ids per pair, dataset and model and the folder holding the rerun jobs
    (<root>/<pair>/<dataset>/<model>.offline.jsonl, written by `streamreplace rerun-capped`)."""
    capped: dict                   # pair -> dataset -> model -> {"cap": int, "ids": [...]}
    root: Path
    pairs: list
    datasets: list
    ids_file: Path | None = None
    shared: dict | None = None     # pair -> datasets with one merge for the whole group (see rerun_ids)

    @classmethod
    def from_profile(cls, profile, outputs: Path, ids_path: str | Path | None = None,
                     rows_root: str | Path | None = None) -> "TruncationRerun | None":
        """The profile's overlays.truncation_rerun. Ids: `ids_path`, else the capped.json that `rerun-capped select`
        wrote into the rerun folder, else the profile's frozen list (the paper's: paper/frozen/rerun_capped/)."""
        spec = (profile.overlays or {}).get("truncation_rerun")
        if not spec:
            return None
        root = Path(rows_root) if rows_root else Path(outputs) / spec.get("rows", "rerun_capped")
        if ids_path:
            ids_file = Path(ids_path)
        elif (root / "capped.json").is_file():
            ids_file = root / "capped.json"
        else:
            ids_file = profile.path(spec["ids"])
        data = json.loads(ids_file.read_text(encoding="utf-8"))
        return cls(capped=data.get("pairs", data), root=root, pairs=list(spec["pairs"]),
                   datasets=list(spec["datasets"]), ids_file=ids_file, shared=dict(spec.get("shared") or {}))

    def ids(self, pair: str, dataset: str, parent: str, child: str) -> frozenset:
        """The rerun ids of the contrast of (parent, child) on a dataset (`rerun_ids`); none outside the overlay."""
        if pair not in self.pairs or dataset not in self.datasets:
            return frozenset()
        return rerun_ids(self.capped.get(pair, {}).get(dataset, {}), parent, child,
                         shared=dataset in (self.shared or {}).get(pair, ()))


class Arms:
    """The arms P (parent offline), C (child offline) and S (child stream) of each child, loaded once; with a
    TruncationRerun the offline arms of the affected (pair, dataset) are the merged ones."""

    def __init__(self, outputs: str | Path, catalog: Catalog, profile, rerun: TruncationRerun | None = None):
        self.outputs, self.catalog, self.profile, self.rerun = Path(outputs), catalog, profile, rerun
        self.jobs = Jobs(outputs, catalog)
        self.merged: dict = {}                     # (dataset, model, child) -> merged job

    def formal(self, dataset: str, model: str, mode: str) -> Job | None:
        return self.jobs(dataset, model, mode)

    def merge_ids(self, dataset: str, child: str) -> frozenset:
        if self.rerun is None:
            return frozenset()
        parent = self.profile.parents[child]
        return self.rerun.ids(self.profile.pair_of(child), dataset, parent, child)

    def job(self, dataset: str, model: str, mode: str, child: str) -> Job | None:
        """The arm `model.mode` as used in the contrasts of `child`."""
        ids = self.merge_ids(dataset, child)
        if mode != "offline" or not ids or model not in (child, self.profile.parents[child]):
            return self.formal(dataset, model, mode)
        key = (dataset, model, child)
        if key not in self.merged:
            formal = self.formal(dataset, model, mode)
            pair = self.profile.pair_of(child)
            rerun = load_job(self.rerun.root / pair, dataset, f"{model}.offline", self.catalog)
            if formal is None or rerun is None:
                raise FileNotFoundError(f"truncation rerun: {dataset} {model}.offline missing (formal or "
                                        f"{self.rerun.root / pair})")
            merged = formal.with_overlays(Replace(ids, rerun, add=True, label=f"rerun-{pair}-{child}"))
            self.merged[key] = merged
        return self.merged[key]

    def merged_jobs(self) -> list[tuple[str, str, str, Job]]:
        """(dataset, model, child, job) of every merged arm built so far."""
        return [(d, m, c, j) for (d, m, c), j in self.merged.items()]
