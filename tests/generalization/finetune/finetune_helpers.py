"""Shared helpers of the fine-tuning tests: the experiment specification over the synthetic benchmark
(finetune_demo.py), writers of its prepared splits and of fake scored evaluation rows."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import finetune_demo

from streamreplace.core.io import write_jsonl
from streamreplace.core.schema import score_row

DEMO = finetune_demo.NAME
SPEC = {
    "outputs": "finetune", "datasets": [DEMO], "budgets": [6, 12], "test_size": 12,
    "pairs": {"p": ["par", "kid"]},
    "lora": {"r": 16, "alpha": 32},
    "train": {"lr": 1.0e-4, "weight_decay": 0.0, "warmup_ratio": 0.03, "epochs": 1, "grad_accum": 2, "clip": 1.0,
              "seed": 0, "save_every_steps": 25, "save_every_minutes": 10},
    "controls": {"blank": {"outputs": "blank", "size": 4, "budgets": [0, 12], "set": {"video": {"blank": True}}}},
    "extra_units": [{"models": ["par"], "dataset": DEMO, "budget": 6, "epochs": 3, "tag": "e3"}],
    "seeds": {"outputs": "seeds", "datasets": [DEMO], "models": ["par", "kid"], "budgets": [6], "seeds": [1, 2],
              "shuffle": True},
}


def spec(**changes) -> dict:
    out = deepcopy(SPEC)
    out.update(changes)
    return out


def write_prepared(paths, test=None, train=None, name: str = DEMO) -> None:
    """The benchmark's prepared splits under the data root (as `streamreplace prepare` writes them)."""
    folder = Path(paths.dataset_dir(name)) / "prepared"
    write_jsonl(folder / "samples.jsonl", [s.to_dict() for s in (finetune_demo.questions() if test is None else test)])
    write_jsonl(folder / "train.jsonl", [s.to_dict() for s in (finetune_demo.training() if train is None else train)])


def write_scored(folder: Path, model_id: str, ids: list[str], right: int | set, dataset: str = DEMO,
                 scored: bool = True, errors: set | None = None, info: dict | None = None) -> None:
    """An evaluation job of `model_id` on `ids` (generation rows + score rows): the first `right` ids (or the ids in
    the set `right`) are correct; ids in `errors` errored; scored=False writes no score file (unscored)."""
    right = set(ids[:right]) if isinstance(right, int) else set(right)
    errors = set(errors or ())
    job = f"{model_id}.offline"
    rows = [{"id": i, "prompt": f"prompt {i}", "raw": "" if i in errors else "A", "tokens": 1, "finish": "eos",
             "latency": 0.1, "info": dict(info or {}), "error": "boom" if i in errors else None} for i in ids]
    write_jsonl(Path(folder) / dataset / f"{job}.jsonl", rows)
    scores = Path(folder) / dataset / f"{job}.scores.jsonl"
    if scored:
        write_jsonl(scores, [score_row(i) if i in errors else score_row(i, pred="A", parse="letter", correct=i in right)
                             for i in ids])
    else:
        scores.unlink(missing_ok=True)
