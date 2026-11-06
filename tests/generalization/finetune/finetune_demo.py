"""Synthetic benchmark of the fine-tuning tests, registered as `ftdemo` (finetune_helpers.DEMO): two-option
multiple-choice questions grouped in cases and filed in two categories, with a training split drawn from two subsets.
No media: the tests write the prepared splits directly (finetune_helpers.write_prepared)."""
from __future__ import annotations

from streamreplace.core.schema import Sample

NAME = "ftdemo"
MEDIA = "video"
JUDGE = None
GROUP = "case"                    # the leakage filter and the bootstrap clusters
STRATUM_KEYS = ("category",)      # the stratified test subset and the bootstrap strata
INTERLEAVE = "subset"             # the training order alternates the subsets
CHECK_NEAR_DUPLICATES = False     # a test switches it on with frame lists of its own


def questions(n: int = 24) -> list[Sample]:
    """q00..: two questions per case, the first 16 in category k1, the rest in k2."""
    return [Sample(f"q{i:02d}", f"Question {i}?", "AB"[i % 2], "mcq", {"A": "left", "B": "right"},
                   metadata={"case": f"c{i // 2}", "category": "k1" if i < 16 else "k2"}) for i in range(n)]


def training(n: int = 60) -> list[Sample]:
    """t000..: three per case; the first case is also a test case (c0), so the leakage filter drops it."""
    return [Sample(f"t{i:03d}", f"Training question {i}?", "AB"[i % 2], "mcq", {"A": "left", "B": "right"},
                   metadata={"case": "c0" if i < 3 else f"tc{i // 3}", "subset": "action" if i % 3 else "scenery"})
            for i in range(n)]


def prepare(raw, media) -> list[Sample]:
    return questions()


def prepare_train(raw, media) -> list[Sample]:
    return training()
