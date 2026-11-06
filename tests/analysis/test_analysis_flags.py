"""Cell flags (analysis/flags.py): record-derived qualifications and the external flag tables."""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from streamreplace.analysis import arms as A
from streamreplace.analysis.contrasts import Contrasts
from streamreplace.analysis.flags import External, Flagger, chance_level, poisson_binomial_upper


def test_poisson_binomial_upper_matches_enumeration():
    probs = np.array([0.1, 0.5, 0.25, 0.9])
    for k in range(5):
        exact = sum(np.prod([p if x else 1 - p for p, x in zip(probs, xs)])
                    for xs in itertools.product([0, 1], repeat=4) if sum(xs) >= k)
        assert poisson_binomial_upper(probs, k) == pytest.approx(exact)


def test_chance_level_is_the_mean_inverse_number_of_options(synthetic):
    _, cat, _ = synthetic
    assert chance_level(cat.samples("img")) == pytest.approx(50.0)


def flagged(synthetic, external=None):
    root, cat, prof = synthetic
    arms = A.Arms(root, cat, prof)
    rows = Contrasts(prof, arms, cat).compute()
    Flagger(prof, arms, cat, external).annotate(rows)
    return {(r["dataset"], r["gap"]): r for r in rows}


def test_record_derived_flags(synthetic):
    rows = flagged(synthetic)
    assert rows[("img", "C-P")]["flags"] == ["invalid"]          # the child capped 5/40 = 12.5% > 10% of its rows
    assert "12.5%" in rows[("img", "C-P")]["flag_notes"][0]
    assert rows[("vid", "S-C")]["flags"] == ["forced"]           # one forced stream answer
    assert rows[("vid", "C-P")]["flags"] == []
    assert rows[("saf", "S-P")]["flags"] == []                   # safety: invalid is an understanding-only flag


def test_external_flag_tables(synthetic, tmp_path):
    folder = tmp_path / "flags"
    folder.mkdir()
    (folder / "artifacts.csv").write_text("dataset,child,reason\nsaf,kid,judge-dependent format\n")
    (folder / "privacy_audit.csv").write_text("child,mode,verdict\nkid,offline,non-answering\n")
    rows = flagged(synthetic, External(folder))
    assert "artifact" in rows[("saf", "C-P")]["flags"] and "artifact" in rows[("saf", "S-C")]["flags"]
    assert External(tmp_path / "missing").artifacts == set()       # absent tables contribute no flags
