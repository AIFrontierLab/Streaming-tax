"""Module contracts of the 12 understanding benchmarks: registry entries, judge policies, expected sizes, private
metadata, metric units (identical to the paper's analysis profile) and the docs pages."""
from __future__ import annotations

import pytest
import yaml

from streamreplace.benchmarks.base import JudgePolicy, MetricSpec
from streamreplace.benchmarks.registry import get, module_path

from .helpers import BENCHMARKS, IMAGE, REPO

EXPECTED_COUNT = {"dynamath": 5010, "mathvista": 1000, "mmbench": 4329, "mme": 2374, "mmmu": 900, "mmstar": 1500,
                  "scienceqa": 2017, "visulogic": 1000, "videomme": 900, "tomato": 1484, "timeblind": 2400,
                  "videoreasonbench": 1440}
JUDGE = {**{name: JudgePolicy.FALLBACK for name in BENCHMARKS},
         "videomme": JudgePolicy.NONE,            # the official regex only ("no third-party models")
         "videoreasonbench": JudgePolicy.ALWAYS}  # the judge decides every reply that has a final answer
HEADLINE = {"dynamath": "avg_acc", "mmbench": "circular_acc", "mme": "perception", "timeblind": "I_Acc"}
PRIVATE = {"timeblind": {"role"}, "videoreasonbench": {"states", "coords", "moves"}}


@pytest.mark.parametrize("name", BENCHMARKS)
def test_module_constants(name):
    bench = get(name)
    group = "image" if name in IMAGE else "video"
    assert module_path(name) == f"streamreplace.benchmarks.{group}.{name}"
    assert (bench.NAME, bench.MEDIA, bench.JUDGE) == (name, group, JUDGE[name])
    assert bench.EXPECTED_COUNT == EXPECTED_COUNT[name] and not bench.LOWER_IS_BETTER
    assert bench.METRICS and all(isinstance(m, MetricSpec) for m in bench.METRICS)
    assert bench.METRICS[0].name == HEADLINE.get(name, "acc")
    assert set(bench.PRIVATE_METADATA) == PRIVATE.get(name, set())
    if name in IMAGE:                     # image benchmarks have no streaming condition: C-P contrasts only
        assert all(m.modes == ("offline",) for m in bench.METRICS)


def test_mme_cognition_subtasks():
    """The analysis splits MME's categories into perception and cognition with this set."""
    assert get("mme").COGNITION == {"commonsense_reasoning", "numerical_calculation", "text_translation",
                                    "code_reasoning"}


@pytest.mark.parametrize("name", BENCHMARKS)
def test_metric_units_match_the_analysis_profile(name):
    path = REPO / "configs" / "analysis.yaml"
    if not path.is_file():
        pytest.skip("configs/analysis.yaml is not present")
    entries = [m for m in yaml.safe_load(path.read_text(encoding="utf-8"))["metrics"] if m["dataset"] == name]
    defaults = MetricSpec("x", "acc")
    fields = ("name", "kind", "cluster", "group", "size", "split", "scale", "lower_is_better", "modes")

    def normal(entry: dict) -> tuple:
        values = {f: entry.get(f, getattr(defaults, f)) for f in fields}
        values["modes"], values["scale"] = tuple(values["modes"]), float(values["scale"])
        return tuple(values[f] for f in fields)

    assert [normal(vars(m)) for m in get(name).METRICS] == [normal(e) for e in entries]


@pytest.mark.parametrize("name", BENCHMARKS)
def test_every_benchmark_has_a_docs_page(name):
    page = REPO / "docs" / "benchmarks" / f"{name}.md"
    assert page.is_file() and page.read_text(encoding="utf-8").startswith("# ")
