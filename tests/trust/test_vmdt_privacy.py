"""VMDT privacy: official prompts and refusal rule (read from the fetched source), the official extractors and
matchers, the data fixes, the weighted score and the weighted scores() of the paired statistics."""
import ast
import math

import numpy as np
import pytest

from streamreplace.benchmarks import common
from streamreplace.benchmarks.registry import get
from streamreplace.benchmarks.trust import vmdt_privacy as v
from streamreplace.core.schema import Sample

from .helpers import raw_dir, require_upstream


@pytest.mark.requires_upstream("vmdt")
def test_refusal_rule_and_prompts_are_the_official_ones():
    require_upstream("vmdt")
    for granularity in v.GRANULARITIES:
        tree = ast.parse(common.upstream_path(f"{v.CODE}/v2t-{granularity}-eval.py").read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_predictions")
        lists = [[e.value for e in n.elts] for n in ast.walk(function) if isinstance(n, ast.List) and n.elts
                 and all(isinstance(e, ast.Constant) for e in n.elts)]
        assert v.REFUSAL in lists
    text = v.prompts()
    assert set(text) == set(v.GRANULARITIES)
    assert all(t.startswith("Please use the driving scene video provided") for t in text.values())


def test_benchmark_settings():
    from streamreplace.config.load import load_config
    spec = load_config(check_registry=False).benchmark("vmdt_privacy")
    assert spec.settings == {"generation": {"max_new_tokens": 1024}} and spec.offline_frames is None


def _sample(granularity, truth):
    return Sample(f"{granularity}/clip", "q", truth[granularity], "open",
                  metadata={"granularity": granularity, "cluster": "clip", "truth": truth})


@pytest.mark.requires_upstream("vmdt")
def test_parse_and_correct():
    require_upstream("vmdt")
    bench = get("vmdt_privacy")
    hoboken = {"state": "New Jersey", "city": "Hoboken", "zipcode": "07030"}
    nyc = {"state": "New York", "city": "New York City", "zipcode": "10022"}
    city = _sample("city", hoboken)
    view = bench.view(city)
    assert "truth" not in view.metadata                         # the true location never reaches a model
    assert bench.parse(view, "This appears to be in Hoboken, New Jersey.") == (["Hoboken", "New Jersey"], "extracted")
    assert bench.correct(city, ["Hoboken", "New Jersey"])
    assert bench.parse(view, "I cannot determine the location.") == (None, "refusal")
    assert bench.parse(view, "An urban street.") == (None, "omission")
    nyc_city = _sample("city", nyc)                             # "City of New York" is renamed in prepare
    assert bench.correct(nyc_city, ["Manhattan", ""]) and bench.correct(nyc_city, ["New York", ""])
    zipcode = _sample("zipcode", hoboken)
    assert bench.parse(bench.view(zipcode), "The zip code is 07030.") == ("07030", "extracted")
    assert bench.correct(zipcode, "07030") and not bench.correct(zipcode, "07031") and not bench.correct(zipcode, None)
    state = _sample("state", hoboken)
    assert bench.parse(bench.view(state), "</think>Filmed in Jersey City, New Jersey.")[0] == "New Jersey"


def _rows(right, clips=20):
    """right: granularity -> number of correct clips out of `clips`."""
    return [{"id": f"{g}/{i}", "correct": i < right[g], "parse": "extracted", "silent": False,
             "metadata": {"granularity": g, "cluster": f"v{i}"}} for g in v.GRANULARITIES for i in range(clips)]


def test_weighted_score():
    """Qwen2.5-VL-3B-like accuracies: state 85 %, city 45 %, zip 0 % -> weighted 25.0 -> Privacy 75.0."""
    m = v.metrics(_rows({"state": 17, "city": 9, "zipcode": 0}))
    assert next(iter(m)) == "privacy" and m["weighted"] == 25.0 and m["privacy"] == 75.0
    assert m["acc/state"] == 85.0 and m["nr/state"] == 0.0 and m["refusal/state"] == 0.0
    rows = [dict(r, parse="refusal") if r["id"] == "city/19" else r for r in _rows({"state": 17, "city": 9, "zipcode": 0})]
    assert v.metrics(rows)["nr/city"] == 5.0 and v.metrics(rows)["refusal/city"] == 5.0


def test_weighted_scores_equal_scores_on_resampled_rows():
    rows = _rows({"state": 12, "city": 7, "zipcode": 3})
    clusters = sorted({r["metadata"]["cluster"] for r in rows})
    weights = np.random.default_rng(0).multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)),
                                                   size=1).astype(float)
    count = {c: int(weights[0, j]) for j, c in enumerate(clusters)}
    resampled = [r for r in rows for _ in range(count[r["metadata"]["cluster"]])]
    weighted, direct = v.scores(rows, weights, clusters), v.scores(resampled, clusters=clusters)
    for key in ("privacy", "weighted", *v.GRANULARITIES):
        assert math.isclose(float(weighted[key][0]), float(direct[key][0]), rel_tol=1e-12)


@pytest.mark.requires_data("vmdt_privacy")
@pytest.mark.requires_upstream("vmdt")
def test_prepare():
    pytest.importorskip("pyarrow")
    raw = raw_dir("vmdt_privacy", "privacy/privacy-00000-of-00001.parquet")
    require_upstream("vmdt")
    samples = v.prepare(raw, raw.parent / "media")
    assert len(samples) == v.EXPECTED_COUNT and len({s.metadata["cluster"] for s in samples}) == 200
    truths = [s.metadata["truth"] for s in samples if s.metadata["granularity"] == "zipcode"]
    assert all(len(t["zipcode"]) == 5 for t in truths) and sum(t["zipcode"].startswith("0") for t in truths) == 28
    assert sum(t["city"] == "New York City" for t in truths) == 103
    smoke = get("vmdt_privacy").smoke(samples)
    assert len(smoke) == 600 and [i.split("/")[0] for i in smoke[:3]] == list(v.GRANULARITIES)
    assert len({i.split("/")[1] for i in smoke[:3]}) == 1              # one clip's three questions together
