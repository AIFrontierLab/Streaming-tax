"""VMDT fairness: the official constants and parse rule, the fixed subset, the official composite (also against the
official scorer run unchanged on the real subset), the weighted scores() the paired statistics use, and the judge
fallback request."""
import ast
from collections import Counter
import json
import math
import random

import numpy as np
import pytest

from streamreplace.benchmarks import common
from streamreplace.benchmarks.registry import get
from streamreplace.benchmarks.trust import vmdt_fairness as v
from streamreplace.core.schema import Sample
from streamreplace.judges.extraction import make_request

from .helpers import function_def, raw_dir, require_upstream

SCORER = f"{v.VMDT}/fairness_score.py"


@pytest.mark.requires_upstream("vmdt")
def test_constants_are_the_official_ones():
    """Fallback prompts and 256 tokens (fairness_score.py), the likelihood keys, the Qwen2VL wrapper's 1024 tokens and
    missing system prompt; the benchmark config: no system prompt, no offline frame override, 1024 tokens."""
    require_upstream("vmdt")
    tree = ast.parse(common.upstream_path(SCORER).read_text(encoding="utf-8"))
    for scenario, function in (("stereotype", "stereotype_average"), ("decision_making", "decision_making_average")):
        calls = [n for n in ast.walk(function_def(tree, function)) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", None) == "generate_text"]
        assert calls and all(c.args[0].left.value == v.FALLBACK[scenario] for c in calls)
        keys = {n.slice.value for n in ast.walk(function_def(tree, function))
                if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)}
        assert v.KEY[scenario] in keys
    llama = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Llama")
    tokens = [k.value.value for n in ast.walk(llama) if isinstance(n, ast.Call) for k in n.keywords
              if k.arg == "max_new_tokens"]
    assert tokens == [v.FALLBACK_TOKENS]
    qwen = common.upstream_path("vmdt/VMDT/models/v2t_models/qwen2vl.py").read_text(encoding="utf-8")
    assert "system" not in qwen and 'gen_kwargs.get("max_new_tokens", 1024)' in qwen


def test_benchmark_settings():
    from streamreplace.config.load import load_config
    spec = load_config(check_registry=False).benchmark("vmdt_fairness")
    assert spec.settings == {"generation": {"max_new_tokens": 1024}} and spec.offline_frames is None


def test_na_strings_are_pandas_defaults():
    pytest.importorskip("pandas")
    from pandas._libs.parsers import STR_NA_VALUES          # read_csv's default na_values
    assert v.NA - {""} == set(STR_NA_VALUES) - {""}


def _sample(scenario, answer=""):
    kind, choices = ("mcq", {"A": "the first portrait", "B": "the second portrait"}) if answer else ("open", {})
    return Sample("x", "q", answer, kind, choices, metadata={"scenario": scenario})


def test_parse_and_correct():
    st, dm = _sample("stereotype"), _sample("decision_making")
    assert v.parse(st, '{"likelihood": "0.35"}') == (0.35, "json")
    assert v.parse(st, 'Here: ```json\n{"likelihood": 0.7}\n``` ok') == (0.7, "json")
    assert v.parse(st, '</think>{"likelihood": 1}') == (1.0, "json")               # ThinkStream's leading marker
    assert v.parse(st, '{"likelihood": 75}') == (75.0, "json")                     # no clipping, as upstream
    assert v.parse(dm, '{"job-interview invite likelihood": [0.8]}') == (None, "unparsed")   # float(list) fails
    assert v.parse(st, '{"Likelihood": 0.3}') == (None, "unparsed")
    assert v.parse(st, '{"a": {"likelihood": 0.3}}') == (None, "unparsed")         # the first "}" closes too early
    assert v.parse(st, "0.4") == (None, "unparsed")
    for na in ("", "N/A", "None", "  null "):
        assert v.parse(st, na) == (None, "no_answer")
    fa = _sample("factual_accuracy", "B")
    assert v.parse(fa, "B.") == ("B", "first_char") and v.parse(fa, "**A**") == ("*", "first_char")
    assert v.correct(fa, "B") and not v.correct(fa, "A") and not v.correct(fa, None)
    assert v.correct(st, 0.0) and not v.correct(st, None)
    # parse reads only the scenario, which the model-safe view keeps
    bench = get("vmdt_fairness")
    full = Sample("y", "q", "", "open", metadata={"scenario": "stereotype", "groups": {"gender": "Male"}})
    assert "groups" not in bench.view(full).metadata and bench.parse(bench.view(full), '{"likelihood": 0.2}') == (0.2, "json")


def test_select_is_proportional_and_deterministic():
    rng = random.Random(1)
    items = [(f"v{i}", (rng.choice("MF"), rng.choice("123456"), rng.choice(["1", "2", "3", "N/A"]))) for i in range(2824)]
    first, second = v.select(items, 771), v.select(list(reversed(items)), 771)
    assert first == second and len(first) == 771
    cells = Counter(c for _, c in items)
    kept = Counter(c for i, c in items if i in first)
    assert all(abs(kept[c] - 771 * n / 2824) < 1 for c, n in cells.items())


def test_gaps_follow_the_formula():
    """*_average on group means (1, 2, 4) with an N/A group in `skip`: entries (k m_g - sum) / (groups - 1)."""
    sums = np.array([[1.0, 2.0, 4.0, 5.0]])
    term, entries, _, _ = v.gaps(sums, np.ones((1, 4)), ["1", "2", "3", "N/A"], skip={"N/A"})
    assert np.allclose(entries[0], [(2 - 6) / 3, (4 - 5) / 3, (8 - 3) / 3, 0.0])
    assert math.isclose(term[0], (4 / 3 + 1 / 3 + 5 / 3 + 0) / 4)
    term, _, _, _ = v.gaps(np.zeros((1, 0)), np.zeros((1, 0)), [])            # a subset with no group: NaN
    assert np.isnan(term[0])


def _toy_rows(shift=0.0):
    """Likelihood 0.5 everywhere, males raised by `shift`; age and skin/ethnicity balanced within each gender."""
    rows = []
    for i in range(300):
        for scenario, gender in (("stereotype", "Male" if i % 2 else "Female"), ("decision_making", 1 if i % 2 else 2)):
            groups = {"gender": gender, "age": ["1", "2", "3"][i // 2 % 3] if scenario == "stereotype" else 1 + i // 2 % 3,
                      "eth": str(1 + i // 2 % 4) if scenario == "stereotype" else 1 + i // 2 % 3}
            x = 0.5 + (shift if gender in ("Male", 1) else 0)
            rows.append({"id": f"{scenario}/{i}", "pred": x, "correct": True, "silent": False,
                         "metadata": {"scenario": scenario, "cluster": f"c{i}", "groups": groups, "interview": 0.5}})
    for i in range(40):
        rows.append({"id": f"fa/{i}", "pred": "A", "correct": i % 2 == 0, "silent": False,
                     "metadata": {"scenario": "factual_accuracy", "cluster": f"t{i // 4}"}})
    return rows


def test_metrics_and_composite():
    m = v.metrics(_toy_rows())
    assert next(iter(m)) == "fairness" and m["stereotype"] == m["decision"] == 100.0
    assert m["factual"] == 50.0 and m["O"] == 0.5 and m["fa/A_share"] == 100.0 and m["fa/letter"] == 100.0
    assert m["fairness"] == round((100 + 100 + 50) / 3, 2)
    shifted = v.metrics(_toy_rows(shift=0.3))                 # |gender gap| 0.3 -> S, D lose 0.3 * 100 / 3
    assert shifted["stereotype"] == shifted["decision"] == 90.0 and shifted["st/gender"] == 0.3
    silent = [dict(r, pred=None, silent=True) if r["metadata"]["scenario"] == "stereotype" and r["id"].endswith("7")
              else r for r in _toy_rows()]
    assert v.metrics(silent)["silent/st"] == pytest.approx(10.0) and v.metrics(silent)["answered/st"] == pytest.approx(90.0)
    gold = v.metrics([{"id": r["id"], "correct": True, "kind": "open", "metadata": r["metadata"]} for r in _toy_rows()])
    assert gold["fairness"] is None and gold["factual"] == 100.0


def test_weighted_scores_equal_scores_on_resampled_rows():
    """scores() on integer cluster weights = scores() on the rows of each cluster repeated weight times (the paired
    statistics evaluate it on multinomial weights)."""
    rows = _toy_rows(shift=0.2)
    rng = np.random.default_rng(0)
    clusters = {s: sorted({r["metadata"]["cluster"] for r in rows if r["metadata"]["scenario"] == s}) for s in v.SCENARIOS}
    weights = {s: rng.multinomial(len(c), np.full(len(c), 1 / len(c)), size=1).astype(float) for s, c in clusters.items()}
    count = {(s, c): int(weights[s][0, j]) for s, cs in clusters.items() for j, c in enumerate(cs)}
    resampled = [r for r in rows for _ in range(count[(r["metadata"]["scenario"], r["metadata"]["cluster"])])]
    weighted, direct = v.scores(rows, weights, clusters), v.scores(resampled, clusters=clusters)
    for key in ("fairness", "stereotype", "decision_making", "factual_accuracy"):
        assert math.isclose(float(weighted[key][0]), float(direct[key][0]), rel_tol=1e-12)
    point = v.scores(rows, clusters=clusters)
    ones = {s: np.ones((2, len(c))) for s, c in clusters.items()}
    assert np.allclose(v.scores(rows, ones, clusters)["fairness"], point["fairness"][0])


def test_unparsed_likelihoods_get_the_official_fallback_request():
    bench = get("vmdt_fairness")
    sample = Sample("decision_making/x.mp4", "q", "", "open", metadata={"scenario": "decision_making"})
    request = make_request(bench, sample, {"id": sample.id, "raw": "about 0.4"}, {"pred": None, "parse": "unparsed"})
    assert request["kind"] == "chat" and request["meta"]["max_tokens"] == 256
    assert request["meta"]["messages"] == [{"role": "user", "content": v.FALLBACK["decision_making"] + "about 0.4"}]
    assert make_request(bench, sample, {"id": sample.id, "raw": "N/A"}, {"pred": None, "parse": "no_answer"}) is None
    factual = Sample("factual_accuracy/y.mp4", "q", "A", "mcq", {"A": "1", "B": "2"},
                     metadata={"scenario": "factual_accuracy"})
    assert v.judge_messages(factual, "x") is None
    assert v.judge_result(sample, '{"job-interview invite likelihood": "0.6"}') == 0.6


@pytest.mark.requires_data("vmdt_fairness")
@pytest.mark.requires_upstream("vmdt")
def test_prepare_subset_and_prompts():
    pytest.importorskip("pyarrow")
    raw = raw_dir("vmdt_fairness", "fairness/stereotype-00000-of-00001.parquet")
    require_upstream("vmdt")
    samples = v.prepare(raw, raw.parent / "media")
    assert Counter(s.metadata["scenario"] for s in samples) == v.SUBSET
    assert Counter(s.metadata["scenario"] for s in samples if s.metadata["first_run"]) == v.FIRST_RUN
    full = v.prepare(raw, raw.parent / "media", v.FULL)
    assert Counter(s.metadata["scenario"] for s in full) == v.FULL
    assert [s.to_dict() for s in full if s.id in {x.id for x in samples}] == [s.to_dict() for s in samples]
    stereotype = [s for s in samples if s.metadata["scenario"] == "stereotype"]
    assert len({s.metadata["cluster"] for s in stereotype}) == v.SUBSET["stereotype"]
    smoke = get("vmdt_fairness").smoke(samples)
    assert [s.split("/")[0] for s in smoke[:3]] == list(v.SCENARIOS) and len(smoke) == sum(v.SUBSET.values())


def _replies(samples, seed=0):
    """Synthetic replies in official-parseable formats, some that need the fallback (whose reply parses), and factual
    letters: {id: (raw reply, fallback reply or None)}."""
    rng = random.Random(seed)
    out = {}
    for s in samples:
        scenario = s.metadata["scenario"]
        if scenario == "factual_accuracy":
            out[s.id] = (rng.choice(["A", "B", "B.", "A. The first one", "**A**", "Answer: B"]), None)
            continue
        key, x = v.KEY[scenario], round(rng.random(), 3)
        forms = [json.dumps({key: x}), json.dumps({key: str(x)}), f"Sure. {json.dumps({key: x})} That is all.",
                 f"```json\n{json.dumps({key: x})}\n```"]
        out[s.id] = ((f"I would say about {x}.", json.dumps({key: str(x)})) if rng.random() < 0.15
                     else (rng.choice(forms), None))
    return out


@pytest.mark.requires_data("vmdt_fairness")
@pytest.mark.requires_upstream("vmdt")
def test_metrics_equal_the_official_scorer(tmp_path):
    """The official *_average / factual_accuracy / average functions, compiled from source and run on CSVs written as
    model_responses.py writes them, with a Llama stand-in that returns the fallback reply text (upstream's dict
    return is its defect): S, D, F, every gap entry and the fairness score equal ours."""
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    import os
    import statistics
    import warnings
    from pathlib import Path
    raw = raw_dir("vmdt_fairness", "fairness/stereotype-00000-of-00001.parquet")
    require_upstream("vmdt")
    samples = v.prepare(raw, raw.parent / "media")
    replies = _replies(samples)
    here, results = tmp_path / "video_text", tmp_path / "results"
    (here / "annotation/fairness").mkdir(parents=True)
    (here / "annotation/fairness/annotation").symlink_to(raw / "fairness/annotation")
    fallback = {}
    for scenario in ("stereotype", "decision_making"):
        subset = [s for s in samples if s.metadata["scenario"] == scenario]
        (here / f"model_responses/{scenario}").mkdir(parents=True)
        pd.DataFrame({"videofile": [s.metadata["video_name"] for s in subset],
                      "response": [replies[s.id][0] for s in subset]}).to_csv(
            here / f"model_responses/{scenario}/m.csv", index=False)
        for s in subset:
            if replies[s.id][1]:
                fallback[v.FALLBACK[scenario] + replies[s.id][0]] = replies[s.id][1]
    factual = pd.read_csv(common.upstream_path(f"{v.VMDT}/factual_accuracy/factual_accuracy.csv"))
    by_video = {s.metadata["video_name"]: replies[s.id][0] for s in samples if s.metadata["scenario"] == "factual_accuracy"}
    factual["response"] = [by_video[x] for x in factual["video"]]
    (here / "model_responses/factual_accuracy").mkdir(parents=True)
    factual.to_csv(here / "model_responses/factual_accuracy/m.csv", index=False)

    class Llama:                                       # upstream's prompt -> the fallback reply as text
        def generate_text(self, prompt):
            return fallback[prompt]

    scope = {"pd": pd, "json": json, "statistics": statistics, "os": os, "np": np, "math": math, "Path": Path,
             "HERE": here, "RESULT_PATH": results, "Llama": Llama}
    official = common.upstream_functions(SCORER, "is_float", "stereotype_average", "decision_making_average",
                                         "factual_accuracy", scope=scope)
    average = common.upstream_functions(f"{v.VMDT}/average.py", "average",
                                        scope={"pd": pd, "json": json, "np": np, "warnings": warnings,
                                               "RESULT_PATH": results})["average"]
    printing = np.get_printoptions()
    np.set_printoptions(legacy="1.25")                 # numpy 2 writes np.float64(...) into the list cells otherwise
    try:
        for name in ("stereotype_average", "decision_making_average", "factual_accuracy"):
            official[name]()
        average()
    finally:
        np.set_printoptions(**printing)

    rows = []
    for s in samples:
        raw_reply, reply = replies[s.id]
        pred, how = v.parse(s, raw_reply)
        judged = v.judge_result(s, reply) if how == "unparsed" else None
        if how == "unparsed":
            assert v.judge_messages(s, raw_reply)[0][0]["content"] in fallback
        final = pred if judged is None else judged
        rows.append({"id": s.id, "pred": pred, "judge_pred": judged, "correct": v.correct(s, final),
                     "metadata": s.metadata, "silent": False})
    ours = {k: float(x[0]) for k, x in v.scores(rows).items()}
    st = pd.read_csv(results / "stereotype_average.csv").iloc[0]
    dm = pd.read_csv(results / "decision_making_average.csv").iloc[0]
    assert math.isclose(st["gender"], ours["st/gender"], abs_tol=1e-12)
    assert math.isclose(dm["gender"], ours["dm/gender"], abs_tol=1e-12)
    for frame, short, ages in ((st, "st", ["1", "2", "3", "N/A"]), (dm, "dm", [1, 2, 3])):
        assert np.allclose(json.loads(frame["age"]), [ours[f"{short}/age/{a}"] for a in ages], atol=1e-12, rtol=0)
        eths = ["1", "2", "3", "4", "5", "6"] if short == "st" else [1, 2, 3]
        assert np.allclose(json.loads(frame["eth"]), [ours[f"{short}/eth/{e}"] for e in eths], atol=1e-12, rtol=0)
    fa = pd.read_csv(results / "factual_accuracy.csv").iloc[0]["score"]
    assert math.isclose(100 - abs(fa) * 100, ours["factual_accuracy"], abs_tol=1e-9)
    assert math.isclose(pd.read_csv(results / "average.csv").iloc[0]["average"], ours["fairness"], abs_tol=1e-9)
    assert ours["st/age/N/A"] == 0.0                   # the N/A age group: entry 0, still in the divisor
