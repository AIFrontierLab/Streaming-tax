"""HarmBench: the official direct-request test cases, the prompt actually sent, ASR metrics and the smoke order."""
from collections import Counter

import pytest

from streamreplace.benchmarks.registry import get
from streamreplace.benchmarks.safety import harmbench as hb
from streamreplace.core.schema import Sample

from .helpers import metric_row, prepared_samples


def behavior(functional, context="", behavior_id="b", tags=""):
    return {"BehaviorID": behavior_id, "Behavior": "do x", "ContextString": context, "FunctionalCategory": functional,
            "SemanticCategory": "chemical", "Tags": tags}


def test_official_test_case_strings():
    """DirectRequest / MultiModalDirectRequest: the behaviour, context + separator, or the behaviour plus one space."""
    assert hb.test_case(behavior("standard")) == "do x"
    assert hb.test_case(behavior("contextual", context="Some context")) == "Some context\n\n---\n\ndo x"
    assert hb.test_case(behavior("multimodal")) == "do x "


def test_the_sent_prompt_strips_the_multimodal_trailing_space():
    """The shared open-question prompt is question.strip(): the paper's runs sent the multimodal test cases without
    upstream's trailing space (kept for reproduction)."""
    bench = get("harmbench")
    sample = Sample("m", hb.test_case(behavior("multimodal")), "", "open", images=("x.png",),
                    metadata={"behavior": behavior("multimodal"), "functional": "multimodal", "semantic": "s"})
    assert bench.prompt(bench.view(sample)) == "do x"
    assert bench.JUDGE.value == "classifier" and bench.LOWER_IS_BETTER and bench.MEDIA == "image"


def test_asr_metrics():
    rows = [metric_row(f"{f}{i}", ok, {"functional": f, "semantic": s})
            for i, (f, s, ok) in enumerate([("standard", "a", True), ("standard", "b", False),
                                            ("contextual", "a", True), ("multimodal", "b", False)])]
    m = hb.metrics(rows)
    assert next(iter(m)) == "asr" and m["asr"] == 50.0 and m["text_asr"] == 33.33
    assert m["func/standard"] == 50.0 and m["func/copyright"] is None and m["func/multimodal"] == 100.0
    assert (m["sem/a"], m["sem/b"]) == (0.0, 100.0)
    assert hb.metrics([dict(r, correct=True) for r in rows])["asr"] == 0      # all safe: ASR 0


def test_smoke_is_round_robin_over_functional_categories():
    samples = [Sample(f"{c}-{i}", "q", "", "open", metadata={"functional": c}) for c in hb.CATEGORIES for i in range(3)]
    order = hb.smoke(samples)
    assert len(order) == 12 and [i.split("-")[0] for i in order[:4]] == list(hb.CATEGORIES)


@pytest.mark.requires_data("harmbench")
def test_prepared_behaviours_are_the_official_test_cases():
    samples = {s.id: s for s in prepared_samples("harmbench")}
    assert len(samples) == hb.EXPECTED_COUNT
    assert Counter(s.metadata["functional"] for s in samples.values()) == {
        "standard": 200, "contextual": 100, "copyright": 100, "multimodal": 110}
    stripped = 0
    for s in samples.values():
        b = s.metadata["behavior"]
        assert s.question == hb.test_case(b)
        assert len(s.images) == (b["FunctionalCategory"] == "multimodal")
        stripped += get("harmbench").prompt(s) != s.question
    assert stripped == 112          # 110 multimodal + 2 text test cases end in whitespace
