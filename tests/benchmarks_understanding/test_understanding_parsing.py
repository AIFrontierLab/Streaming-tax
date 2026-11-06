"""Benchmark-specific parsing and correctness rules (no data, no upstream code). Expected values are what the code
of the paper's runs returned for the same inputs; the common parsers (clean, parse_choice, parse_number, ...) are
tested with benchmarks/common."""
from __future__ import annotations

import pytest

from streamreplace.benchmarks.registry import get
from streamreplace.benchmarks.image.dynamath import short_answer, split_choices, stable_id
from streamreplace.core.schema import Sample

NUMERIC = Sample("d1", "q", "2.0944", "numeric")
MCQ = Sample("d2", "q", "B", "mcq", {"A": "1", "B": "2", "C": "3"})
TEXT = Sample("d3", "q", "Hello World", "text")


def test_dynamath_ids_and_choices_are_stable():
    assert stable_id("dynamath", "sample_variant1/default", "6:1") == "4632251fe0dcaf354fd88f5b"
    assert split_choices("Question Choices: (A) one (B) two") == ("Question", {"A": "one", "B": "two"})
    assert split_choices("Pick one\nA. red\nB. blue\n") == ("Pick one", {"A": "red", "B": "blue"})
    with pytest.raises(ValueError):
        split_choices("(B) one (A) two")              # labels must run A, B, ...


@pytest.mark.parametrize("sample, text, parsed, correct", [
    (NUMERIC, '{"solution": "period = 2pi/3", "short answer": "2.094"}', ("2.094", "json_numeric"), True),
    # VLMEvalKit's preprocessing drops every backslash before json.loads (LaTeX in the solution)
    (NUMERIC, '```json\n{"solution": "use \\\\frac{2\\\\pi}{3}", "short answer": "2.094"}\n```',
     ("2.094", "json_numeric"), True),
    # invalid JSON: the last "short answer" field is read with a regex
    (NUMERIC, '{"solution": "x = "quoted" y", "short answer": "2.0944"}', ("2.0944", "json_numeric"), True),
    (NUMERIC, "The answer is 2.1", ("2.1", "numeric"), False),          # no JSON: the common rules
    (NUMERIC, '{"solution": "...", "short answer": ""}\nFinal answer: 2.094', ("2.094", "numeric"), True),
    (MCQ, '{"solution": "...", "short answer": "B"}', ("B", "json_letter"), True),
    (MCQ, '{"solution": "...", "short answer": "(C) 3"}', ("C", "json_letter"), False),
    # free-form answers: the prediction must be a (case-insensitive) substring of the answer, as upstream
    (TEXT, '{"solution": "...", "short answer": "world"}', ("world", "json_text"), True),
    (TEXT, '{"solution": "...", "short answer": "planet"}', ("planet", "json_text"), False),
    (TEXT, "no json here", ("no json here", "text"), False),
])
def test_dynamath_reads_the_json_short_answer(sample, text, parsed, correct):
    bench = get("dynamath")
    assert bench.parse(bench.view(sample), text) == parsed
    assert bench.correct(sample, parsed[0]) is correct


def test_dynamath_short_answer_accepts_non_string_values():
    assert short_answer('{"short answer": 3.5}') == "3.5"
    assert short_answer("no braces") is None


@pytest.mark.parametrize("answer, prediction, correct", [
    ("B", "B", True), ("B", "b", False),                          # multiple choice: the exact letter
    ("['$MgS$', 'MgS']", "MgS", True), ("['$MgS$', 'MgS']", "$MgS$", True),   # a list of accepted forms
    ("100", "100.9", True), ("100", "102", False),                # 1% relative tolerance
    ("$5", "5", True),                                            # dollar signs are ignored
    ("[1, 2", "[1, 2", True),                                     # an unparseable list is compared as is
])
def test_mmmu_correctness(answer, prediction, correct):
    kind, choices = ("mcq", {"A": "x", "B": "y"}) if answer == "B" else ("text", {})
    assert get("mmmu").correct(Sample("m", "q", answer, kind, choices), prediction) is correct


def test_videoreasonbench_rule_fallback_is_exact_after_normalisation():
    bench = get("videoreasonbench")
    sample = Sample("o", "q", "Yes", "open")
    assert [bench.correct(sample, p) for p in ("yes", "Yes.", None, "no")] == [True, True, False, False]
    assert bench.correct(Sample("o", "q", "", "open"), "x") is False      # operation questions have no answer text


def test_default_hooks_fill_in_for_modules_that_do_not_define_them():
    """MME and ScienceQA use the common prompt / parse / correct, as in the paper's runs."""
    mme, scienceqa = get("mme"), get("scienceqa")
    yesno = Sample("x#1", "Is it red? Please answer yes or no.", "yes", "yesno")
    assert mme.parse(mme.view(yesno), "Yes, it is.") == ("yes", "first_word")
    assert mme.correct(yesno, "yes") is True
    mcq = Sample("00000-1", "Which?", "A", "mcq", {"A": "iron", "B": "wood"})
    assert scienceqa.parse(scienceqa.view(mcq), "iron") == ("A", "option_text")
