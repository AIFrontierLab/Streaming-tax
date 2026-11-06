"""The shared benchmark rules (benchmarks/common): answer parsing tables, the default prompt, metric helpers and the
deterministic smoke order. These rules score every benchmark that does not override them, so each expected value
here is what the code of the paper's runs returned."""
from __future__ import annotations

import hashlib

import pytest

from streamreplace.benchmarks import common
from streamreplace.benchmarks.common.parsing import clean, final_answer, parse_choice, parse_number, parse_yesno
from streamreplace.core.schema import Sample

CHOICES = {"A": "red", "B": "blue", "C": "green"}


@pytest.mark.parametrize("text,expected", [
    ("A", "A"), ("(B)", "B"), ("C. green", "C"), ("The answer is B.", "B"), ("Final answer: C", "C"),
    (r"\boxed{A}", "A"), ("blue", "B"), ("B\n\nBecause the sky is blue.", "B"),
    ("<think>maybe A</think>The answer is C", "C"), ("Answer: (A)", "A"),
    ("A or B", None), ("I cannot answer this question.", None), ("", None),
    ("I think (B)", "B"), ("B blue", "B"), ("A red", "A"), ("A blue sky", None), ("(b)", None),
])
def test_choice_parsing(text, expected):
    assert parse_choice(text, CHOICES)[0] == expected


@pytest.mark.parametrize("text,status", [
    ("B", "letter"), ("blue", "option_text"), ("I think (B)", "letter_in_short_answer"), ("A or B", "ambiguous_choice"),
    ("", "no_marker"),
])
def test_choice_parse_status_strings(text, status):
    assert parse_choice(text, CHOICES)[1] == status


@pytest.mark.parametrize("text,expected", [
    ("Yes", "yes"), ("No, it is not.", "no"), ("The answer is yes.", "yes"), ("yes or no", None), ("maybe", None)])
def test_yesno_parsing(text, expected):
    assert parse_yesno(text)[0] == expected


@pytest.mark.parametrize("text,answer,correct", [
    ("Final answer: 1.001", "1", True), ("1.0011", "1", False), ("1/2", "0.5", True),
    (r"\frac{1}{2}", "0.5", True), (r"2\pi", "6.2832", True), ("[1, 2]", "(1, 2)", True),
    ("[2, 1]", "(1, 2)", False), ("1e-3", "0.001", True), ("x = 3", "3", True), ("12%", "12", True),
    ("-0.5<|im_end|>", "-0.5", True), ("1,234", "1234", True),
])
def test_numeric_scoring(text, answer, correct):
    sample = Sample("s", "q", answer, "numeric")
    assert common.correct(sample, common.parse(sample, text)[0]) is correct


@pytest.mark.parametrize("text", ["", "NaN", "inf", "1/0", "I counted 4 then 5 then 6", "5\n6"])
def test_numeric_is_not_guessed(text):
    assert parse_number(text)[0] is None


def test_numeric_tolerances_come_from_the_sample():
    loose = Sample("s", "q", "100", "numeric", metadata={"relative_tolerance": 0.01})
    assert common.correct(loose, "100.9") and not common.correct(loose, "102")
    assert not common.correct(Sample("s", "q", "1", "numeric", metadata={"tolerance": 0}), "1.0001")


def test_clean_and_final_answer():
    assert clean("<think>a</think> <|im_end|>B<|endoftext|>") == "B"
    assert clean("reasoning cut here</think>C") == "C"                 # an unterminated reasoning prefix
    assert clean("<silent><response>D</s>") == "D"
    assert final_answer(r"so \boxed{1} then \boxed{\frac{1}{2}}") == r"\frac{1}{2}"
    assert final_answer("line one\nThe final answer is: 42") == "42"
    assert final_answer("two\nlines") == "" and final_answer("one line") == "one line"


def test_default_parse_of_each_kind():
    assert common.parse(Sample("s", "q", "B", "mcq", CHOICES), "  ") == (None, "empty")
    assert common.parse(Sample("s", "q", "yes", "yesno"), "Yes.") == ("yes", "first_word")
    assert common.parse(Sample("s", "q", "Paris", "text"), "Answer: Paris") == ("Paris", "text")
    assert common.parse(Sample("s", "q", "x", "open"), "line\nline") == (None, "no_marker")
    assert not common.correct(Sample("s", "q", "B", "mcq", CHOICES), None)


def test_default_prompts_and_reference_never_shown():
    sample = Sample("s", "Which?", "B", "mcq", {"A": "x", "B": "y"}, metadata={"secret": "zzz"})
    text = common.prompt(sample)
    assert text == "Which?\nA. x\nB. y\nAnswer with the option's letter from the given choices directly."
    assert "zzz" not in text and text.endswith(common.MCQ)
    assert common.prompt(Sample("s", " Is it red? ", "yes", "yesno")) == "Is it red?\nAnswer with yes or no."
    assert common.prompt(Sample("s", "Red, yes or no?", "yes", "yesno")) == "Red, yes or no?"
    assert common.prompt(Sample("s", "How many?", "3", "numeric")).endswith("\n" + common.NUMERIC)
    assert common.prompt(Sample("s", "Name it.", "x", "text")).endswith("\n" + common.TEXT)
    assert common.prompt(Sample("s", "Describe.", "x", "open")) == "Describe."


def test_grouped_metrics_skip_incomplete_groups():
    rows = [{"correct": True, "metadata": {"g": "a"}}] * 2 + [{"correct": True, "metadata": {"g": "b"}}]
    assert common.group_all(rows, "g", size=2) == (100.0, 1)
    assert common.group_all(rows, "g") == (100.0, 2)


def test_metric_helpers_round_to_two_decimals():
    rows = [{"correct": c, "metadata": {"category": k}} for c, k in
            [(True, "x"), (False, "x"), (True, "x"), (True, "y")]]
    assert common.pct([True, False, False]) == 33.33 and common.pct([]) is None
    assert common.metrics(rows) == {"acc": 75.0, "cat/x": 66.67, "cat/y": 100.0}
    assert list(common.metrics([{"correct": False, "metadata": {}}])) == ["acc"]
    assert common.by(rows, "missing") == {"unknown": 75.0}


def test_smoke_order_is_deterministic_and_keeps_groups_whole():
    samples = [Sample(f"q{i}-{r}", "q", "A", "mcq", {"A": "a"}, metadata={"question": i}) for i in range(40)
               for r in range(3)]
    order = common.smoke(samples, group="question", n=10)
    keys = sorted((str(i) for i in range(40)), key=lambda k: hashlib.sha256(k.encode()).hexdigest())[:4]
    assert order == [f"q{k}-{r}" for k in keys for r in range(3)]       # whole groups until about n ids are chosen
    shuffled = common.smoke(list(reversed(samples)), group="question", n=10)
    assert [i.split("-")[0] for i in shuffled] == [i.split("-")[0] for i in order]
    singles = common.smoke(samples[:5], n=50)
    assert singles == sorted((s.id for s in samples[:5]), key=lambda k: hashlib.sha256(k.encode()).hexdigest())
