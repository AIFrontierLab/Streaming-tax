"""The video benchmarks' official scorers, read from the fetched upstream code (skipped when it is not fetched):
Video-MME's extract_characters_regex, TOMATO's pre_parser and judge prompt, TimeBlind's extract_answer and judge
prompt, VideoReasonBench's extract_final_answer. Expected values are what the code of the paper's runs returned for
the same inputs; the judge prompts and max_tokens are part of the judge cache keys. The rendered judge prompts are
compared by sha256: their upstream text has no licence and is not copied into this repository."""
from __future__ import annotations

import hashlib

import pytest

from streamreplace.benchmarks.registry import get
from streamreplace.core.schema import Sample

from .helpers import require_upstream


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.mark.requires_upstream("videomme")
@pytest.mark.parametrize("text, expected", [
    ("A", ("A", "official")),
    ("The best answer is B.", ("B", "official")),
    ("(C)", ("C", "official")),
    # more than 10 words and no capital A-D anywhere: no answer
    ("I think the answer is clearly not determinable from the given frames at all", (None, "official_unparsed")),
    # upstream takes the FIRST capital A-D anywhere, here the B of "Based" (kept: it produced the paper's numbers)
    ("Based on the video, the answer is D", ("B", "official")),
    ("<think>A</think>C", ("C", "official")),       # parsed after clean()
    ("", (None, "official_unparsed")),
    ("<|im_end|>", (None, "official_unparsed")),
])
def test_videomme_official_regex(text, expected):
    require_upstream("videomme")
    bench = get("videomme")
    sample = Sample("v", "q", "A", "mcq", {"A": "a", "B": "b", "C": "c", "D": "d"})
    assert bench.parse(bench.view(sample), text) == expected


def test_videomme_is_never_judged():
    assert get("videomme").JUDGE == "none" and not get("videomme").has("judge_messages")


TOMATO = Sample("count:x", "How many?", "B", "mcq", {"A": "1", "B": "2", "C": "3"})


@pytest.mark.requires_upstream("tomato")
@pytest.mark.parametrize("text, expected", [
    ("B", ("B", "official_pre_parser")),
    ("b", ("B", "official_pre_parser")),
    ("B: 2", ("B", "official_pre_parser")),
    ("Answer is:(C)", ("C", "official_pre_parser")),
    ("Answer is (C)", (None, "official_unparsed")),        # upstream's regex allows no space before "("
    ("It is two.", (None, "official_unparsed")),
    ("<think>no</think> A", ("A", "official_pre_parser")),
    ("", (None, "official_unparsed")),
])
def test_tomato_official_pre_parser(text, expected):
    require_upstream("tomato")
    bench = get("tomato")
    assert bench.parse(bench.view(TOMATO), text) == expected


@pytest.mark.requires_upstream("tomato")
def test_tomato_judge_prompt_and_seeded_random_fallback():
    require_upstream("tomato")
    bench = get("tomato")
    messages, max_tokens = bench.judge_messages(TOMATO, "It is two.")
    assert max_tokens == 16 and [m["role"] for m in messages] == ["user"]
    # parse_result.py's gpt_parser prompt with the response, all_choices and index2answer filled in
    content = messages[0]["content"]
    assert "response: It is two.\n" in content and "all_choices: ['A', 'B', 'C']" in content
    assert "index2answer: {'A': '1', 'B': '2', 'C': '3'}" in content
    assert sha256(content) == "dc6b4239354ebf89a4d189bc142469c777a19b08e35d2ddfe13bdc6b4170f02e"
    assert bench.judge_result(TOMATO, "C") == "C"
    # a reply outside all_choices: random.Random(int(sha256("tomato-random:<id>")[:8], 16)).choice(all_choices)
    assert bench.judge_result(TOMATO, "two") == bench.judge_result(TOMATO, "D") == "B"
    other = Sample("rotation:0001-01", "Which way?", "A", "mcq", {"A": "x", "B": "y", "C": "z"})
    assert bench.judge_result(other, "none") == "A"


YESNO = Sample("7", "Is it?", "no", "yesno")
CHOICE = Sample("8", "Which? Option A: x Option B: y", "A", "mcq", {"A": "x", "B": "y"})


@pytest.mark.requires_upstream("timeblind")
@pytest.mark.parametrize("sample, text, expected", [
    (YESNO, "Yes.", ("yes", "official")),
    (YESNO, "No, it does not.", ("no", "official")),
    (YESNO, "Final answer: yes", ("yes", "official")),
    (YESNO, "maybe", (None, "official_invalid")),
    (YESNO, "yes or no", ("yes", "official")),
    (CHOICE, "A", ("A", "official")),
    (CHOICE, "(B)", ("B", "official")),
    (CHOICE, "Option B", ("B", "official")),
    (CHOICE, "I choose the second one", (None, "official_invalid")),
    (CHOICE, "A man walks in", ("A", "official")),           # upstream reads a standalone capital A as the answer
    (CHOICE, "the boy", (None, "official_invalid")),         # option text is not read (the judge may extract it)
    (CHOICE, "", (None, "official_invalid")),
], ids=lambda value: value.id if isinstance(value, Sample) else None)
def test_timeblind_official_extractor(sample, text, expected):
    require_upstream("timeblind")
    bench = get("timeblind")
    assert bench.parse(bench.view(sample), text) == expected


@pytest.mark.requires_upstream("timeblind")
def test_timeblind_judge_prompt_and_reply():
    require_upstream("timeblind")
    bench = get("timeblind")
    messages, max_tokens = bench.judge_messages(YESNO, "hmm")
    assert max_tokens == 10 and [m["role"] for m in messages] == ["system", "user"]
    # llm_judge.py's system message and its yes/no extraction prompt, the model output at the end
    assert [sha256(m["content"]) for m in messages] == [
        "70104c2bccd8e5e27632fe6d5f26b0bb7f36249493f099b037c2558b318009b8",
        "cfb3360413f7e8acea9a0562ab29d55ad3b4baf1b96a23a6689c7e96b464abd0"]
    assert messages[1]["content"].endswith("\nhmm\n")
    choice = bench.judge_messages(CHOICE, "hmm")[0]                   # the A/B extraction prompt
    assert choice[0] == messages[0] and choice[1]["content"].endswith("\nhmm\n")
    assert sha256(choice[1]["content"]) == "8120dc81f2b5959f1ed603ea9588f11a06cbcbafcf91f3774ca08f957c1275e2"
    assert [bench.judge_result(YESNO, r) for r in ("no", "unsure")] == ["no", None]
    assert [bench.judge_result(CHOICE, r) for r in ("B", "b", "C")] == ["B", "B", None]


@pytest.mark.requires_upstream("videoreasonbench")
@pytest.mark.parametrize("text, expected", [
    # the FIRST "Final Answer", colon optional and kept, through the end of the reply
    ("reasoning\nFinal Answer: B, then C", (": B, then C", "final_answer_marker")),
    ("Final Answer 3 moves", ("3 moves", "final_answer_marker")),
    ("Final Answer: x\nmore\nFinal Answer: y", (": x\nmore\nFinal Answer: y", "final_answer_marker")),
    ("1st: 10, cup0", (None, "no_final_answer")),
    ("", (None, "no_final_answer")),
    ("final answer: lower", (None, "no_final_answer")),          # case-sensitive
    ("<think>Final Answer: no</think>Final Answer: yes", (": yes", "final_answer_marker")),
])
def test_videoreasonbench_official_final_answer(text, expected):
    require_upstream("videoreasonbench")
    assert get("videoreasonbench").parse(None, text) == expected
