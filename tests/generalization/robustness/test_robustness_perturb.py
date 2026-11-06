"""Pure text perturbations of the prompt-robustness experiment (generalization/robustness/perturb.py)."""
from __future__ import annotations

import random
import re

import pytest

from streamreplace.generalization.robustness import perturb

# Outputs of the code of the paper's runs on made-up sentences (seeded with random.Random(str), i.e. SHA-512): any
# change of the draws or the operations changes the prompts of every typo condition.
GOLDEN_TYPOS = [
    ("How many people are standing near the old wooden table?", "0|demo|q1|typo", 0.2, 4,
     "How many people ared standinh near the old wooden table?"),
    ("What is the man holding in his right hand at the end of the clip?", "0|demo|q2|typo", 0.2, 4,
     "What is teh man holding in his right hand at th end of the clip?"),
    ("Answer with the option's letter from the given choices directly.", "0|demo|7|ins_typo", 0.2, 4,
     "Anwser with th option's letter from the given choices directly."),
    ("Which animal appears first in the video?", "s", 1.0, 4, "Wihch animal appoears firsr in the videl?"),
]


@pytest.mark.parametrize("text,seed,rate,max_words,expected", GOLDEN_TYPOS)
def test_typo_reproduces_the_original_draws(text, seed, rate, max_words, expected):
    assert perturb.typo(text, seed, rate, max_words) == expected


def test_typo_is_deterministic_bounded_and_spares_protected_tokens():
    text = 'How many "Stop Sign" boards near the TV in <image 1> show 25 cars, according to the narrator?'
    out = perturb.typo(text, "seed", rate=1.0, max_words=4)
    assert out == perturb.typo(text, "seed", rate=1.0, max_words=4) and out != text
    for kept in ['"Stop Sign"', "TV", "<image 1>", "25"]:
        assert kept in out
    words, new = re.findall(r"\S+", text), re.findall(r"\S+", out)
    assert len(words) == len(new) and 1 <= sum(a != b for a, b in zip(words, new)) <= 4
    assert sum(a != b for a, b in zip(words, re.findall(r"\S+", perturb.typo(text, "x", rate=0.01)))) == 1
    assert perturb.typo("Is it OK?", "s") == "Is it OK?"                 # nothing eligible: unchanged


def test_corrupt_keeps_first_letter_and_always_changes():
    rng = random.Random(0)
    for word in ["the", "video", "Running", "aaa", "Nottingham"]:
        for _ in range(200):
            out = perturb.corrupt(word, rng)
            assert out != word and out[0] == word[0] and abs(len(out) - len(word)) <= 1
    assert set(perturb.NEIGHBOURS["a"]) == set("qwsz") and set(perturb.NEIGHBOURS["s"]) == set("awedzx")
    assert set(perturb.NEIGHBOURS["p"]) == set("ol") and set(perturb.NEIGHBOURS["m"]) == set("njk")


def test_stress_sentence_and_append():
    assert perturb.STRESS_TEST == " and true is true " * 5
    assert perturb.append("Which animal appears first in the video?", perturb.STRESS_TEST) == (
        "Which animal appears first in the video? and true is true  and true is true  and true is true  "
        "and true is true  and true is true")
    assert perturb.append("What is shown?  ", " and true is true ") == "What is shown? and true is true  "


def test_irrelevant_pool_and_pick():
    pool = perturb.irrelevant_pool()
    assert len(pool) == 50 and len(set(pool)) == 50 and not any(re.search(r"\d", s) for s in pool)
    banned = {"before", "after", "first", "last", "next", "then", "left", "right", "video", "image", "man", "woman"}
    assert not any(banned & set(re.findall(r"[a-z]+", s.lower())) for s in pool)
    stem = "What is my friend planning for the weekend trip?"
    picked = perturb.pick_irrelevant(stem, "k", pool)
    assert picked in pool and not set(perturb.CONTENT.findall(picked.lower())) & set(perturb.CONTENT.findall(stem.lower()))
    assert picked == perturb.pick_irrelevant(stem, "k", pool)
    # the choices of the code of the paper's runs
    assert perturb.pick_irrelevant("Which animal appears first in the video?\nA. a dog\nB. a cat",
                                   "0|demo|q3|irrelevant", pool) == "My friend recommended a podcast about history."
    assert perturb.pick_irrelevant("What colour is the car?\nA. red\nB. blue\nAnswer with the option's letter from the "
                                   "given choices directly.", "0|demo|9|ins_irrelevant", pool) == (
        "I heard the museum has a new exhibition.")
    with pytest.raises(ValueError, match="every irrelevant sentence overlaps"):
        perturb.pick_irrelevant(" ".join(pool), "k", pool)


def test_paraphrase_rules():
    stem = "How many apples are on the table at the end of the video?"
    assert perturb.paraphrase_problems(stem, "At the end of the clip, how many apples sit on the table?") == []
    assert "too similar" in perturb.paraphrase_problems(stem, "How many apples are on the table at the end of the video ?")
    assert "numbers changed" in perturb.paraphrase_problems("Is 3 the answer?", "Is three the correct answer?")
    assert "length" in perturb.paraphrase_problems(stem, "Apples?")
    assert "adds the answer text" in perturb.paraphrase_problems(
        "What fruit is shown?", "Which fruit, a green apple, appears in the clip?", answer_text="green apple")
    assert perturb.paraphrase_problems(stem, "") == ["empty"]
    assert perturb.word_edit_ratio("How many apples are on the table?", "On the table, how many apples are there?") == 0.75
