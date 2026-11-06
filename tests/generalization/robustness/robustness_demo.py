"""A synthetic MMBench-like image benchmark for the prompt-robustness tests: 12 made-up questions, each asked in two
circular rotations of its options, in three categories (metadata l2). No data or media are needed; tests register it
with streamreplace.benchmarks.registry.register(NAME, "robustness_demo") and write its prepared samples directly."""
from __future__ import annotations

from streamreplace.core.schema import Sample

NAME = "rdemo"
MEDIA = "image"
JUDGE = "none"
POST_PROMPT = "\nAnswer with the option's letter from the given choices directly."
QUESTIONS = [
    ("What colour is the large ball lying on the grass?", ("red", "blue"), "shapes"),
    ("How many chairs stand around the kitchen table?", ("four", "six"), "counting"),
    ("Which object is placed on top of the bookshelf?", ("a lamp", "a clock"), "objects"),
    ("What shape does the window frame have in this picture?", ("round", "square"), "shapes"),
    ("How many birds are sitting on the wire?", ("two", "three"), "counting"),
    ("Which animal is sleeping under the wooden bench?", ("a dog", "a cat"), "objects"),
    ("What colour are the curtains hanging beside the door?", ("green", "yellow"), "shapes"),
    ("How many candles are burning on the birthday cake?", ("seven", "nine"), "counting"),
    ("Which tool is resting against the garden fence?", ("a rake", "a shovel"), "objects"),
    ("What pattern covers the surface of the carpet?", ("stripes", "dots"), "shapes"),
    ("How many boats are floating near the harbour wall?", ("five", "eight"), "counting"),
    ("Which fruit is lying inside the glass bowl?", ("an apple", "a pear"), "objects"),
]


def samples() -> list[Sample]:
    """Rotation 0 has the true answer under A, rotation 1 under B; ids: question number + 1000 * rotation."""
    out = []
    for q, (question, (right, wrong), category) in enumerate(QUESTIONS, 1):
        for rotation in (0, 1):
            choices = {"A": right, "B": wrong} if rotation == 0 else {"A": wrong, "B": right}
            out.append(Sample(str(rotation * 1000 + q), question, "A" if rotation == 0 else "B", "mcq", choices,
                              metadata={"question": q, "rotation": rotation, "rotations": 2, "l2": category,
                                        "hint": None, "raw_question": question}))
    return out


def prepare(raw, media) -> list[Sample]:
    return samples()


def prompt(sample) -> str:
    options = "\n".join(["There are several options:"] + [f"{k}. {v}" for k, v in sample.choices.items()])
    return f"{sample.metadata['raw_question']} {options}\n{POST_PROMPT}"


def parse(sample, text):
    letter = (text or "").strip()[:1]
    return (letter, "letter") if letter in sample.choices else (None, "no_letter")


def correct(sample, pred) -> bool:
    return pred == sample.answer
