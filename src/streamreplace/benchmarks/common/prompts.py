"""Prompt rendering shared by the benchmarks that do not define their own (ported unchanged).

The prompt text is part of the request identity: for every benchmark x model x mode the rendered prompt must be
byte-identical to the one recorded in the original outputs, and P, C and S send the same text."""
from __future__ import annotations

MCQ = "Answer with the option's letter from the given choices directly."
YESNO = "Answer with yes or no."
NUMERIC = "Answer with a single number."
TEXT = "Answer the question using a single word or phrase."


def options_block(choices: dict[str, str]) -> str:
    return "\n".join(f"{k}. {v}" for k, v in choices.items())


def prompt(sample) -> str:
    q = sample.question.strip()
    if sample.kind == "mcq":
        return f"{q}\n{options_block(sample.choices)}\n{MCQ}"
    if sample.kind == "yesno":
        return q if "yes or no" in q.lower() else f"{q}\n{YESNO}"
    if sample.kind == "numeric":
        return f"{q}\n{NUMERIC}"
    if sample.kind == "text":
        return f"{q}\n{TEXT}"
    return q
