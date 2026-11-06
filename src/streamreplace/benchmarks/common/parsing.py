"""Answer parsing shared by all benchmarks (every P/C/S score of the image sets, and the default parse/correct of any
benchmark that does not override them). Conservative: ambiguous answers are parse failures (the LLM judge may later
extract them); no Python eval, no guessing from long reasoning.

Ported unchanged from the paper's runs: the regexes, their order, the tolerances and the status strings all feed
paper numbers. clean() is the released rule (strip <think>...</think>, keep the text after an unterminated </think>,
drop special tokens), which every reported number uses except four ThinkStream cells of the supplement's master
tables (docs/reproducing-the-paper.md)."""
from __future__ import annotations

import math
import re

_SPECIAL = re.compile(r"<\|[^|>]*\|>|</?s>|<silent>|<response>|<\|silent\|>")
_THINK = re.compile(r"<think>.*?</think>", re.S)
# \uff1a is the full-width colon some models write after "answer" (an escape for the literal character).
_MARKER = re.compile(r"(?:final\s+answer(?:\s+is)?\s*[:\uff1a]?|the\s+(?:correct\s+|final\s+)?answer\s+is\s*[:\uff1a]?"
                     r"|\banswer\s*[:\uff1a])\s*(.+)", re.I)
_BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def clean(text: str) -> str:
    text = _THINK.sub("", str(text or ""))
    if "</think>" in text:                       # unterminated reasoning prefix
        text = text.rsplit("</think>", 1)[1]
    return _SPECIAL.sub("", text).strip()


def normalize(value) -> str:
    value = str(value).replace("−", "-").replace("–", "-").strip().casefold()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s*=\s*", "=", value)
    return value.strip(" \t\r\n.`\"'*")


def final_answer(text: str) -> str:
    """The explicit answer span: last \\boxed{}, else text after the last answer marker,
    else the whole response if it is a single line; '' when none applies."""
    text = clean(text)
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    markers = _MARKER.findall(text)
    if markers:
        return markers[-1].strip().strip("`*").strip()
    return text if "\n" not in text else ""


def parse_choice(text: str, choices: dict[str, str]) -> tuple[str | None, str]:
    """Letter or option text. Order matters: exact option text beats a leading article-like "A ..."."""
    letters = set(choices)
    candidate = final_answer(text) or clean(text).split("\n")[0].strip()
    if not candidate:
        return None, "no_marker"
    bare = candidate.strip("()[] .*")
    if bare in letters:                                               # "B", "(B)", "B."
        return bare, "letter"
    full = clean(text)                                                # multi-line options (code, posts)
    exact = [k for k, v in choices.items() if v and v.strip().rstrip(".") in {candidate.strip().rstrip("."), full.rstrip(".")}]
    if len(exact) == 1:
        return exact[0], "option_text"
    loose = [k for k, v in choices.items() if v and normalize(v) in {normalize(candidate), normalize(full)}]
    if len(loose) == 1:
        return loose[0], "option_text"
    marked = re.match(r"^\(?([A-Z])(?:[.:)]|\))\s*(.*)$", candidate, re.S)     # "B. text", "(B) text", "B: text"
    others = {m for m in re.findall(r"\(([A-Z])\)|(?<![A-Za-z])([A-Z])[.)](?=\s|$)", candidate) for m in m if m in letters}
    if marked and marked[1] in letters and not (len(candidate.split()) <= 6 and others - {marked[1]}):
        return marked[1], "letter"
    spaced = re.match(r"^([A-Z])\s+(.+)$", candidate, re.S)            # "B second" only if the rest is B's text
    if spaced and spaced[1] in letters and normalize(spaced[2]).startswith(normalize(choices[spaced[1]])[:20] or "\0"):
        return spaced[1], "letter"
    if len(others) == 1 and len(candidate.split()) <= 6:                # "I think (B)"
        return others.pop(), "letter_in_short_answer"
    return None, "ambiguous_choice"


def parse_yesno(text: str) -> tuple[str | None, str]:
    candidate = normalize(final_answer(text) or clean(text).split("\n")[0])
    words = re.findall(r"[a-z]+", candidate)
    found = {w for w in words if w in {"yes", "no"}}
    if words and words[0] in {"yes", "no"} and (len(words) > 6 or len(found) == 1):
        return words[0], "first_word"
    return (found.pop(), "single_word") if len(found) == 1 else (None, "no_yesno")


_NUMBER = r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def scalar(value) -> float | None:
    text = str(value).strip().replace("−", "-").replace("–", "-").replace("$", "").replace(" ", "")
    boxed = _BOXED.fullmatch(text)
    if boxed:
        text = boxed[1]
    text = re.sub(r"\\frac\{(" + _NUMBER + r")\}\{(" + _NUMBER + r")\}", r"\1/\2", text).replace(r"\pi", "π")
    text = text.rstrip("%°").replace(",", "") if re.fullmatch(_NUMBER + r"[%°]?", text) else text
    try:
        fraction = re.fullmatch(r"\(?(" + _NUMBER + r")/(" + _NUMBER + r")\)?", text)
        if fraction:
            result = float(fraction[1]) / float(fraction[2])
        elif "π" in text:
            match = re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)?)[*×]?π(?:/(" + _NUMBER + r"))?", text)
            if not match:
                return None
            coefficient = {"": 1, "+": 1, "-": -1}.get(match[1])
            result = (coefficient if coefficient is not None else float(match[1])) * math.pi
            result /= float(match[2]) if match[2] else 1
        elif re.fullmatch(_NUMBER, text):
            result = float(text)
        else:
            return None
        return result if math.isfinite(result) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def numeric_sequence(value) -> list[float] | None:
    parts = re.split(r"[,;]", str(value).strip().strip("[](){}"))
    if len(parts) < 2:
        return None
    values = [scalar(p) for p in parts]
    return values if all(v is not None for v in values) else None


def parse_number(text: str) -> tuple[str | None, str]:
    candidate = final_answer(text).rstrip(".")
    if not candidate:                    # multi-line reply without an answer marker: the judge fallback extracts it
        return None, "no_marker"
    if scalar(candidate) is not None or numeric_sequence(candidate) is not None:
        return candidate, "numeric"
    # "x = 3", "3 cm": accept a single number inside a short answer
    numbers = re.findall(_NUMBER, candidate)
    if len(numbers) == 1 and len(candidate.split()) <= 4:
        return numbers[0], "number_in_short_answer"
    return None, "invalid_numeric"


def equivalent(prediction: str, answer: str, tolerance: float = 1e-3, relative: float = 0.0) -> bool:
    first, second = scalar(prediction), scalar(answer)
    if first is not None and second is not None:
        return abs(first - second) <= max(tolerance, relative * abs(second)) + 1e-12
    first, second = numeric_sequence(prediction), numeric_sequence(answer)
    if first is not None and second is not None:
        return len(first) == len(second) and all(abs(a - b) <= tolerance + 1e-12 for a, b in zip(first, second))
    return normalize(prediction) == normalize(answer)


# ---------- defaults used by every benchmark that does not override them ----------

def parse(sample, text):
    if not clean(text):
        return None, "empty"
    if sample.kind == "mcq":
        return parse_choice(text, sample.choices)
    if sample.kind == "yesno":
        return parse_yesno(text)
    if sample.kind == "numeric":
        return parse_number(text)
    candidate = final_answer(text)
    return (candidate, "text") if candidate else (None, "no_marker")


def correct(sample, prediction) -> bool:
    if prediction is None:
        return False
    if sample.kind in {"mcq", "yesno"}:
        return normalize(prediction) == normalize(sample.answer)
    if sample.kind == "numeric":
        return equivalent(prediction, sample.answer, float(sample.metadata.get("tolerance", 1e-3)),
                          float(sample.metadata.get("relative_tolerance", 0.0)))
    return equivalent(prediction, sample.answer)
