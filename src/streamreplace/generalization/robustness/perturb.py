"""Pure text perturbations of the prompt-robustness experiment (Supp. Tables eval_robustness / eval_instruction and
the prompt-robustness columns of Table 2). No I/O except reading irrelevant.txt; every function is deterministic in
its seed string.

typo             DeepWordBug's four character operations (swap adjacent letters, substitute, delete, insert; Gao et
                 al. 2018, the character level of PromptRobust, Zhu et al. 2023) on randomly drawn words. The official
                 recipes rank words by querying the model, which would give each model a different prompt (no
                 parent/child pairing) and would add TextAttack for a few lines of string code, so words are drawn at
                 random and the operations are written here. Substituted / inserted letters are QWERTY neighbours
                 (keyboard slips); the first letter of a word is never touched.
STRESS_TEST      the fixed sentence of PromptBench's StressTest attack that the `stress` condition appends.
pick_irrelevant  a natural, topic-irrelevant sentence from irrelevant.txt sharing no content word with the item.
append           where PromptBench inserts sentence-level text: after the last word of the question stem.
paraphrase_problems  the rule checks a paraphrase must pass before the LLM equivalence check (paraphrase.py).

random.Random(str) seeds through SHA-512, so the draws repeat on every machine and Python version; the edits are then
frozen in each condition folder's prompt_overrides.jsonl, so those files, not this code, define what the models saw."""
from __future__ import annotations

import hashlib
from pathlib import Path
import random
import re

# The sentence the `stress` condition appends: the third of the three fixed texts of PromptBench's
# StressTestTransformation, copied verbatim from microsoft/promptbench @ fcda538bd779ad11612818e0645a387a462b5c3b
# (promptbench/prompt_attack/transformations.py). PromptBench: MIT License, Copyright (c) Microsoft Corporation.
STRESS_TEST = " and true is true " * 5
IRRELEVANT = Path(__file__).with_name("irrelevant.txt")
ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
OPERATIONS = ("swap", "substitute", "delete", "insert")
WORD = re.compile(r"[A-Za-z]+")
PROTECTED = re.compile(r'"[^"]*"|“[^”]*”|<[^>]*>|\S*\d\S*')       # quoted text, <tags>, tokens with digits
CONTENT = re.compile(r"[a-z]{4,}")
TOKEN = re.compile(r"[a-z0-9]+")
NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _neighbours() -> dict[str, str]:
    """Keys touching each letter on a QWERTY keyboard (each row sits half a key right of the one above)."""
    where = {ch: (r, c) for r, row in enumerate(ROWS) for c, ch in enumerate(row)}
    near = lambda r, c: [(r, c - 1), (r, c + 1), (r - 1, c), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c)]
    return {ch: "".join(ROWS[i][j] for i, j in near(r, c) if 0 <= i < len(ROWS) and 0 <= j < len(ROWS[i]))
            for ch, (r, c) in where.items()}


NEIGHBOURS = _neighbours()


def typo(text: str, seed: str, rate: float = 0.2, max_words: int = 4) -> str:
    """round(rate x eligible words), clipped to [1, max_words], each get one character operation. Eligible: runs of
    >= 3 ASCII letters, not all capitals (acronyms, option letters), outside quotes, <tags> and tokens with digits.
    Returns the text unchanged when no word is eligible."""
    rng = random.Random(seed)
    blocked = [m.span() for m in PROTECTED.finditer(text)]
    words = [m for m in WORD.finditer(text) if len(m.group()) >= 3 and not m.group().isupper()
             and not any(a < m.end() and m.start() < b for a, b in blocked)]
    if not words:
        return text
    n = min(max_words, max(1, round(rate * len(words))))
    parts, last = [], 0
    for m in sorted(rng.sample(words, n), key=lambda m: m.start()):
        parts += [text[last:m.start()], corrupt(m.group(), rng)]
        last = m.end()
    return "".join(parts) + text[last:]


def corrupt(word: str, rng: random.Random) -> str:
    """One operation at a position after the first letter; the result always differs from `word` (len >= 3)."""
    op = rng.choice(OPERATIONS)
    if op == "swap":
        i = rng.randrange(1, len(word) - 1)
        if word[i] != word[i + 1]:
            return word[:i] + word[i + 1] + word[i] + word[i + 2:]
        op = "substitute"                                    # swapping two equal letters changes nothing
    if op == "delete":
        i = rng.randrange(1, len(word))
        return word[:i] + word[i + 1:]
    if op == "insert":                                       # a neighbour of the letter before the gap
        i = rng.randrange(1, len(word) + 1)
        return word[:i] + _like(rng.choice(NEIGHBOURS[word[i - 1].lower()]), word[i - 1]) + word[i:]
    i = rng.randrange(1, len(word))
    return word[:i] + _like(rng.choice(NEIGHBOURS[word[i].lower()]), word[i]) + word[i + 1:]


def _like(letter: str, model: str) -> str:
    return letter.upper() if model.isupper() else letter


def append(text: str, sentence: str) -> str:
    """`sentence` after the last word of `text` (where PromptBench inserts sentence-level text); trailing whitespace of
    `text` is kept after it."""
    body = text.rstrip()
    return f"{body} {sentence.strip()}{text[len(body):]}"


def irrelevant_pool() -> list[str]:
    """The 50 distractor sentences of irrelevant.txt (comment lines skipped)."""
    return [line.strip() for line in IRRELEVANT.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]


def pick_irrelevant(avoid: str, seed: str, pool: list[str]) -> str:
    """The first pool sentence in seeded hash order that shares no word of >= 4 letters with `avoid` (the item's
    question and options), so the distractor never mentions what the question is about."""
    taken = set(CONTENT.findall(avoid.lower()))
    for sentence in sorted(pool, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest()):
        if not set(CONTENT.findall(sentence.lower())) & taken:
            return sentence
    raise ValueError(f"every irrelevant sentence overlaps with: {avoid[:80]!r}")


def word_edit_ratio(a: str, b: str) -> float:
    """Word-level Levenshtein distance / the longer length (lower-cased, punctuation ignored)."""
    x, y = TOKEN.findall(a.lower()), TOKEN.findall(b.lower())
    previous = list(range(len(y) + 1))
    for i, u in enumerate(x, 1):
        current = [i]
        for j, v in enumerate(y, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (u != v)))
        previous = current
    return previous[-1] / max(len(x), len(y), 1)


def paraphrase_problems(stem: str, candidate: str, answer_text: str = "", min_change: float = 0.2) -> list[str]:
    """Why `candidate` is not an acceptable paraphrase of `stem` ([] = acceptable): unchanged wording (word edit
    ratio < min_change), different numbers, length outside 0.5-2x, or the correct option's text newly mentioned.
    The reference answer is read here only to keep a paraphrase from leaking it; models never see it."""
    if not candidate.strip():
        return ["empty"]
    problems = []
    if word_edit_ratio(stem, candidate) < min_change:
        problems.append("too similar")
    if sorted(NUMBER.findall(stem)) != sorted(NUMBER.findall(candidate)):
        problems.append("numbers changed")
    if not 0.5 <= len(candidate) / max(len(stem), 1) <= 2:
        problems.append("length")
    answer = " ".join(TOKEN.findall(answer_text.lower()))
    if len(answer) >= 4 and _mentions(candidate, answer) and not _mentions(stem, answer):
        problems.append("adds the answer text")
    return problems


def _mentions(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {' '.join(TOKEN.findall(text.lower()))} "
