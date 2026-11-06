"""Shared helpers of the prompt-robustness tests: the experiment specification over the synthetic benchmark, a stub
chat model and a writer of fake scored outputs."""
from __future__ import annotations

import robustness_demo

from streamreplace.core.io import write_jsonl
from streamreplace.core.schema import score_row
from streamreplace.generalization.robustness import edits, make, paraphrase

DEMO = robustness_demo.NAME
REWRITES = ["Reply with the letter of the right option, directly.", "Give the option's letter as a direct answer.",
            "State the letter of the correct choice directly."]
SPEC = {
    "outputs": "robustness", "seed": 0, "datasets": {DEMO: 8}, "pairs": ["p"],
    "conditions": {
        "clean": {},
        "typo": {"edit": "typo", "family": "typo", "rate": 0.2, "max_words": 4},
        "para": {"edit": "para", "family": "para", "pivot": "Chinese"},
        "stress": {"edit": "stress", "family": "distraction"},
        "irrelevant": {"edit": "irrelevant", "family": "distraction"},
        "ins_typo": {"edit": "typo", "target": "instruction", "twin": "typo", "datasets": [DEMO], "units": 4,
                     "rate": 0.2, "max_words": 4},
        "ins_para": {"edit": "choice", "target": "instruction", "twin": "para", "datasets": [DEMO], "units": 4,
                     "texts": REWRITES},
        "ins_irrelevant": {"edit": "irrelevant", "target": "instruction", "twin": "irrelevant", "datasets": [DEMO],
                           "units": 4},
        "ins8_typo": {"edit": "typo", "target": "instruction", "twin": "typo", "datasets": [DEMO], "units": 8,
                      "rate": 0.2, "max_words": 4, "seed_as": "ins_typo"},
    },
    "paraphrase": {"min_change": 0.2},
}
PAIRS = {"p": ("parent", "child")}


def paraphrase_all(exp, reject_stem: str | None = None) -> dict:
    """Paraphrases every para unit with a stub model ("translation" = the stem with a prefix); the stem
    `reject_stem` is judged not equivalent in both attempts, so its unit is skipped."""
    def chat(prompt):
        if prompt.startswith("Translate the following question into"):
            return "ZH:" + prompt.split("\n\n", 1)[1]
        if prompt.startswith("Translate the following text into English"):
            return "Tell me please, " + prompt.split("\n\n", 1)[1].removeprefix("ZH:")
        if prompt.startswith("Rewrite the following question"):
            return "Kindly answer this: " + prompt.split("\n\n", 1)[1]
        return "No." if reject_stem and f"Version 1: {reject_stem}\n" in prompt else "Yes."
    return paraphrase.run(paraphrase.requests(exp), exp.root / edits.PARAPHRASE_CACHE, chat, threads=2,
                          log=lambda line: None)


def prepared(exp, reject: bool = True) -> str | None:
    """subset + paraphrases + edits; returns the stem whose paraphrase was rejected."""
    make.make_subset(exp)
    stem = make.DATASETS[DEMO].stem(next(iter(exp.units(DEMO, "para").values()))[0]) if reject else None
    paraphrase_all(exp, stem)
    edits.make_edits(exp)
    return stem


def stub_chat(replies: dict):
    """A chat function answering each prompt by the first key it contains."""
    def chat(prompt):
        return next(v for k, v in replies.items() if k in prompt)
    return chat


def write_job(exp, condition: str, model: str, correct=lambda sample: True, logits=None, error_ids=(),
              unscored_ids=(), dataset: str = DEMO) -> None:
    """Fake scored outputs of `model` in a condition folder: the reply is the gold letter when correct(sample) is
    true, else the other letter; logits(sample) -> info.letter_logits (optional)."""
    from streamreplace.data.prompts import read_overrides, render_prompts
    samples = exp.samples(dataset, condition)
    prompts = render_prompts(dataset, samples, (read_overrides(exp.folder(condition)) or {}).get(dataset))
    gen, scores = [], []
    for s in samples:
        if s.id in error_ids:
            gen.append({"id": s.id, "prompt": prompts[s.id], "raw": None, "tokens": None, "finish": None,
                        "latency": 0.1, "info": None, "error": "boom", "t": None})
            continue
        ok = bool(correct(s))
        letter = s.answer if ok else next(k for k in s.choices if k != s.answer)
        info = {"letter_logits": logits(s)} if logits else {}
        gen.append({"id": s.id, "prompt": prompts[s.id], "raw": letter, "tokens": 2, "finish": "eos",
                    "latency": 0.1, "info": info, "error": None, "t": None})
        if s.id not in unscored_ids:
            scores.append(score_row(s.id, letter, "letter", ok))
    folder = exp.folder(condition) / dataset
    write_jsonl(folder / f"{model}.offline.jsonl", gen)
    write_jsonl(folder / f"{model}.offline.scores.jsonl", scores)
