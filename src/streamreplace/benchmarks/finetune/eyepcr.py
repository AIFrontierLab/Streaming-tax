"""EyePCR (Gui et al. 2025, "EyePCR: A Comprehensive Benchmark for Fine-Grained Perception, Knowledge Comprehension and
Clinical Reasoning in Ophthalmic Surgery"; HF EvergreenTree/EyePCR @5a25c1e8, CC-BY-4.0): ophthalmic-surgery video QA
on clips of surgery cases, the fine-tuning experiment's medical domain (docs/generalization/finetune.md).

The official split files (LLaMA-Factory conversations, one clip each) are test.json (154 cases, test_cases.txt) and
train.json; their multiple-choice conversations hold 3,508 / 39,733 questions. There is no official evaluation
script, so:
  sample    one per multiple-choice question (earlier turns of its conversation are not shown)
  prompt    the conversation's opening instruction (INTRO, without <video>) and the question turn as the files write it,
            "{question}\\nCHOICE: {str(options)}", in one user turn (the files' acknowledgement turn in between is left
            out); training targets the answer turn, the option's text
  options   reordered per question by sha256(id:option): in the released order the answer is the third option for 40%
            of the questions (train and test), a position prior fine-tuning would pass off as adaptation
  frames    16 uniform frames, np.linspace(0, total - 1, 16, dtype=int) (configs/benchmarks.yaml offline_frames)
  scoring   the reply (clean()) names an option: equal to it after normalize(), else the only option it contains as
            a phrase (an option inside another named option does not count); correct iff that option is the answer.
            Two rules were added after the first zero-shot results had been seen (post hoc, kept for reproduction):
            the option a marked answer span starts with ("The correct answer is 'Scarred'. The cornea ...": ThinkStream
            states its answer, then runs on to the token limit naming other options), and a lone letter ("C",
            "The answer is C.") read as that position in the listed order (zero-shot MOSS / ThinkStream answered so,
            right 48% / 40% of the time, so the letters index the list). SENSITIVITY keeps the first-round parser
            (neither rule) so every table can also be reported without them. Decoding greedy, max_new_tokens 512
            (128 cut ThinkStream's reasoning; no official value)
  dropped   questions whose answer is no option (one equal after normalize is respelled as that option) and questions
            with two equal options
  category  the MCQ.zip folder filing the question; 264 test questions are filed twice
            ("Intraoperative_Anatomy+Pathological_Diagnosis"); stratifies the <= 2K test subset"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

from .. import common
from ...core.schema import Sample

MEDIA = "video"
JUDGE = None                  # no official scorer: replies are matched to option texts (parse)
EXPECTED_COUNT = 3492
GROUP = "case"                # one surgery: the official split's unit and the cluster of the paired statistics
STRATUM_KEYS = ("category",)
INTRO = ("You are a professional assistant with expertise in ophthalmic surgery and analyze the provided ophthalmology "
         "surgery video. Based on the video, I will ask you some questions and provide a list of choices for each "
         "question, you need to select the most appropriate one.")
REPLY = ("Of course! Please go ahead and share the ophthalmology surgery video. I'll carefully analyze and pick the most "
         "appropriate answer for each one.")
CHOICE = "\nCHOICE: "
# \uff1a is the full-width colon some models write after "answer" (an escape for the literal character)
LETTER = re.compile(r"(?i:(?:the\s+)?(?:correct\s+|final\s+|best\s+)?(?:answer|option|choice)(?:\s+is)?\s*[:\uff1a]?\s*)?"
                    r"\(?([A-Z])\)?[.:)]?")    # a lone upper-case letter, optionally after "the answer is"
CLIPS = "videos_resized_slice.zip"


def reorder(sample_id: str, options: list[str]) -> list[str]:
    return sorted(options, key=lambda o: hashlib.sha256(f"{sample_id}:{o}".encode()).hexdigest())


def categories(raw: Path) -> dict[tuple, str]:
    """(case, clip, question, sorted options) -> the MCQ.zip folder(s) filing it, "A+B" when filed twice."""
    found: dict[tuple, set] = {}
    with zipfile.ZipFile(raw / "MCQ.zip") as archive:
        for name in [n for n in archive.namelist() if n.endswith(".json")]:
            category, case = name.split("/")[0], Path(name).stem
            for clip, questions in json.loads(archive.read(name)).items():
                for q in questions:                       # two schemas: options/answer and Options/Answer
                    options = tuple(sorted(map(str, q.get("options", q.get("Options")) or [])))
                    found.setdefault((case, clip, q.get("Q", "").strip(), options), set()).add(category)
    return {key: "+".join(sorted(value)) for key, value in found.items()}


def _samples(raw: Path, split: str) -> list[Sample]:
    normalize = common.normalize
    common.extract_zip(raw / CLIPS, raw)
    filed, held = categories(raw), set((raw / "test_cases.txt").read_text().split())
    samples, dropped, respelled = [], {"answer_not_an_option": 0, "equal_options": 0}, 0
    for conversation in json.loads((raw / f"{split}.json").read_text(encoding="utf-8")):
        turns = [t["content"] for t in conversation["messages"]]
        if turns[0] != "<video>" + INTRO:                 # an open-ended conversation
            continue
        assert turns[1] == REPLY and len(turns) % 2 == 0 and len(conversation["videos"]) == 1
        path = conversation["videos"][0]
        _, case, clip = path.split("/")
        clip = clip.removesuffix(".mp4")
        assert (case in held) == (split == "test"), f"{case} breaks the official case split"
        for turn, (asked, answer) in enumerate(zip(turns[2::2], turns[3::2])):
            question, _, listed = asked.rpartition(CHOICE)
            options = ast.literal_eval(listed)
            assert asked == f"{question}{CHOICE}{options}"   # the files write str() of the option list
            if len({normalize(o) for o in options}) < len(options):
                dropped["equal_options"] += 1
                continue
            if answer not in options:
                same = [o for o in options if normalize(o) == normalize(answer)]
                if len(same) != 1:
                    dropped["answer_not_an_option"] += 1
                    continue
                answer, respelled = same[0], respelled + 1
            sample_id = f"{case}_{clip}_{turn}"
            samples.append(Sample(sample_id, question, answer, "text", video=str(raw / path),
                                  metadata={"case": case, "clip": clip, "turn": turn,
                                            "category": filed[(case, clip, question.strip(), tuple(sorted(options)))],
                                            "options": reorder(sample_id, options)}))
    print(f"eyepcr {split}: {len(samples)} questions; dropped {dropped}; {respelled} answers respelled as their option",
          flush=True)
    return samples


def prepare(raw: Path, media: Path) -> list[Sample]:
    return _samples(raw, "test")


def prepare_train(raw: Path, media: Path) -> list[Sample]:
    return _samples(raw, "train")


def prompt(sample) -> str:
    return f"{INTRO}\n{sample.question}{CHOICE}{sample.metadata['options']}"


def match(options: list[str], text: str) -> tuple[str | None, str]:
    """The option a reply names: equal after normalize, else the only option it contains as a phrase."""
    normalize = common.normalize
    reply = normalize(text)
    equal = [o for o in options if normalize(o) == reply]
    if len(equal) == 1:
        return equal[0], "option_text"
    named = [o for o in options if normalize(o) and re.search(rf"(?<!\w){re.escape(normalize(o))}(?!\w)", reply)]
    named = [o for o in named if not any(o != p and normalize(o) in normalize(p) for p in named)]
    if len(named) == 1:
        return named[0], "option_in_reply"
    return None, "several_options" if named else "no_option"


def leading(options: list[str], span: str) -> str | None:
    """The option a marked answer span starts with ("'Scarred'. The cornea is ..."), the longest if several."""
    normalize = common.normalize
    head = normalize(span).lstrip("'\"*`( ")
    starts = [o for o in options if normalize(o) and re.match(rf"{re.escape(normalize(o))}(?!\w)", head)]
    return max(starts, key=len) if starts else None


def parse(sample, text, lenient: bool = True):
    """The explicit answer span (final_answer()) first, else the whole cleaned reply. lenient (the post hoc revision)
    adds: the option a marked span starts with, then a lone letter."""
    options, reply = sample.metadata["options"], common.clean(text)
    if not reply:
        return None, "empty"
    span = common.final_answer(text)
    found = match(options, span) if span else (None, "no_option")
    found = found if found[0] else match(options, reply)
    if found[0] or not lenient:
        return found
    if span and span != reply and (first := leading(options, span)):     # after an answer marker, not a whole reply
        return first, "marked_option"
    for candidate in (span, reply):
        hit = LETTER.fullmatch(candidate.strip().strip("*` "))
        if hit and ord(hit[1]) - ord("A") < len(options):
            return options[ord(hit[1]) - ord("A")], "letter"
    return found


# The first-round parser (no marked-option or letter rule): the post hoc rules' sensitivity analysis. The key is the
# heading of its table in the fine-tuning report.
SENSITIVITY = {"with the first-round parser (no marked-option or letter rule)":
               lambda sample, text: parse(sample, text, lenient=False)}


def correct(sample, prediction) -> bool:
    return prediction is not None and prediction == sample.answer


def metrics(rows) -> dict:
    return {"acc": common.pct(r["correct"] for r in rows), **common.by(rows, "category", "cat/")}


def smoke(samples):
    return common.smoke(samples, group=GROUP, n=50)
