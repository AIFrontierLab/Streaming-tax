"""How each streaming child's raw output becomes the reply of condition S: pure functions, no model code.

These rules decide what counts as an answer and what counts as silence in the native streams (paper condition S, and
the plugin arms built on them). A silent stream row is scored wrong without parsing or judging, so every rule here
moves paper numbers and is ported unchanged from the paper's runs:

- MOSS-VL-Realtime: `read_reply`, the official client's display rule (control tokens hidden, the first visible text
  after the question up to the next <|silence|> / <|im_end|>). Provenance: the paper's Video-MME, TOMATO, TimeBlind and
  VideoReasonBench rows were generated with an earlier reading that took only the text after <|response|> and counted
  a reply without it as silence. Their silent rows were then re-read, without new inference, with this same
  read_reply applied to the saved output tail (`info.tail`, from its last <|round_start|>); those rows carry
  `info.reread`. 226 of the 230 silent Video-MME rows held an answer; the TOMATO, TimeBlind and VideoReasonBench
  silent rows stayed silent. The safety and VMDT rows were read live with read_reply. Regenerated rows are all read
  live; for the reply shapes of those runs the live reading gives the same text as the re-read
  (docs/models/stream-adapters.md).
- MMDuet2: `is_no_reply`, silence is a reply equal to "NO REPLY" (any case, trailing dots), or an empty reply.
- AURA: `visible`, the reply with AURA's <|silent|> marker removed; an empty result is silence.
- ThinkStream: `answer_text` (the text between <response> and <|im_end|>), `option_tokens` (the tokens its
  restricted answer readout scores) and `stream_seed` (its per-question sampling seed).
- Question timing (setting stream.question_at, a diagnostic of the native streams): `question_at`, `question_index`
  and `timing_info`, shared by the five children whose interface takes a question during the stream.
"""
from __future__ import annotations

import re

from ..core.schema import seed_of

# --------------------------------------------------------------------------------------------- MOSS-VL-Realtime

MOSS_TOKEN = re.compile(r"(<\|[a-z_]+\|>|<go>)")        # control tokens of the realtime session's output
MOSS_TURN_END = {"<|silence|>", "<|im_end|>"}


def read_reply(stream: str) -> tuple[str, dict]:
    """The answer in MOSS-VL-Realtime's raw output after the question's <|round_start|>, read as the official client
    shows it (realtime_inference/run_online_inference.py render_output: control tokens hidden, <|response|>
    included): the first visible text, up to the next <|silence|> / <|im_end|>.

    <|response|> is not required. The session's push_prompt_frame puts the frame and the question into two queues one
    after the other; when the generation thread wakes up in between, the question is taken in a later cycle than its
    frame, no <|silence|> is injected, and the model writes the answer straight away without <|response|>.

    Returns (text, info): info = {"response_tag": <|response|> seen, "prompt_with_frame": the injected <|silence|>
    follows <|round_start|> (question and frame arrived together), "complete": the reply reached its end token}.
    No visible text means the model stayed silent."""
    text, started, complete, tag = [], False, False, False
    for piece in MOSS_TOKEN.split(stream):
        if not piece:
            continue
        if piece in MOSS_TURN_END:
            if started:
                complete = True
                break
        elif MOSS_TOKEN.fullmatch(piece):
            tag = tag or piece == "<|response|>"
        elif started or piece.strip():
            started = True
            text.append(piece)
    return "".join(text).strip(), {"response_tag": tag, "prompt_with_frame": stream.startswith("<|silence|>"),
                                   "complete": complete}


def has_visible_text(chunk: str) -> bool:
    """Does an output chunk of the realtime session hold visible text (what counts against the token budget)?"""
    return bool(MOSS_TOKEN.sub("", chunk).strip())


# --------------------------------------------------------------------------------------------- MMDuet2

def is_no_reply(text: str) -> bool:
    """MMDuet2's silence: the literal "NO REPLY" of proactive_eval/evaluate.py (NO_REPLY_STR), in any case and with
    trailing dots, or an empty reply. Upstream's demo tests startswith("NO") instead, which would also swallow a
    genuine "No." to a yes/no question; the exact rule is the one the paper's runs used."""
    return not text or text.rstrip(".").strip().upper() == "NO REPLY"


# --------------------------------------------------------------------------------------------- AURA

AURA_SILENT = "<|silent|>"      # AURA's silence token (vLLM drops it from the text; another checkpoint writes it)


def visible(content: str | None) -> tuple[str, bool]:
    """The reply as AURA's client shows it: vLLM already drops AURA's special <|silent|> token; a checkpoint without
    that token writes it as text, which is removed the same way. Returns (text, whether a literal marker was
    removed). An empty text is silence."""
    content = content or ""
    return content.replace(AURA_SILENT, "").strip(), AURA_SILENT in content


# --------------------------------------------------------------------------------------------- ThinkStream

def answer_text(turn: str | None) -> str:
    """ThinkStream's answer: the decoded answer turn between <response> and <|im_end|>, stripped ("" without a
    turn). The turn is decoded with special tokens kept, so the split markers are present."""
    return turn.split("<response>", 1)[-1].split("<|im_end|>", 1)[0].strip() if turn else ""


def options_of(kind: str, choices) -> list[str]:
    """The options ThinkStream's restricted readout scores: the choice letters of a multiple-choice request,
    ["No", "Yes"] (in this order: ties go to the first) for yes/no, none for other kinds."""
    return list(choices) if kind == "mcq" else (["No", "Yes"] if kind == "yesno" else [])


def option_tokens(tokenizer, req) -> dict[str, int]:
    """Options of a multiple-choice / yes-no request -> the token ThinkStream's restricted readout scores each by: the
    LAST token of the option text (eval_common's restricted_token_ids). Empty for other kinds. `req` needs .kind and
    .choices; `tokenizer(text, add_special_tokens=False).input_ids` must tokenize."""
    return {option: tokenizer(option, add_special_tokens=False).input_ids[-1]
            for option in options_of(req.kind, req.choices)}


def stream_seed(sample_id: str, replica=None) -> int:
    """ThinkStream's per-question sampling seed, int(sha256(id).hexdigest()[:8], 16), set once before the question's
    stream. Replicate k > 0 (settings stream.seed = k, a run-to-run variance check) uses the id f"{id}#seed{k}";
    unset or 0 gives the seed of the paper's runs."""
    return seed_of(f"{sample_id}#seed{replica}" if replica else sample_id)


# --------------------------------------------------------------------------------------------- question timing

def question_at(stream: dict | None) -> str | None:
    """The setting stream.question_at: None (the paper's S: the question arrives at the end of the video) or
    "start" | "middle" (the question is posed once at the first step or at step n // 2, the stream runs on under the
    child's own response policy, and the last reply is scored; Supp. "Question timing")."""
    where = (stream or {}).get("question_at")
    if where not in (None, "start", "middle"):
        raise ValueError(f"stream.question_at must be start or middle, not {where!r}")
    return where


def question_index(where: str, steps: int) -> int:
    """The step (frame, turn or segment) the question is posed at: the first, or step n // 2."""
    return 0 if where == "start" else steps // 2


def timing_info(where: str, index: int, t: float, replies: list[dict], end: float, fallback: bool) -> dict:
    """Row info of a question-timing stream. replies: the child's visible replies after the question, in order, as
    {"t": media time in seconds, "text"}; fallback: no reply came before the end, so an answer was requested there
    as in S. The last 20 replies are kept, each cut to 300 characters; the first reply is kept separately."""
    cut = [{**r, "text": r["text"][:300]} for r in replies]
    return {"question_at": where, "question_index": index, "question_t": round(float(t), 2),
            "replies": cut[-20:], "n_replies": len(replies),
            "first_reply_t": replies[0]["t"] if replies else None, "first_reply": cut[0]["text"] if replies else None,
            "answer_t": replies[-1]["t"] if replies else round(float(end), 2), "fallback": bool(fallback)}
