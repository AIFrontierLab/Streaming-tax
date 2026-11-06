"""Respond (part of S_fix): the child's own answer turn, committed.

1. Silence bans: the answer turn may not open with the child's silence marker (a refusal still can). The marker is
   the child's declared `answer_turn.silence` (configs/plugin.yaml; recipe.AnswerTurn), never inferred from its id:
   - one token (AURA <|silent|>, MOSS-VL-Realtime <|silence|>): it and <|im_end|> are banned at the first step, or on
     the opening steps until the first visible token for a child whose logits path is `opening` (hooks.py);
   - "NO REPLY" (MMDuet2, whose silence is a reply reading "NO REPLY"; logits path `processor` only): at the first
     step every token whose stripped text is "NO", plus <|im_end|>; at the second step, after a first token reading
     "no" in any casing, every token of stripped length >= 2 that starts "REPLY" case-insensitively (so "No, ..."
     still answers);
   - null: no ban (MMDuet, whose forced reply already answers; LiveStar, no marker; ThinkStream, which answers after
     its thought).
2. Option readout, for multiple-choice and yes/no items only: when the reply is silent (info.no_answer) or holds no
   valid option by the benchmark's own rule parser, the reply becomes the option whose first token has the highest
   first-step logit (letters, or "Yes"/"No" without a leading space; the restricted readout ThinkStream's own
   evaluation uses). Open questions are never touched."""
from __future__ import annotations

import re

from .recipe import CLOSED_KINDS, NO_REPLY

LETTER = re.compile(r"^[\s\"'(\[*]*([A-Z])(?:[\s)\].:,*]|$)")
YESNO = re.compile(r"^[\s\"'(\[*]*(yes|no)\b", re.I)


def options(req) -> list[str]:
    """The options a closed-choice reply must hold: the letters of a multiple-choice request, Yes / No for yes/no."""
    if req.kind == "mcq":
        return list(req.choices)
    if req.kind == "yesno":
        return ["Yes", "No"]
    return []


def opens_with_option(text: str, opts: list[str]) -> bool:
    if not opts:
        return True
    if opts == ["Yes", "No"]:
        return YESNO.match(text or "") is not None
    match = LETTER.match(text or "")
    return match is not None and match.group(1) in opts


def answered(req, text: str, opts: list[str]) -> bool:
    """Does the reply hold a valid option by the benchmark's own rule parser (the first step of its official scoring,
    on the model-safe sample view)? A request outside any benchmark counts as answered when it opens with an option."""
    if not getattr(req, "dataset", None) or getattr(req, "sample", None) is None:
        return opens_with_option(text, opts)
    from ..benchmarks.registry import get
    pred = get(req.dataset).parse(req.sample, text)[0]
    return pred is not None and str(pred).strip().lower() in {o.lower() for o in opts}


def option_ids(tokenizer, opts: list[str]) -> dict[str, int]:
    """Each option's first token at the start of an answer (no leading space)."""
    ids = {}
    for opt in opts:
        tokens = tokenizer.encode(opt, add_special_tokens=False)
        if tokens:
            ids[opt] = tokens[0]
    return ids


def read(logits, ids: dict[str, int]) -> tuple[str, dict]:
    """The option with the highest first-token logit (the first on ties), and every option's logit."""
    scores = {opt: round(float(logits[i]), 4) for opt, i in ids.items()}
    return max(scores, key=scores.get), scores


def is_closed(kind: str | None) -> bool:
    return kind in CLOSED_KINDS


# ---------- silence bans ----------

def mmduet2_silence_marker(tok) -> tuple[list[int], dict[int, list[int]]]:
    """MMDuet2's "NO REPLY" ban, built once from the vocabulary: (first-step ids, {first id: second-step ids})."""
    text = {i: tok.decode([i]).strip() for i in range(len(tok))}
    first = [i for i, t in text.items() if t == "NO"]
    follow = [i for i, t in text.items() if len(t) >= 2 and "REPLY".startswith(t.upper())]
    return first, {i: follow for i, t in text.items() if t.upper() == "NO"}


EOT = "<|im_end|>"


def silence_bans(marker: str | None, tokenizer) -> tuple[list[int], dict[int, list[int]] | None]:
    """(first-step bans, second-step bans keyed by the first token) of an answer turn whose silence marker is `marker`
    (the child's answer_turn.silence): None -> none; "NO REPLY" -> MMDuet2's two-step ban; any other marker must be
    one token of the tokenizer, banned together with <|im_end|>. Raises for a marker the tokenizer does not hold."""
    conv = tokenizer.convert_tokens_to_ids
    if marker is None:
        return [], None
    if marker == NO_REPLY:
        first, follow = mmduet2_silence_marker(tokenizer)
        return first + [conv(EOT)], follow
    token = conv(marker)
    if token is None or token == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"silence marker {marker!r} is not a token of the child's tokenizer (a marker is one token, "
                         f"or {NO_REPLY!r})")
    return [token, conv(EOT)], None
