"""Conditions, arms and job names (core/protocol.py) and the data contracts (core/schema.py)."""
from __future__ import annotations

import hashlib

import pytest

from streamreplace.core.protocol import (FIXED_REFUSAL, GAPS, VARIANTS, Arm, Condition, Mode, Role, condition_arm,
                                         conditions_for)
from streamreplace.core.schema import (FINISH, GEN_KEYS, SCORE_ROW_KEYS, Generation, Request, Sample, generation_row,
                                       score_row, seed_of)


def test_enums_format_as_their_values():
    assert str(Mode.OFFLINE) == "offline" and f"{Mode.STREAM}" == "stream" and f"{Role.CHILD:>6}" == " child"
    assert Condition("S+") is Condition.S_PLUS and Mode("stream") is Mode.STREAM and Mode.OFFLINE == "offline"


@pytest.mark.parametrize("arm, name", [
    (Arm("aura_8b", Mode.STREAM), "aura_8b.stream"),
    (Arm("qwen3vl_8b", "offline"), "qwen3vl_8b.offline"),
    (Arm("aura_8b", Mode.STREAM, "sfix_prs"), "aura_8b@sfix_prs.stream"),
    (Arm(FIXED_REFUSAL, Mode.OFFLINE), "fixed_refusal.offline"),
    (Arm("qwen25vl_3b@lingoqa-1k", Mode.OFFLINE), "qwen25vl_3b@lingoqa-1k.offline"),   # a fine-tuned id keeps its '@'
])
def test_job_names_round_trip(arm, name):
    assert arm.job_name() == str(arm) == name
    assert Arm.parse(name) == arm


@pytest.mark.parametrize("args", [("", Mode.OFFLINE), ("a.b", Mode.OFFLINE), ("m", Mode.STREAM, "shadow"),
                                  ("m@sfix", Mode.STREAM)])
def test_invalid_arms(args):
    with pytest.raises(ValueError):
        Arm(*args)


def test_job_name_without_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        Arm.parse("aura_8b")
    with pytest.raises(ValueError):
        Arm.parse("aura_8b.live")


def test_conditions_map_to_arms():
    assert GAPS == {"C-P": ("P", "C"), "S-C": ("C", "S"), "S-P": ("P", "S")}
    assert VARIANTS == ("sfix", "sfix_prs", "record", "inert", "hf")
    assert condition_arm("P", "qwen3vl_8b", "aura_8b") == Arm("qwen3vl_8b", Mode.OFFLINE)
    assert condition_arm(Condition.C, "qwen3vl_8b", "aura_8b") == Arm("aura_8b", Mode.OFFLINE)
    assert condition_arm("S", "qwen3vl_8b", "aura_8b") == Arm("aura_8b", Mode.STREAM)
    assert condition_arm("S_fix", "qwen3vl_8b", "aura_8b").job_name() == "aura_8b@sfix.stream"
    assert condition_arm("S_HF", "qwen3vl_8b", "aura_8b").job_name() == "aura_8b@hf.stream"
    with pytest.raises(ValueError, match="assembled"):
        condition_arm("S+", "qwen3vl_8b", "aura_8b")
    assert conditions_for("video") == (Condition.P, Condition.C, Condition.S)
    assert conditions_for("image") == (Condition.P, Condition.C)
    with pytest.raises(ValueError):
        conditions_for("audio")


def test_samples_validate_and_round_trip():
    sample = Sample("q1", "Which?", "B", "mcq", {"A": "x", "B": "y"}, ("a.png",), None, {"truth": 1, "cat": "c"})
    again = Sample.from_dict(sample.to_dict())
    assert again.images == ("a.png",) and again.to_dict() == sample.to_dict()
    view = sample.view(frozenset({"truth"}))
    assert view.metadata == {"cat": "c"} and not hasattr(view, "answer") and view.choices == sample.choices
    with pytest.raises(ValueError, match="unknown kind"):
        Sample("q", "?", "x", "essay")
    with pytest.raises(ValueError, match="not an option"):
        Sample("q", "?", "C", "mcq", {"A": "x", "B": "y"})


def test_request_seed_is_the_per_question_seed():
    assert seed_of("019-3") == int(hashlib.sha256(b"019-3").hexdigest()[:8], 16)
    assert Request("019-3", "videomme", "p").seed == seed_of("019-3")
    assert Request("019-3", "videomme", "p", seed=7).seed == 7
    assert Request("x", "d", "p").reference is None


def test_generation_and_score_rows():
    gen = Generation.from_dict({"text": "B", "tokens": 3, "finish": None, "info": None})
    assert (gen.finish, gen.info) == ("unknown", {}) and gen.to_dict()["text"] == "B"
    row = generation_row("q1", "prompt", Generation("B", 3, "eos", {"frames": 4}), 1.5, t="2026-01-01T00:00:00Z")
    assert list(row) == list(GEN_KEYS) and row["error"] is None and row["info"] == {"frames": 4}
    error = generation_row("q1", "prompt", None, 0.2, error="CUDA out of memory")
    assert list(error) == list(GEN_KEYS) and error["raw"] is None and error["info"] is None
    assert error["error"] == "CUDA out of memory" and error["t"].endswith("Z")
    assert generation_row("q1", "p", None, 0.0)["error"] == "unknown error"
    score = score_row("q1", "B", "letter", True, judge={"name": "extract", "key": "0" * 16})
    assert list(score) == list(SCORE_ROW_KEYS) and score["judge_pred"] is None
    assert "guard" in FINISH and "readout" in FINISH
