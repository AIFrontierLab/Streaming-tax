"""The streams' reply rules (models/readouts.py): what counts as an answer and what as silence in condition S."""
from __future__ import annotations

from types import SimpleNamespace

from streamreplace.core.schema import seed_of
from streamreplace.models.readouts import (answer_text, has_visible_text, is_no_reply, option_tokens, read_reply,
                                           stream_seed, visible)


# ------------------------------------------------------------------------------------------- MOSS-VL-Realtime

def test_reply_with_response_tag():
    text, info = read_reply("<|silence|><|response|>A.<|im_end|>")
    assert text == "A."
    assert info == {"response_tag": True, "prompt_with_frame": True, "complete": True}


def test_reply_when_question_came_without_its_frame():
    # push_prompt_frame race: no injected <|silence|>, no <|response|>; the client still shows "A"
    assert read_reply("A<|silence|>") == ("A", {"response_tag": False, "prompt_with_frame": False, "complete": True})
    assert read_reply("C.<|im_end|><|im_end|><|im_end|><|silence|>")[0] == "C."


def test_silence_is_no_answer():
    assert read_reply("<|silence|>") == ("", {"response_tag": False, "prompt_with_frame": True, "complete": False})
    assert read_reply("<|silence|><|silence|><|silence|>")[0] == ""
    assert read_reply("")[0] == ""


def test_reply_keeps_multi_chunk_text_and_stops_at_turn_end():
    text, info = read_reply("<|silence|><|response|>The ball  moves left.<|silence|> later text")
    assert text == "The ball  moves left."
    assert info["complete"]
    assert read_reply("<|silence|><|response|>B")[1]["complete"] is False


def test_reply_skips_leading_blank_text_and_go_tokens():
    assert read_reply("<|silence|> \n<go><|response|> B. <|im_end|>")[0] == "B."


def test_visible_chunks_count_against_the_budget():
    assert has_visible_text("<|response|>A") and not has_visible_text("<|silence|><go> ")


# ------------------------------------------------------------------------------------------- MMDuet2

def test_mmduet2_no_reply_is_exact():
    assert is_no_reply("NO REPLY") and is_no_reply("NO REPLY.") and is_no_reply("") and is_no_reply("no reply..")
    assert not is_no_reply("No.") and not is_no_reply("No") and not is_no_reply("NO, the cup moves left.")
    assert not is_no_reply("NO REPLY, but the cup moves left.")


# ------------------------------------------------------------------------------------------- AURA

def test_aura_visible_text():
    assert visible("<|silent|>") == ("", True) and visible(" A.") == ("A.", False)
    assert visible("<|silent|> B") == ("B", True) and visible(None) == ("", False)


# ------------------------------------------------------------------------------------------- ThinkStream

def test_thinkstream_answer_is_between_response_and_end_of_turn():
    assert answer_text("<think>the cup moves</think><response>B<|im_end|>") == "B"
    assert answer_text("<think>x</think><response> The cup moves left. <|im_end|>tail") == "The cup moves left."
    assert answer_text(None) == "" and answer_text("") == ""
    # an answer turn ended by a <response> sampled inside the thought (upstream breaks on it): split as upstream's
    assert answer_text("<think>a<response></think><silent><|im_end|>") == "</think><silent>"


def test_thinkstream_option_tokens_and_replicate_seeds():
    class Tok:
        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=[0, {"A": 32, "B": 33, "No": 2753, "Yes": 9454}[text]])

    mcq = SimpleNamespace(kind="mcq", choices={"A": "x", "B": "y"})
    assert option_tokens(Tok(), mcq) == {"A": 32, "B": 33}                       # the LAST token of each option
    yesno = option_tokens(Tok(), SimpleNamespace(kind="yesno", choices={}))
    assert yesno == {"No": 2753, "Yes": 9454} and list(yesno) == ["No", "Yes"]   # order decides argmax ties
    assert option_tokens(Tok(), SimpleNamespace(kind="open", choices={})) == {}
    assert stream_seed("q1") == stream_seed("q1", None) == stream_seed("q1", 0) == seed_of("q1")
    assert seed_of("q1") == int(__import__("hashlib").sha256(b"q1").hexdigest()[:8], 16)
    assert stream_seed("q1", 2) == seed_of("q1#seed2")
    assert len({stream_seed("q1"), stream_seed("q1", 1), stream_seed("q1", 2)}) == 3
