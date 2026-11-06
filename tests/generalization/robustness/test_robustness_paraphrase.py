"""The paraphrase procedure (paraphrase.py) with a stub model, its cache, and the OpenAI-compatible request."""
from __future__ import annotations

import io
import json

import pytest
from robustness_helpers import DEMO, stub_chat

from streamreplace.core.io import read_jsonl
from streamreplace.generalization.robustness import edits, make, paraphrase

REQUEST = {"key": "k", "stem": "What colour is the car?", "pivot": "Chinese", "options": "A. red\nB. blue",
           "answer_text": "red", "min_change": 0.2}


def test_back_translation_accepted():
    ok = paraphrase.paraphrase(stub_chat({"into Chinese": "[pivot] car colour?", "into English": "Which colour does the car have?",
                                          "Do the two versions": "Yes."}), REQUEST)
    assert ok["ok"] and ok["method"] == "back_translation" and ok["text"] == "Which colour does the car have?"
    assert ok["pivot_text"] == "[pivot] car colour?" and ok["verdicts"] == {"back_translation": "yes"}


def test_fallback_to_a_direct_rewording():
    fallback = paraphrase.paraphrase(stub_chat({"into Chinese": "x", "into English": "What colour is the car?",
                                                "Rewrite": "Of which colour is the vehicle?", "Do the two versions": "yes"}),
                                     REQUEST)
    assert fallback["ok"] and fallback["method"] == "direct" and fallback["problems"]["back_translation"] == ["too similar"]
    second = stub_chat({"into Chinese": "x", "into English": "Which colour does the car have?",
                        "Rewrite": "Of what colour is the auto?"})
    verdicts = iter(["no", "yes"])                    # the back-translation is judged different, the rewording the same
    rescued = paraphrase.paraphrase(lambda p: next(verdicts) if "Do the two versions" in p else second(p), REQUEST)
    assert rescued["ok"] and rescued["method"] == "direct" and rescued["text"] == "Of what colour is the auto?"


def test_rejected_paraphrase_is_a_skip_reason():
    rejected = paraphrase.paraphrase(stub_chat({"into Chinese": "x", "into English": "Which colour does the car have?",
                                                "Rewrite": "Which shade has the vehicle?",
                                                "Do the two versions": "No, they differ."}), REQUEST)
    assert not rejected["ok"] and rejected["reason"] == "judged not equivalent" and rejected["text"] is None
    assert rejected["verdicts"] == {"back_translation": "no", "direct": "no"}
    rules = paraphrase.paraphrase(stub_chat({"into Chinese": "x", "into English": "What colour is the car?",
                                             "Rewrite": "What colour is the car ?"}), REQUEST)
    assert rules["reason"] == "rules: too similar" and rules["verdict"] is None


def test_clean_reply():
    assert paraphrase.clean_reply('Translation: "Which one is it?"') == "Which one is it?"
    assert paraphrase.clean_reply("“Which one?”") == "Which one?"


def test_requests_cache_and_resume(experiment):
    exp = experiment
    make.make_subset(exp)
    pending = paraphrase.requests(exp)
    assert len(pending) == 8 and {"key", "stem", "pivot", "options", "answer_text", "min_change"} == set(pending[0])
    unit, samples = next(iter(exp.units(DEMO, "para").items()))
    assert any(r["key"] == edits.paraphrase_key(DEMO, unit, samples[0].metadata["raw_question"], "Chinese")
               for r in pending)
    calls = []

    def chat(prompt):
        calls.append(prompt)
        if "Do the two versions" in prompt:
            return "yes"
        if prompt.startswith("Translate the following question"):
            return "[pivot text]"
        if "boats" in prompt or "birds" in prompt:
            raise RuntimeError("server error")         # left for the next run
        return "Please say: " + prompt.split("\n\n", 1)[1]
    status = paraphrase.run(pending, exp.root / edits.PARAPHRASE_CACHE, chat, threads=3, log=lambda line: None)
    cached = read_jsonl(exp.root / edits.PARAPHRASE_CACHE)
    assert status["requests"] == 8 and len(cached) + status["errors"] == 8
    assert len(paraphrase.requests(exp)) == status["errors"]          # only the failed requests are asked again


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_endpoint_request_body_and_retries(monkeypatch):
    seen, failures = [], iter([True, False])

    def urlopen(request, timeout):
        seen.append((request.full_url, json.loads(request.data), dict(request.header_items()), timeout))
        if next(failures):
            raise OSError("connection reset")
        return FakeResponse(json.dumps({"choices": [{"message": {"content": "  yes \n"}}]}).encode())
    monkeypatch.setattr(paraphrase.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(paraphrase.time, "sleep", lambda s: None)
    options = {**paraphrase.DEFAULTS, "endpoint": "http://127.0.0.1:8000/v1/", "api_key_env": "TEST_KEY"}
    monkeypatch.setenv("TEST_KEY", "secret")
    chat = paraphrase.Endpoint.from_settings(options)
    assert chat("Hello?") == "yes" and len(seen) == 2
    url, body, headers, timeout = seen[0]
    assert url == "http://127.0.0.1:8000/v1/chat/completions"
    assert body == {"model": "judge", "messages": [{"role": "user", "content": "Hello?"}], "temperature": 0,
                    "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}}
    assert headers["Authorization"] == "Bearer secret"
    with pytest.raises(ValueError, match="no paraphrase endpoint"):
        paraphrase.Endpoint.from_settings(paraphrase.DEFAULTS)
