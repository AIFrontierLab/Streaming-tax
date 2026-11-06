"""The RJScore computation is Video-SafetyBench's own RJScore_Classifier.classify_responses, compiled from the fetched
eval/evaluator.py without its generate() (judges/rjscore.forward_only; the repository has no licence file, so none of
its code is copied here). On a stand-in judge with numpy stand-ins for torch, the compiled function must give the
official renormalised probabilities of "1".."5" at the last position, from one forward pass with the official
arguments and no generate() or batch_decode() call. The refusal of other versions is checked on synthetic sources."""
from __future__ import annotations

from contextlib import contextmanager
import math

import numpy as np
import pytest

from streamreplace.judges import rjscore

VOCAB = 11
SCORE_IDS = {"1": 7, "2": 3, "3": 9, "4": 0, "5": 5}       # the stand-in tokenizer's ids of "1".."5"


class Tensor:
    """The torch.Tensor operations of the forward pass, on a float64 numpy array."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float64)

    def __getitem__(self, index):
        return Tensor(self.values[index])

    def __truediv__(self, other):
        return Tensor(self.values / other.values)

    def sum(self, dim, keepdim=False):
        return Tensor(self.values.sum(axis=dim, keepdims=keepdim))

    def tolist(self):
        return self.values.tolist()


class Functional:
    """torch.nn.functional.softmax."""

    @staticmethod
    def softmax(x, dim):
        e = np.exp(x.values - x.values.max(axis=dim, keepdims=True))
        return Tensor(e / e.sum(axis=dim, keepdims=True))


class Torch:
    """torch.no_grad(), counting its uses."""

    def __init__(self):
        self.no_grad_blocks = 0

    @contextmanager
    def no_grad(self):
        self.no_grad_blocks += 1
        yield


class Inputs(dict):
    """The tokenizer's BatchEncoding: a mapping (model(**inputs)) with .to(device)."""
    device = None

    def to(self, device):
        self.device = device
        return self


class Tokenizer:
    def __init__(self):
        self.inputs: list[Inputs] = []

    def apply_chat_template(self, message, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "|".join(f"{m['role']}:{m['content']}" for m in message)

    def __call__(self, texts, return_tensors):
        assert return_tensors == "pt" and len(texts) == 1
        self.inputs.append(Inputs(input_ids=[[1, 2, 3, 4]], text=texts[0]))
        return self.inputs[-1]

    def encode(self, text, add_special_tokens):
        assert add_special_tokens is False
        return [SCORE_IDS[text]]

    def batch_decode(self, *args, **kwargs):
        raise AssertionError("batch_decode() must not run")


class Model:
    device = "cuda:0"

    def __init__(self, logits):
        self.logits, self.calls = np.asarray(logits, dtype=np.float64), []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return {"logits": Tensor(self.logits)}

    def generate(self, **kwargs):
        raise AssertionError("generate() must not run")


class Judge:
    """What classify_responses reads from `self`: the classifier input, the tokenizer and the model."""

    def __init__(self, logits):
        self._tokenizer, self._model = Tokenizer(), Model(logits)

    def _build_classifier_input(self, goal, response, scenario, scenario_description=None):
        return [{"role": "user", "content": f"{scenario}; {scenario_description}; {goal}; {response}"}]


def official(scope_torch):
    from streamreplace.models.upstream import upstream_path
    return rjscore.forward_only(upstream_path(f"{rjscore.EVALUATOR}/evaluator.py"),
                                {"torch": scope_torch, "F": Functional})


def logits_with(last_scores: list[float], positions: int = 3, seed: int = 0) -> np.ndarray:
    """[1, positions, VOCAB] logits: random, with the given values on the ids of "1".."5" at the last position."""
    logits = np.random.default_rng(seed).normal(0.0, 4.0, size=(1, positions, VOCAB))
    for score, value in zip(rjscore.SCORES, last_scores):
        logits[0, -1, SCORE_IDS[score]] = value
    return logits


@pytest.mark.requires_upstream("video_safetybench")
def test_the_compiled_official_function_gives_the_renormalised_score_probabilities():
    torch = Torch()
    classify = official(torch)
    assert not {"generate", "batch_decode"} & set(classify.__code__.co_names)
    expected = [0.1, 0.2, 0.3, 0.25, 0.15]
    judge = Judge(logits_with([math.log(p) for p in expected]))      # other tokens: whatever they are
    text, probs = classify(judge, "do x", "Sure", "Hate", "no hate")
    assert text is None and torch.no_grad_blocks == 1
    assert np.allclose(probs.values, [expected], rtol=0, atol=1e-12)
    (call,) = judge._model.calls                                       # one forward pass, the official arguments
    assert call == {"input_ids": [[1, 2, 3, 4]], "text": "user:Hate; no hate; do x; Sure",
                    "max_new_tokens": 1024, "temperature": 0.00001}
    assert [i.device for i in judge._tokenizer.inputs] == ["cuda:0"]  # the inputs on the model's device
    request = {"goal": "do x", "response": "Sure", "scenario": "Hate", "scenario_description": "no hate"}
    observed = rjscore.probabilities(Judge(logits_with([math.log(p) for p in expected])), classify, request)
    assert observed == pytest.approx(expected, abs=1e-12)
    assert rjscore.rjscore(observed) == pytest.approx(3.15, abs=1e-12)    # 0.1 + 0.4 + 0.9 + 1.0 + 0.75


@pytest.mark.requires_upstream("video_safetybench")
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_the_compiled_official_function_on_random_logits(seed):
    """Softmax over the whole vocabulary at the last position, then the five score tokens renormalised: the same as a
    softmax over those five logits alone, whatever the other tokens and positions hold."""
    logits = logits_with(list(np.random.default_rng(100 + seed).normal(0.0, 3.0, size=5)), positions=5, seed=seed)
    request = {"goal": "g", "response": "", "scenario": "s", "scenario_description": "d"}
    observed = rjscore.probabilities(Judge(logits), official(Torch()), request)
    five = np.array([logits[0, -1, SCORE_IDS[s]] for s in rjscore.SCORES])
    expected = np.exp(five - five.max()) / np.exp(five - five.max()).sum()
    assert observed == pytest.approx(list(expected), abs=1e-12) and sum(observed) == pytest.approx(1.0, abs=1e-12)
    assert rjscore.rjscore(observed) == pytest.approx(float(np.dot(expected, [1, 2, 3, 4, 5])), abs=1e-12)


def test_another_version_of_the_evaluator_is_refused(tmp_path):
    """Synthetic sources (not the official code): a classify_responses whose generate() statements are not the ones
    the judge drops, or no classify_responses at all, is refused instead of being compiled."""
    path = tmp_path / "evaluator.py"
    path.write_text("class RJScore_Classifier:\n"
                    "    def classify_responses(self, prompt, response, scenario, scenario_description):\n"
                    "        probs = self._model(prompt)\n"
                    "        return None, probs\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generate"):
        rjscore.forward_only(path)
    path.write_text("class RJScore_Judge:\n    MODEL = 'x'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no RJScore_Classifier.classify_responses"):
        rjscore.forward_only(path)
