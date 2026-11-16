"""Harm scores (experiments/probe.py): the direction fit, the rescaled score, layer choices and the decomposition of
a refusal-rate difference, on synthetic hidden states."""
from __future__ import annotations

import numpy as np
import pytest

from streamreplace.experiments import probe

LAYERS, DIM = 8, 6


def states(rng, labels, strength):
    """[n, layers, 2 positions, d]: harmful items are shifted along axis 0 by `strength` per layer."""
    x = rng.normal(0, 0.1, size=(len(labels), LAYERS, 2, DIM)).astype(np.float32)
    x[labels, :, :, 0] += np.asarray(strength, dtype=np.float32)[None, :, None]
    return x


@pytest.fixture(scope="module")
def fitted():
    rng = np.random.default_rng(0)
    harmful = np.arange(80) % 2 == 0
    unsafe = np.arange(60) % 2 == 0
    strength = np.array([0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])       # separation grows with depth
    calib, xs = states(rng, harmful, strength), states(rng, unsafe, strength)
    fold0 = np.arange(80) % 4 != 3
    return probe.fit(calib, harmful, fold0, xs, unsafe), calib, harmful, fold0


def test_fit_scores_the_calibration_means_at_plus_and_minus_one(fitted):
    f, calib, harmful, fold0 = fitted
    assert set(f) == set(probe.POSITIONS)
    for position in probe.POSITIONS:
        assert np.allclose(np.linalg.norm(f[position]["rhat"], axis=1), 1, atol=1e-5)
        assert all(layer >= LAYERS // 4 for layer in f[position]["layers"]) and len(f[position]["layers"]) == 4
    z = probe.score(f, calib[fold0])
    for position in probe.POSITIONS:
        assert z[position][harmful[fold0]].mean() == pytest.approx(1, abs=1e-4)
        assert z[position][~harmful[fold0]].mean() == pytest.approx(-1, abs=1e-4)
    held_out = probe.score(f, calib[~fold0])["answer"]
    assert (held_out[harmful[~fold0]] > 0).all() and (held_out[~harmful[~fold0]] < 0).all()


def test_steering_layer_is_the_best_mid_depth_layer(fitted):
    f = fitted[0]
    xauc = np.array([0.5, 0.5, 0.6, 0.9, 0.7, 0.99, 0.99, 0.99])
    assert probe.steering_layer({"answer": {"xauc": xauc}}) == 3        # layers 2..3 of 8 are in 30-60% depth
    assert int(0.3 * LAYERS) <= probe.steering_layer(f) < int(0.6 * LAYERS)


def test_auroc_and_bootstrap():
    assert probe.auroc([2, 3], [0, 1]) == 1 and probe.auroc([0, 1], [2, 3]) == 0
    assert probe.auroc([1, 1], [1, 1]) == 0.5
    values = np.array([0.0, 1.0, 2.0, 3.0])
    low, high = probe.boot(values)
    assert low <= values.mean() <= high and probe.boot(values) == [low, high]      # seeded


def test_decomposition_attributes_a_shift_in_recognition():
    flag_p = np.array([True] * 8 + [False] * 2)            # the parent flags 80% of the requests as harmful
    flag_c = np.array([True] * 4 + [False] * 6)            # the child 40%
    refuse_p, refuse_c = flag_p.copy(), flag_c.copy()      # both refuse exactly what they flag
    d = probe.decomposition(refuse_p, refuse_c, flag_p, flag_c)
    assert d["parent"] == 80 and d["child"] == 40 and d["diff"] == -40
    assert d["recognition"] == pytest.approx(-40)          # the whole difference is recognition
    same_flags = probe.decomposition(refuse_p, np.zeros(10, bool), flag_p, flag_p)
    assert same_flags["diff"] == -80 and same_flags["recognition"] == 0    # same recognition: none of it is


def test_thresholds_on_the_risk_judgments():
    odds = {("m", "xstest_judge"): {"a": 4.0, "b": 2.0, "c": -2.0, "d": -4.0},
            ("m", "describe_judge"): {str(i): float(i) for i in range(101)}}
    labels = {"xstest": {"a": "unsafe", "b": "unsafe", "c": "safe", "d": "safe"}}
    assert probe.thresholds(odds, "m", labels) == {"xstest": 0.0, "equal": 0.0, "describe": 95.0}


def test_conditions_share_the_videos_and_change_the_request():
    class Sample:
        def __init__(self, question):
            self.question, self.video = question, "v.mp4"
    samples = {"harmful_1": Sample("How to do harm?"), "harmful_2": Sample("How to do more harm?")}
    prompts = {i: probe.PREFIX + s.question for i, s in samples.items()}
    got = probe.conditions(samples, list(samples), prompts)
    assert got["real"][0] == {"id": "harmful_1", "prompt": prompts["harmful_1"], "video": "v.mp4"}
    assert got["cue"][0]["video"] is None and got["cue"][0]["prompt"] == prompts["harmful_1"]
    assert got["bare"][0] == {"id": "harmful_1", "prompt": "How to do harm?", "video": None}
    assert got["howto"][1]["prompt"] == probe.PREFIX + probe.HOWTO[1] and got["howto"][1]["video"] == "v.mp4"
    assert got["describe"][0]["prompt"] == probe.PREFIX + probe.DESCRIBE
    assert "How to do harm?" in got["judge"][0]["prompt"] and "Yes or No" in got["judge"][0]["prompt"]
    with pytest.raises(ValueError, match="video-framing sentence"):
        probe.conditions(samples, ["harmful_1"], {"harmful_1": "How to do harm?"})
