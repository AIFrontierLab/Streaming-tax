"""Calibration recipe pieces that run on CPU without models: inputs, layer / strength selection, harmless-request requests."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from streamreplace.plugin import artefacts
from streamreplace.plugin.calibrate import directions as Dn
from streamreplace.plugin.calibrate import harmless, inputs
from streamreplace.plugin.recipe import VIDEO_PREFIX

LAYERS = {"mmduet_7b": 28, "thinkstream_3b": 36, "mmduet2_3b": 36, "aura_8b": 36, "livestar_8b": 32, "moss_vl_rt": 48}


def test_refusal_direction_auc_and_layer_window():
    rng = np.random.default_rng(0)
    L, d = 10, 6
    h0 = rng.normal(size=(40, L, d)).astype(np.float32)
    labels = ["harmful"] * 20 + ["harmless"] * 20
    h0[:20, :, 0] += 3.0
    r, rhat = Dn.refusal_direction(h0, labels)
    assert np.allclose(np.linalg.norm(rhat, axis=1), 1) and r.shape == (L, d) and (rhat[:, 0] > 0.5).all()
    aucs = [0.1, 0.9, 0.2, 0.7, 0.8, 0.8, 0.95, 0.3, 0.2, 0.1]
    assert Dn.pick_r_layer(aucs, list(range(L))) == 4                  # [3, 6): the first maximum wins
    assert Dn.pick_r_layer(aucs, [0, 1, 2, 3, 5, 6, 7, 8, 9]) == 5     # a skipped layer is never chosen
    assert Dn.auc([2, 3], [0, 1]) == 1.0 and Dn.auc([1], [1]) == 0.5


@pytest.mark.parametrize("child", sorted(LAYERS))
def test_shipped_layers_follow_the_recipe(child):
    L = LAYERS[child]
    art = artefacts.load(child)
    assert int(0.3 * L) <= art.r_layer < int(0.6 * L)
    assert art.p_layer in Dn.candidate_layers(L, list(range(L)))
    assert art.alpha in Dn.ALPHAS


def test_candidates_snap_to_runnable_layers_and_ties_go_first():
    # x * 35 for x in linspace(0.2, 0.8, 7) = 7.0, 10.500000000000002, 14.0, 17.5, 21.0, 24.500000000000004, 28.0:
    # float noise rounds two of them up; 17.5 rounds half to even
    assert Dn.candidate_layers(36, list(range(36))) == [7, 11, 14, 18, 21, 25, 28]
    assert Dn.candidate_layers(36, [l for l in range(36) if l != 14]) == [7, 11, 13, 18, 21, 25, 28]
    kl = {"base": 0.5, "L7_a0.5": 0.3, "L7_a1.0": 0.2, "L10_a0.5": 0.2}
    assert Dn.pick_prs(kl) == (7, 1.0)
    lpp = np.log(np.array([0.5, 0.25, 0.25], dtype=np.float16))
    lpc = np.log(np.array([0.25, 0.5, 0.25], dtype=np.float32))
    expected = float((np.exp(lpp.astype(np.float32)) * (lpp.astype(np.float32) - lpc)).sum())
    assert Dn.kl_to_parent(lpp, lpc) == expected and expected > 0


def write_sources(tmp_path):
    adv = io.StringIO()
    w = csv.writer(adv)
    w.writerow(["goal", "target"])
    for k in range(520):
        w.writerow([f"harmful goal {k}", "Sure"])
    (tmp_path / "advbench.csv").write_text(adv.getvalue())
    alpaca = [{"instruction": f"task {k}", "input": "" if k % 3 else "context", "output": ""} for k in range(1200)]
    (tmp_path / "alpaca.json").write_text(json.dumps(alpaca))
    return tmp_path / "advbench.csv", tmp_path / "alpaca.json"


def test_random0_procedure_matches_sampling_the_texts(tmp_path):
    adv, alp = write_sources(tmp_path)
    rows = inputs.sample_texts(adv, alp)
    rng = random.Random(0)                                              # the paper's procedure, on the texts
    harmful = rng.sample(inputs.advbench_goals(adv), 300)
    harmless = rng.sample(inputs.alpaca_without_input(alp)[0], 300)
    assert [r["text"] for r in rows] == harmful + harmless
    assert [r["fold"] for r in rows[:6]] == [1, 0, 0, 1, 0, 0] and sum(r["fold"] == 0 for r in rows) == 400
    data = json.loads(alp.read_text())
    assert all(data[r["row"]]["instruction"] == r["text"] for r in rows if r["source"] == "alpaca")
    frozen = {"sources": {"advbench": {"sha256": inputs.sha256_file(adv)}, "alpaca": {"sha256": inputs.sha256_file(alp)}},
              "items": [{k: v for k, v in r.items() if k != "text"} for r in rows]}
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    assert inputs.texts_from_frozen(tmp_path / "frozen.json", adv, alp) == rows
    frozen["sources"]["alpaca"]["sha256"] = "0" * 64
    (tmp_path / "frozen.json").write_text(json.dumps(frozen))
    with pytest.raises(ValueError, match="sha256"):
        inputs.texts_from_frozen(tmp_path / "frozen.json", adv, alp)


def test_calibration_videos_and_folds():
    samples = [SimpleNamespace(metadata={"video": f"yt{k}"}, video=f"media/yt{k}.mp4") for k in range(300) for _ in (0, 1)]
    videos = inputs.calibration_videos(samples)
    assert len(videos) == 300 and [v for v, _ in videos] == sorted(
        (f"yt{k}" for k in range(300)), key=lambda v: hashlib.sha256(v.encode()).hexdigest())
    texts = [{"id": f"t{i}", "text": "x", "label": "harmful" if i < 300 else "harmless", "fold": 0 if i % 3 else 1}
             for i in range(600)]
    f0, f1 = inputs.calib_inputs(texts, videos)
    assert (len(f0), len(f1)) == (496, 248)
    assert f0[400] == ("vid0", inputs.GENERIC[0], videos[0][1], None) and f1[200][0] == "vid96"
    assert f1[-1] == ("vid143", inputs.GENERIC[143 % 12], videos[143][1], None)
    req = inputs.request(f0[400], {"generation": {"max_new_tokens": 512}})
    assert req.prompt == VIDEO_PREFIX + inputs.GENERIC[0] and req.video == videos[0][1] and req.kind == "open"
    assert inputs.request(f0[0], {}).prompt == "x"


def test_harmless_requests_and_clip_command():
    items = harmless.load(artefacts.root_dir().parent / "paper/frozen/plugin/harmless_requests.json")
    howto = next(it for it in items if it["set"] == "howto")
    describe = next(it for it in items if it["set"] == "describe")
    assert harmless.request_text(howto) == VIDEO_PREFIX + harmless.HOWTO[howto["template"]]
    assert harmless.request_text(describe) == VIDEO_PREFIX + "Describe what happens in the video."
    assert sorted({it["template"] for it in items if it["set"] == "howto"}) == list(range(8))
    reqs = harmless.requests(items[:2], "clips", {"stream": {"fps": 1}, "generation": {"max_new_tokens": 1024}})
    assert reqs[0].settings["generation"]["max_new_tokens"] == 160 and reqs[0].video.endswith(f"{howto['vid']}.mp4")
    cmd = harmless.cut_command("ffmpeg", "src.mp4", 22.252, "out.mp4")
    assert cmd[cmd.index("-ss") + 1] == "22.252" and cmd[cmd.index("-frames:v") + 1] == "313"
    assert "scale=1440:1440:flags=lanczos" in cmd[cmd.index("-vf") + 1] and "-an" in cmd
