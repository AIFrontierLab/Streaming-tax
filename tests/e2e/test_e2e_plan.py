"""Planning against the shipped configs (no data, no GPU): the paper's 235 jobs, identities, plugin arms; plus the
pieces the dispatcher relies on (adapter constants read from source, worker environment, the mock model)."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from streamreplace.config.load import load_config
from streamreplace.config.schema import ModelSpec
from streamreplace.core.identity import run_identity
from streamreplace.core.protocol import Mode
from streamreplace.core.schema import Request
from streamreplace.models.mock import MockModel
from streamreplace.paths import resolve
from streamreplace.run.dispatch import payload_differences
from streamreplace.run.envs import src_dir, worker_vars
from streamreplace.run.plan import AdapterConstants, _source_constants, adapter_constants, plan

REPO = Path(__file__).resolve().parents[2]
IMAGE = ("dynamath", "mathvista", "mmbench", "mme", "mmmu", "mmstar", "scienceqa", "harmbench", "visulogic")
VIDEO = ("videomme", "tomato", "timeblind", "video_safetybench", "mcv_safetybench", "vmdt_fairness", "vmdt_privacy",
         "videoreasonbench")
PARENTS = ("llava_ov_7b", "qwen25vl_3b", "qwen3vl_8b", "livestar_iv_8b", "moss_vl")
CHILDREN = ("mmduet_7b", "thinkstream_3b", "mmduet2_3b", "aura_8b", "livestar_8b", "moss_vl_rt")


@pytest.fixture
def shipped(tmp_path):
    return load_config(REPO / "configs"), resolve(home=tmp_path / "home")


def test_paper_plan_has_the_235_jobs(shipped):
    config, paths = shipped
    jobs = plan(config, paths, pairs="all", protocols="P,C,S", benchmarks="paper-core,paper-supp")
    names = [job.name for job in jobs]
    assert len(names) == len(set(names)) == 235
    per = Counter(job.dataset for job in jobs)
    assert {d: per[d] for d in IMAGE} == {d: 11 for d in IMAGE}           # P/C: every model offline
    assert {d: per[d] for d in VIDEO} == {d: 17 for d in VIDEO}           # + the six children's streams
    expected = {f"{d}/{m}.offline" for d in IMAGE + VIDEO for m in PARENTS + CHILDREN}
    expected |= {f"{d}/{m}.stream" for d in VIDEO for m in CHILDREN}
    assert set(names) == expected
    assert not any(job.arm.mode is Mode.STREAM and job.arm.model in PARENTS for job in jobs)
    stream = next(job for job in jobs if job.name == "videomme/aura_8b.stream")
    assert stream.adapter == "aura_stream" and stream.env == "aura" and stream.settings["stream"] == {"fps": 1}
    assert all(job.outputs == paths.outputs for job in jobs)


def test_conditions_select_arms(shipped):
    config, paths = shipped
    names = lambda **kw: [job.name for job in plan(config, paths, benchmarks="videomme,mmbench", **kw)]
    assert names(pairs="p3", protocols="P") == ["videomme/qwen3vl_8b.offline", "mmbench/qwen3vl_8b.offline"]
    assert names(pairs="p3", protocols="C,S") == ["videomme/aura_8b.offline", "videomme/aura_8b.stream",
                                                  "mmbench/aura_8b.offline"]
    assert names(models="aura_8b", protocols="S") == ["videomme/aura_8b.stream"]
    with pytest.raises(ValueError, match="unknown protocols"):
        names(pairs="p3", protocols="P,X")
    with pytest.raises(ValueError, match="choose --pairs"):
        names(protocols="P")


def test_image_identities_ignore_the_pairs_video_protocol(shipped):
    """Image benchmarks never read video settings: changing a pair's video protocol changes the identities of its
    video jobs only."""
    config, paths = shipped
    before = {j.name: run_identity(j.payload) for j in plan(config, paths, pairs="all", benchmarks="mmbench,videomme")}
    for pair_id, pair in list(config.pairs.items()):
        config.pairs[pair_id] = replace(pair, video={"fps": 7, "max_pixels": 1})
    after = {j.name: run_identity(j.payload) for j in plan(config, paths, pairs="all", benchmarks="mmbench,videomme")}
    assert all(before[n] == after[n] for n in before if n.startswith("mmbench/"))
    assert not any(before[n] == after[n] for n in before if n.startswith("videomme/"))


def test_set_overrides_enter_settings_and_identity(shipped):
    config, paths = shipped
    base, = plan(config, paths, models="qwen25vl_3b", protocols="P", benchmarks="mathvista")
    capped, = plan(config, paths, models="qwen25vl_3b", protocols="P", benchmarks="mathvista",
                   sets={"generation": {"max_new_tokens": 8192}})
    assert base.settings["generation"]["max_new_tokens"] == 2048
    assert capped.settings["generation"]["max_new_tokens"] == 8192
    assert payload_differences(base.payload, capped.payload) == ["settings.generation.max_new_tokens: 2048 -> 8192"]


def test_plugin_arms_of_the_shipped_config(shipped):
    """AURA's plugin arms run on its Hugging Face rebuild with the full-history Recall; ThinkStream's Recall widens
    its window. The plugin block (recipe, Recall, artefact hashes) is part of the identity."""
    pytest.importorskip("streamreplace.plugin.hooks")
    config, paths = shipped
    jobs = {j.name: j for j in plan(config, paths, pairs="p2,p3", protocols="", variants="sfix",
                                    benchmarks="videomme")}
    assert sorted(jobs) == ["videomme/aura_8b@sfix.stream", "videomme/mmduet2_3b@sfix.stream",
                            "videomme/thinkstream_3b@sfix.stream"]
    aura = jobs["videomme/aura_8b@sfix.stream"]
    assert aura.adapter == "aura_stream_hf" and aura.payload["model"]["adapter"] == "aura_stream_hf"
    assert aura.settings["stream"] == {"fps": 1, "window": "full"}
    thinkstream = jobs["videomme/thinkstream_3b@sfix.stream"]
    assert thinkstream.settings["stream"] == {"fps": 1, "window": "full", "max_len": 81920}
    assert jobs["videomme/mmduet2_3b@sfix.stream"].settings["stream"] == {"fps": 1}
    assert all(j.payload["model"]["plugin"]["recipe"]["name"] == "sfix" for j in jobs.values())
    assert all(j.plugin == {"child": j.arm.model, "variant": "sfix", "section": config.section("plugin")}
               for j in jobs.values())


def test_adapter_constants_are_read_from_the_source(tmp_path, monkeypatch):
    assert adapter_constants("mock") == AdapterConstants(1, ("stream.mock_load",))
    package = tmp_path / "fakeadapters"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "a.py").write_text(
        "from streamreplace.models.base import BaseAdapter\n"
        "class A(BaseAdapter):\n    LOAD_KEYS = ('video.max_pixels',)\n"
        "class B(A):\n    PROTOCOL_VERSION = 3\n"
        "class C(A):\n    LOAD_KEYS = A.LOAD_KEYS + ('stream.max_len',)\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert _source_constants("fakeadapters.a", "A") == {"LOAD_KEYS": ("video.max_pixels",), "PROTOCOL_VERSION": 1}
    assert _source_constants("fakeadapters.a", "B") == {"PROTOCOL_VERSION": 3, "LOAD_KEYS": ("video.max_pixels",)}
    assert _source_constants("fakeadapters.a", "C") is None             # not a literal: asked from the model env
    assert _source_constants("fakeadapters.missing", "A") is None


def test_worker_environment(tmp_path):
    paths = resolve(home=tmp_path)
    env = worker_vars(paths, "local", 3, 4, ffmpeg=True, deterministic=True, base={"PATH": "/usr/bin"})
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["PYTHONPATH"] == str(src_dir())
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == env["OPENBLAS_NUM_THREADS"] == "4"
    assert env["HF_HUB_OFFLINE"] == "1" and env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["PATH"].endswith("/usr/bin") and env["FLASHINFER_WORKSPACE_BASE"] == str(paths.cache / "flashinfer")
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and env["FLASH_ATTENTION_DETERMINISTIC"] == "1"
    assert env["STREAMREPLACE_CACHE"] == str(paths.cache) and env["STREAMREPLACE_OUTPUTS"] == str(paths.outputs)
    assert "LD_LIBRARY_PATH" not in env                                 # no FFmpeg libraries under <envs>/ffmpeg
    (paths.envs / "ffmpeg" / "lib").mkdir(parents=True)
    env = worker_vars(paths, "local", 0, 1, ffmpeg=True, base={"LD_LIBRARY_PATH": "/opt/lib"})
    assert env["LD_LIBRARY_PATH"].split(":") == [str(paths.envs / "ffmpeg" / "lib"), "/opt/lib"]
    assert "CUBLAS_WORKSPACE_CONFIG" not in worker_vars(paths, "local", 0, 1, base={})


def test_mock_model_is_deterministic():
    spec = ModelSpec.parse("m", {"role": "child", "checkpoint": "test/m",
                                 "offline": {"adapter": "mock", "env": "local"},
                                 "options": {"accuracy": 0.5, "seed": 3, "fail": ["bad"], "concurrency": 4}})
    model = MockModel(spec, {"stream": {"fps": 2}}, None)
    request = lambda i: Request(i, "demo", "prompt", reference="A", settings={"stream": {"fps": 2}})
    replies = [model.generate(request(f"id{i}")).text for i in range(40)]
    assert replies == [MockModel(spec, {}, None).generate(request(f"id{i}")).text for i in range(40)]
    assert set(replies) == {"The answer is A.", "The answer is B."} and model.concurrency == 4
    assert model.stream(request("id0")).info["stream_fps"] == 2
    with pytest.raises(RuntimeError, match="configured mock failure"):
        model.generate(request("bad"))
