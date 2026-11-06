"""Configuration loading and validation (config/load.py, config/schema.py): the shipped configs describe the paper's
11 models, 5 pairs and every benchmark; typos and inconsistent files are refused; local and extra files merge."""
from __future__ import annotations

import re

import pytest

from streamreplace.benchmarks import registry as benchmarks
from streamreplace.config import CheckpointRef, load_config
from streamreplace.config.schema import ModelSpec
from streamreplace.core.protocol import Role
from streamreplace.models import registry as adapters

REVISION = re.compile(r"^[0-9a-f]{40}$")
PAIRS = {"p1": ("llava_ov_7b", ("mmduet_7b",)), "p2": ("qwen25vl_3b", ("thinkstream_3b", "mmduet2_3b")),
         "p3": ("qwen3vl_8b", ("aura_8b",)), "p4": ("livestar_iv_8b", ("livestar_8b",)),
         "p5": ("moss_vl", ("moss_vl_rt",))}


@pytest.fixture(scope="module")
def shipped():
    return load_config()


def test_shipped_pairs_and_models(shipped):
    assert {pid: (p.parent, p.children) for pid, p in shipped.pairs.items()} == PAIRS
    assert len(shipped.models) == 11
    for pair in shipped.pairs.values():
        assert shipped.model(pair.parent).role is Role.PARENT and shipped.model(pair.parent).stream is None
        for child in pair.children:
            spec = shipped.model(child)
            assert spec.role is Role.CHILD and spec.offline is not None and spec.stream is not None
            assert spec.pair == pair.id and spec.offline.adapter == shipped.model(pair.parent).offline.adapter


def test_every_checkpoint_is_pinned_to_a_revision(shipped):
    for spec in shipped.models.values():
        for role, ref in spec.checkpoints().items():
            assert REVISION.match(ref.revision or ""), f"{spec.id}.{role}: {ref}"
        assert spec.decoding.repetition_penalty in (1.0, 1.05)


def test_benchmarks_suites_and_registry_agree(shipped):
    assert list(shipped.benchmarks) == list(benchmarks.BENCHMARKS)
    assert len(shipped.suites["paper-core"]) == 15 and shipped.suites["paper-supp"] == ("visulogic", "videoreasonbench")
    assert shipped.suites["finetune"] == ("lingoqa", "eyepcr", "openuav_qa")
    assert shipped.expand_benchmarks("paper-core,paper-supp,mme") == [*shipped.suites["paper-core"],
                                                                      *shipped.suites["paper-supp"]]
    assert shipped.expand_benchmarks(["all"]) == list(shipped.benchmarks)
    assert shipped.expand_pairs("all") == ["p1", "p2", "p3", "p4", "p5"]
    for spec in shipped.benchmarks.values():
        assert spec.media in ("image", "video")
    assert {m.adapter(mode).adapter for m in shipped.models.values() for mode in m.modes} <= set(adapters.names())


def test_sections_are_kept_whole(shipped):
    assert {"judges", "analysis", "plugin"} <= set(shipped.sections)
    section = shipped.section("plugin")
    section["changed"] = True
    assert "changed" not in shipped.section("plugin")            # a copy


def test_small_project_loads(tmp_config):
    config = tmp_config()
    assert config.pair_of("child").id == "p" and config.model("parent").modes == ("offline",)
    assert config.settings("demo", "child", "stream")["video"] == {"nframes": 8, "max_pixels": 200704, "fps": 2,
                                                                     "max_frames": 16}


@pytest.mark.parametrize("parts, message", [
    ({"models": {"m": {"role": "parent", "checkpoint": "a/b@1", "offline": {"adapter": "mock", "env": "local"},
                       "typo": 1}}, "pairs": {}}, "unknown keys"),
    ({"models": {"m": {"role": "parent", "checkpoint": "a/b@1"}}, "pairs": {}}, "offline and/or a stream block"),
    ({"models": {"m.x": {"role": "parent", "checkpoint": "a/b@1", "offline": {"adapter": "mock", "env": "local"}}},
      "pairs": {}}, "invalid model id"),
    ({"models": {"m@sfix": {"role": "child", "checkpoint": "a/b@1", "offline": {"adapter": "mock", "env": "local"}}},
      "pairs": {}}, "invalid model id"),
    ({"models": {"m": {"role": "parent", "checkpoint": "no-org", "offline": {"adapter": "mock", "env": "local"}}},
      "pairs": {}}, "repository reference"),
    ({"benchmarks": {"demo": {"media": "audio"}}}, "media"),
    ({"benchmarks": {"demo": {"media": "video", "offline_frames": {"max_pixels": 1}}}}, "frame-sampling keys"),
    ({"benchmarks": {"demo": {"media": "video", "settings": {"gpus": [0]}}}}, "not generation settings"),
    ({"pairs": {"p": {"parent": "parent", "children": "child"}}}, "non-empty list"),
])
def test_typos_are_refused(tmp_config, parts, message):
    with pytest.raises(ValueError, match=message):
        tmp_config(**parts)


def test_cross_file_problems_are_listed_together(tmp_config):
    models = {
        "parent": {"role": "parent", "checkpoint": "t/p@0", "offline": {"adapter": "mock", "env": "local"},
                   "stream": {"adapter": "mock", "env": "local"}},
        "child": {"role": "parent", "checkpoint": "t/c@0", "offline": {"adapter": "nope", "env": "local"}},
        "other": {"role": "child", "checkpoint": "t/o@0", "offline": {"adapter": "mmduet_stream", "env": "local"}},
    }
    pairs = {"p": {"parent": "parent", "children": ["child", "ghost"]}, "q": {"parent": "other", "children": ["child"]}}
    with pytest.raises(ValueError) as error:
        tmp_config(models=models, pairs=pairs, suites={"demo": ["demo"], "s": ["missing"]})
    text = str(error.value)
    for expected in ("unknown model 'ghost'", "is in pairs p and q", "parents are evaluated offline only",
                     "child 'child' has role parent", "parent 'other' has role child", "unknown adapter 'nope'",
                     "does not serve offline mode", "suites.demo: the name is taken", "unknown benchmark 'missing'"):
        assert expected in text


def test_registry_check_can_be_skipped(tmp_config):
    models = {"m": {"role": "parent", "checkpoint": "t/m@0", "offline": {"adapter": "not_written_yet", "env": "x"}}}
    assert tmp_config(models=models, pairs={}, check_registry=False).model("m").offline.adapter == "not_written_yet"


def test_core_keys_are_defined_once(config_dir):
    folder = config_dir()
    (folder / "suites.yaml").write_text("suites: {}\n")
    with pytest.raises(ValueError, match="core key"):
        load_config(folder)
    (folder / "suites.yaml").unlink()
    (folder / "more.yaml").write_text("x: 1\n")
    (folder / "pairs.yaml").write_text("pairs: {}\nmodels: {}\n")
    with pytest.raises(ValueError, match="already defined"):
        load_config(folder)


def test_local_and_extra_files_merge_over_the_folder(config_dir, tmp_path):
    folder = config_dir(sections={"analysis": {"engine": {"reps": 10000, "seed": 0}}},
                        local={"defaults": {"generation": {"max_new_tokens": 32}}, "suites": {"quick": ["demo"]},
                               "analysis": {"engine": {"reps": 100}}, "paths": {"home": "elsewhere"}})
    extra = tmp_path / "extra.yaml"
    extra.write_text("models:\n  ft:\n    role: child\n    checkpoint: test/child@0\n"
                     "    offline: {adapter: mock, env: local}\n    options: {lora: adapters/ft, deterministic: true}\n"
                     "analysis:\n  engine: {seed: 1}\n")
    config = load_config(folder, extra=[extra])
    assert config.defaults["generation"]["max_new_tokens"] == 32 and config.suites["quick"] == ("demo",)
    assert config.section("analysis") == {"engine": {"reps": 100, "seed": 1}}
    assert config.paths_block == {"home": "elsewhere"}
    assert config.model("ft").options == {"lora": "adapters/ft", "deterministic": True} and config.model("ft").pair is None
    assert config.sources[-2:] == [folder / "local.yaml", extra]
    with pytest.raises(ValueError, match="unknown path keys"):
        load_config(config_dir(local={"paths": {"modelz": "x"}}))


def test_checkpoint_refs():
    ref = CheckpointRef.parse("Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3")
    assert (ref.repo, ref.revision) == ("Qwen/Qwen2.5-VL-3B-Instruct", "66285546d2b821cf421d4f5eb2576359d3770cd3")
    assert str(ref) == "Qwen/Qwen2.5-VL-3B-Instruct@66285546d2b821cf421d4f5eb2576359d3770cd3"
    assert CheckpointRef.parse({"repo": "a/b"}) == CheckpointRef("a/b") and str(CheckpointRef("a/b")) == "a/b"
    assert CheckpointRef.parse(ref) is ref and ref.as_dict()["repo"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    for bad in ("", "noslash@1", 3, {"revision": "1"}):
        with pytest.raises(ValueError):
            CheckpointRef.parse(bad)


def test_identity_block_of_a_model(shipped):
    block = shipped.model("mmduet_7b").identity_block("stream", 2)
    assert block == {"checkpoint": {"repo": "wangyueqian/MMDuet", "revision": "23669174afcaed24044225a6b89eae8427557cce"},
                     "base": {"repo": "lmms-lab/llava-onevision-qwen2-7b-ov",
                              "revision": "0b07bf7565e244cf4f39982249eafe8cd799d6dd"},
                     "vision": {"repo": "google/siglip-so400m-patch14-384",
                                "revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"},
                     "adapter": "mmduet_stream", "adapter_protocol": 2, "decoding": {"repetition_penalty": 1.05},
                     "plugin": None}
    spec = ModelSpec.parse("x", {"role": "parent", "checkpoint": "a/b@1", "offline": {"adapter": "mock", "env": "e"},
                                 "options": {"accuracy": 0.5}})
    assert spec.identity_block("offline", 1)["options"] == {"accuracy": 0.5}
    with pytest.raises(ValueError, match="no stream adapter"):
        spec.adapter("stream")
