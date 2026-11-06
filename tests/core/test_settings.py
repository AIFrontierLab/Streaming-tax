"""Settings resolution (config/settings.py): the merge rule, --set parsing, the layer order, and the resolved settings
of real paper jobs on the shipped configs, compared with the settings the original runs recorded in their meta.json
files (without the judge block those files also held)."""
from __future__ import annotations

import pytest

from streamreplace.config import load_config
from streamreplace.config.settings import SAMPLING_KEYS, check_layer, compose_settings, merge, parse_sets

DEFAULTS = {"generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
            "video": {"nframes": 64, "max_pixels": 200704}, "stream": {"fps": 1}}


def test_merge_is_recursive_for_dicts_and_replaces_everything_else():
    base = {"video": {"fps": 2, "pad": True}, "list": [1, 2], "x": 1}
    over = {"video": {"fps": 1}, "list": [3], "y": {"a": 1}}
    assert merge(base, over) == {"video": {"fps": 1, "pad": True}, "list": [3], "x": 1, "y": {"a": 1}}
    assert merge(None, {}, {"a": {"b": 1}}) == {"a": {"b": 1}}
    assert merge({"a": {"b": 1}}, {"a": 5}) == {"a": 5}              # a scalar replaces a dict ...
    assert merge({"a": 5}, {"a": {"b": 1}}) == {"a": {"b": 1}}       # ... and a dict replaces a scalar


def test_merge_copies_its_inputs():
    inner = {"fps": 2}
    out = merge({"video": inner}, {"list": [inner]})
    out["video"]["fps"] = 9
    out["list"][0]["fps"] = 8
    assert inner == {"fps": 2}


def test_parse_sets_reads_yaml_values_into_nested_sections():
    assert parse_sets(["video.nframes=32", "video.sampler=qwen_vl_utils", "generation.max_new_tokens=8192",
                       "stream.window=full", "system='Be brief: answer'"]) == {
        "video": {"nframes": 32, "sampler": "qwen_vl_utils"}, "generation": {"max_new_tokens": 8192},
        "stream": {"window": "full"}, "system": "Be brief: answer"}         # values are YAML: quote a colon
    assert parse_sets(["video.pad_square=true", "video.fps=0.5"]) == {"video": {"pad_square": True, "fps": 0.5}}
    assert parse_sets(None) == {}


@pytest.mark.parametrize("items, message", [
    (["video.nframes"], "field=value"),
    (["gpus=0,1"], "not generation settings"),
    (["judge.gpus=[0]"], "not generation settings"),
])
def test_parse_sets_refuses_malformed_and_non_generation_keys(items, message):
    with pytest.raises(ValueError, match=message):
        parse_sets(items)


def test_check_layer_validates_sections():
    assert check_layer(None, "x") == {}
    with pytest.raises(ValueError, match="must be a mapping"):
        check_layer([1], "x")
    with pytest.raises(ValueError, match="video must be a mapping"):
        check_layer({"video": 3}, "x")
    assert check_layer({"system": "text"}, "x") == {"system": "text"}


def test_layer_order_defaults_benchmark_pair_model_extra_then_overrides():
    out = compose_settings(DEFAULTS, {"generation": {"max_new_tokens": 1024}}, {"fps": 2, "max_pixels": 401408},
                           {"video": {"max_pixels": 1}}, "stream", None, {"video": {"fps": 3}},
                           {"stream": {"window": "full"}, "video": {"max_pixels": 2}}, media="video")
    assert out == {"generation": {"max_new_tokens": 1024}, "image": {"max_pixels": 1003520},
                   "video": {"nframes": 64, "max_pixels": 2, "fps": 3}, "stream": {"fps": 1, "window": "full"}}


def test_offline_frames_replace_only_the_sampling_keys_and_only_offline():
    pair = {"fps": 2, "max_frames": 160, "pad_square": True}
    frames = {"nframes": 16, "sampler": "linspace_int"}
    offline = compose_settings(DEFAULTS, None, pair, None, "offline", frames)
    assert offline["video"] == {"max_pixels": 200704, "pad_square": True, "nframes": 16, "sampler": "linspace_int"}
    assert not (set(offline["video"]) & (SAMPLING_KEYS - set(frames)))
    stream = compose_settings(DEFAULTS, None, pair, None, "stream", frames)
    assert stream["video"] == {"nframes": 64, "max_pixels": 200704, "fps": 2, "max_frames": 160, "pad_square": True}


def test_image_benchmarks_pin_video_and_stream_to_the_defaults():
    out = compose_settings(DEFAULTS, {"generation": {"max_new_tokens": 16}}, {"fps": 2, "max_pixels": 9},
                           {"stream": {"fps": 7}}, "offline", None, {"video": {"nframes": 3}}, media="image")
    assert out == {**DEFAULTS, "generation": {"max_new_tokens": 16}}
    kept = compose_settings(DEFAULTS, None, {"fps": 2}, None, "offline", media="image", pin_image=False)
    assert kept["video"]["fps"] == 2


# Settings recorded by the original runs in <dataset>/<model>.<mode>.meta.json (the "judge" key removed).
RECORDED = {
    ("videomme", "llava_ov_7b", "offline"): {
        "generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 200704, "fps": 2, "max_frames": 160, "pad_square": True},
        "stream": {"fps": 1}},
    ("videomme", "mmduet_7b", "stream"): {
        "generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 200704, "fps": 2, "max_frames": 160, "pad_square": True},
        "stream": {"fps": 1}},
    ("videomme", "thinkstream_3b", "stream"): {
        "generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 401408, "sampler": "qwen_vl_utils", "fps": 2, "min_pixels": 200704},
        "stream": {"fps": 1}},
    ("videomme", "aura_8b", "stream"): {
        "generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 602112, "sampler": "qwen_vl_utils", "fps": 2}, "stream": {"fps": 1}},
    ("tomato", "livestar_8b", "offline"): {
        "generation": {"max_new_tokens": 1024}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 200704, "fps": 1}, "stream": {"fps": 1},
        "system": "You are an expert in understanding dynamics of objects."},
    ("video_safetybench", "moss_vl", "offline"): {
        "generation": {"max_new_tokens": 1024}, "image": {"max_pixels": 1003520},
        "video": {"max_pixels": 16777216, "video_max_pixels": 65536000, "nframes": 16, "sampler": "linspace_int"},
        "stream": {"fps": 1}},
    ("video_safetybench", "moss_vl_rt", "stream"): {
        "generation": {"max_new_tokens": 1024}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 16777216, "fps": 1, "video_max_pixels": 65536000},
        "stream": {"fps": 1}},
    ("mcv_safetybench", "llava_ov_7b", "offline"): {
        "generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
        "video": {"max_pixels": 200704, "pad_square": True, "nframes": 33, "sampler": "linspace_int"},
        "stream": {"fps": 1}},
    ("vmdt_privacy", "mmduet2_3b", "stream"): {
        "generation": {"max_new_tokens": 1024}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 401408, "sampler": "qwen_vl_utils", "fps": 2, "min_pixels": 200704},
        "stream": {"fps": 1}},
    ("mme", "qwen25vl_3b", "offline"): {
        "generation": {"max_new_tokens": 16}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 200704}, "stream": {"fps": 1}},
    ("harmbench", "aura_8b", "offline"): {
        "generation": {"max_new_tokens": 512}, "image": {"max_pixels": 1003520},
        "video": {"nframes": 64, "max_pixels": 200704}, "stream": {"fps": 1}},
}


@pytest.fixture(scope="module")
def shipped():
    return load_config()


@pytest.mark.parametrize("job", sorted(RECORDED), ids=lambda j: f"{j[0]}/{j[1]}.{j[2]}")
def test_resolved_settings_equal_the_original_runs(shipped, job):
    assert shipped.settings(*job) == RECORDED[job]


def test_set_overrides_and_extra_layers_on_the_shipped_configs(shipped):
    out = shipped.settings("videomme", "thinkstream_3b", "stream", overrides=parse_sets(["stream.window=8"]),
                           extra={"stream": {"window": "full", "max_len": 81920}})
    assert out["stream"] == {"fps": 1, "window": 8, "max_len": 81920}      # --set wins over the plugin's Recall layer
    with pytest.raises(ValueError, match="unknown mode"):
        shipped.settings("videomme", "thinkstream_3b", "live")
    with pytest.raises(ValueError, match="Unknown benchmark"):
        shipped.settings("nope", "thinkstream_3b", "stream")
