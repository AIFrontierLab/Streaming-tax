"""The offline adapters' input rules on CPU stand-ins (no torch, transformers or checkpoints): Qwen's resize rule and
frame handling, LiveStar's chat text, MOSS's offline call, the decoding arguments, the Qwen2.5 position patch."""
from __future__ import annotations

import sys
import types

import pytest

from streamreplace.media.digest import frame_digest
from streamreplace.models import internvl_offline, llava_offline, moss_offline, qwen_offline, registry
from streamreplace.models.decoding import greedy
from streamreplace.models.patches import qwen25_rope
from offline_fakes import FakeChatModel, FakeQwenProcessor, FakeTokenizer, fake_offline_frames, frames, request

P2 = {"sampler": "qwen_vl_utils", "fps": 2, "min_pixels": 200704, "max_pixels": 401408}
P5 = {"fps": 1, "max_pixels": 16777216, "video_max_pixels": 65536000}
MCV_P2 = {"min_pixels": 200704, "max_pixels": 401408, "nframes": 33, "sampler": "linspace_int"}
OFFLINE = {"qwen_offline": "QwenOffline", "llava_offline": "LlavaOffline", "internvl_offline": "InternVLOffline",
           "moss_offline": "MossOffline"}


@pytest.mark.parametrize("name", OFFLINE)
def test_registry_names_import_with_the_core_dependencies_only(name):
    cls = registry.load_class(name)
    assert cls.__name__ == OFFLINE[name] and registry.modes(name) == {"offline"}
    assert cls.PROTOCOL_VERSION == 1 and cls.LOAD_KEYS == () and cls.CAPTURES_LETTER_LOGITS
    for method in ("build_inputs", "generation_kwargs", "generate", "encode", "train_forward", "close"):
        assert callable(getattr(cls, method)), method


@pytest.mark.parametrize("height, width, factor, low, high, expected", [
    (720, 1280, 28, 200704, 401408, (448, 840)),      # p2 video frames of a 720p clip (the formal rows: 840x448)
    (720, 1280, 32, 131072, 602112, (576, 1024)),     # p3 (formal rows: 1024x576)
    (720, 1280, 28, 3136, 1003520, (728, 1288)),      # an image within the budget: nearest multiples
    (720, 1280, 32, 3136, 1003520, (704, 1280)),      # 720 / 32 = 22.5 rounds half to even (22), as Qwen's code
    (20, 20, 28, 3136, 1003520, (56, 56)),            # below the floor: scaled up, ceiled
])
def test_smart_resize_goldens(height, width, factor, low, high, expected):
    assert qwen_offline.smart_resize(height, width, factor, low, high) == expected


def qwen(qwen3: bool = False):
    adapter = object.__new__(qwen_offline.QwenOffline)
    adapter.qwen3, adapter.factor = qwen3, 32 if qwen3 else 28
    adapter.rope_fix = None if qwen3 else True
    adapter.attention, adapter.processor, adapter.repetition_penalty = "sdpa", FakeQwenProcessor(), 1.05
    return adapter


def test_qwen25_framework_frames_pad_an_odd_count_after_the_digest(monkeypatch):
    """MCV-SafetyBench's 33 frames: the digest covers the 33 sampled frames (the same for every model of the pair);
    the model gets 34 (the last frame repeated, temporal patch 2) at fps = 34 / duration."""
    sampled = fake_offline_frames(33, fps=4.0, size=(1280, 720))
    monkeypatch.setattr(qwen_offline, "offline_frames", sampled)
    adapter = qwen()
    info = adapter.build_inputs(request(path="clip.mp4", video=MCV_P2)).info
    expected_digest = frame_digest({"indices": list(range(33))})
    assert info["frame_indices"] == expected_digest and info["frames"] == 34
    assert info["frame_size"] == [840, 448] and info["rope_fix"] is True
    call = adapter.processor.calls[-1]
    assert call["videos"][0].shape == (34, 448, 840, 3)
    assert call["kwargs"] == {"do_sample_frames": False, "do_resize": False, "fps": [34 / 8.25]}
    assert info["sample_fps"] == round(34 / 8.25, 4) and info["duration"] == 8.25
    assert list(info)[:2] == ["frame_indices", "rope_fix"] and info["attention"] == "sdpa"


def test_qwen3_framework_frames_pass_the_padded_metadata(monkeypatch):
    monkeypatch.setattr(qwen_offline, "offline_frames", fake_offline_frames(33, fps=4.0))
    adapter = qwen(qwen3=True)
    info = adapter.build_inputs(request(path="clip.mp4", video=MCV_P2)).info
    metadata = adapter.processor.calls[-1]["kwargs"]["video_metadata"][0]
    assert metadata == {"fps": 4.0, "frames_indices": list(range(33)) + [32], "total_num_frames": 33,
                        "duration": 8.25}
    assert "rope_fix" not in info and "fps" not in adapter.processor.calls[-1]["kwargs"]


def test_qwen_images_system_prompt_and_content_order(monkeypatch):
    monkeypatch.setattr(qwen_offline, "open_rgb", lambda path: frames(1, (1280, 720))[0])
    adapter = qwen()
    req = request(images=["a.png", "b.png"], system="You are an expert.")
    info = adapter.build_inputs(req).info
    system, user = adapter.processor.messages
    assert system == {"role": "system", "content": "You are an expert."}
    assert [c["type"] for c in user["content"]] == ["image", "image", "text"] and user["content"][-1]["text"] == "Q?"
    assert adapter.processor.calls[-1]["kwargs"] == {"do_resize": False}
    assert info["image_sizes"] == [[1288, 728], [1288, 728]] and info["input_tokens"] > 0
    adapter.build_inputs(request(images=["a.png"]))
    assert [m["role"] for m in adapter.processor.messages] == ["user"]       # no system turn without a system prompt


def test_qwen_refuses_frame_lists_for_its_own_sampler():
    with pytest.raises(ValueError, match="frame list"):
        qwen()._fetch_video("clip.frames.json", P2)


def test_training_inputs_append_the_answer_and_end_of_turn(monkeypatch):
    monkeypatch.setattr(qwen_offline, "open_rgb", lambda path: frames(1)[0])
    adapter = qwen()
    enc = adapter.encode(request(images=["a.png"]), answer="B")
    assert enc["n"] == len("B<|im_end|>") and adapter.processor.calls[-1]["text"][0].endswith("<assistant>B<|im_end|>")


@pytest.mark.parametrize("module, cls", [(qwen_offline, "QwenOffline"), (llava_offline, "LlavaOffline"),
                                         (internvl_offline, "InternVLOffline")])
def test_generation_kwargs_are_greedy_plus_the_repetition_penalty(module, cls):
    adapter = object.__new__(getattr(module, cls))
    adapter.repetition_penalty = 1.05
    req = request(generation={"max_new_tokens": 2048})
    assert adapter.generation_kwargs(req) == {**greedy(req.settings), "repetition_penalty": 1.05}
    assert adapter.generation_kwargs(req) == {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                                              "num_beams": 1, "max_new_tokens": 2048, "repetition_penalty": 1.05}


def test_a_missing_checkpoint_folder_is_not_read_as_a_penalty_mismatch(tmp_path):
    from streamreplace.models.decoding import check_repetition_penalty, checkpoint_repetition_penalty
    with pytest.raises(FileNotFoundError, match="download models mmduet_7b"):
        check_repetition_penalty(1.05, tmp_path / "absent", "mmduet_7b")
    with pytest.raises(FileNotFoundError, match="no such folder"):
        checkpoint_repetition_penalty(tmp_path / "absent")
    assert checkpoint_repetition_penalty(tmp_path) == 1.0              # a folder without a generation config: 1.0
    with pytest.raises(ValueError, match="has no generation_config.json.*download models llava_ov_7b"):
        check_repetition_penalty(1.05, tmp_path, "llava_ov_7b")       # e.g. an incomplete download


def test_llava_reports_a_missing_lora_base_before_the_penalty_check(tmp_path, monkeypatch):
    """MMDuet's LoRA downloaded without its LLaVA-OneVision base: the base is reported missing (with the download
    command), not as a configured repetition penalty that disagrees with the checkpoint."""
    from streamreplace.config.schema import CheckpointRef
    constants = types.ModuleType("llava.constants")
    constants.DEFAULT_IM_END_TOKEN = constants.DEFAULT_IM_START_TOKEN = constants.DEFAULT_IMAGE_PATCH_TOKEN = "<x>"
    monkeypatch.setitem(sys.modules, "llava", types.ModuleType("llava"))
    monkeypatch.setitem(sys.modules, "llava.constants", constants)
    monkeypatch.setattr(llava_offline.paths, "checkpoint_dir", lambda ref: tmp_path / ref.repo)
    (tmp_path / "wangyueqian" / "MMDuet").mkdir(parents=True)
    adapter = object.__new__(llava_offline.LlavaOffline)
    adapter.torch = None
    adapter.spec = types.SimpleNamespace(
        id="mmduet_7b", checkpoint=CheckpointRef("wangyueqian/MMDuet"),
        base=CheckpointRef("lmms-lab/llava-onevision-qwen2-7b-ov"), vision=CheckpointRef("google/siglip-so400m"),
        decoding=types.SimpleNamespace(repetition_penalty=1.05), options={})
    with pytest.raises(FileNotFoundError, match=r"mmduet_7b: the LoRA base checkpoint lmms-lab/llava-onevision-qwen2-7b"
                                                r"-ov not found at .* \(no config.json\); run `streamreplace download "
                                                r"models mmduet_7b`"):
        adapter._setup(load=False)


def livestar(monkeypatch, n_frames: int = 2):
    monkeypatch.setattr(internvl_offline, "offline_frames", fake_offline_frames(n_frames, fps=1.0))
    monkeypatch.setattr(internvl_offline, "open_rgb", lambda path: frames(1)[0])
    adapter = object.__new__(internvl_offline.InternVLOffline)
    adapter.model, adapter._pixels = FakeChatModel(), lambda images: [["tile"] for _ in images]
    return adapter


def test_livestar_video_text_is_numbered_frames_then_the_prompt(monkeypatch):
    adapter = livestar(monkeypatch)
    req = request(path="clip.mp4", video={"fps": 1})
    tiles, prefix, info = adapter._media(req)
    assert prefix == "Frame1: <image>\nFrame2: <image>\n" and info["tiles"] == 2 and info["frames"] == 2
    query, n = adapter._query(req, tiles, prefix)
    image = "<img>" + "<IMG_CONTEXT>" * 16 + "</img>"
    assert query == ("<|im_start|>system\nDEFAULT SYSTEM<|im_end|>\n<|im_start|>user\n"
                     f"Frame1: {image}\nFrame2: {image}\nQ?<|im_end|>\n<|im_start|>assistant\n") and n == 0


def test_livestar_images_text_only_and_system_prompt(monkeypatch):
    adapter = livestar(monkeypatch)
    one = request(images=["a.png"])
    assert adapter._media(one)[1] == "<image>\n"
    assert adapter._media(request(images=["a.png", "b.png"]))[1] == "Image-1: <image>\nImage-2: <image>\n"
    text_only = request(kind="open", system="You are an expert.")
    tiles, prefix, info = adapter._media(text_only)
    query, _ = adapter._query(text_only, tiles, prefix)
    assert tiles == [] and info == {"tiles": 0} and "<img>" not in query and "<image>" not in query
    assert query.startswith("<|im_start|>system\nYou are an expert.<|im_end|>")


def test_livestar_training_answer(monkeypatch):
    adapter = livestar(monkeypatch)
    adapter.tokenizer = FakeTokenizer()
    req = request(images=["a.png"])
    tiles, prefix, _ = adapter._media(req)
    query, n = adapter._query(req, tiles, prefix, answer="A")
    assert query.endswith("<|im_start|>assistant\nA<|im_end|>") and n == len("A<|im_end|>")


def moss():
    adapter = object.__new__(moss_offline.MossOffline)
    adapter.repetition_penalty = 1.0
    return adapter


def test_moss_video_call_uses_the_pair_budgets():
    req = request(path="clip.mp4", video=P5, generation={"max_new_tokens": 1024})
    call = moss()._video_call(req, frames(74))
    assert call == {"prompt": "Q?", "video": moss_offline.FRAMES, "longest_edge": 16777216,
                    "video_max_pixels": 65536000, "max_frames": 74, "max_new_tokens": 1024, "do_sample": False,
                    "repetition_penalty": 1.0}
    with_floor = request(path="clip.mp4", video={"fps": 1, "max_pixels": 401408, "min_pixels": 200704},
                         system="You are an expert in understanding dynamics of objects.")
    call = moss()._video_call(with_floor, frames(3))
    assert call["video_max_pixels"] == 3 * 401408 and call["shortest_edge"] == 200704
    assert call["system_prompt"] == "You are an expert in understanding dynamics of objects."


def test_moss_image_query_has_no_system_message():
    query, overrides = moss()._image_query(request(images=["a.png"], system="ignored"), ["image"])
    assert "system_prompt" not in query and query["media_kwargs"] == {"max_pixels": 1003520}
    assert query["generate_kwargs"] == {"max_new_tokens": 512, "do_sample": False, "vision_chunked_length": 64,
                                        "repetition_penalty": 1.0}
    assert overrides == {"size": {"shortest_edge": 4096, "longest_edge": 1003520}}


ROPE_SOURCE = """def get_rope_index(self, second_per_grid_t, range_tensor):
    return torch.as_tensor(
        second_per_grid_t, dtype=range_tensor.dtype, device=range_tensor.device
    )
"""


def fake_rope(monkeypatch, source: str, version: str = "4.57.3"):
    module = types.ModuleType("fake_qwen2_5_vl")
    module.__file__ = "fake_qwen2_5_vl.py"
    monkeypatch.setattr(qwen25_rope, "_source", lambda: (module, source))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(__version__=version))


def test_qwen25_rope_patch_rebinds_the_method_with_float32(monkeypatch):
    fake_rope(monkeypatch, ROPE_SOURCE)
    model = types.SimpleNamespace()
    assert qwen25_rope.needs_fix() and qwen25_rope.apply(model)
    assert "float32" in model.get_rope_index.__func__.__code__.co_names
    assert model.get_rope_index.__self__ is model


@pytest.mark.parametrize("source", [ROPE_SOURCE.replace("range_tensor.dtype", "torch.float64"), ROPE_SOURCE * 2])
def test_qwen25_rope_patch_refuses_other_sources(monkeypatch, source):
    fake_rope(monkeypatch, source, version="4.99.0")
    assert not qwen25_rope.needs_fix()
    with pytest.raises(RuntimeError, match="exactly once"):
        qwen25_rope.apply(types.SimpleNamespace())


def test_pad_to_square_matches_mmduet_geometry():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cv2")
    frame = np.full((720, 1280, 3), 255, np.uint8)
    out = llava_offline.pad_to_square(frame)
    assert out.shape == (384, 384, 3)
    rows = np.where(out[:, 192, 0] == 255)[0]                        # 16:9 -> 384x216 content, centered
    assert rows[0] == 84 and rows[-1] == 299 and out[:84].max() == 0 and out[300:].max() == 0
