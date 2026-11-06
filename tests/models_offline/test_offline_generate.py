"""The offline adapters' generate() paths and the LLaVA-OneVision / MMDuet input and weight rules on CPU stand-ins
(no torch, transformers, LLaVA-NeXT or checkpoints): the first-token option-letter logits recorded for multiple-choice
requests (info.letter_logits, read by the prompt-robustness analysis), the head state around generation, budget
stops and reply clean-up, LLaVA's conversation text, frames and images, MMDuet's LoRA/projector weight check, and
LiveStar's and MOSS's chat calls. The LLaVA and LiveStar environments cannot always be run where the tests run, so
these paths are covered here with fakes."""
from __future__ import annotations

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from streamreplace.media.digest import frame_digest
from streamreplace.models import internvl_offline, llava_offline, moss_offline, qwen_offline
from streamreplace.models.decoding import greedy
from offline_fakes import (FakeChatModel, FakeHead, FakeQwenProcessor, FakeTokenizer, fake_array, fake_offline_frames,
                           first_step_logits, frames, request)
# the adapter finds the conversation template in the module of the chat model's class, as with the remote code
from offline_fakes import get_conv_template  # noqa: F401

EOT_ID = 1                                  # the fakes' <|im_end|> token
FAKE_TORCH = SimpleNamespace(inference_mode=nullcontext, bfloat16="bfloat16",
                             cat=lambda tiles: fake_array(np.zeros((len(tiles), 1))))


class FakeGenerator:
    """A model whose generate() calls the LM head once per new token (the first call carries `logits`) and returns
    the new token ids, after the prompt ids when `echo_prompt` (HF generate on input ids) or alone (LLaVA's
    generate on embeddings)."""
    def __init__(self, head, new_ids, logits, echo_prompt=True):
        self.head, self.new_ids, self.logits, self.echo_prompt = head, list(new_ids), logits, echo_prompt
        self.generation_config = SimpleNamespace(eos_token_id=[EOT_ID, 2])
        self.dtype = "float16"
        self.seen: dict = {}

    def generate(self, input_ids=None, *args, **kwargs):
        self.seen = {"kwargs": kwargs, "capture": self.head.capture, "keep": self.head.keep}
        for _ in self.new_ids:
            self.head.forward(self.logits)
        new = np.array([self.new_ids])
        return fake_array(np.concatenate([np.asarray(input_ids), new], axis=1) if self.echo_prompt else new)


def decode_with_eot(ids, skip_special_tokens=False):
    return "".join("<|im_end|>" if int(i) == EOT_ID else chr(int(i)) for i in ids)


def qwen(new_ids, logits):
    adapter = object.__new__(qwen_offline.QwenOffline)
    adapter.qwen3, adapter.factor, adapter.rope_fix, adapter.attention = False, 28, True, "sdpa"
    adapter.processor, adapter.repetition_penalty, adapter.torch = FakeQwenProcessor(), 1.05, FAKE_TORCH
    adapter.processor.tokenizer.decode = decode_with_eot
    adapter.head = FakeHead()
    adapter.model = FakeGenerator(adapter.head, new_ids, logits)
    return adapter


def test_qwen_records_the_first_token_letter_logits_of_multiple_choice(monkeypatch):
    monkeypatch.setattr(qwen_offline, "open_rgb", lambda path: frames(1)[0])
    adapter = qwen([ord(" "), ord("B"), EOT_ID], first_step_logits(A=1.5, B=2.25, C=9.0))
    req = request(images=["a.png"])                       # choices A, B
    out = adapter.generate(req)
    assert adapter.model.seen["capture"] is True and adapter.head.capture is False
    assert out.info["letter_logits"] == {"A": 1.5, "B": 2.25}           # only the request's options
    assert (out.text, out.tokens, out.finish) == ("B", 3, "eos")        # <|im_end|> removed, stripped
    assert adapter.model.seen["kwargs"] == {**greedy(req.settings), "repetition_penalty": 1.05}


def test_qwen_open_questions_capture_nothing_and_report_budget_stops(monkeypatch):
    monkeypatch.setattr(qwen_offline, "open_rgb", lambda path: frames(1)[0])
    adapter = qwen([ord("x")] * 4, first_step_logits(A=1.0))
    out = adapter.generate(request(images=["a.png"], kind="open", generation={"max_new_tokens": 4}))
    assert adapter.model.seen["capture"] is False and "letter_logits" not in out.info
    assert (out.text, out.tokens, out.finish) == ("xxxx", 4, "length")


# ---------- LLaVA-OneVision / MMDuet (LLaVA-NeXT faked) ----------

class FakeConversation:
    """LLaVA-NeXT's conv_templates["qwen_1_5"]: ChatML with a default system line."""
    def __init__(self):
        self.system = "<|im_start|>system\nYou are a helpful assistant."
        self.roles, self.sep, self.messages = ("<|im_start|>user", "<|im_start|>assistant"), "<|im_end|>", []

    def append_message(self, role, message):
        self.messages.append([role, message])

    def get_prompt(self):
        out = self.system + self.sep + "\n"
        for role, message in self.messages:
            out += role + "\n" + (message + self.sep + "\n" if message else "")
        return out


class FakeIds(list):
    def unsqueeze(self, dim):
        return fake_array([list(self)])


@pytest.fixture
def llava(monkeypatch):
    """A LlavaOffline whose LLaVA-NeXT helpers are fakes; returns (adapter, the global template)."""
    template = FakeConversation()
    constants = ModuleType("llava.constants")
    constants.DEFAULT_IMAGE_TOKEN, constants.IMAGE_TOKEN_INDEX = "<image>", -200
    conversation = ModuleType("llava.conversation")
    conversation.conv_templates = {"qwen_1_5": template}
    mm_utils = ModuleType("llava.mm_utils")
    mm_utils.process_images = lambda images, processor, config: [fake_array(np.zeros((2, 3, 4, 4))) for _ in images]

    def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors=None):
        ids = FakeIds()
        for i, chunk in enumerate(prompt.split("<image>")):
            ids += ([image_token_index] if i else []) + tokenizer(chunk).input_ids
        return ids
    mm_utils.tokenizer_image_token = tokenizer_image_token
    for name, module in (("llava", ModuleType("llava")), ("llava.constants", constants),
                         ("llava.conversation", conversation), ("llava.mm_utils", mm_utils)):
        monkeypatch.setitem(sys.modules, name, module)
    adapter = object.__new__(llava_offline.LlavaOffline)
    adapter.tokenizer, adapter.config, adapter.info = FakeTokenizer(), SimpleNamespace(), {"dtype": "torch.float16"}
    adapter.processor = SimpleNamespace(preprocess=lambda video, return_tensors=None: {
        "pixel_values": fake_array(np.zeros((len(video), 3, 4, 4)))})
    adapter.repetition_penalty, adapter.torch, adapter.head = 1.05, FAKE_TORCH, FakeHead()
    return adapter, template


def test_llava_video_text_frames_and_padding(llava, monkeypatch):
    adapter, template = llava
    monkeypatch.setattr(llava_offline, "offline_frames", fake_offline_frames(3, fps=2.0, size=(64, 48)))
    padded = []
    monkeypatch.setattr(llava_offline, "pad_to_square", lambda a: padded.append(a.shape) or np.zeros((384, 384, 3)))
    enc = adapter._encode(request(path="clip.mp4", video={"fps": 2, "max_frames": 160, "pad_square": True}))
    x, info = enc["inputs"], enc["info"]
    assert padded == [(48, 64, 3)] * 3 and x["images"][0].shape == (3, 3, 4, 4)
    assert x["image_sizes"] == [(384, 384)] and x["modalities"] == ["video"]
    text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\nQ?<|im_end|>\n"
            "<|im_start|>assistant\n")
    expected = [ord(c) for c in text[:text.index("<image>")]] + [-200] + [ord(c) for c in text.split("<image>")[1]]
    assert x["input_ids"].tolist() == [expected] and info["input_tokens"] == len(expected)
    assert info == {"dtype": "torch.float16", "frames": 3, "duration": 1.5, "pad_square": True,
                    "frame_indices": frame_digest({"indices": [0, 1, 2]}), "input_tokens": len(expected)}
    assert template.messages == []                         # the shared template is copied, never changed


def test_llava_images_text_only_system_prompt_and_training_answer(llava, monkeypatch):
    adapter, _ = llava
    monkeypatch.setattr(llava_offline, "open_rgb", lambda path: frames(1, (640, 360))[0])
    enc = adapter._encode(request(images=["a.png", "b.png"], system="You are an expert."))
    x = enc["inputs"]
    assert len(x["images"]) == 2 and x["image_sizes"] == [(640, 360)] * 2 and x["modalities"] == ["image"] * 2
    tokens = x["input_ids"].tolist()[0]
    text = "".join(chr(t) if t > 0 else "<image>" for t in tokens)
    assert text.startswith("<|im_start|>system\nYou are an expert.<|im_end|>\n<|im_start|>user\n<image><image>\nQ?")
    text_only = adapter._encode(request(kind="open"))["inputs"]
    assert text_only["images"] is None and text_only["modalities"] is None and -200 not in text_only["input_ids"][0]
    train = adapter._encode(request(images=["a.png"]), answer="B")
    assert train["n"] == len("B<|im_end|>")
    assert "".join(chr(t) for t in train["inputs"]["input_ids"][0][-12:]) == "\nB<|im_end|>"


def test_llava_generate_projects_only_the_last_video_position_and_reads_letters(llava, monkeypatch):
    adapter, _ = llava
    monkeypatch.setattr(llava_offline, "offline_frames", fake_offline_frames(2, fps=2.0))
    monkeypatch.setattr(llava_offline, "pad_to_square", lambda a: np.zeros((384, 384, 3)))
    adapter.model = FakeGenerator(adapter.head, [ord("A")], first_step_logits(A=3.0, B=-1.0), echo_prompt=False)
    req = request(path="clip.mp4", video={"fps": 2, "pad_square": True})
    out = adapter.generate(req)
    seen = adapter.model.seen
    assert seen["keep"] == 1 and seen["capture"] is True and adapter.head.keep == 0 and not adapter.head.capture
    assert {k: seen["kwargs"][k] for k in greedy(req.settings)} == greedy(req.settings)
    assert seen["kwargs"]["repetition_penalty"] == 1.05 and seen["kwargs"]["modalities"] == ["video"]
    assert (out.text, out.tokens, out.finish, out.info["letter_logits"]) == ("A", 1, "eos", {"A": 3.0, "B": -1.0})
    adapter.model = FakeGenerator(adapter.head, [ord("y")] * 3, first_step_logits(), echo_prompt=False)
    monkeypatch.setattr(llava_offline, "open_rgb", lambda path: frames(1)[0])
    out = adapter.generate(request(images=["a.png"], kind="open", generation={"max_new_tokens": 3}))
    assert adapter.model.seen["keep"] == 0 and "letter_logits" not in out.info and out.finish == "length"


class FakeTensor(np.ndarray):
    def cpu(self):
        return self

    def to(self, dtype):
        return self


def tensor(value) -> FakeTensor:
    return np.asarray(value, dtype=np.float32).view(FakeTensor)


def test_mmduet_weights_map_onto_the_peft_names_and_must_load_exactly(monkeypatch):
    lora = "base_model.model.model.layers.0.self_attn.q_proj"
    weights = {f"{lora}.lora_A.weight": tensor([1.0]), f"{lora}.lora_B.weight": tensor([2.0]),
               "base_model.model.model.mm_projector.0.weight": tensor([3.0]),
               "base_model.model.model.informative_head.weight": tensor([4.0]),
               "base_model.model.model.relevance_head.weight": tensor([5.0])}
    loaded = {f"{lora}.lora_A.mmduet.weight": tensor([1.0]), f"{lora}.lora_B.mmduet.weight": tensor([2.0]),
              "base_model.model.model.mm_projector.modules_to_save.mmduet.0.weight": tensor([3.0])}
    calls = []

    class PeftModel:
        @staticmethod
        def from_pretrained(model, path, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(state_dict=lambda: loaded)
    peft = ModuleType("peft")
    peft.PeftModel = PeftModel
    safetensors_torch = ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda path: dict(weights)
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "safetensors", ModuleType("safetensors"))
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    adapter = object.__new__(llava_offline.LlavaOffline)
    adapter.torch, adapter.info, adapter.model = SimpleNamespace(equal=np.array_equal), {}, object()
    adapter._apply_mmduet(llava_offline.Path("mmduet"))
    assert adapter.info["mmduet_weights"] == {"mapped": 3, "excluded_heads": 2}
    assert calls == [{"adapter_name": "mmduet", "is_trainable": False, "autocast_adapter_dtype": False}]
    loaded["base_model.model.model.mm_projector.modules_to_save.mmduet.0.weight"] = tensor([3.5])
    with pytest.raises(RuntimeError, match="not loaded exactly: base_model.model.model.mm_projector.0.weight"):
        adapter._apply_mmduet(llava_offline.Path("mmduet"))
    del loaded[f"{lora}.lora_B.mmduet.weight"]
    with pytest.raises(RuntimeError, match="lora_B"):
        adapter._apply_mmduet(llava_offline.Path("mmduet"))


# ---------- LiveStar (remote code faked) ----------

class FakeLiveStarChat(FakeChatModel):
    def __init__(self, head, reply, steps, logits):
        self.head, self.reply, self.steps, self.logits, self.seen = head, reply, steps, logits, {}

    def chat(self, tokenizer, pixels, question, generation_config, **kwargs):
        self.seen = {"system": self.system_message, "question": question, "config": generation_config,
                     "capture": self.head.capture, **kwargs}
        for _ in range(self.steps):
            self.head.forward(self.logits)
        return self.reply


def livestar(monkeypatch, reply=" B ", steps=2, logits=None):
    monkeypatch.setattr(internvl_offline, "offline_frames", fake_offline_frames(2, fps=1.0))
    monkeypatch.setattr(internvl_offline, "open_rgb", lambda path: frames(1)[0])
    adapter = object.__new__(internvl_offline.InternVLOffline)
    adapter.head, adapter.tokenizer, adapter.torch, adapter.repetition_penalty = FakeHead(), FakeTokenizer(), \
        FAKE_TORCH, 1.05
    adapter.model = FakeLiveStarChat(adapter.head, reply, steps, logits or first_step_logits(A=0.5, B=0.75))
    adapter._pixels = lambda images: [["tile"] for _ in images]
    return adapter


def test_livestar_generate_swaps_the_system_message_only_for_the_call(monkeypatch):
    adapter = livestar(monkeypatch)
    req = request(path="clip.mp4", video={"fps": 1}, system="You are an expert.")
    out = adapter.generate(req)
    seen = adapter.model.seen
    assert seen["system"] == "You are an expert." and adapter.model.system_message == "DEFAULT SYSTEM"
    assert seen["question"] == "Frame1: <image>\nFrame2: <image>\nQ?" and seen["num_patches_list"] == [1, 1]
    assert seen["config"] == {**greedy(req.settings), "repetition_penalty": 1.05} and seen["history"] is None
    assert seen["capture"] is True and not adapter.head.capture
    assert (out.text, out.tokens, out.finish) == ("B", 2, "eos")
    assert out.info["letter_logits"] == {"A": 0.5, "B": 0.75} and out.info["frames"] == 2
    query, _ = adapter._query(req, [["tile"], ["tile"]], "Frame1: <image>\nFrame2: <image>\n")
    assert out.info["input_tokens"] == len(query)


def test_livestar_restores_the_system_message_when_chat_fails(monkeypatch):
    adapter = livestar(monkeypatch)

    def broken(*args, **kwargs):
        raise RuntimeError("out of memory")
    adapter.model.chat = broken
    with pytest.raises(RuntimeError):
        adapter.generate(request(images=["a.png"], system="You are an expert."))
    assert adapter.model.system_message == "DEFAULT SYSTEM" and not adapter.head.capture


def test_livestar_budget_stop_from_the_head_calls(monkeypatch):
    adapter = livestar(monkeypatch, reply="long", steps=8)
    out = adapter.generate(request(images=["a.png"], kind="open", generation={"max_new_tokens": 8}))
    assert (out.tokens, out.finish) == (8, "length") and "letter_logits" not in out.info


# ---------- MOSS (offline API faked) ----------

def test_moss_image_generate_reads_letters_and_counts_tokens(monkeypatch):
    monkeypatch.setattr(moss_offline, "open_rgb", lambda path: frames(1)[0])
    adapter = object.__new__(moss_offline.MossOffline)
    adapter.head, adapter.torch, adapter.repetition_penalty, adapter.attention = FakeHead(), FAKE_TORCH, 1.0, "sdpa"
    adapter.processor = SimpleNamespace(tokenizer=FakeTokenizer())
    seen = {}

    def offline_generate(processor, query, image_processor_overrides=None):
        seen.update(query=query, overrides=image_processor_overrides, capture=adapter.head.capture)
        for _ in range(3):
            adapter.head.forward(first_step_logits(A=-2.0, B=4.0))
        return " A. a "
    adapter.model = SimpleNamespace(_offline_generate_one_with_processor_overrides=offline_generate)
    out = adapter.generate(request(images=["a.png"]))
    assert seen["capture"] is True and not adapter.head.capture
    assert seen["query"]["generate_kwargs"] == {"max_new_tokens": 512, "do_sample": False, "vision_chunked_length": 64,
                                                "repetition_penalty": 1.0}
    assert (out.text, out.tokens, out.finish) == ("A. a", 3, "eos")
    assert out.info == {"images": 1, "letter_logits": {"A": -2.0, "B": 4.0}, "attention": "sdpa"}
