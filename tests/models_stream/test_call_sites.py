"""The stream adapters' control flow on CPU fakes of the upstream objects: the native path (hooks=None) builds the
row the paper's runs wrote, and with hooks each adapter arms the plugin exactly around its answer turn."""
from __future__ import annotations

import subprocess
import sys
import types

import numpy as np
import pytest

from streamreplace.models import (aura_stream, aura_stream_hf, livestar_stream, mmduet2_stream, mmduet_stream,
                                  mossrt_stream, registry, thinkstream_stream)
from streamreplace.models.base import StreamAdapter
from streamreplace.models.readouts import stream_seed
from stream_fakes import RecordingHooks, fake_torch, request

STREAM_ADAPTERS = ["mmduet_stream", "thinkstream_stream", "mmduet2_stream", "aura_stream", "aura_stream_hf",
                   "livestar_stream", "mossrt_stream"]


@pytest.mark.parametrize("name", STREAM_ADAPTERS)
def test_registry_names_import_without_model_libraries(name):
    """Each adapter module imports with the core dependencies only; load=False gives a schedule-only adapter."""
    cls = registry.load_class(name)
    adapter = cls(None, {"stream": {}}, None, load=False)
    assert isinstance(adapter, StreamAdapter)
    assert adapter.PROTOCOL_VERSION == (2 if name == "mmduet_stream" else 1)
    assert adapter.concurrency == (8 if name == "aura_stream" else 1)


# ------------------------------------------------------------------------------------------- MMDuet

class FakeHead:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)


class FakeInfer:
    """LiveInferForDemo's surface: frames are popped from a queue; a frame in `fire` triggers a reply from inside
    input_one_frame (through self._generate_response, as upstream); every reply runs `steps` LM-head forwards."""

    def __init__(self, fire=(), steps=3):
        self.fire, self.steps, self.calls, self.head_outputs = set(fire), steps, [], []
        self.head = FakeHead()
        self.model = types.SimpleNamespace(get_output_embeddings=lambda: self.head)
        self.tokenizer = "mmduet tokenizer"

    def reset(self):
        self.frame_embeds_queue = []

    def input_video_stream(self, images):
        self.frame_embeds_queue = list(images)

    def encode_given_query(self, prompt):
        self.calls.append(("query", len(self.frame_embeds_queue)))

    def input_one_frame(self):
        frame = self.frame_embeds_queue.pop(0)
        self.calls.append(("frame", frame))
        response = self._generate_response() if frame in self.fire else None
        return {"frame_idx": frame + 1, "time": frame / 2, "informative_score": 0.123456, "relevance_score": 0.2,
                "response": response}

    def _generate_response(self):
        self.calls.append(("generate",))
        for _ in range(self.steps):           # torch uses a forward hook's non-None result as the module's output
            self.head_outputs.append([hook(self.head, (), "logits") for hook in self.head.hooks])
        return " reply "


def mmduet(monkeypatch, infer, frames=5):
    monkeypatch.setattr(mmduet_stream, "measure", lambda path: {"frames": frames, "fps": 2.0, "header_frames": frames})
    adapter = mmduet_stream.MMDuetStream(None, {}, None, load=False)
    adapter.torch, adapter.infer = fake_torch(), infer
    adapter._load_video = lambda path, output_fps: (list(range(frames)), None)
    return adapter


FORCED = {"generation": {"max_new_tokens": 8}, "stream": {"fps": 1, "force_answer": True}}


def test_mmduet_native_stays_silent_when_its_heads_do_not_fire(monkeypatch):
    """Condition S: the heads decide. Below the threshold on the last frame MMDuet says nothing (no forced reply)."""
    infer = FakeInfer(fire=())
    gen = mmduet(monkeypatch, infer).stream(request())
    assert infer.calls == [("frame", 0), ("frame", 1), ("frame", 2), ("frame", 3), ("query", 1), ("frame", 4)]
    assert gen.text == "" and gen.info["no_answer"] is True and gen.info["forced"] is False
    fired = mmduet(monkeypatch, FakeInfer(fire={4})).stream(request())
    assert fired.text == "reply" and fired.info["no_answer"] is False and fired.info["forced"] is False
    assert gen.info["schedule"] != mmduet(monkeypatch, FakeInfer()).stream(request(settings=FORCED)).info["schedule"]


def test_mmduet_forced_control_replies_on_the_last_frame(monkeypatch):
    """S-dagger (stream.force_answer): one reply is generated at the last frame when the heads stayed silent."""
    infer = FakeInfer(fire=())
    gen = mmduet(monkeypatch, infer).stream(request(settings=FORCED))
    assert infer.calls == [("frame", 0), ("frame", 1), ("frame", 2), ("frame", 3), ("query", 1), ("frame", 4),
                           ("generate",)]
    assert gen.text == "reply" and gen.tokens is None and gen.finish == "unknown"
    info = dict(gen.info)
    assert len(info.pop("schedule")) == 16
    assert info == {"stream_fps": 2, "frames": 5, "replies_before_question": 0, "events": [],
                    "last_frame_scores": {"informative_score": 0.1235, "relevance_score": 0.2}, "forced": True,
                    "no_answer": False}
    assert "_generate_response" not in vars(infer) and infer.head.hooks == []          # nothing installed


def test_mmduet_question_timing_scores_the_last_reply(monkeypatch):
    """stream.question_at: the question is encoded once before frame q; the last reply after it is the answer, and
    one reply is requested after the last frame only when the heads never fired."""
    start = {"generation": {"max_new_tokens": 8}, "stream": {"fps": 1, "question_at": "start"}}
    infer = FakeInfer(fire={1, 3})
    gen = mmduet(monkeypatch, infer).stream(request(settings=start))
    assert infer.calls[0] == ("query", 5) and gen.text == "reply"
    assert gen.info["n_replies"] == 2 and gen.info["first_reply_t"] == 0.5 and gen.info["answer_t"] == 1.5
    assert gen.info["fallback"] is False and gen.info["question_index"] == 0
    middle = {"generation": {"max_new_tokens": 8}, "stream": {"fps": 1, "question_at": "middle"}}
    infer = FakeInfer(fire={0})                 # a reply before the question does not count
    gen = mmduet(monkeypatch, infer).stream(request(settings=middle))
    assert infer.calls[:3] == [("frame", 0), ("generate",), ("frame", 1)] and infer.calls[3] == ("query", 3)
    assert gen.info["fallback"] is True and gen.info["n_replies"] == 0 and infer.calls[-1] == ("generate",)
    with pytest.raises(ValueError, match="start or middle"):
        mmduet(monkeypatch, FakeInfer()).stream(request(settings={"generation": {"max_new_tokens": 8},
                                                                  "stream": {"question_at": "end"}}))


def test_mmduet_arms_the_first_reply_after_the_question_only(monkeypatch, fake_plugin_guard, hooks):
    infer = FakeInfer(fire={1, 4})              # a reply before the question, then the heads fire on the last frame
    gen = mmduet(monkeypatch, infer).stream(request(), hooks)
    assert not gen.info["forced"] and gen.info["replies_before_question"] == 1
    assert hooks.calls[0] == ("attach", ("layers of", infer.model), "mmduet tokenizer")
    assert hooks.names()[1:] == ["arm", "step", "step", "step", "disarm", "finish"]
    assert hooks.calls[1] == ("arm", True) and [c[1] for c in hooks.calls if c[0] == "step"] == [0, 1, 2]
    infer2 = FakeInfer(fire=())                 # Respond's forced reply: that one is the answer turn
    hooks2 = RecordingHooks()
    adapter = mmduet(monkeypatch, infer2)
    adapter.stream(request(settings=FORCED), hooks2)
    assert hooks2.names() == ["attach", "arm", "step", "step", "step", "disarm", "finish"]
    adapter.stream(request(settings=FORCED), hooks2)                            # attached once per hooks object
    assert hooks2.names().count("attach") == 1 and hooks2.names().count("arm") == 2
    hooks3 = RecordingHooks()                   # without Respond a silent question has no answer turn
    mmduet(monkeypatch, FakeInfer(fire=())).stream(request(), hooks3)
    assert hooks3.names() == ["attach", "finish"]


def test_mmduet_lm_head_result_replaces_the_logits_only_in_the_answer_turn(monkeypatch, fake_plugin_guard):
    hooks = RecordingHooks(step_result="halted logits")
    infer = FakeInfer(fire={1})                 # a reply before the question (unarmed), then the forced answer
    mmduet(monkeypatch, infer).stream(request(settings=FORCED), hooks)
    assert infer.head_outputs == [[None]] * 3 + [["halted logits"]] * 3
    assert [hook(infer.head, (), "logits") for hook in infer.head.hooks] == [None]   # after the request: inert


# ------------------------------------------------------------------------------------------- ThinkStream

class Seqlens:
    def __init__(self, value):
        self.value = value

    def clone(self):
        return Seqlens(self.value)

    def copy_(self, other):
        self.value = other.value


class FakeDecoder:
    def __init__(self):
        self.cache_seqlens, self.cache, self.used = Seqlens(7), "cache", []
        self.static_input_ids = self.static_position_ids = self.static_logits = types.SimpleNamespace(
            copy_=lambda value: None)

    def step(self, input_ids, position_ids):
        self.used.append("graph")
        return "graph logits"


class FakeEngine:
    """StreamingWindowInferenceEngine's surface: generate() prefills through self.prefill, then decodes."""

    def __init__(self):
        self.decoder, self.next_start_pos, self.prefills = FakeDecoder(), None, 0
        self._window_count = [3]
        self.model = lambda **kwargs: types.SimpleNamespace(logits="eager logits")

    def prefill(self, **kwargs):
        self.prefills += 1
        return "prefill logits"

    def generate(self, **kwargs):
        self.prefill(**kwargs)
        self.decoder.step("token", "position")
        return [["tokens"]]


class Tokenizer:
    ids = {"</think>": 1, "<|im_end|>": 2, "<silent>": 3, "<response>": 4, "<|video_pad|>": 5, "<|vision_end|>": 6}

    def convert_tokens_to_ids(self, token):
        return self.ids[token]

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=[100 + ord(text[0])])


def install_torchcodec(monkeypatch, duration=3.0, frames=90, fps=30.0):
    decoders = types.ModuleType("torchcodec.decoders")
    meta = types.SimpleNamespace(duration_seconds=duration, num_frames=frames, average_fps=fps)
    decoders.VideoDecoder = lambda path: types.SimpleNamespace(metadata=meta)
    package = types.ModuleType("torchcodec")
    package.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", package)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)


def thinkstream(monkeypatch, chunks=3, deadline=False, answer="<think>t</think><response>B<|im_end|>"):
    install_torchcodec(monkeypatch)
    adapter = thinkstream_stream.ThinkStreamStream(None, {}, None, load=False)
    seeds = []
    adapter.torch = fake_torch(manual_seed=seeds.append,
                               cuda=types.SimpleNamespace(get_rng_state=lambda: "rng", set_rng_state=lambda s: None))
    adapter.processor = types.SimpleNamespace(tokenizer=Tokenizer(), decode=lambda ids, skip_special_tokens: answer)
    adapter.model = types.SimpleNamespace(config=types.SimpleNamespace())
    adapter._restricted_sample, adapter._system, adapter._template = "restricted", "system", "template"
    engines = []

    def engine(stream, window):
        engines.append(FakeEngine())
        return engines[-1]
    adapter._engine = engine

    def chat(engine, sample_kwargs, **kwargs):
        for index in range(chunks):
            last = index == chunks - 1
            engine.generate(input_ids="ids", position_ids="pos",
                            sample_kwargs=dict(sample_kwargs, is_query_window=last, allow_deferral=not last))
            if last and deadline:
                raise RuntimeError("Model failed to answer when deferral was disallowed (Deadline exceeded).")
            yield {"is_answer": last, "generated_tokens": [["t"]]}
    adapter._chat = chat
    return adapter, seeds, engines


def test_thinkstream_native_answer_seed_and_info(monkeypatch):
    adapter, seeds, _ = thinkstream(monkeypatch)
    gen = adapter.stream(request(sample_id="v-7"))
    assert gen.text == "B" and seeds == [stream_seed("v-7")]
    info = dict(gen.info)
    assert len(info.pop("schedule")) == 16
    assert info == {"chunks": 3, "window_chunks": 20, "kept_chunks": 3, "duration": 3.0, "seed": stream_seed("v-7"),
                    "restricted": True, "raw_turn": "<think>t</think><response>B<|im_end|>", "no_answer": False}


def test_thinkstream_native_deadline_is_unanswered_and_no_chunk_is_an_error(monkeypatch):
    adapter, _, _ = thinkstream(monkeypatch, chunks=3, deadline=True)
    gen = adapter.stream(request(kind="open"))
    assert gen.text == "" and gen.info["no_answer"] and gen.info["chunks"] == 2
    assert gen.info["failure"].endswith("(Deadline exceeded).") and gen.info["restricted"] is False
    adapter, _, _ = thinkstream(monkeypatch, chunks=1, deadline=True)
    with pytest.raises(RuntimeError, match="Deadline"):                         # nothing streamed: an error
        adapter.stream(request())


def test_thinkstream_arms_the_deadline_chunk_and_reads_out_a_missed_deadline(monkeypatch, fake_plugin_guard, hooks):
    adapter, _, engines = thinkstream(monkeypatch, chunks=3)
    adapter.stream(request(), hooks)
    assert hooks.names() == ["attach", "arm", "disarm", "finish"]               # observation chunks untouched
    assert engines[0].decoder.used == ["graph", "graph", "graph"]
    adapter, _, engines = thinkstream(monkeypatch, chunks=1, deadline=True)    # one-chunk clip: same unanswered turn
    adapter._fork = lambda turn: ("fork logits", turn.pre[0].value, turn.chunk["input_ids"])
    hooks = RecordingHooks()
    gen = adapter.stream(request(), hooks)
    assert gen.info["no_answer"] and gen.info["chunks"] == 0 and "Deadline" in gen.info["failure"]
    assert hooks.names() == ["attach", "arm", "disarm", "deadline", "finish"]
    assert hooks.readout == ("fork logits", 7, "ids")                          # state taken before the prefill


def test_thinkstream_eager_decode_and_answer_turn_errors(monkeypatch, fake_plugin_guard):
    hooks = RecordingHooks()
    hooks.eager_decode = True
    adapter, _, engines = thinkstream(monkeypatch, chunks=2)
    adapter.stream(request(), hooks)
    assert engines[0].decoder.used == ["graph"]                                 # the query chunk decoded eagerly
    engine, turn = FakeEngine(), thinkstream_stream._AnswerTurn(RecordingHooks(), "Q?")

    def broken(**kwargs):
        raise RuntimeError("CUDA out of memory")
    engine.generate = broken
    adapter._wrap(engine, turn)
    with pytest.raises(thinkstream_stream.AnswerTurnError, match="out of memory"):
        engine.generate(sample_kwargs={"is_query_window": True, "allow_deferral": False})
    assert turn.hooks.names() == ["arm", "disarm"]


# ------------------------------------------------------------------------------------------- MMDuet2

class FakeClient:
    """ProactiveInferenceAPIClient's surface: a reply is generated when a turn's images are complete."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.generations = []
        self.model = types.SimpleNamespace(generate=self._generate)
        self.processor = types.SimpleNamespace(tokenizer="mmduet2 tokenizer")

    def _generate(self, **kwargs):
        self.generations.append(kwargs)
        return self.replies.pop(0)

    def reset(self):
        self.history, self.buffer, self.text = [{"role": "system", "content": "proactive"}], [], ""

    def set_fps(self, fps):
        self.fps = fps

    def add_text(self, text):
        self.text = text

    def add_image(self, image):
        self.buffer.append(image)
        if len(self.buffer) < self.num_frames_per_turn:
            return
        content = [*self.buffer, *([self.text] if self.text else [])]
        self.buffer, self.text = [], ""
        self.history.append({"role": "user", "content": content})
        reply = self.model.generate(content=content)
        self.history.append({"role": "assistant", "content": reply})


def mmduet2(monkeypatch, replies):
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(5)]
    meta = {"indices": [0, 4, 8, 12, 16], "timestamps": [0, 1, 2, 3, 4], "fps": 4.0, "duration": 5.0}
    monkeypatch.setattr(mmduet2_stream, "sample_frames", lambda path, fps: (frames, meta))
    adapter = mmduet2_stream.MMDuet2Stream(None, {}, None, load=False)
    adapter.client, adapter.resize_frame = FakeClient(replies), lambda array, max_size: array
    return adapter


def test_mmduet2_native_final_reply_and_exact_silence(monkeypatch):
    adapter = mmduet2(monkeypatch, ["NO REPLY", "The cup.", "B"])
    gen = adapter.stream(request())
    info = dict(gen.info)
    assert len(info.pop("schedule")) == 16
    assert gen.text == "B" and info == {"stream_fps": 1.0, "frames": 5, "duration": 5.0, "turns": 3,
                                        "replies_before_question": 1, "no_answer": False}
    assert adapter.client.history[-2]["content"][-1] == "Q?"                    # the question after the last image
    assert len(adapter.client.history[-2]["content"]) == 2 and adapter.client.max_new_tokens == 8
    gen = mmduet2(monkeypatch, ["NO REPLY", "NO REPLY", "NO REPLY."]).stream(request())
    assert gen.info["no_answer"] and gen.text == "NO REPLY."


def test_mmduet2_arms_only_the_final_turn(monkeypatch, fake_plugin_guard, fake_transformers, hooks):
    adapter = mmduet2(monkeypatch, ["NO REPLY", "NO REPLY", "B"])
    adapter.stream(request(), hooks)
    assert hooks.names() == ["attach", "processors", "arm", "disarm", "finish"]
    calls = adapter.client.generations
    assert [("logits_processor" in c) for c in calls] == [False, False, True]
    assert list(calls[-1]["logits_processor"]) == ["first-step processor"]


# ------------------------------------------------------------------------------------------- LiveStar

class Frame:
    def __init__(self, t):
        self.t = t


class Clip:
    def __init__(self, frames):
        self.frames = frames


class FakeSession:
    created = 0

    def __init__(self, model, tokenizer, gen, tiles, args):
        FakeSession.created += 1
        self.gen, self.args, self.narration = gen, args, [(0, "a man"), (2, "he walks")]

    def run(self, n):
        self.clips = [Clip([Frame(0), Frame(1)]), Clip([Frame(2), Frame(3), Frame(4)])]

    def _captioned_clips(self):
        return self.clips

    def _derive_history(self):
        return [("".join(f"Frame-{f.t}: <image>\n" for f in c.frames), f"caption {i}") for i, c in enumerate(self.clips)]

    def _committed_gids(self):
        return [f.t for c in self.clips for f in c.frames]

    def _pixels(self, ids):
        return ("pixels", tuple(ids)), [1] * len(ids)

    def _vis(self, ids):
        return ("features", tuple(ids))


def livestar(monkeypatch):
    infer = types.ModuleType("streaming_infer")
    infer.StreamingSession = FakeSession
    infer.build_transform = lambda input_size: (lambda tile: tile)
    infer.dynamic_preprocess = lambda image, image_size, max_num, use_thumbnail: [image]
    monkeypatch.setitem(sys.modules, "streaming_infer", infer)
    monkeypatch.setattr(livestar_stream, "sample_frames",
                        lambda path, fps: (list(range(5)), {"indices": [0, 30, 60, 90, 120], "fps": 30.0}))
    adapter = livestar_stream.LiveStarStream(None, {}, None, load=False)
    chats = []

    def chat(tokenizer, pixels, question, gen, **kwargs):
        chats.append((question, gen, kwargs))
        return " The man walks. "
    adapter.model = types.SimpleNamespace(chat=chat, language_model="language model")
    adapter.torch = fake_torch(stack=lambda tiles: tiles)
    adapter.tokenizer, adapter.max_tiles = "livestar tokenizer", 1
    FakeSession.created = 0
    return adapter, chats


def test_livestar_native_question_turn_and_info(monkeypatch):
    adapter, chats = livestar(monkeypatch)
    gen = adapter.stream(request())
    question, gen_cfg, kwargs = chats[0]
    assert question == "Frame-3: <image>\nFrame-4: <image>\nQ?" and gen.text == "The man walks."
    assert gen_cfg == {"do_sample": False, "num_beams": 1, "max_new_tokens": 8, "repetition_penalty": 1.05}
    assert kwargs["history"][-1][0] == "Frame-2: <image>\n"                    # the last narration keeps its first frame
    assert kwargs["num_patches_list"] == [1] * 5 and kwargs["visual_features"] == ("features", (0, 1, 2, 3, 4))
    info = dict(gen.info)
    assert len(info.pop("schedule")) == 16
    assert info == {"stream_fps": 1, "frames": 5, "context_frames": 5, "frames_in_question": 2, "narrations": 2,
                    "alpha": 1.04, "num_runs": 3}
    adapter.stream(request(sample_id="q2"))
    assert FakeSession.created == 2                                             # native: one session per question


def test_livestar_plugin_shares_one_session_per_video(monkeypatch, fake_plugin_guard, fake_transformers, hooks):
    adapter, chats = livestar(monkeypatch)
    first = adapter.stream(request(), hooks)
    adapter.stream(request(sample_id="q2"), hooks)
    assert FakeSession.created == 1 and first.info["session_shared"] is True
    adapter.stream(request(sample_id="q3", video="other.mp4"), hooks)
    assert FakeSession.created == 2
    assert hooks.calls[0] == ("attach", ("layers of", "language model"), "livestar tokenizer")
    assert hooks.names()[1:5] == ["processors", "arm", "disarm", "finish"]
    assert list(chats[0][1]["logits_processor"]) == ["first-step processor"]


# ------------------------------------------------------------------------------------------- MOSS-VL-Realtime

IM_START = 5


class FakeRealtimeSession:
    def __init__(self, model, outputs, drains):
        self.model, self.outputs, self.drains, self.queue = model, outputs, drains, []
        self.pending_frames, self.active, self.pushed, self.closed = 0, True, [], None

    def start(self):
        pass

    def poll_output(self, timeout):
        return self.queue.pop(0) if self.queue else None

    def push_frame(self, frame, timestamp, drop_oldest):
        self.pushed.append((frame, timestamp, drop_oldest))

    def push_prompt_frame(self, prompt, frame, timestamp, drop_oldest):
        self.pushed.append((prompt, frame, timestamp))
        if self.drains:      # the session thread: the update that drains the question, then the answer's forwards
            self.model._update_model_kwargs_for_real_time_generation(input_ids=np.array([IM_START, 9]))
            for _ in range(2):
                for hook in self.model.lm_head.hooks:
                    hook(self.model.lm_head, (), "logits")
        self.queue.extend(self.outputs)

    def close(self, timeout):
        self.closed = timeout


class FakeMossModel:
    def __init__(self, outputs, drains=True):
        self.outputs, self.drains = outputs, drains
        self.lm_head = FakeHead()
        self.sessions = []

    def create_realtime_session(self, processor, **kwargs):
        self.session_kwargs = kwargs
        self.sessions.append(FakeRealtimeSession(self, self.outputs, self.drains))
        return self.sessions[-1]

    def _update_model_kwargs_for_real_time_generation(self, input_ids):
        return (np.append(input_ids, IM_START),)


def moss(monkeypatch, outputs, drains=True):
    meta = {"indices": [0, 1, 2], "timestamps": [0.0, 0.01, 0.02], "duration": 0.03, "fps": 100.0}
    monkeypatch.setattr(mossrt_stream, "sample_frames", lambda path, fps: (["f0", "f1", "f2"], meta))
    adapter = mossrt_stream.MossRTStream(None, {}, None, load=False)
    adapter.model = FakeMossModel(outputs, drains)
    adapter.processor = types.SimpleNamespace(tokenizer=types.SimpleNamespace(
        convert_tokens_to_ids=lambda token: {"<|im_start|>": IM_START}[token]))
    return adapter


def test_moss_native_reads_the_reply_after_round_start(monkeypatch):
    adapter = moss(monkeypatch, ["<|silence|>", "<|round_start|><|silence|>", "<|response|>A.", "<|im_end|>"])
    gen = adapter.stream(request())
    session = adapter.model.sessions[0]
    assert session.pushed[:2] == [("f0", 0.0, False), ("f1", 0.01, False)] and session.pushed[2] == ("Q?", "f2", 0.02)
    assert adapter.model.session_kwargs == {"initial_prompt": "", "frame_queue_size": 256, "max_tokens_per_turn": 12,
                                            "max_new_tokens": 4096, "do_sample": False}
    assert gen.text == "A." and gen.tokens == 1 and gen.finish == "unknown" and session.closed == 30
    info = dict(gen.info)
    assert len(info.pop("schedule")) == 16 and info.pop("tail").endswith("<|im_end|>")
    assert info == {"stream_fps": 1, "frames": 3, "events": 4, "no_answer": False, "response_tag": True,
                    "prompt_with_frame": True}


def test_moss_arms_when_the_question_drains(monkeypatch, fake_plugin_guard, hooks):
    adapter = moss(monkeypatch, ["<|round_start|>B<|silence|>"])
    gen = adapter.stream(request(), hooks)
    assert gen.text == "B" and not gen.info["prompt_with_frame"]
    assert hooks.names() == ["attach", "arm", "step", "step", "disarm", "finish"]
    assert [c[1] for c in hooks.calls if c[0] == "step"] == [0, 1]
    silent = moss(monkeypatch, ["<|round_start|>B<|silence|>"], drains=False)
    with pytest.raises(RuntimeError, match="never took the question"):
        silent.stream(request(), RecordingHooks())


def test_moss_always_disarms_and_a_late_drain_cannot_arm(monkeypatch, fake_plugin_guard):
    """The stream's end disarms the hooks whether or not the turn was armed, and a drain that comes after it (the
    session thread taking the question at the answer timeout or at close) no longer arms them: an armed guard would
    otherwise steer and record through the next request's whole stream."""
    silent, hooks = moss(monkeypatch, ["<|round_start|>B<|silence|>"], drains=False), RecordingHooks()
    with pytest.raises(RuntimeError, match="never took the question"):
        silent.stream(request(), hooks)
    assert hooks.names() == ["attach", "disarm"]
    hooks = RecordingHooks()
    late = mossrt_stream._AnswerTurn(hooks)
    late.end()
    late.arm()
    assert hooks.names() == ["disarm"] and not hooks.armed and not late.asked and not late.armed


# ------------------------------------------------------------------------------------------- AURA

def aura_meta(monkeypatch, frames=34, fps=10.0):
    monkeypatch.setattr(aura_stream, "video_info", lambda path: {"frames": frames, "fps": fps, "header_frames": frames,
                                                                 "duration": frames / fps})


def test_aura_vllm_request_and_visible_reply(monkeypatch, tmp_path):
    aura_meta(monkeypatch)
    clip = tmp_path / "s.mp4"
    clip.write_bytes(b"0")
    monkeypatch.setattr(aura_stream, "segment", lambda video, start, length: clip)
    sent, replies = [], ["<|silent|>", "A"]

    def create(**kwargs):
        sent.append(kwargs)
        message = types.SimpleNamespace(content=replies.pop(0))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")],
                                     usage=types.SimpleNamespace(completion_tokens=3, prompt_tokens=100))
    adapter = aura_stream.AuraStream(None, {"stream": {"aura_concurrency": 1}}, None, load=False)
    adapter.client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))
    assert adapter.concurrency == 1
    gen = adapter.stream(request())
    messages = sent[-1]["messages"]
    assert messages[0] == {"role": "system", "content": aura_stream.SYSTEM} and len(messages) == 1 + 3 + 2
    assert [m["role"] for m in messages[1:]] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["content"][-1] == {"type": "text", "text": "Q?"} and messages[2]["content"] == "<|silent|>"
    assert sent[-1]["temperature"] == 0 and sent[-1]["max_tokens"] == 8
    assert sent[-1]["extra_body"] == {"mm_processor_kwargs": {"fps": 1.0, "size": {"shortest_edge": 4096,
                                                                                    "longest_edge": 602112}}}
    assert gen.text == "" and gen.info["no_answer"] and gen.info["silent_text"]
    assert (gen.info["segments"], gen.info["window_start"], gen.info["duration"]) == (3, 0.4, 3.4)
    gen = adapter.stream(request())
    assert gen.text == "A" and not gen.info["no_answer"] and "silent_text" not in gen.info
    assert (gen.tokens, gen.finish, gen.info["prompt_tokens"]) == (3, "stop", 100)
    with pytest.raises(ValueError, match="aura_stream_hf"):
        adapter.stream(request(), RecordingHooks())


def test_aura_segments_need_imageio_ffmpeg(monkeypatch):
    """No fallback to another FFmpeg build (it could cut different frames into the shared segment cache)."""
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)            # import imageio_ffmpeg -> ImportError
    with pytest.raises(RuntimeError, match="imageio-ffmpeg"):
        aura_stream.ffmpeg_executable()
    fake = types.ModuleType("imageio_ffmpeg")
    fake.get_ffmpeg_exe = lambda: "/env/bin/ffmpeg-bundled"
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake)
    assert aura_stream.ffmpeg_executable() == "/env/bin/ffmpeg-bundled"


class FakeServer:
    """A `vllm serve` process that keeps loading; `stubborn` ignores SIGTERM."""

    def __init__(self, stubborn: bool):
        self.calls, self.stubborn = [], stubborn

    def poll(self):
        return None

    def terminate(self):
        self.calls.append("terminate")

    def wait(self, timeout):
        self.calls.append("wait")
        if self.stubborn and self.calls.count("wait") == 1:
            raise subprocess.TimeoutExpired("vllm", timeout)

    def kill(self):
        self.calls.append("kill")


@pytest.mark.parametrize("stubborn", [False, True])
def test_aura_server_that_never_comes_up_is_stopped(monkeypatch, tmp_path, stubborn):
    """A failed construction is never closed by the worker: the adapter stops its own server (killed when SIGTERM is
    not enough), so the next AURA job does not start a second server on the same GPU."""
    servers = []

    def popen(*args, **kwargs):
        servers.append(FakeServer(stubborn))
        return servers[-1]

    def unreachable():
        raise ConnectionError("still loading")
    openai = types.ModuleType("openai")
    openai.OpenAI = lambda **kwargs: types.SimpleNamespace(models=types.SimpleNamespace(list=unreachable))
    monkeypatch.setitem(sys.modules, "openai", openai)
    monkeypatch.setattr(aura_stream, "subprocess", types.SimpleNamespace(
        Popen=popen, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(aura_stream, "SERVER_WAIT", (2, 0.0))
    monkeypatch.setattr(aura_stream.paths, "cache_dir", lambda: tmp_path)
    monkeypatch.setattr(aura_stream.paths, "checkpoint_dir", lambda ref: tmp_path / "AURA")
    with pytest.raises(RuntimeError, match="did not come up"):
        aura_stream.AuraStream(types.SimpleNamespace(checkpoint="aurateam/AURA"), {}, None)
    assert servers[0].calls == (["terminate", "wait", "kill", "wait"] if stubborn else ["terminate", "wait"])


class FakeBatch(dict):
    def to(self, device):
        return self


def aura_hf(monkeypatch):
    aura_meta(monkeypatch)
    monkeypatch.setattr(aura_stream_hf, "segment", lambda video, start, length: f"{start:.2f}_{length:.2f}.mp4")
    monkeypatch.setattr(aura_stream_hf, "decode", lambda path, length: np.zeros((2, 4, 4, 3), np.uint8))
    adapter = aura_stream_hf.AuraStreamHF(None, {}, None, load=False)
    seen = {}

    def apply_chat_template(messages, tokenize, add_generation_prompt):
        seen["messages"] = messages
        return "prompt"

    def process(**kwargs):
        seen["processor"] = kwargs
        return FakeBatch(input_ids=np.zeros((1, 7), np.int64))

    def generate(**kwargs):
        seen["generate"] = kwargs
        return np.zeros((1, 10), np.int64)
    class Processor:
        tokenizer = types.SimpleNamespace(decode=lambda ids, skip_special_tokens: "<|silent|> B")

        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return apply_chat_template(messages, tokenize, add_generation_prompt)

        def __call__(self, **kwargs):
            return process(**kwargs)

    adapter.processor = Processor()
    adapter.model = types.SimpleNamespace(generate=generate)
    adapter.torch = fake_torch()
    return adapter, seen


def test_aura_hf_rebuilds_the_vllm_prompt(monkeypatch):
    adapter, seen = aura_hf(monkeypatch)
    gen = adapter.stream(request())
    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
    assert seen["messages"][-1]["content"] == [{"type": "video"}, {"type": "text", "text": "Q?"}]
    assert seen["processor"]["video_metadata"][0] == {"fps": 1.0, "frames_indices": [0, 1], "total_num_frames": 2,
                                                      "duration": 2.0}
    assert seen["processor"]["do_sample_frames"] is False and seen["processor"]["do_resize"] is True
    assert seen["processor"]["size"] == {"shortest_edge": 4096, "longest_edge": 602112}
    assert "logits_processor" not in seen["generate"] and seen["generate"]["max_new_tokens"] == 8
    assert gen.text == "B" and gen.tokens == 3 and gen.finish == "eos"
    assert {k: gen.info[k] for k in ("segments", "evicted", "window_start", "duration", "prompt_tokens", "engine")} == \
        {"segments": 3, "evicted": 0, "window_start": 0.4, "duration": 3.4, "prompt_tokens": 7, "engine": "hf"}


def test_aura_hf_arms_its_single_generate(monkeypatch, fake_plugin_guard, fake_transformers, hooks):
    adapter, seen = aura_hf(monkeypatch)
    adapter.stream(request(), hooks)
    assert hooks.names() == ["attach", "processors", "arm", "disarm", "finish"]
    assert list(seen["generate"]["logits_processor"]) == ["first-step processor"]
