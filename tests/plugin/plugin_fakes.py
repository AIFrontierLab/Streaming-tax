"""CPU stand-ins for driving the stream adapters with the real plugin hooks (tests/plugin/test_plugin_adapters.py).

A toy decoder with a child's real number of layers and hidden size (so the shipped artefacts attach), a toy
tokenizer holding every special token the children use, a Hugging Face style greedy generate, a recorder of the hook
calls and of every forward of the decoder (which positions carry the PRS vector, the gate layer's state), and the
upstream objects each adapter drives (MMDuet's live inference, ThinkStream's engine, MMDuet2's client, LiveStar's
model and session, MOSS-VL-Realtime's session, AURA's processor). The upstream surfaces follow the fakes of
tests/models_stream/stream_fakes.py and test_call_sites.py, but every model call runs the toy decoder. Needs torch."""
from __future__ import annotations

import sys
import types

import numpy as np
import torch

from streamreplace.core.schema import Request

# ---------- tokenizer ----------

VOCAB = ["A", "B", "C", "D", "Yes", "No", "NO", " REPLY", ",", " The", " man", " walks", ".", "<|im_end|>",
         "<|silent|>", "<|silence|>", "<|response|>", "<|im_start|>", "<|round_start|>", "<think>", "</think>",
         "<silent>", "<response>", "<|vision_end|>", "<|video_pad|>", "x", "y", "z"]
SPECIAL = {t for t in VOCAB if t.startswith("<") and t.endswith(">")}
ID = {t: i for i, t in enumerate(VOCAB)}
EOS = ID["<|im_end|>"]


class Tok:
    """Every token is one vocabulary entry; text outside the vocabulary has no tokens."""

    def __len__(self):
        return len(VOCAB)

    def convert_tokens_to_ids(self, token):
        return ID[token]

    def encode(self, text, add_special_tokens=False):
        return [ID[text]] if text in ID else []

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=self.encode(text))

    def decode(self, ids, skip_special_tokens=False):
        ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
        return "".join(VOCAB[i] for i in ids if not (skip_special_tokens and VOCAB[i] in SPECIAL))


def ids(*tokens) -> torch.Tensor:
    return torch.tensor([[ID[t] for t in tokens]])


# ---------- toy decoder ----------

class ToyDecoderLayer(torch.nn.Module):
    """Adds its own constant vector and a little of the causal mean of its input, so a position's state depends on
    the whole forward; returns a tuple like the decoder layers of transformers."""

    def __init__(self, index: int, hidden: int):
        super().__init__()
        generator = torch.Generator().manual_seed(1000 + index)
        self.register_buffer("shift", torch.randn(hidden, generator=generator) * 0.05)

    def math(self, h):
        mean = h.cumsum(dim=1) / torch.arange(1, h.shape[1] + 1, dtype=h.dtype).view(1, -1, 1)
        return h + self.shift + 0.02 * mean

    def forward(self, h):
        return (self.math(h),)


class VisionLayer(torch.nn.Module):
    """A vision tower's layer list, longer than the decoder's: plugin.guard.find_layers must skip it."""

    def forward(self, h):
        return h


class ScriptedHead(torch.nn.Module):
    """The LM head: a fixed projection, plus on each forward the next entry of `script` ({token id: added logit}) at
    every position, so a test decides what each decoding step would choose natively."""

    def __init__(self, hidden: int):
        super().__init__()
        generator = torch.Generator().manual_seed(7)
        self.register_buffer("weight", torch.randn(len(VOCAB), hidden, generator=generator) * 0.001)
        self.script: list[dict] = []

    def logits(self, h):
        out = h.float() @ self.weight.T
        if self.script:
            for token, value in self.script.pop(0).items():
                out[..., token] += value
        return out

    def forward(self, h):
        return self.logits(h)


class ToyLM(torch.nn.Module):
    """A child's decoder at its real depth and width (position-wise toy layers, no cache)."""

    def __init__(self, layers: int, hidden: int):
        super().__init__()
        generator = torch.Generator().manual_seed(3)
        self.register_buffer("embedding", torch.randn(len(VOCAB), hidden, generator=generator) * 0.05)
        self.layers = torch.nn.ModuleList([ToyDecoderLayer(i, hidden) for i in range(layers)])
        self.visual = torch.nn.ModuleList([VisionLayer() for _ in range(layers + 2)])
        self.lm_head = ScriptedHead(hidden)

    def forward(self, input_ids):
        h = self.embedding[input_ids]
        for layer in self.layers:
            h = layer(h)[0]
        return self.lm_head(h)

    def replay(self, input_ids):
        """The same forward without calling any module, as a replayed CUDA graph: no forward hook runs."""
        h = self.embedding[input_ids]
        for layer in self.layers:
            h = layer.math(h)
        return self.lm_head.logits(h)

    def state(self, input_ids, layer: int):
        """Decoder layer `layer`'s output at the last position, without hooks (what the gate should read)."""
        h = self.embedding[input_ids]
        for block in self.layers[: layer + 1]:
            h = block.math(h)
        return h[0, -1].float()


def toy_for(child: str) -> ToyLM:
    from streamreplace.plugin import artefacts
    manifest = artefacts.read_manifest(child)
    return ToyLM(int(manifest["layers"]), int(manifest["hidden_size"]))


def greedy_generate(lm, input_ids, max_new_tokens: int, logits_processor=None):
    """transformers' greedy generate on the toy decoder: the prompt's prefill, then one forward per new token; the
    logits processors get (every id so far, the last position's scores)."""
    out, step = input_ids, input_ids
    for _ in range(max_new_tokens):
        scores = lm(step)[:, -1, :]
        if logits_processor is not None:
            scores = logits_processor(out, scores)
        token = scores.argmax(-1, keepdim=True)
        out = torch.cat([out, token], dim=1)
        if int(token) == EOS:
            break
        step = token
    return out


# ---------- what happened: hook calls and decoder forwards in one timeline ----------

HOOK_CALLS = ("begin", "attach", "arm", "logits_processors", "on_step_logits", "deadline_readout", "disarm", "finish")


class Timeline:
    """Records the hook calls (spy) and every decoder forward (probe) in the order they happen. A forward entry is
    ("forward", positions, steered positions)."""

    def __init__(self):
        self.events: list[tuple] = []

    def spy(self, hooks):
        """Wraps the hooks' protocol methods so each call is recorded before the real method runs."""
        for name in HOOK_CALLS:
            method = getattr(hooks, name)

            def call(*args, _method=method, _name=name, **kwargs):
                self.events.append((_name,))
                return _method(*args, **kwargs)
            setattr(hooks, name, call)
        return hooks

    def probe(self, lm: ToyLM, p_layer: int, prs) -> None:
        """Forward pre-hooks on layer p_layer and the next one: the output of layer p_layer as the next layer receives
        it (after the plugin's own forward hooks) minus what the layer computed is the PRS vector at the steered
        positions and zero elsewhere. Pre-hooks do not interact with the plugin's forward hooks."""
        prs = torch.as_tensor(np.asarray(prs), dtype=torch.float32)
        seen: dict = {}

        def before_p(_module, args):
            seen["in"] = args[0]

        def after_p(_module, args):
            delta = args[0] - lm.layers[p_layer].math(seen.pop("in"))
            steered = [i for i in range(delta.shape[1]) if torch.allclose(delta[0, i], prs, atol=1e-4)]
            clean = [i for i in range(delta.shape[1]) if torch.allclose(delta[0, i], torch.zeros_like(prs), atol=1e-5)]
            assert len(steered) + len(clean) == delta.shape[1], "a position changed by something other than PRS"
            self.events.append(("forward", delta.shape[1], tuple(steered)))

        lm.layers[p_layer].register_forward_pre_hook(before_p)
        lm.layers[p_layer + 1].register_forward_pre_hook(after_p)

    def calls(self) -> list[str]:
        """The hook calls in order, runs of on_step_logits collapsed."""
        out = []
        for event in self.events:
            if event[0] != "forward" and not (out and out[-1] == event[0] == "on_step_logits"):
                out.append(event[0])
        return out

    def forwards(self, inside: bool) -> list[tuple]:
        """(positions, steered positions) of the decoder forwards inside or outside an armed answer turn."""
        out, armed = [], False
        for event in self.events:
            if event[0] in ("arm", "disarm"):
                armed = event[0] == "arm"
            elif event[0] == "forward" and armed == inside:
                out.append(event[1:3])
        return out


def request(kind="mcq", video="v.mp4", settings=None, sample_id="q1", prompt="Q?") -> Request:
    settings = settings or {"generation": {"max_new_tokens": 8}, "stream": {"fps": 1}}
    choices = {"A": "a", "B": "b", "C": "c", "D": "d"} if kind == "mcq" else {}
    return Request(sample_id, "demo", prompt, video=video, settings=settings, kind=kind, choices=choices)


def torch_proxy(**overrides):
    """The torch attributes the adapters' control flow uses, on the CPU (zeros ignores the CUDA device)."""
    attributes = {"zeros": lambda *a, **k: torch.zeros(*a, **{x: v for x, v in k.items() if x != "device"}),
                  "long": torch.long, "inference_mode": torch.inference_mode, "manual_seed": lambda seed: None,
                  "stack": lambda tiles: tiles,
                  "cuda": types.SimpleNamespace(get_rng_state=lambda: "rng", set_rng_state=lambda state: None)}
    attributes.update(overrides)
    return types.SimpleNamespace(**attributes)


# ---------- MMDuet: LiveInferForDemo ----------

GEN_PROMPT = ("<|im_end|>", "<|im_start|>", "z")       # the short generation prompt MMDuet's reply opens with


class MMDuetRoot(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def get_output_embeddings(self):
        return self.lm.lm_head

    def forward(self, input_ids):
        return self.lm(input_ids)


class FakeLiveInfer:
    """LiveInferForDemo on the toy decoder: every frame and the question run a forward; a frame in `fire` replies
    from inside input_one_frame (through self._generate_response, as upstream); _generate_response is MMDuet's
    fast greedy loop: the generation prompt, then one token per forward, argmax of the LM head's (hooked) output,
    stop at eos. `scripts`: per reply, the head script of its forwards."""

    def __init__(self, lm, fire=(), scripts=()):
        self.lm, self.model, self.tokenizer = lm, MMDuetRoot(lm), Tok()
        self.eos_token_id = EOS
        self.fire, self.scripts = set(fire), [list(s) for s in scripts]
        self.answer_prompt = None
        self.queried = False

    def reset(self):
        self.frame_embeds_queue, self.queried, self.answer_prompt = [], False, None

    def input_video_stream(self, images):
        self.frame_embeds_queue = list(images)

    def encode_given_query(self, prompt):
        self.model(ids("<|im_start|>", "y", "y"))
        self.queried = True

    def input_one_frame(self):
        frame = self.frame_embeds_queue.pop(0)
        self.model(ids("<|im_start|>", "x"))
        response = self._generate_response() if frame in self.fire else None
        return {"frame_idx": frame + 1, "time": frame / 2, "informative_score": 0.1, "relevance_score": 0.2,
                "response": response}

    def _generate_response(self):
        if self.queried and self.answer_prompt is None:
            self.answer_prompt = ids(*GEN_PROMPT)                 # the answer turn's first forward
        self.lm.lm_head.script = self.scripts.pop(0) if self.scripts else []
        step, tokens = ids(*GEN_PROMPT), []
        for _ in range(self.inplace_output_ids.shape[1]):
            token = int(self.model(step)[0, -1].argmax())
            tokens.append(token)
            if token == self.eos_token_id:
                break
            step = torch.tensor([[token]])
        return self.tokenizer.decode([t for t in tokens if t != EOS])


def mmduet(monkeypatch, lm, frames=4, **kwargs):
    from streamreplace.models import mmduet_stream
    monkeypatch.setattr(mmduet_stream, "measure", lambda path: {"frames": frames, "fps": 2.0, "header_frames": frames})
    adapter = mmduet_stream.MMDuetStream(None, {}, None, load=False)
    adapter.torch, adapter.infer = torch_proxy(), FakeLiveInfer(lm, **kwargs)
    adapter._load_video = lambda path, output_fps: (list(range(frames)), None)
    return adapter


# ---------- ThinkStream: StreamingWindowInferenceEngine + streaming_video_chat ----------

class FakeDecoder:
    def __init__(self, lm):
        self.lm, self.cache, self.cache_seqlens = lm, "static cache", torch.tensor([7])
        self.static_input_ids = torch.zeros(1, 1, dtype=torch.long)
        self.static_position_ids = torch.zeros(1, 1, dtype=torch.long)
        self.static_logits = torch.zeros(1, 1, len(VOCAB))
        self.replayed = 0

    def step(self, input_ids, position_ids):
        """A decode step by CUDA-graph replay: no module is called, so no forward hook runs."""
        self.replayed += 1
        self.static_logits.copy_(self.lm.replay(input_ids))
        return self.static_logits


class EngineModel:
    """engine.model: the forward the adapter's eager decode step calls."""

    def __init__(self, lm):
        self.lm, self.calls = lm, 0

    def __call__(self, input_ids, **kwargs):
        self.calls += 1
        return types.SimpleNamespace(logits=self.lm(input_ids))


class FakeEngine:
    """generate(): the chunk's prefill through self.prefill (an instance attribute the adapter may wrap), then decode
    steps through self.decoder.step until eos or `steps`; the query chunk runs `answer_script` and, with
    `deadline`, raises upstream's missed-deadline error after its decode steps."""

    def __init__(self, lm, answer_script, deadline=False, steps=4):
        self.lm, self.model, self.decoder = lm, EngineModel(lm), FakeDecoder(lm)
        self.next_start_pos, self._window_count = None, [3]
        self.answer_script, self.deadline, self.steps = answer_script, deadline, steps

    def prefill(self, input_ids, **kwargs):
        return self.lm(input_ids)

    def generate(self, input_ids, position_ids, sample_kwargs, **kwargs):
        query = bool(sample_kwargs.get("is_query_window")) and sample_kwargs.get("allow_deferral") is False
        self.lm.lm_head.script = [dict(s) for s in self.answer_script] if query else []
        token = self.prefill(input_ids=input_ids, position_ids=position_ids)[:, -1:].argmax(-1)
        tokens = [int(token)]
        for _ in range(self.steps):
            if tokens[-1] == EOS:
                break
            token = self.decoder.step(token, torch.zeros(1, 1, dtype=torch.long))[:, -1:].argmax(-1)
            tokens.append(int(token))
        if query and self.deadline:
            raise RuntimeError("Model failed to answer when deferral was disallowed (Deadline exceeded).")
        return torch.tensor([tokens])


def install_torchcodec(monkeypatch, duration=3.0, frames=90, fps=30.0):
    decoders = types.ModuleType("torchcodec.decoders")
    meta = types.SimpleNamespace(duration_seconds=duration, num_frames=frames, average_fps=fps)
    decoders.VideoDecoder = lambda path: types.SimpleNamespace(metadata=meta)
    package = types.ModuleType("torchcodec")
    package.decoders = decoders
    monkeypatch.setitem(sys.modules, "torchcodec", package)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", decoders)


def chunk_ids(index: int, last: bool) -> torch.Tensor:
    return ids("<|im_start|>", *(["x"] * (index + 1)), "<|vision_end|>", *(["y", "y"] if last else []))


def thinkstream(monkeypatch, lm, answer_script, chunks=3, deadline=False):
    from streamreplace.models import thinkstream_stream
    install_torchcodec(monkeypatch)
    adapter = thinkstream_stream.ThinkStreamStream(None, {}, None, load=False)
    tok = Tok()
    adapter.torch = torch_proxy()
    adapter.processor = types.SimpleNamespace(tokenizer=tok, decode=lambda t, skip_special_tokens: tok.decode(t))
    adapter.model = MMDuetRoot(lm)                     # any module holding the decoder: find_layers' root
    adapter.model.config = types.SimpleNamespace()
    adapter._restricted_sample, adapter._system, adapter._template = "restricted", "system", "template"
    engines = []

    def engine(stream, window):
        engines.append(FakeEngine(lm, answer_script, deadline))
        return engines[-1]
    adapter._engine = engine

    def chat(engine, sample_kwargs, **kwargs):
        for index in range(chunks):
            last = index == chunks - 1
            x = chunk_ids(index, last)
            out = engine.generate(input_ids=x, position_ids=torch.zeros(3, 1, x.shape[1], dtype=torch.long),
                                  sample_kwargs=dict(sample_kwargs, is_query_window=last, allow_deferral=not last))
            yield {"is_answer": last, "generated_tokens": out}
    adapter._chat = chat
    return adapter, engines


# ---------- MMDuet2: ProactiveInferenceAPIClient ----------

class ClientModel(torch.nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def generate(self, input_ids, max_new_tokens, logits_processor=None, **kwargs):
        return greedy_generate(self.lm, input_ids, max_new_tokens, logits_processor)


class FakeProactiveClient:
    """The client's surface on the toy decoder: a turn generates its reply when its images are complete; the
    question (add_text) joins the final turn after its images; each turn's reply runs the next of `scripts`."""

    def __init__(self, lm, scripts):
        self.lm, self.model, self.scripts = lm, ClientModel(lm), [list(s) for s in scripts]
        self.processor = types.SimpleNamespace(tokenizer=Tok())
        self.answer_prompt = None

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
        question = bool(self.text)
        prompt = ids("<|im_start|>", *(["x"] * len(self.buffer)), *(["y", "y"] if question else []), "<|im_start|>")
        if question:
            self.answer_prompt = prompt
        self.history.append({"role": "user", "content": [*self.buffer, *([self.text] if question else [])]})
        self.buffer, self.text = [], ""
        self.lm.lm_head.script = self.scripts.pop(0) if self.scripts else []
        out = self.model.generate(input_ids=prompt, max_new_tokens=self.max_new_tokens)
        reply = self.processor.tokenizer.decode(out[0, prompt.shape[1]:], skip_special_tokens=True)
        self.history.append({"role": "assistant", "content": reply})


def mmduet2(monkeypatch, lm, scripts, frames=5):
    from streamreplace.models import mmduet2_stream
    images = [np.zeros((4, 4, 3), np.uint8) for _ in range(frames)]
    meta = {"indices": list(range(0, 4 * frames, 4)), "timestamps": list(range(frames)), "fps": 4.0,
            "duration": float(frames)}
    monkeypatch.setattr(mmduet2_stream, "sample_frames", lambda path, fps: (images, meta))
    adapter = mmduet2_stream.MMDuet2Stream(None, {}, None, load=False)
    adapter.client, adapter.resize_frame = FakeProactiveClient(lm, scripts), lambda array, max_size: array
    return adapter


# ---------- LiveStar: InternVL chat() and the StreamingSession ----------

class Frame:
    def __init__(self, t):
        self.t = t


class Clip:
    def __init__(self, frames):
        self.frames = frames


class FakeSession:
    """StreamingSession's surface; its narration runs one forward of the language model per clip."""
    created = 0

    def __init__(self, model, tokenizer, gen, tiles, args):
        FakeSession.created += 1
        self.model, self.narration = model, [(0, "a man"), (2, "he walks")]

    def run(self, n):
        self.clips = [Clip([Frame(0), Frame(1)]), Clip([Frame(2), Frame(3), Frame(4)])]
        for _ in self.clips:
            self.model.language_model(ids("<|im_start|>", "x", "x"))

    def _captioned_clips(self):
        return self.clips

    def _derive_history(self):
        return [("".join(f"Frame-{f.t}: <image>\n" for f in c.frames), f"caption {i}") for i, c in enumerate(self.clips)]

    def _committed_gids(self):
        return [f.t for c in self.clips for f in c.frames]

    def _pixels(self, gids):
        return ("pixels", tuple(gids)), [1] * len(gids)

    def _vis(self, gids):
        return ("features", tuple(gids))


class LiveStarModel(torch.nn.Module):
    """chat(): the question turn's prompt through the language model, greedy generate with the given config."""

    def __init__(self, lm, scripts):
        super().__init__()
        self.language_model, self.scripts = lm, [list(s) for s in scripts]
        self.answer_prompts: list = []

    def chat(self, tokenizer, pixels, question, gen, **kwargs):
        prompt = ids("<|im_start|>", *(["x"] * question.count("<image>")), "y", "y", "<|im_start|>")
        self.answer_prompts.append(prompt)
        self.language_model.lm_head.script = self.scripts.pop(0) if self.scripts else []
        out = greedy_generate(self.language_model, prompt, gen["max_new_tokens"], gen.get("logits_processor"))
        return tokenizer.decode(out[0, prompt.shape[1]:], skip_special_tokens=True)


def livestar(monkeypatch, lm, scripts):
    from streamreplace.models import livestar_stream
    infer = types.ModuleType("streaming_infer")
    infer.StreamingSession = FakeSession
    infer.build_transform = lambda input_size: (lambda tile: tile)
    infer.dynamic_preprocess = lambda image, image_size, max_num, use_thumbnail: [image]
    monkeypatch.setitem(sys.modules, "streaming_infer", infer)
    monkeypatch.setattr(livestar_stream, "sample_frames",
                        lambda path, fps: (list(range(5)), {"indices": [0, 30, 60, 90, 120], "fps": 30.0}))
    adapter = livestar_stream.LiveStarStream(None, {}, None, load=False)
    adapter.model, adapter.torch = LiveStarModel(lm, scripts), torch_proxy()
    adapter.tokenizer, adapter.max_tiles = Tok(), 1
    FakeSession.created = 0
    return adapter


# ---------- MOSS-VL-Realtime: the realtime session ----------

class FakeRealtimeSession:
    """The session thread, inline: frames run a forward each; push_prompt_frame drains the question through the
    model's input update, then decodes greedily from the (hooked) LM head until <|silence|> / <|im_end|> and queues
    the output as the session prints it."""

    def __init__(self, model, budget=8):
        self.model, self.budget, self.queue, self.pushed = model, budget, [], False
        self.pending_frames = 0

    @property
    def active(self):
        return bool(self.queue) or not self.pushed

    def start(self):
        pass

    def poll_output(self, timeout):
        return self.queue.pop(0) if self.queue else None

    def push_frame(self, frame, timestamp, drop_oldest):
        self.model(ids("<|im_start|>", "x"))

    def push_prompt_frame(self, prompt, frame, timestamp, drop_oldest):
        model, tok = self.model, Tok()
        model._update_model_kwargs_for_real_time_generation(input_ids=torch.tensor([ID["x"], ID["x"]]))
        model.lm.lm_head.script = list(model.answer_script)
        model.answer_prompt = step = ids("<|im_start|>", "y", "y")
        tokens = []
        for _ in range(self.budget):
            token = int(model(step)[0, -1].argmax())
            tokens.append(token)
            if VOCAB[token] in ("<|silence|>", "<|im_end|>"):
                break
            step = torch.tensor([[token]])
        self.pushed = True
        self.queue.append("<|round_start|>" + tok.decode(tokens))

    def close(self, timeout):
        pass


class FakeMossModel(torch.nn.Module):
    def __init__(self, lm, answer_script):
        super().__init__()
        self.lm, self.lm_head, self.answer_script = lm, lm.lm_head, answer_script
        self.answer_prompt = None

    def forward(self, input_ids):
        return self.lm(input_ids)

    def create_realtime_session(self, processor, **kwargs):
        return FakeRealtimeSession(self)

    def _update_model_kwargs_for_real_time_generation(self, input_ids):
        return (torch.cat([input_ids, torch.tensor([ID["<|im_start|>"]])]),)


def moss(monkeypatch, lm, answer_script):
    from streamreplace.models import mossrt_stream
    meta = {"indices": [0, 1, 2], "timestamps": [0.0, 0.01, 0.02], "duration": 0.03, "fps": 100.0}
    monkeypatch.setattr(mossrt_stream, "sample_frames", lambda path, fps: (["f0", "f1", "f2"], meta))
    adapter = mossrt_stream.MossRTStream(None, {}, None, load=False)
    adapter.model = FakeMossModel(lm, answer_script)
    adapter.processor = types.SimpleNamespace(tokenizer=Tok())
    return adapter


# ---------- AURA: the transformers rebuild ----------

class FakeBatch(dict):
    def to(self, device):
        return self


class AuraModel(torch.nn.Module):
    def __init__(self, lm, script):
        super().__init__()
        self.lm, self.script = lm, script

    def generate(self, input_ids, max_new_tokens, logits_processor=None, **kwargs):
        self.lm.lm_head.script = list(self.script)
        return greedy_generate(self.lm, input_ids, max_new_tokens, logits_processor)


def aura_prompt(segments: int) -> torch.Tensor:
    """The toy prompt of an AURA request over `segments` segments (what the processor returns)."""
    return ids("<|im_start|>", *(["x"] * min(segments, 4)), "y", "y", "<|im_start|>")


class AuraProcessor:
    tokenizer = Tok()

    def __init__(self):
        self.videos = None
        self.answer_prompt = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "prompt"

    def __call__(self, text, videos, **kwargs):
        self.videos = len(videos)
        self.answer_prompt = aura_prompt(len(videos))
        return FakeBatch(input_ids=self.answer_prompt)


def aura_hf(monkeypatch, lm, script, frames=34, fps=10.0, settings=None):
    from streamreplace.models import aura_stream, aura_stream_hf
    monkeypatch.setattr(aura_stream, "video_info", lambda path: {"frames": frames, "fps": fps, "header_frames": frames,
                                                                 "duration": frames / fps})
    monkeypatch.setattr(aura_stream_hf, "segment", lambda video, start, length: f"{start:.2f}_{length:.2f}.mp4")
    monkeypatch.setattr(aura_stream_hf, "decode", lambda path, length: np.zeros((2, 4, 4, 3), np.uint8))
    adapter = aura_stream_hf.AuraStreamHF(None, settings or {}, None, load=False)
    adapter.processor, adapter.model, adapter.torch = AuraProcessor(), AuraModel(lm, script), torch
    return adapter
