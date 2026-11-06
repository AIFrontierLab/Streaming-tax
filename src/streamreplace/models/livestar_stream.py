"""LiveStar's native stream (paper condition S for LiveStar-8B, pair p4).

LiveStar has no official "question at time t" evaluation. The adapter runs its official StreamingSession (upstream
inference/streaming_infer.py: SVeD narration gate) with the accuracy-evaluation settings of its benchmark script, then
asks the question once, as in its fixed-schedule evaluation:

- frames: the framework's fixed-step frames at `stream.fps` (1 fps; the same indices as the p4 offline frames), one
  448x448 tile per frame (max_dynamic_patch 1), 16 tokens per frame after token merging.
- memory: no compression and no retrieval (no_tshm, no_retrieval): every frame and every narration stays in context;
  the ViT features are cached per frame (no_viscache False), no LLM KV cache (kv False).
- narration: SVeD decides per frame whether to narrate: a new caption (384 tokens at most, greedy, repetition penalty
  1.05, with upstream's narration TASK_PROMPT) is written when the frame's perplexity exceeds the reference times
  alpha = 1.04. The perplexity is averaged over num_runs = 3 passes as upstream (a single pass changed the narration
  count on 6 of 8 smoke samples, so it is not equivalent).
- question: frames that arrived after the last narration's first frame move from that narration's turn into the
  question turn: "Frame-<t>: <image>\\n" for each of them (0-based t), then the benchmark prompt. The answer is one
  chat() call on the narration history with the cached visual features, greedy, repetition penalty 1.05, up to the
  benchmark's token budget. LiveStar always answers: it has no silence marker, so `info.no_answer` is never set.
- nondeterminism: the narration count can change from run to run (the gate's perplexity passes on GPU).

Plugin hook points (hooks=None is exactly the native path above):
- attach: once, with the decoder layers of `model.language_model` and the tokenizer.
- answer turn: the final chat() call. `hooks.arm()` runs immediately before it (the gate reads the last position of
  its prefill) and `hooks.disarm()` right after; the call's generation config gets
  `logits_processor=hooks.logits_processors()`. The narration calls run untouched.
- shared session: with hooks, one StreamingSession per (video, fps) serves all of that video's consecutive questions,
  as in the paper's plugin runs (the native path streams the video again for each question). This is a protocol
  change against native S that the paper reports; the narration is one draw of the same stochastic session.
- `hooks.finish(req, generation)` makes the final row.
"""
from __future__ import annotations

from contextlib import ExitStack
import copy
from dataclasses import dataclass
import json
from pathlib import Path
import struct
from types import SimpleNamespace

from .. import paths
from ..core.io import atomic_text, fingerprint
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.frames import frame_indices, sample_frames
from ..media.video_meta import video_info
from .base import AnswerTurnHooks, BaseAdapter

ADAPTER = "livestar_stream"
CODE = "livestar/inference"           # upstream remote code, tokenizer files and streaming_infer.py
NARRATION_TOKENS = 384
REPETITION_PENALTY = 1.05             # the official demo's greedy decoding
TILE = 448
# StreamingSession arguments of upstream's accuracy evaluation: no memory compression, no retrieval, alpha 1.04
SESSION_ARGS = dict(no_tshm=True, l_max=160, sigma=0.75, beta=0.3, beam_k=3, max_recall=2, num_runs=3, trace=False,
                    no_retrieval=True, no_viscache=False, kv=False, recall_verbatim=False, recall_min_gap=0, alpha=1.04)


def _plan(indices: list[int], source_fps: float, stream_fps: float) -> dict:
    plan = {"adapter": ADAPTER, "stream_fps": stream_fps, "frames": len(indices), "indices": indices,
            "frame_indices": frame_digest({"indices": indices}), "question_after_frame": len(indices) - 1,
            "question_time": indices[-1] / source_fps}
    plan["digest"] = fingerprint(plan)
    return plan


def schedule(req: Request, meta: dict | None = None) -> dict:
    """CPU-only: the frames LiveStar's session is fed. `meta` = {"frames" (decodable), "fps" (decord)} as in a
    video_meta manifest row; measured from the file when None. Which frames end up in the question turn depends on
    the narration and is only known after the stream."""
    meta = video_info(req.video) if meta is None else meta
    fps = float(req.settings["stream"]["fps"])
    return _plan(frame_indices({"frames": meta["frames"], "fps": meta["fps"]}, fps=fps), meta["fps"], fps)


def overlay(checkpoint: Path, code: Path) -> Path:
    """The released checkpoints ship weights and config only: a folder of symlinks to the checkpoint's files and the
    upstream code (modeling, tokenizer), with model.safetensors.index.json rebuilt from the shard headers."""
    target = paths.cache_dir() / "overlays" / checkpoint.name
    target.mkdir(parents=True, exist_ok=True)
    for source in (checkpoint, code):
        for path in source.iterdir():
            link = target / path.name
            if path.is_file() and path.suffix in {".py", ".json", ".model", ".safetensors"} \
                    and path.name != "model.safetensors.index.json" and not link.exists():
                try:
                    link.symlink_to(path)
                except FileExistsError:                    # another worker made it first
                    pass
    weight_map, size = {}, 0
    for shard in sorted(checkpoint.glob("*.safetensors")):
        with shard.open("rb") as handle:
            header = json.loads(handle.read(struct.unpack("<Q", handle.read(8))[0]))
        weight_map.update({k: shard.name for k in header if k != "__metadata__"})
        size += shard.stat().st_size
    atomic_text(target / "model.safetensors.index.json",
                json.dumps({"metadata": {"total_size": size}, "weight_map": weight_map}))
    return target


@dataclass
class _Context:
    """What the answer call needs from a finished session."""
    history: list
    pixels: object
    patches: list
    visual: object
    pending: list[int]
    info: dict


class LiveStarStream(BaseAdapter):
    """Condition S of LiveStar-8B (env livestar: transformers 4.37.2, flash-attn 2.7.4.post1, bf16).
    load=False builds a schedule-only adapter (no model)."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    concurrency = 1

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        self._paths = ExitStack()
        self._attached: AnswerTurnHooks | None = None
        self._shared: tuple | None = None               # ((video, fps), _Context) of the last shared session
        if load:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer, GenerationConfig
        self.torch = torch
        # streaming_infer and the remote code import their siblings (tshm, conversation) lazily
        self._paths.enter_context(self.upstream.use(CODE))
        code = self.upstream.path(CODE)
        checkpoint = paths.checkpoint_dir(self.spec.checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(str(code), trust_remote_code=True, local_files_only=True)
        self.model = AutoModel.from_pretrained(str(overlay(checkpoint, code)), trust_remote_code=True,
                                               local_files_only=True, torch_dtype=torch.bfloat16).cuda().eval()
        config = json.loads((checkpoint / "config.json").read_text())
        for key in ("template", "system_message"):
            if key in config:
                setattr(self.model, key, config[key])
        # the release has no generation_config: the language model's defaults (incl. BOS) come from llm_config
        lm_config = self.model.language_model.config.__class__.from_dict(config["llm_config"])
        self.model.language_model.generation_config = GenerationConfig.from_model_config(lm_config)
        self.max_tiles = int(config.get("max_dynamic_patch", 1))
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")

    def schedule(self, req: Request) -> dict:
        return schedule(req)

    def _pixels(self, images) -> list:
        from streaming_infer import build_transform, dynamic_preprocess
        transform = build_transform(input_size=TILE)
        return [self.torch.stack([transform(t) for t in dynamic_preprocess(i, image_size=TILE, max_num=self.max_tiles,
                                                                          use_thumbnail=True)]) for i in images]

    def _gen(self, settings: dict) -> dict:
        """The official demo's decoding: greedy with repetition_penalty 1.05."""
        return {"do_sample": False, "num_beams": 1, "max_new_tokens": settings["generation"]["max_new_tokens"],
                "repetition_penalty": REPETITION_PENALTY}

    def _session(self, req: Request) -> _Context:
        """Runs the StreamingSession over the video and derives the question turn's context: the last narration keeps
        only its first frame; the frames after it are asked about in the question turn."""
        from streaming_infer import StreamingSession
        s = req.settings
        frames, meta = sample_frames(req.video, fps=s["stream"]["fps"])
        tiles = self._pixels(frames)
        args = SimpleNamespace(**SESSION_ARGS)
        with self.torch.inference_mode():
            session = StreamingSession(self.model, self.tokenizer, dict(self._gen(s), max_new_tokens=NARRATION_TOKENS),
                                       tiles, args)
            session.run(len(tiles))
            last = session._captioned_clips()[-1]
            kept, last.frames = last.frames, last.frames[:1]
            history, answered = session._derive_history(), session._committed_gids()
            last.frames = kept
            pending = [f.t for f in kept[1:]]
            ids = answered + pending
            pixels, patches = session._pixels(ids)
            visual = session._vis(ids)
        info = {"stream_fps": s["stream"]["fps"], "frames": len(frames), "context_frames": len(ids),
                "frames_in_question": len(pending), "narrations": len(session.narration), "alpha": args.alpha,
                "num_runs": args.num_runs,
                "schedule": _plan(meta["indices"], meta["fps"], float(s["stream"]["fps"]))["digest"]}
        return _Context(history, pixels, list(patches), visual, pending, info)

    def _attach(self, hooks: AnswerTurnHooks) -> None:
        if hooks is self._attached:
            return
        from ..plugin.guard import find_layers
        hooks.attach(find_layers(self.model.language_model), self.tokenizer)
        self._attached = hooks

    def stream(self, req: Request, hooks: AnswerTurnHooks | None = None) -> Generation:
        if hooks is None:
            context = self._session(req)
        else:
            self._attach(hooks)
            key = (req.video, req.settings["stream"]["fps"])
            if self._shared is None or self._shared[0] != key:
                self._shared = (key, self._session(req))
            context = self._shared[1]
        question = "".join("Frame-%d: <image>\n" % t for t in context.pending) + req.prompt
        gen = self._gen(req.settings)
        if hooks is not None:
            from transformers import LogitsProcessorList
            gen = dict(gen, logits_processor=LogitsProcessorList(hooks.logits_processors()))
            hooks.arm()
        try:
            with self.torch.inference_mode():
                # chat() reads the history without changing it (return_history=False); the copy keeps a shared
                # session's history intact in any case
                text = self.model.chat(self.tokenizer, context.pixels, question, gen,
                                       num_patches_list=list(context.patches), history=copy.deepcopy(context.history),
                                       return_history=False, visual_features=context.visual)
        finally:
            if hooks is not None:
                hooks.disarm()
        info = dict(context.info, **({"session_shared": True} if hooks is not None else {}))
        generation = Generation(text.strip(), None, "unknown", info)
        return hooks.finish(req, generation) if hooks is not None else generation

    def close(self) -> None:
        self.model = None
        self._shared = None
        self._paths.close()
