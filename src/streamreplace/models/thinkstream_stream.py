"""ThinkStream's native stream (paper condition S for ThinkStream-3B, pair p2).

ThinkStream ships an official "question at time t, answer now" evaluation (thinkstream/eval/eval_common.py,
mcq_predict_streaming, used for OVO-Bench and StreamingBench-RT). The adapter runs the same upstream code
(thinkstream.model.inference: StreamingWindowInferenceEngine + streaming_video_chat) with the official settings:

- frames: the video is cut into chunks of max(1, ceil(duration / 120)) seconds, at most 120 chunks, 2 frames per chunk
  read uniformly over the whole video by qwen-vl-utils (torchcodec), at 200,704-401,408 pixels per frame (the
  official eval's values, forced on the video processor; the checkpoint's own config would resize again). Each chunk
  is one user turn. `schedule()` reproduces the chunking and frame selection without decoding.
- memory: KV cache of 24,576 tokens holding the last 20 video chunks (the checkpoint's video_flex_window_size,
  default 20); older chunks are evicted (their visual tokens; thoughts and template tokens stay). Settings
  `stream.window` (chunks, or "full" = all 120) and `stream.max_len` change the budget: the full-history control
  S_full is {window: full, max_len: 81920}. Setting `stream.reservoir: k` is Stream++'s Recall: within the same
  window budget the newest k chunks stay and, when the budget is full, the older chunk the reservoir would not keep
  is evicted instead of the oldest (`recall_engine`, plugin/recall.py evict_index); the row records the retained
  chunks (`info.retained`). `stream.memory_seed` varies the reservoir.
- question: asked at the end of the video (timestamp = torchcodec duration, which puts it in the last chunk), as
  "\\n" + the benchmark prompt after that chunk's frames; ThinkStream's own system prompt, the dataset's is not used.
- decoding: every chunk thinks for at most 20 sampled tokens (T 1.0, top-k 50, top-p 0.95, through flashinfer), seeded
  once per question with torch.manual_seed(int(sha256(id)[:8], 16)) (`readouts.stream_seed`). Chunks before the
  question are forced to </think><silent>. The question chunk is the deadline: after </think> it must answer.
  Multiple-choice / yes-no: <response> then the argmax over the option tokens (eval_common's restricted readout).
  Open questions have no official procedure: <response> is forced after </think> and the answer then decodes freely
  (still sampled at T = 1) up to the benchmark's budget.
- answer: the text between <response> and <|im_end|> (`readouts.answer_text`). When the thought ends with
  <|im_end|> before </think>, upstream raises "Deadline exceeded": the row is unanswered (`info.no_answer`,
  `info.failure`), scored wrong. A RuntimeError before any chunk was streamed is an error, not silence.
- nondeterminism: sampled thoughts; reproducible on the same GPU type and flashinfer version only.
- question timing (setting `stream.question_at: start | middle`, a diagnostic; Supp. "Question timing"): the
  question is posed once at 0 s or at half the duration and ThinkStream's official deferral runs to the video's end
  (the deadline, where an answer is forced, is the last chunk). Its query interface returns one answer and ends the
  run (break_on_answer), so the first answer is the scored one.
- deliberate deviations from the official eval: the video is not extended by slack_time past the question, the answer
  budget is the benchmark's (official 30 tokens), and the question is the harness prompt (official: options plus
  "\\nPlease select the correct answer.").

Plugin hook points (hooks=None is exactly the native path above):
- attach: once, with the language model's decoder layers and the tokenizer.
- answer turn: the query chunk's `engine.generate` (is_query_window and no deferral left: the deadline chunk).
  `hooks.arm()` runs immediately before it (the gate reads the last position of its prefill, after eviction) and
  `hooks.disarm()` right after. Other chunks run untouched.
- decode steps: the engine decodes by CUDA-graph replay, where forward hooks do not run. When the hooks set
  `eager_decode` (true for parent-restoration steering), the query chunk's decode steps run the same forward eagerly,
  so the steering also reaches the sampled thinking tokens; every other chunk keeps the graph.
- deadline readout: when the turn ends unanswered, `hooks.deadline_readout(fork)` gets a callable that re-prefills the
  query chunk from its state before the prefill (cache length, next MROPE start and CUDA RNG restored) with the
  question followed by "<think></think><response>", unsteered, and returns the logits at <response> (float32, CPU).
  On a clip of one chunk upstream raises before its first yield; with hooks that is the same unanswered turn.
- `hooks.finish(req, generation)` makes the final row.
"""
from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path

from .. import paths
from ..core.io import fingerprint
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.qwen_sampling import fetch_video_selection
from ..media.video_meta import torchcodec_meta
from .base import AnswerTurnHooks, BaseAdapter
from .readouts import answer_text, option_tokens, question_at, stream_seed

ADAPTER = "thinkstream_stream"
THINKSTREAM_PIXELS = (200704, 401408)       # official benchmark eval (scripts/eval/{ovo,rtvu}/eval.sh)
FRAMES_PER_CHUNK = 2
MAX_CHUNKS = 120
MAX_LEN = 24576                              # static KV cache of the official eval
DEFAULT_WINDOW = 20                          # thinkstream.model.DEFAULT_VIDEO_FLEX_WINDOW_SIZE
MAX_THINK_TOKENS = 20
SLACK_TIME = 3.0
DEADLINE = "Deadline exceeded"               # upstream's RuntimeError text for a turn that ends without an answer


class AnswerTurnError(Exception):
    """A failure inside the plugin's answer turn. Not a RuntimeError on purpose: the native path reads a RuntimeError
    raised after the first chunk as upstream's missed deadline (an unanswered row), and only that may pass as one."""


def window_chunks(stream: dict, config_window: int | None = None) -> int:
    """KV window in video chunks: settings stream.window ("full" = all MAX_CHUNKS), else the checkpoint's
    video_flex_window_size, else 20."""
    window = stream.get("window") or config_window or DEFAULT_WINDOW
    return MAX_CHUNKS if window == "full" else int(window)


def chunking(duration: float, video_fps: float, total_frames: int, frames_per_chunk: int = FRAMES_PER_CHUNK,
             max_chunks: int = MAX_CHUNKS) -> dict:
    """preload_video's chunking (thinkstream/data/stream_data_processor.py) for video_start = 0 and video_end = the
    torchcodec duration, with its arithmetic unchanged: chunk length max(1, ceil(d / 120)) s, floor(d / length)
    chunks, and fewer, longer chunks when the video has fewer frames than 2 per chunk (low-fps clips)."""
    video_start, video_end = 0.0, duration
    if video_end <= video_start:     # degenerate clip: a window of 4 frames around the start, as upstream
        t, max_sec = video_start, (total_frames - 1) / video_fps
        half_span = 2.0 / video_fps
        video_start = max(0.0, t - half_span)
        video_end = min(max_sec, t + half_span)
    video_duration = video_end - video_start
    chunk_seconds = max(1.0, float(math.ceil(video_duration / max_chunks)))
    chunks = max(1, int(video_duration // chunk_seconds))
    total_nframes = chunks * frames_per_chunk
    start_frame = math.ceil(max(0.0, video_start) * video_fps)
    end_frame = min(math.floor(video_end * video_fps), total_frames - 1)
    available = max(end_frame - start_frame + 1, frames_per_chunk)
    if total_nframes > available:
        chunks = max(1, available // frames_per_chunk)
        total_nframes = chunks * frames_per_chunk
        chunk_seconds = video_duration / chunks
    windows = []
    for index in range(chunks):
        end = video_end if index == chunks - 1 else video_start + (index + 1) * chunk_seconds
        windows.append([video_start + index * chunk_seconds, end])
    return {"video_start": video_start, "video_end": video_end, "chunk_seconds": chunk_seconds, "chunks": chunks,
            "frames": total_nframes, "windows": windows}


def measure(path: str) -> dict:
    """torchcodec's exact-mode metadata, what both the adapter (duration) and preload_video (fps, frame count) read."""
    return {"torchcodec": torchcodec_meta(path)}


def schedule(req: Request, meta: dict | None = None, config_window: int | None = None) -> dict:
    """CPU-only: ThinkStream's chunks, their frames (qwen-vl-utils selection), the question chunk and the seed.
    `meta` = {"torchcodec": {"frames", "fps", "duration"}} as in a video_meta manifest row; measured when None."""
    tc = (measure(req.video) if meta is None else meta)["torchcodec"]
    plan = chunking(tc["duration"], tc["fps"], tc["frames"])
    video_end = plan["video_end"] if plan["video_end"] > plan["video_start"] else plan["video_start"] + 1e-3
    selection = fetch_video_selection({"video_start": plan["video_start"], "video_end": video_end,
                                       "nframes": plan["frames"]}, tc["frames"], tc["fps"])
    indices = selection["frames_indices"][:plan["chunks"] * FRAMES_PER_CHUNK]
    stream = req.settings.get("stream", {})
    plan.update(adapter=ADAPTER, duration=tc["duration"], frames_per_chunk=FRAMES_PER_CHUNK, indices=indices,
                frame_indices=frame_digest({"indices": indices}), question_chunk=plan["chunks"] - 1,
                question_time=tc["duration"], window_chunks=window_chunks(stream, config_window),
                max_len=int(stream.get("max_len") or MAX_LEN), seed=stream_seed(req.id, stream.get("seed")))
    if stream.get("reservoir"):                 # Recall: the newest k chunks + a reservoir within the window budget
        plan.update(reservoir=int(stream["reservoir"]), memory_seed=int(stream.get("memory_seed", 0)))
    where = question_at(stream)
    if where is not None:
        plan.update(question_at=where, question_time=0.0 if where == "start" else tc["duration"] / 2)
    plan["digest"] = fingerprint(plan)
    return plan


def _open_answer_sample(next_token, logits, step, generated_tokens, generated_length, think_end_token_id,
                        max_think_tokens, eos_token_id, silent_token_id, response_token_id, is_query_window=True,
                        allow_deferral=False, **kwargs):
    """ThinkStream sampler for open questions (there is no official one): observation chunks exactly as the official
    think_budget_sample_restricted (thinking budget, then <silent>); in the answer chunk the thinking budget applies,
    <response> is forced right after </think>, and the answer then decodes freely up to max_new_tokens."""
    import torch
    from thinkstream.model.inference import think_budget_sample, think_budget_sample_restricted
    if not is_query_window:
        return think_budget_sample_restricted(next_token, logits, step, generated_tokens, generated_length,
                                              think_end_token_id, max_think_tokens, eos_token_id, silent_token_id,
                                              response_token_id, restricted_token_ids=None, is_query_window=False)
    next_token = think_budget_sample(next_token, logits, step, generated_tokens, generated_length,
                                     think_end_token_id, max_think_tokens)
    if step == 0:
        return next_token
    think = generated_tokens[:, :step] == think_end_token_id
    first_after = think.any(dim=1) & (step - think.to(torch.long).argmax(dim=1) - 1 == 0)
    return torch.where(first_after.unsqueeze(1), torch.full_like(next_token, response_token_id), next_token)


def recall_engine(base):
    """ThinkStream's window engine with Recall's eviction: when the visual budget is full, the older chunk the
    reservoir would not keep goes instead of the oldest (plugin/recall.py evict_index). The KV move is the engine's
    own cache compaction plus its window bookkeeping; text KV stays, as natively. The instance attribute `recall` =
    {"recent", "stream_id", "seed", "retained": arrival indices kept (oldest first), "arriving": the index of the
    chunk the next generate adds}."""
    from ..plugin.recall import evict_index

    class RecallEngine(base):
        def _maybe_evict(self):
            if int(self._window_count[0]) < self.video_flex_window_size:
                return
            r = self.recall
            index = evict_index(r["retained"], r["arriving"], r["recent"], r["stream_id"], r["seed"])
            starts = self._window_starts[:, index].clone()
            ends = self._window_ends[:, index].clone()
            amount = (ends - starts).unsqueeze(1)
            self._cache_eviction.evict(starts, ends)
            count = int(self._window_count[0])
            keep = [i for i in range(count) if i != index]
            old_starts = self._window_starts[:, keep].clone()
            old_ends = self._window_ends[:, keep].clone()
            self._window_starts[:, :count - 1] = old_starts - (old_starts >= ends.unsqueeze(1)) * amount
            self._window_ends[:, :count - 1] = old_ends - (old_ends >= ends.unsqueeze(1)) * amount
            self._window_count -= 1
            r["retained"].pop(index)

        def _record_video_windows(self, *args):
            super()._record_video_windows(*args)
            self.recall["retained"].append(self.recall["arriving"])

    return RecallEngine


class _AnswerTurn:
    """Plugin state of one request: the query chunk's inputs and the engine state the deadline fork restores."""

    def __init__(self, hooks: AnswerTurnHooks, prompt: str):
        self.hooks, self.prompt = hooks, prompt
        self.query = False          # inside the query chunk's generate
        self.pre = None             # (cache_seqlens, next MROPE start, CUDA RNG) after eviction, before the prefill
        self.chunk: dict | None = None
        self.engine = self.prefill = None

    def deadline_missed(self, exc: BaseException) -> bool:
        return DEADLINE in str(exc) and self.pre is not None


class ThinkStreamStream(BaseAdapter):
    """Condition S of ThinkStream-3B (env modern; flash-attn and flashinfer). load=False: schedule-only adapter."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    concurrency = 1

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        self._paths = ExitStack()
        self._attached: AnswerTurnHooks | None = None
        if load:
            self._load()

    def _load(self) -> None:
        """As the official benchmark eval (eval_common.py load_model_and_processor): the video processor is set to
        200,704-401,408 pixels, otherwise the checkpoint's demo config (100,352-150,528) resizes again."""
        import torch
        from transformers import AutoProcessor
        self.torch = torch
        self._paths.enter_context(self.upstream.use("thinkstream"))
        from thinkstream.data.stream_data_processor import QWEN_TEMPLATE_WO_SYSTEM, SYSTEM_PROMPT
        from thinkstream.model import DEFAULT_VIDEO_FLEX_WINDOW_SIZE, MODEL_CLS
        # importing the inference module also registers the "flash_attention_2_infer" attention set below
        from thinkstream.model.inference import (StreamingWindowInferenceEngine, streaming_video_chat,
                                                 think_budget_sample_restricted)
        self._engine_cls, self._chat = StreamingWindowInferenceEngine, streaming_video_chat
        self._restricted_sample = think_budget_sample_restricted
        self._default_window = DEFAULT_VIDEO_FLEX_WINDOW_SIZE
        self._system, self._template = SYSTEM_PROMPT, QWEN_TEMPLATE_WO_SYSTEM
        checkpoint = str(paths.checkpoint_dir(self.spec.checkpoint))
        self.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
        vp = self.processor.video_processor
        vp.min_pixels, vp.max_pixels = THINKSTREAM_PIXELS
        vp.size["shortest_edge"], vp.size["longest_edge"] = THINKSTREAM_PIXELS
        self.model = MODEL_CLS["qwen2.5vl"].from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
                                                           attn_implementation="flash_attention_2", device_map="cuda:0",
                                                           local_files_only=True).eval()
        self.model.config.text_config._attn_implementation = "flash_attention_2_infer"
        tokenizer = self.processor.tokenizer
        self._vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self._nothink = tokenizer("<think></think><response>", add_special_tokens=False).input_ids

    def _config_window(self) -> int | None:
        """The checkpoint's video_flex_window_size, else upstream's default (20); None without a loaded model, where
        window_chunks() applies the same default."""
        model = getattr(self, "model", None)
        if model is None:
            return None
        return getattr(model.config, "video_flex_window_size", getattr(self, "_default_window", None))

    def schedule(self, req: Request) -> dict:
        return schedule(req, config_window=self._config_window())

    def _attach(self, hooks: AnswerTurnHooks) -> None:
        if hooks is self._attached:
            return
        from ..plugin.guard import find_layers
        hooks.attach(find_layers(self.model), self.processor.tokenizer)
        self._attached = hooks

    def _engine(self, stream: dict, window: int):
        """The request's KV engine: upstream's window engine, or with stream.reservoir its Recall subclass."""
        tokenizer = self.processor.tokenizer
        cfg = getattr(self.model.config, "text_config", self.model.config)
        engine_cls = recall_engine(self._engine_cls) if stream.get("reservoir") else self._engine_cls
        return engine_cls(
            self.model, batch_size=1, max_len=int(stream.get("max_len") or MAX_LEN), num_hidden_layers=cfg.num_hidden_layers,
            num_key_value_heads=cfg.num_key_value_heads, head_dim=cfg.hidden_size // cfg.num_attention_heads,
            vocab_size=cfg.vocab_size, pad_token_id=self.model.generation_config.pad_token_id,
            eos_token_ids=self.model.generation_config.eos_token_id,
            video_token_id=tokenizer.convert_tokens_to_ids("<|video_pad|>"), video_flex_window_size=window)

    def _wrap(self, engine, turn: _AnswerTurn) -> None:
        """The plugin's call sites on this request's engine (see the module docstring); other chunks pass through."""
        torch = self.torch
        prefill, generate, decoder = engine.prefill, engine.generate, engine.decoder
        graph_step = decoder.step
        eager = bool(getattr(turn.hooks, "eager_decode", False))

        def eager_step(input_ids, position_ids):
            # the captured decode call, run eagerly so that forward hooks run
            with torch.inference_mode():
                decoder.static_input_ids.copy_(input_ids)
                decoder.static_position_ids.copy_(position_ids)
                logits = engine.model(input_ids=decoder.static_input_ids, position_ids=decoder.static_position_ids,
                                      past_key_values=decoder.cache, attn_cache_seqlens=decoder.cache_seqlens,
                                      use_cache=True).logits
                decoder.static_logits.copy_(logits)
            return decoder.static_logits

        def snapshot_prefill(**kwargs):
            if turn.query:                  # the query chunk, after eviction, before its prefill
                turn.pre = (decoder.cache_seqlens.clone(),
                            engine.next_start_pos.clone() if engine.next_start_pos is not None else None,
                            torch.cuda.get_rng_state())
            return prefill(**kwargs)

        def answer_generate(**kwargs):
            sample_kwargs = kwargs.get("sample_kwargs") or {}
            turn.query = bool(sample_kwargs.get("is_query_window")) and sample_kwargs.get("allow_deferral") is False
            if not turn.query:
                return generate(**kwargs)
            turn.chunk = {k: kwargs[k] for k in ("input_ids", "position_ids", "pixel_values_videos", "video_grid_thw")
                          if k in kwargs}
            turn.hooks.arm()
            if eager:
                decoder.step = eager_step
            try:
                return generate(**kwargs)
            except RuntimeError as exc:
                if DEADLINE in str(exc):
                    raise
                raise AnswerTurnError(f"ThinkStream answer turn: {type(exc).__name__}: {exc}") from exc
            finally:
                turn.hooks.disarm()
                decoder.step = graph_step
                turn.query = False
                turn.engine, turn.prefill = engine, prefill

        engine.prefill = snapshot_prefill    # engine.generate calls self.prefill: the instance attribute
        engine.generate = answer_generate

    def _fork(self, turn: _AnswerTurn):
        """The deadline readout: after the stream (outside the native generator), the query chunk prefilled again
        from its pre-prefill state with the question followed by an empty thought and <response>; the logits at
        <response>, unsteered (the hooks are disarmed)."""
        torch, tok = self.torch, self.processor.tokenizer
        engine, prefill, x = turn.engine, turn.prefill, turn.chunk
        ids, pos = x["input_ids"], x["position_ids"]
        pv = int((ids[0] == self._vision_end).nonzero()[-1]) + 1
        text = tok.decode(ids[0, pv:].tolist())
        if not text.startswith("\n" + turn.prompt):
            raise AnswerTurnError("the query chunk does not end with the prompt")
        tail = text[len("\n" + turn.prompt):]
        suffix = tok("\n" + turn.prompt + tail, add_special_tokens=False).input_ids + self._nothink
        fork_ids = torch.cat([ids[:, :pv], torch.tensor([suffix], device=ids.device)], dim=1)
        text_pos = pos[:, :, pv:pv + 1] + torch.arange(len(suffix), device=pos.device).view(1, 1, -1)
        fork_pos = torch.cat([pos[:, :, :pv], text_pos], dim=2)
        seqlens, start, rng = turn.pre
        engine.decoder.cache_seqlens.copy_(seqlens)
        engine.next_start_pos = start.clone() if start is not None else None
        torch.cuda.set_rng_state(rng)
        with torch.inference_mode():
            p_ids = engine._process_position_ids(input_ids=fork_ids, position_ids=fork_pos, num_generations=1,
                                                 adjust_position_ids=True)
            logits = prefill(input_ids=fork_ids, num_generations=1, position_ids=p_ids,
                             attention_mask=torch.ones_like(fork_ids), pixel_values_videos=x.get("pixel_values_videos"),
                             video_grid_thw=x.get("video_grid_thw"))
        return (logits[0, -1] if logits.dim() == 3 else logits[0]).float().cpu()

    def stream(self, req: Request, hooks: AnswerTurnHooks | None = None) -> Generation:
        from torchcodec.decoders import VideoDecoder
        torch, tokenizer = self.torch, self.processor.tokenizer
        stream = req.settings.get("stream", {})
        config_window = self._config_window()
        window = window_chunks(stream, config_window)
        engine = self._engine(stream, window)
        reservoir = stream.get("reservoir")
        if reservoir:                           # Recall: the newest `reservoir` chunks + a sample of older ones
            engine.recall = {"recent": int(reservoir), "stream_id": Path(req.video).name,
                             "seed": int(stream.get("memory_seed", 0)), "retained": [], "arriving": 0}
        token = tokenizer.convert_tokens_to_ids
        kwargs = {"think_end_token_id": token("</think>"), "max_think_tokens": MAX_THINK_TOKENS,
                  "eos_token_id": token("<|im_end|>"), "silent_token_id": token("<silent>"),
                  "response_token_id": token("<response>")}
        options = option_tokens(tokenizer, req)
        if options:
            sample = self._restricted_sample
            kwargs["restricted_token_ids"] = list(options.values())
        else:
            sample = _open_answer_sample
        metadata = VideoDecoder(req.video).metadata
        duration = metadata.duration_seconds
        # question timing: asked once at 0 s or at half the duration; the deadline is then the last chunk
        where = question_at(stream)
        if where is not None and hooks is not None:
            raise ValueError("stream.question_at is a diagnostic of the native stream: it takes no plugin hooks")
        ask_t = duration if where is None else 0.0 if where == "start" else duration / 2
        slack = SLACK_TIME if where is None else duration - ask_t + 1.0
        answer_t = answer_chunk = None
        seed = stream_seed(req.id, stream.get("seed"))
        torch.manual_seed(seed)
        turn = None
        if hooks is not None:
            self._attach(hooks)
            turn = _AnswerTurn(hooks, req.prompt)
            self._wrap(engine, turn)
        answer, chunks, failure = None, 0, None
        try:
            for event in self._chat(
                    engine=engine, processor=self.processor, video_path=req.video,
                    queries=[{"content": req.prompt, "timestamp": ask_t}], video_start=0.0, video_end=duration,
                    frames_per_chunk=FRAMES_PER_CHUNK, max_chunks=MAX_CHUNKS, min_pixels=THINKSTREAM_PIXELS[0],
                    max_pixels=THINKSTREAM_PIXELS[1], max_new_tokens=req.settings["generation"]["max_new_tokens"],
                    system_prompt=self._system, chat_template_wo_system=self._template, sample=sample,
                    sample_kwargs=kwargs, model_type="qwen2.5vl", slack_time=slack, break_on_answer=True):
                chunks += 1
                if reservoir:                   # the arrival index of the next generate's chunk
                    engine.recall["arriving"] = chunks
                if event["is_answer"]:
                    answer = self.processor.decode(event["generated_tokens"][0], skip_special_tokens=False)
                    if where is not None:
                        answer_t, answer_chunk = round(float(event["window_end"]), 2), int(event["chunk_idx"])
        except RuntimeError as exc:         # upstream raises when there is no <response> by the deadline
            # nothing streamed at all (e.g. FlashInfer refusing the GPU) is an error, not silence; with hooks, the
            # deadline of a one-chunk clip (raised before upstream's first yield) is the same unanswered turn
            if not chunks and not (turn is not None and turn.deadline_missed(exc)):
                raise
            failure = str(exc)
        info = {"chunks": chunks, "window_chunks": window, "kept_chunks": int(engine._window_count[0]),
                "duration": round(duration, 2), "seed": seed, "restricted": bool(options),
                "raw_turn": (answer or "")[:2000], "no_answer": answer is None}
        if failure:
            info["failure"] = failure
        if reservoir:
            info.update(reservoir=int(reservoir), retained=list(engine.recall["retained"]))
        if where is not None:
            info.update(question_at=where, question_t=round(ask_t, 2), answer_t=answer_t, answer_chunk=answer_chunk,
                        deadline_answer=answer_t is not None and answer_t >= round(duration, 2) - 1e-6)
        try:                                 # the row's schedule digest; never changes the native result
            info["schedule"] = schedule(req, {"torchcodec": {"frames": int(metadata.num_frames),
                                                             "fps": float(metadata.average_fps),
                                                             "duration": float(duration)}},
                                        config_window=config_window)["digest"]
        except ValueError:
            pass
        generation = Generation(answer_text(answer), None, "unknown", info)
        if turn is None:
            return generation
        if info["no_answer"] and turn.pre is not None:
            hooks.deadline_readout(lambda: self._fork(turn))
        return hooks.finish(req, generation)

    def close(self) -> None:
        self.model = None
        self._paths.close()
