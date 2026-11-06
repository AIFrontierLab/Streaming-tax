"""MOSS-VL-Realtime's native stream (paper condition S for MOSS-VL-Realtime, pair p5).

MOSS-VL-Realtime has no official benchmark script. The adapter drives the official realtime session of the
checkpoint's remote code (create_realtime_session) the way its realtime_inference README asks (playback speed 1):

- frames: the framework's fixed-step frames at `stream.fps` (1 fps; the same indices as the p5 offline frames), close
  to native resolution. All frames but the last are pushed on the wall clock at their media timestamps
  (time.monotonic(), drop_oldest False, a 256-frame queue). The adapter then waits until the session has consumed
  every frame and the wall clock has reached the video's duration (timeout max(120 s, duration)).
- question: pushed together with the last frame (`push_prompt_frame`). The session's default system prompt is used;
  the dataset's is not.
- decoding: the session's greedy decoding, paced at 12 tokens per second (max_tokens_per_turn 12), with a total budget
  of max(4096, 12 * frames + the benchmark's budget) tokens.
- answer: the output after the question's <|round_start|>, read by the official client's display rule
  (`readouts.read_reply`): control tokens hidden, the first visible text up to <|silence|> / <|im_end|>, <|response|>
  not required. Reading stops when the reply is complete, after 3 s without output, after 180 s, or when the chunks
  with visible text reach the benchmark's budget (`finish: length`). No visible text sets `info.no_answer` (wrong).
- nondeterminism: the session runs on the wall clock, so how much the model generates between frames, and whether the
  question is taken in the same cycle as its frame (`info.prompt_with_frame`; push_prompt_frame is not atomic),
  depend on GPU speed. Reruns are not bit-identical.

Plugin hook points (hooks=None is exactly the native path above). MOSS-VL-Realtime's plugin arms ship but are not
part of the paper's reported S+.
- attach: once, with the decoder layers and the tokenizer.
- answer turn: starts at the first update of the session's model inputs (`_update_model_kwargs_for_real_time_
  generation`) whose <|im_start|> count rises during the request, i.e. when the question is drained, whatever cycle
  the prompt arrives in. `hooks.arm()` runs right after that update (the gate reads the next multi-token forward: the
  question's prefill) and `hooks.disarm()` when the stream ends, unconditionally; a drain racing the end of the
  request (answer timeout, close) cannot arm the hooks after that disarm.
- logits: the session decodes by argmax over the LM head inside its own thread, so every LM-head forward after arming
  calls `hooks.on_step_logits(step, logits)` (step 0, 1, ...; logits = the head's output, shape [1, positions,
  vocab]); a returned tensor replaces the output (the plugin's opening-step bans and readout act there).
- with hooks, a stream whose session never took the question raises.
- `hooks.finish(req, generation)` makes the final row.
"""
from __future__ import annotations

import threading
import time

from .. import paths
from ..core.io import fingerprint
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.frames import frame_indices, sample_frames
from ..media.video_meta import video_info
from .base import AnswerTurnHooks, BaseAdapter
from .readouts import has_visible_text, read_reply

ADAPTER = "mossrt_stream"
FRAME_QUEUE = 256
TOKENS_PER_TURN = 12                 # the session's pace: at most 12 tokens per (1 s) turn
MIN_TOTAL_TOKENS = 4096
ANSWER_TIMEOUT = 180.0               # seconds to wait for the reply after the question
IDLE_STOP = 3.0                      # stop reading after this long without output, once the reply has started
CLOSE_TIMEOUT = 30


def _plan(indices: list[int], timestamps: list[float], duration: float, stream_fps: float) -> dict:
    plan = {"adapter": ADAPTER, "stream_fps": stream_fps, "frames": len(indices), "indices": indices,
            "frame_indices": frame_digest({"indices": indices}), "timestamps": timestamps,
            "question_with_frame": len(indices) - 1, "question_time": timestamps[-1],
            "question_not_before": duration}
    plan["digest"] = fingerprint(plan)
    return plan


def schedule(req: Request, meta: dict | None = None) -> dict:
    """CPU-only: the frames and their push times. `meta` = {"frames" (decodable), "fps" (decord)} as in a video_meta
    manifest row; measured from the file when None."""
    meta = video_info(req.video) if meta is None else meta
    fps = float(req.settings["stream"]["fps"])
    indices = frame_indices({"frames": meta["frames"], "fps": meta["fps"]}, fps=fps)
    return _plan(indices, [i / meta["fps"] for i in indices], meta["frames"] / meta["fps"], fps)


class _AnswerTurn:
    """Plugin state of one request: waiting for the question to drain, then armed until the stream ends. The session
    thread arms and the main thread ends the turn, both under `lock`, so a drain racing the end of the request cannot
    arm the hooks after their final disarm()."""

    def __init__(self, hooks: AnswerTurnHooks):
        self.hooks, self.waiting, self.asked, self.armed, self.step = hooks, True, False, False, 0
        self.lock = threading.Lock()

    def arm(self) -> None:
        """Session thread, on the update that drains the question."""
        with self.lock:
            if self.waiting:
                self.waiting, self.asked = False, True
                self.hooks.arm()
                self.armed = True

    def end(self) -> None:
        """Main thread, when the stream ends: no later arming, and the hooks are disarmed whatever happened (as the
        paper's plugin code did unconditionally)."""
        with self.lock:
            self.waiting = False
            self.hooks.disarm()
            self.armed = False


class MossRTStream(BaseAdapter):
    """Condition S of MOSS-VL-Realtime (env modern; remote code from the checkpoint, bf16, flash-attn 2 when
    installed). load=False builds a schedule-only adapter (no model)."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    concurrency = 1

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        self._attached: AnswerTurnHooks | None = None
        self._turn: _AnswerTurn | None = None
        if load:
            self._load()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.torch = torch
        try:
            import flash_attn  # noqa: F401
            attention = "flash_attention_2"
        except ImportError:
            attention = "sdpa"
        checkpoint = str(paths.checkpoint_dir(self.spec.checkpoint))
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True,
                                                          torch_dtype=torch.bfloat16, device_map="cuda:0",
                                                          attn_implementation=attention).eval()

    def schedule(self, req: Request) -> dict:
        return schedule(req)

    def _attach(self, hooks: AnswerTurnHooks) -> None:
        """Once per hooks object: hand over layers and tokenizer, then install the two call sites (question-drain
        arming on the session's input update, the LM-head logits hook) once per adapter; both are inert outside a
        request. A hooks.attach() that raises installs nothing, and the next stream() calls it again."""
        if hooks is self._attached:
            return
        from ..plugin.guard import find_layers
        hooks.attach(find_layers(self.model), self.processor.tokenizer)
        if self._attached is None:
            model = self.model
            im_start = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
            update = model._update_model_kwargs_for_real_time_generation

            def drained(*args, **kwargs):
                turn = self._turn
                # a one-frame video's question comes at the first update
                before = int((kwargs["input_ids"] == im_start).sum())
                out = update(*args, **kwargs)
                if turn is not None and turn.waiting and int((out[0] == im_start).sum()) > before:
                    turn.arm()
                return out

            def lm_head(_module, _inputs, output):
                turn = self._turn
                if turn is None or not turn.armed:
                    return None
                step = turn.step
                turn.step += 1
                return turn.hooks.on_step_logits(step, output)

            model._update_model_kwargs_for_real_time_generation = drained
            model.lm_head.register_forward_hook(lm_head)
        self._attached = hooks

    def stream(self, req: Request, hooks: AnswerTurnHooks | None = None) -> Generation:
        s = req.settings
        budget = s["generation"]["max_new_tokens"]
        frames, meta = sample_frames(req.video, fps=s["stream"]["fps"])
        if hooks is not None:
            self._attach(hooks)
            self._turn = _AnswerTurn(hooks)
        session = self.model.create_realtime_session(
            self.processor, initial_prompt="", frame_queue_size=FRAME_QUEUE, max_tokens_per_turn=TOKENS_PER_TURN,
            max_new_tokens=max(MIN_TOTAL_TOKENS, len(frames) * TOKENS_PER_TURN + budget), do_sample=False)
        events, start = [], time.monotonic()

        def drain(timeout=0.0):
            value = session.poll_output(timeout)
            if value is not None:
                events.append(value)
            return value

        in_answer, reply, texts = False, "", 0
        try:
            session.start()
            for frame, stamp in zip(frames[:-1], meta["timestamps"][:-1]):
                while time.monotonic() - start < stamp:
                    drain(max(0.0, min(0.1, stamp - (time.monotonic() - start))))
                session.push_frame(frame, timestamp=stamp, drop_oldest=False)
                drain()
            deadline = time.monotonic() + max(120, meta["duration"])
            while session.pending_frames or time.monotonic() - start < meta["duration"]:
                drain(0.1)
                if time.monotonic() > deadline:
                    raise TimeoutError("MOSS did not consume all frames")
            while drain() is not None:
                pass
            session.push_prompt_frame(req.prompt, frames[-1], timestamp=meta["timestamps"][-1], drop_oldest=False)
            # the reply is usually <|silence|><|response|>text... and ends at the next <|silence|> (the model then waits
            # for input); see readouts.read_reply. Output chunks with visible text count against the budget.
            asked, last = time.monotonic(), time.monotonic()
            while time.monotonic() - asked < ANSWER_TIMEOUT and session.active:
                value = drain(0.1)
                if value is None:
                    if in_answer and time.monotonic() - last > IDLE_STOP:
                        break
                    continue
                last = time.monotonic()
                if not in_answer:
                    if "<|round_start|>" not in value:
                        continue
                    in_answer, value = True, value.split("<|round_start|>", 1)[1]
                reply += value
                texts += has_visible_text(value)
                if read_reply(reply)[1]["complete"] or texts >= budget:
                    break
        finally:
            session.close(CLOSE_TIMEOUT)
            if hooks is not None:
                turn, self._turn = self._turn, None
                turn.end()
        if hooks is not None and not turn.asked:
            raise RuntimeError("the session never took the question")
        text, parsed = read_reply(reply)
        info = {"stream_fps": s["stream"]["fps"], "frames": len(frames), "events": len(events), "no_answer": not text,
                "response_tag": parsed["response_tag"], "prompt_with_frame": parsed["prompt_with_frame"],
                "tail": "".join(events[-12:])[-400:],
                "schedule": _plan(meta["indices"], meta["timestamps"], meta["duration"],
                                  float(s["stream"]["fps"]))["digest"]}
        generation = Generation(text, texts, "length" if texts >= budget else "unknown", info)
        return hooks.finish(req, generation) if hooks is not None else generation

    def close(self) -> None:
        self.model = None
