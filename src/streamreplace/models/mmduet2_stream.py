"""MMDuet2's native stream (paper condition S for MMDuet2, pair p2).

MMDuet2 has no official "question at time t" evaluation. The adapter drives its official in-process proactive client
(demo/api_server.py ProactiveInferenceAPIClient, loaded by patches/mmduet2_client.py) the way its paper describes
inference (one second of video per frame):

- frames: the framework's fixed-step frames at `stream.fps` (1 fps: frame index int(i * fps_source / 1), the same
  rule as the offline frames of pairs p4/p5), each downscaled by the demo's resize_frame (longest side at most 640,
  cv2.INTER_AREA) and then by the client's qwen-vl-utils smart_resize (28-pixel grid, 3,136-12,845,056 pixels).
- turns: two frames per user turn (the last turn may hold one); the client generates a reply after every turn, with
  MMDuet2's own proactive system prompt. The full KV cache is reused across turns and every reply, including
  "NO REPLY", stays in the history. Model: MMDuet2's Qwen2.5-VL variant (drop_method 'none', bf16, flash-attn 2).
- question: added as text to the final turn, after its images (as in MMDuet2's training data and demo API).
- decoding: greedy with the checkpoint's repetition_penalty (1.05), up to the benchmark's token budget (the client's
  hard-coded 512 is patched to the budget).
- answer: the reply to the final turn. Silence = an empty reply or exactly "NO REPLY" (`readouts.is_no_reply`), which
  sets `info.no_answer` and is scored wrong. There is no forcing: forced variants were tried during the study and gave
  answers near chance, biased to "A" / "No", so silence stays silence.
- nondeterminism: none by design (greedy).

Plugin hook points (hooks=None is exactly the native path above):
- attach: once, with the decoder layers of the client's model and its tokenizer.
- answer turn: the client's generate call for the final turn (the one that carries the question). `hooks.arm()` runs
  immediately before it (the gate reads the last position of its prefill: the final frames, the question and the
  assistant opener) and `hooks.disarm()` right after; that call gets `logits_processor=hooks.logits_processors()`
  (first-step capture, the silence bans). Every earlier turn runs untouched.
- `hooks.finish(req, generation)` makes the final row.
"""
from __future__ import annotations

from contextlib import ExitStack

from .. import paths
from ..core.io import fingerprint
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.frames import frame_indices, sample_frames
from ..media.video_meta import video_info
from .base import AnswerTurnHooks, BaseAdapter
from .readouts import is_no_reply

ADAPTER = "mmduet2_stream"
FRAMES_PER_TURN = 2
MAX_FRAME_SIZE = 640        # demo/frontend.py resize_frame(max_size=640)


def turns_of(n: int) -> list[list[int]]:
    """Positions of the n stream frames in each user turn: pairs, the last turn possibly a single frame."""
    return [list(range(i, min(i + FRAMES_PER_TURN, n))) for i in range(0, n, FRAMES_PER_TURN)]


def _plan(indices: list[int], source_fps: float, duration: float, stream_fps: float) -> dict:
    turns = turns_of(len(indices))
    plan = {"adapter": ADAPTER, "stream_fps": stream_fps, "frames": len(indices), "indices": indices,
            "frame_indices": frame_digest({"indices": indices}), "turns": len(turns),
            "turn_frames": [len(t) for t in turns], "question_turn": len(turns) - 1,
            "question_time": indices[-1] / source_fps, "duration": duration}
    plan["digest"] = fingerprint(plan)
    return plan


def schedule(req: Request, meta: dict | None = None) -> dict:
    """CPU-only: MMDuet2's frames and turns for this request. `meta` = {"frames" (decodable), "fps" (decord)} as in a
    video_meta manifest row; measured from the file when None."""
    meta = video_info(req.video) if meta is None else meta
    fps = float(req.settings["stream"]["fps"])
    indices = frame_indices({"frames": meta["frames"], "fps": meta["fps"]}, fps=fps)
    return _plan(indices, meta["fps"], meta["frames"] / meta["fps"], fps)


class MMDuet2Stream(BaseAdapter):
    """Condition S of MMDuet2 (env mmduet2: transformers 4.49, qwen-vl-utils 0.0.8, flash-attn 2.6.3).
    load=False builds a schedule-only adapter (no model)."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    concurrency = 1

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        self._paths = ExitStack()
        self._attached: AnswerTurnHooks | None = None
        self._answer: AnswerTurnHooks | None = None     # set while the final turn is being added
        if load:
            self._load()

    def _load(self) -> None:
        """The official client (in-process, without its Flask routes), greedy, no assistant turns in the input."""
        from .patches.mmduet2_client import load_client_module, load_resize_frame
        # the client's module imports demo/inference.py and the proactive_eval model package lazily
        self._paths.enter_context(self.upstream.use("mmduet2/demo", "mmduet2/proactive_eval"))
        api = load_client_module(self.upstream)
        self.resize_frame = load_resize_frame(self.upstream)
        output_dir = paths.cache_dir() / "mmduet2"     # TrainingArguments needs one; the client writes nothing there
        output_dir.mkdir(parents=True, exist_ok=True)
        self.client = api.ProactiveInferenceAPIClient(api.ProactiveTestAPIArguments(
            output_dir=str(output_dir), llm_pretrained=str(paths.checkpoint_dir(self.spec.checkpoint)),
            attn_implementation="flash_attention_2", do_sample=False, input_assistant_turns=False))

    def schedule(self, req: Request) -> dict:
        return schedule(req)

    def _attach(self, hooks: AnswerTurnHooks) -> None:
        """Once per hooks object: hand over layers and tokenizer, then install the answer-turn call site on the
        client's generate (a pass-through unless the final turn is being added) once per adapter. A hooks.attach()
        that raises installs nothing, and the next stream() calls it again."""
        if hooks is self._attached:
            return
        from ..plugin.guard import find_layers
        hooks.attach(find_layers(self.client.model), self.client.processor.tokenizer)
        if self._attached is None:
            client = self.client
            generate = client.model.generate

            def turn(*args, **kwargs):
                answer = self._answer
                if answer is None:
                    return generate(*args, **kwargs)
                self._answer = None                     # only the final turn's one generate call
                from transformers import LogitsProcessorList
                kwargs = dict(kwargs, logits_processor=LogitsProcessorList(answer.logits_processors()))
                answer.arm()
                try:
                    return generate(*args, **kwargs)
                finally:
                    answer.disarm()

            client.model.generate = turn
        self._attached = hooks

    def stream(self, req: Request, hooks: AnswerTurnHooks | None = None) -> Generation:
        import numpy as np
        from PIL import Image
        client = self.client
        fps = float(req.settings["stream"]["fps"])
        frames, meta = sample_frames(req.video, fps=fps)
        if hooks is not None:
            self._attach(hooks)
        client.reset()
        client.set_fps(fps=fps)
        client.max_new_tokens = req.settings["generation"]["max_new_tokens"]
        try:
            for i in range(0, len(frames), FRAMES_PER_TURN):
                chunk = frames[i:i + FRAMES_PER_TURN]
                client.num_frames_per_turn = len(chunk)
                if i + len(chunk) == len(frames):           # the final turn: its images, then the question
                    self._answer = hooks
                    client.add_text(req.prompt)
                for frame in chunk:
                    client.add_image(Image.fromarray(self.resize_frame(np.asarray(frame), max_size=MAX_FRAME_SIZE)))
        finally:
            self._answer = None
        replies = [str(t["content"]).strip() for t in client.history if t.get("role") == "assistant"]
        final = replies[-1] if replies else ""
        info = {"stream_fps": fps, "frames": len(frames), "duration": round(meta["duration"], 2),
                "turns": len(replies), "replies_before_question": sum(not is_no_reply(r) for r in replies[:-1]),
                "no_answer": is_no_reply(final),
                "schedule": _plan(meta["indices"], meta["fps"], meta["duration"], fps)["digest"]}
        generation = Generation(final, None, "unknown", info)
        return hooks.finish(req, generation) if hooks is not None else generation

    def close(self) -> None:
        self.client = None
        self._paths.close()
