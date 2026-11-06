"""MMDuet's native stream (paper condition S for the child of pair p1; reported as S-dagger because of the forced reply).

MMDuet has no official "question at time t" evaluation. The adapter follows its benchmark loop (upstream
test/inference.py, LiveInferForBenchmark): the question is encoded, then the next frame, and the model replies when its
two score heads fire. Everything runs in the official live-inference class of the demo (demo/liveinfer.py,
LiveInferForDemo) with the MAGQA settings:

- frames: the demo's `load_video(output_fps=2)` (OpenCV): one frame every 0.5 s of video time, the long side resized
  to 384 and padded to a black 384x384 square, at most 400 frames; 49 visual tokens per frame. Videos longer than
  200 s (400 frames at 2 fps) are refused, because load_video would silently cut them. `schedule()` reproduces
  load_video's frame selection without decoding (`load_video_indices`).
- memory: every frame stays in the KV cache; replies are not kept in context (remove_assistant_turns).
- question: encoded after all frames but the last (`encode_given_query`), then the last frame, so the question arrives
  with the last frame as for the other children; the dataset's system prompt is not used (MMDuet's own one is).
- answer: if informative + relevance head probabilities exceed 0.5 on that last frame, the model's own reply is the
  answer. Otherwise one reply is generated there anyway with MMDuet's own `_generate_response` (`info.forced`). This
  forcing is our addition (the paper's S-dagger): the heads fire on their own on only a few percent of questions
  (14 of 900 Video-MME questions), so without it MMDuet would almost never answer.
- decoding: MMDuet's fast_greedy_generate (plain argmax, no repetition penalty), up to the benchmark's token budget.
- silence: an empty reply sets `info.no_answer` (scored wrong); with the forcing it practically never happens.
- nondeterminism: none by design (greedy); GPU kernels can still flip near-tied tokens across GPU types.

Plugin hook points (hooks: plugin AnswerTurnHooks; hooks=None is exactly the native path above):
- attach: once, with the decoder layers of the live model (`infer.model`) and MMDuet's tokenizer.
- answer turn: the FIRST `_generate_response` after the question is encoded, whether the heads fired on the last
  frame or the reply is forced. `hooks.arm(first=True)` runs immediately before it (the gate reads the first forward
  after arming: the short generation prompt on the cached stream) and `hooks.disarm()` right after it.
- logits: MMDuet decodes by argmax over the LM head's output, not through HF generate, so the plugin's first-step
  rules act on the LM head: during the armed turn every LM-head forward calls `hooks.on_step_logits(step, logits)`
  (step 0, 1, ...; logits = the head's output, shape [1, positions, vocab]); a returned tensor replaces the output.
- `hooks.finish(req, generation)` makes the final row (refusal or option readout).
"""
from __future__ import annotations

from contextlib import ExitStack
import json
import math
from pathlib import Path

from .. import paths
from ..core.io import atomic_text, fingerprint
from ..core.schema import Generation, Request
from ..media.digest import frame_digest
from ..media.video_meta import video_info
from .base import AnswerTurnHooks, BaseAdapter

ADAPTER = "mmduet_stream"
OUTPUT_FPS = 2              # load_video(output_fps=2) and LiveTestArguments(frame_fps=2)
MAX_FRAMES = 400            # load_video's own cap (max_num_frames = 400 inside upstream) and LiveTestArguments
HEAD_THRESHOLD = 0.5        # MAGQA: informative + relevance probability > 0.5 triggers a reply
SCORE_HEADS = "informative_score,relevance_score"


def load_video_indices(frame_count: float, input_fps: float, decodable: int, output_fps: float = OUTPUT_FPS,
                       max_num_frames: int = MAX_FRAMES) -> list[int]:
    """Numbers of the decoded frames MMDuet's demo `load_video` keeps, by its own loop (demo/liveinfer.py), so the
    selection is known without decoding.

    `frame_count` and `input_fps` are OpenCV's CAP_PROP_FRAME_COUNT and CAP_PROP_FPS; `decodable` is the number of
    frames cap.read() returns. load_video keeps frame k when the accumulated time k * (1 / input_fps) (summed one frame
    at a time, as floats) reaches the next target time i / output_fps, for i < floor(frame_count / input_fps *
    output_fps), and stops at max_num_frames frames. The float accumulation is kept on purpose: it decides which
    frame lands on a target time and so which frames the paper's runs saw."""
    video_duration = frame_count / input_fps
    output_fps = output_fps if output_fps > 0 else max_num_frames / video_duration
    num_frames_total = math.floor(video_duration * output_fps)
    frame_sec = [i / output_fps for i in range(num_frames_total)]
    kept, cur_time, frame_index = [], 0, 0
    for number in range(decodable):
        if frame_index < len(frame_sec) and cur_time >= frame_sec[frame_index]:
            kept.append(number)
            frame_index += 1
        if len(kept) >= max_num_frames:
            break
        cur_time += 1 / input_fps
    return kept


def measure(path: str) -> dict:
    """The video numbers the schedule needs, read the way the native path reads them: decodable frames and decord fps
    (video_info: the 200-s limit) and OpenCV's header frame count and fps (load_video's selection; omitted when OpenCV
    is not installed, then the decord header values stand in for them)."""
    info = video_info(path)
    meta = {"frames": info["frames"], "fps": info["fps"], "header_frames": info["header_frames"]}
    try:
        import cv2
    except ImportError:
        return meta
    capture = cv2.VideoCapture(str(path))
    try:
        meta["opencv"] = {"frames": capture.get(cv2.CAP_PROP_FRAME_COUNT), "fps": capture.get(cv2.CAP_PROP_FPS)}
    finally:
        capture.release()
    return meta


def schedule(req: Request, meta: dict | None = None) -> dict:
    """CPU-only: what MMDuet's stream is fed for this request. `meta` = {"frames" (decodable), "fps" (decord),
    "header_frames", optionally "opencv": {"frames", "fps"}} as in a video_meta manifest row; measured from the file
    when None. Raises ValueError for videos longer than 400 frames at 2 fps, as the native path does."""
    meta = measure(req.video) if meta is None else meta
    duration = meta["frames"] / meta["fps"] if meta["fps"] else 0.0
    if duration * OUTPUT_FPS > MAX_FRAMES:          # load_video would truncate silently
        raise ValueError(f"unsupported: video longer than MMDuet's {MAX_FRAMES} frames at {OUTPUT_FPS} fps")
    opencv = meta.get("opencv") or {"frames": meta["header_frames"], "fps": meta["fps"]}
    indices = load_video_indices(opencv["frames"], opencv["fps"], meta["frames"])
    n = len(indices)
    plan = {"adapter": ADAPTER, "fps": OUTPUT_FPS, "frames": n, "indices": indices,
            "frame_indices": frame_digest({"indices": indices}),
            "question_before_frame": n - 1, "question_time": (n - 1) / OUTPUT_FPS,
            "answer": "heads on the last frame, else one forced reply there"}
    plan["digest"] = fingerprint(plan)
    return plan


def _overlay(base: Path, vision: Path) -> Path:
    """The base checkpoint with a local vision-tower path: symlinks to its files plus a rewritten config.json.
    MMDuet's model builder branches on the substring "llava" in the model path, so the folder name keeps it."""
    overlay = paths.cache_dir() / "overlays" / "llava-mmduet"
    overlay.mkdir(parents=True, exist_ok=True)
    for path in base.iterdir():
        if path.is_file() and path.name != "config.json" and not (overlay / path.name).exists():
            try:
                (overlay / path.name).symlink_to(path)
            except FileExistsError:                    # another worker made it first
                pass
    config = json.loads((base / "config.json").read_text())
    config["mm_vision_tower"] = str(vision)
    atomic_text(overlay / "config.json", json.dumps(config, indent=2))
    return overlay


class _AnswerTurn:
    """Plugin state of one request's answer turn (see the module docstring)."""

    def __init__(self, hooks: AnswerTurnHooks):
        self.hooks, self.started, self.armed, self.step = hooks, False, False, 0


class MMDuetStream(BaseAdapter):
    """Condition S of MMDuet-7B (env llava, bf16, sdpa). load=False builds a schedule-only adapter (no model)."""
    PROTOCOL_VERSION = 1
    LOAD_KEYS: tuple[str, ...] = ()
    concurrency = 1

    def __init__(self, spec, settings: dict, upstream, load: bool = True):
        super().__init__(spec, settings, upstream)
        self._paths = ExitStack()
        self._attached: AnswerTurnHooks | None = None
        self._turn: _AnswerTurn | None = None
        if load:
            self._load()

    def _load(self) -> None:
        import torch
        self.torch = torch
        # MMDuet's code imports its own packages (demo, models, test) and LLaVA-NeXT's lazily, so both stay importable
        # for the adapter's lifetime.
        self._paths.enter_context(self.upstream.use("mmduet", "llava"))
        from demo.liveinfer import LiveInferForDemo, load_video
        from models.arguments_live import LiveTestArguments
        self._load_video = load_video
        vision = paths.checkpoint_dir(self.spec.vision)
        self.infer = LiveInferForDemo(LiveTestArguments(
            llm_pretrained=str(_overlay(paths.checkpoint_dir(self.spec.base), vision)),
            lora_pretrained=str(paths.checkpoint_dir(self.spec.checkpoint)), vision_pretrained=str(vision),
            frame_fps=OUTPUT_FPS, max_num_frames=MAX_FRAMES, attn_implementation="sdpa", bf16=True, fp16=False,
            score_heads=SCORE_HEADS, stream_end_prob_threshold=HEAD_THRESHOLD,     # MAGQA (scripts/inference/magqa.sh)
            remove_assistant_turns=True))

    def schedule(self, req: Request) -> dict:
        return schedule(req)

    def _attach(self, hooks: AnswerTurnHooks) -> None:
        """Once per hooks object: hand the plugin the decoder layers and tokenizer, then install the two call sites
        (answer-turn arming on _generate_response, the LM-head logits hook) once per adapter. Both are inert outside
        an answer turn. A hooks.attach() that raises installs nothing, and the next stream() calls it again."""
        if hooks is self._attached:
            return
        from ..plugin.guard import find_layers
        hooks.attach(find_layers(self.infer.model), self.infer.tokenizer)
        if self._attached is None:
            infer = self.infer
            original = infer._generate_response

            def generate_response():
                turn = self._turn
                if turn is None or turn.started:
                    return original()
                turn.started = True
                turn.hooks.arm(first=True)
                turn.armed = True
                try:
                    return original()
                finally:
                    turn.armed = False
                    turn.hooks.disarm()

            def lm_head(_module, _inputs, output):
                turn = self._turn
                if turn is None or not turn.armed:
                    return None
                step = turn.step
                turn.step += 1
                return turn.hooks.on_step_logits(step, output)

            infer._generate_response = generate_response        # input_one_frame calls it through the instance
            infer.model.get_output_embeddings().register_forward_hook(lm_head)
        self._attached = hooks

    def stream(self, req: Request, hooks: AnswerTurnHooks | None = None) -> Generation:
        plan = schedule(req)                 # also refuses videos over 400 frames, before decoding
        torch, infer = self.torch, self.infer
        images, _ = self._load_video(req.video, output_fps=OUTPUT_FPS)
        budget = req.settings["generation"]["max_new_tokens"]
        if hooks is not None:
            self._attach(hooks)
        infer.reset()
        infer.inplace_output_ids = torch.zeros(1, budget, device="cuda", dtype=torch.long)
        infer.input_video_stream(images)
        events = []
        with torch.inference_mode():
            while len(infer.frame_embeds_queue) > 1:
                event = infer.input_one_frame()
                if event.get("response"):
                    events.append({"t": event["time"], "text": event["response"]})
            infer.encode_given_query(req.prompt)
            self._turn = _AnswerTurn(hooks) if hooks is not None else None
            try:
                last = infer.input_one_frame()
                answer, forced = last.get("response"), last.get("response") is None
                if forced:                   # the heads stayed below the threshold: reply here anyway (S-dagger)
                    answer = infer._generate_response()
            finally:
                self._turn = None
        scores = {k: round(float(v), 4) for k, v in last.items() if k.endswith("_score")}
        info = {"stream_fps": OUTPUT_FPS, "frames": len(images), "replies_before_question": len(events),
                "events": events[-5:], "last_frame_scores": scores, "forced": forced,
                "no_answer": not (answer or "").strip(), "schedule": plan["digest"]}
        if len(images) != plan["frames"]:   # OpenCV decoded another number of frames than the schedule predicts
            info["schedule_frames"] = plan["frames"]
        generation = Generation((answer or "").strip(), None, "unknown", info)
        return hooks.finish(req, generation) if hooks is not None else generation

    def close(self) -> None:
        self.infer = None
        self._paths.close()
