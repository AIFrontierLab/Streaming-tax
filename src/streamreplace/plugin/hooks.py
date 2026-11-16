"""AnswerTurnHooks (models/base.py) for the plugin arms: the arm's settings delta (Recall, Respond), the gate, PRS,
silence bans and the option readout, wired into a child's native stream adapter. `hooks=None` is exactly the native
stream.

The calls in the order a run makes them (the run worker calls begin(), the stream adapter all the others; each adapter
documents its arming point, docs/plugin.md):
    begin(req)                   per request, before streaming: resets per-request state and disarms the gate / PRS
                                 hooks; the request's answer format picks the live gate's threshold
    attach(layers, tokenizer)    on the adapter's first stream() with these hooks (so after the first begin()): hooks
                                 registered on the decoder layers, inert until armed. Only a successful call counts: a
                                 call that raises keeps and registers nothing, so the next stream() attaches again
    arm(first=False)             immediately before the answer-turn prefill (MMDuet: first=True)
    logits_processors()          logits path `processor` (HF generate: AURA, MMDuet2, LiveStar): first-step capture,
                                 bans, halt
    on_step_logits(step, logits) logits paths `first_step` (MMDuet's greedy loop) and `opening` (MOSS-VL-Realtime's
                                 session), from the LM head: the same rules
    disarm()                     right after the answer turn
    deadline_readout(fork)       ThinkStream only (logits path `deadline`), after the stream: an unsteered fork
                                 returning answer logits
    finish(req, gen)             fixed refusal (live gate) or option readout; writes info.plugin. It refuses to make
                                 the row of an arm whose gate / PRS or Respond never attached
settings(s) merges the arm's settings delta for callers outside the run planner (the planner merges the same delta into the
job's settings, and the worker never calls it). `eager_decode` (true with PRS) makes ThinkStream decode its answer
chunk eagerly, because forward hooks do not run inside the CUDA graph it otherwise replays.

Each child's answer turn is declared in configs/plugin.yaml `answer_turn` (recipe.AnswerTurn): its silence marker and
its logits path, i.e. which of the calls above its adapter makes. An arm with Respond or a live / record gate refuses
a child without a declaration, and a logits call that does not match the declared path raises, so no child runs such
an arm without its bans, option readout or halt.

Per child (arming point / answer position):
    aura_8b         HF generate of the full prompt (AURA's HF rebuild) / last prompt token
    thinkstream_3b  the query (last) chunk's generate / last token of its prefill; readout from the deadline fork
    mmduet2_3b      the generate after the question joins the final turn / last token of its prefill
    livestar_8b     the answer chat() call / last token of its prefill
    mmduet_7b       the answer turn's first _generate_response, arm(first=True) / last position of its first forward
    moss_vl_rt      the forward after the session drains the question / last position of that forward"""
from __future__ import annotations

from typing import Any, Callable

from ..core.schema import Generation
from . import respond as R
from .recipe import INFO_KEY, REFUSAL, AnswerTurn, ArmRecipe, PluginConfig, load_plugin_config

# How the child's answer turn reaches the hooks' logit rules (its declared AnswerTurn.logits): "processor" = an HF
# LogitsProcessor (logits_processors()); "first_step" = the raw LM head, first step only (MMDuet); "opening" = the raw
# LM head on the opening steps until the first visible token (MOSS-VL-Realtime, whose readout logits are those of the
# step after <|response|>); "deadline" = no logit rule, readout from the unsteered deadline fork (ThinkStream).
LM_HEAD = ("first_step", "opening")             # the paths of on_step_logits()
RESPONSE = "<|response|>"


def _last(logits):
    """The last position's logits of a [vocab], [batch, vocab] or [batch, seq, vocab] tensor (a view)."""
    if logits.dim() == 3:
        return logits[0, -1]
    if logits.dim() == 2:
        return logits[0]
    return logits


def merge(base: dict, delta: dict) -> dict:
    from ..config.settings import merge as merge_settings
    return merge_settings(base, delta)


class PluginHooks:
    """One child, one arm. `artefact` (plugin.artefacts.Artefact) is required when the arm has a gate or PRS; `turn`
    (the child's recipe.AnswerTurn, configs/plugin.yaml answer_turn) when it has Respond or a live / record gate."""

    def __init__(self, child: str, arm: ArmRecipe, artefact=None, delta: dict | None = None,
                 refusal: str = REFUSAL, engine: str | None = None, turn: AnswerTurn | None = None):
        if arm.needs_artefacts and artefact is None:
            raise ValueError(f"{child}@{arm.name}: the arm needs the child's artefacts")
        if arm.needs_turn and turn is None:
            raise ValueError(f"{child}@{arm.name}: configs/plugin.yaml declares no answer_turn for {child} (its "
                             "silence marker and the logits path its stream adapter uses; docs/plugin.md, "
                             "'Calibrating a new child')")
        self.child, self.recipe, self.artefact, self.turn = child, arm, artefact, turn
        self.delta, self.refusal, self.engine = dict(delta or {}), refusal, engine
        self.guard = None
        self.tokenizer = None                    # set last by a successful attach(): the "attached" marker
        self.ban: list[int] = []
        self.ban_next: dict[int, list[int]] | None = None
        self.eos: int | None = None
        self.response_id: int | None = None
        self.path = turn.logits if turn is not None else None     # None: undeclared (no rule needs the logits)
        self.capture: dict = {}
        self.kind: str | None = None             # the current request's answer format (begin)
        self.armed = False
        self._steps = 0
        self._opening = False
        self._fork: Callable[[], Any] | None = None

    @property
    def eager_decode(self) -> bool:
        """True when the hooks act on decode-step forwards (PRS): ThinkStream then decodes its answer chunk eagerly
        instead of replaying a CUDA graph, inside which forward hooks would not run."""
        return bool(self.recipe.prs)

    # ---------- identity ----------

    def identity(self) -> dict:
        """The part of a plugin arm's run identity: recipe, settings delta (Recall, Respond), answer turn, artefact
        hashes and thresholds."""
        return {"child": self.child, "recipe": self.recipe.to_dict(), "settings": self.delta,
                "answer_turn": self.turn.to_dict() if self.turn is not None else None,
                "refusal": self.refusal if self.recipe.gate == "live" else None, "engine": self.engine,
                "artefacts": self.artefact.identity() if self.recipe.needs_artefacts else None}

    @property
    def attached(self) -> bool:
        """attach() has succeeded (decoder hooks registered when the arm has a gate or PRS, bans built)."""
        return self.tokenizer is not None

    # ---------- AnswerTurnHooks ----------

    def settings(self, settings: dict) -> dict:
        """The stream settings with the arm's settings delta merged in (Recall, Respond; unchanged without one)."""
        return merge(settings, self.delta) if self.delta else dict(settings)

    def attach(self, decoder_layers, tokenizer, *, eos: int | None = None) -> None:
        """decoder_layers: the decoder ModuleList (or a model, searched with guard.find_layers); tokenizer: the
        child's; eos: the id that ends the answer turn, default <|im_end|> (every child's end of turn; for MMDuet it is
        also the id its greedy loop stops at, since MMDuet's tokenizer builder makes <|im_end|> the eos token). Only
        the first successful call acts (an adapter calls it on its first stream() with these hooks, after that
        request's begin(), so the guard takes the current request's answer format here). Everything is built before
        anything is kept and the guard registers its forward hooks after its last check, so a call that raises (e.g.
        a decoder depth the artefacts were not built on) leaves the hooks unattached with nothing registered: the
        adapter's next stream() calls attach() again, and every request of the arm fails until one succeeds."""
        from .guard import Guard, find_layers
        if self.attached:
            return
        eos_id = int(eos) if eos is not None else int(tokenizer.convert_tokens_to_ids(R.EOT))
        ban, ban_next = R.silence_bans(self.turn.silence, tokenizer) if self.recipe.respond else ([], None)
        response_id = int(tokenizer.convert_tokens_to_ids(RESPONSE)) if self.path == "opening" else None
        guard = None
        if self.recipe.needs_artefacts:
            import torch
            layers = decoder_layers if isinstance(decoder_layers, torch.nn.ModuleList) else find_layers(decoder_layers)
            art = self.artefact
            if len(layers) != int(art.manifest.get("layers", len(layers))):
                raise ValueError(f"{self.child}: {len(layers)} decoder layers, the artefacts were built on "
                                 f"{art.manifest['layers']}")
            guard = Guard(layers, art.rhat, art.r_layer, art.p_layer, art.prs if self.recipe.prs else None,
                          mode=self.recipe.gate, tau=art.tau() if self.recipe.gate == "live" else None)
            guard.reset(kind=self.kind)
        self.eos, self.ban, self.ban_next, self.response_id, self.guard = eos_id, ban, ban_next, response_id, guard
        self.tokenizer = tokenizer                     # last: the hooks count as attached only now

    def begin(self, req) -> None:
        """Per request, before streaming: the answer format picks the live gate's threshold; per-request state. The
        guard is disarmed too (adapters disarm right after their answer turn; this makes sure that no adapter carries
        an armed guard into the next request, whose answer turn arms it again)."""
        self.kind = getattr(req, "kind", None)
        self.capture.clear()
        self._fork, self.armed, self._steps, self._opening = None, False, 0, False
        if self.guard is not None:
            self.guard.disarm()
            self.guard.reset(kind=self.kind)

    def arm(self, first: bool = False) -> None:
        self.capture.clear()
        self.armed, self._steps, self._opening = True, 0, True
        if self.guard is not None:
            self.guard.arm(first)

    def disarm(self) -> None:
        self.armed = False
        if self.guard is not None:
            self.guard.disarm()

    def logits_processors(self) -> list:
        """The answer turn's logits processor (logits path `processor`). Raises for a child declared on another
        path: its rules would never run where its adapter decodes."""
        if self.path not in (None, "processor"):
            raise RuntimeError(f"{self.child}: the stream adapter asks for logits processors, but configs/plugin.yaml "
                               f"declares answer_turn.logits {self.path!r} (processor: an HF generate answer turn)")
        from transformers import LogitsProcessorList
        from .guard import first_step_processor
        return LogitsProcessorList([first_step_processor(self.guard, self.ban, self.eos, self.capture,
                                                         self.ban_next)])

    def on_step_logits(self, step: int, logits):
        """The LM head's logits of one decoding step of the armed answer turn (`step` counts from 0; the hooks keep
        their own count). Returns the logits to decode from (a modified copy when a rule applies). Raises for a
        child declared on the `processor` or `deadline` path: its rules would never see these logits."""
        if not self.armed:
            return logits
        if self.path not in (None, *LM_HEAD):
            raise RuntimeError(f"{self.child}: the stream adapter hands over LM-head logits, but configs/plugin.yaml "
                               f"declares answer_turn.logits {self.path!r} (first_step or opening for an answer turn "
                               "decoded from the LM head)")
        if self.path == "first_step":
            if self._steps:
                return logits
            self._steps = 1
            self.capture["logits"] = _last(logits).detach().float().cpu()
            if self.guard is not None and self.guard.halt():
                logits = logits.clone()
                last = _last(logits)
                last[:] = float("-inf")
                last[self.eos] = 0.0
            elif self.ban:                                 # a declared one-token marker (MMDuet declares none)
                logits = logits.clone()
                _last(logits)[self.ban] = float("-inf")
            return logits
        if self.path == "opening":
            if not self._opening:
                return logits
            self.capture["logits"] = _last(logits).detach().float().cpu()
            if self.guard is not None and self.guard.halt():
                logits = logits.clone()
                last = _last(logits)
                last[:] = float("-inf")
                last[self.eos] = 0.0
                self._opening = False
                return logits
            if self.ban:
                logits = logits.clone()
                _last(logits)[self.ban] = float("-inf")
            if int(_last(logits).argmax()) != self.response_id:
                self._opening = False          # the first visible token comes now; else the next step is the reading one
            return logits
        return logits

    def deadline_readout(self, fork: Callable[[], Any]) -> None:
        """ThinkStream: `fork()` returns the answer-position logits of the unsteered empty-thought fork of the query
        chunk; it is called only when a closed-choice turn ended without an answer."""
        self._fork = fork

    def finish(self, req, gen: Generation) -> Generation:
        """Live gate fired -> the fixed refusal (finish "guard", no readout); otherwise the option readout for a
        closed-choice reply that is silent or holds no option by the benchmark's rule parser (finish "readout").
        The native reply stays in info.plugin.native; info.plugin.guard holds the gate log, info.plugin.readout the
        option logits when the readout answered. Raises when the arm's gate / PRS hooks or its Respond rules never
        attached (attach() failed or was never called): such a row would look valid without them."""
        if self.recipe.needs_artefacts and self.guard is None:
            raise RuntimeError(f"{self.child}@{self.recipe.name}: no gate / PRS hooks on the decoder (attach() never "
                               "succeeded): the row would lack the arm's guard")
        if self.recipe.respond and not self.attached:
            raise RuntimeError(f"{self.child}@{self.recipe.name}: Respond never attached (attach() never succeeded)")
        info = dict(gen.info or {})
        log: dict[str, Any] = {"native": gen.text}
        text, finish = gen.text, gen.finish
        if self.guard is not None:
            log["guard"] = self.guard.log()
            if log["guard"]["fired"]:
                info["no_answer"] = False
                info[INFO_KEY] = log
                return Generation(self.refusal, gen.tokens, "guard", info)
        opts = R.options(req)
        if opts and self.recipe.respond and self.tokenizer is not None:
            ids = R.option_ids(self.tokenizer, opts)
            logits = self.capture.get("logits")
            if logits is None and self._fork is not None and info.get("no_answer"):
                logits = self._fork().float().cpu()
            if logits is not None and ids and (info.get("no_answer") or not R.answered(req, text or "", list(ids))):
                choice, scores = R.read(logits, ids)
                text, finish = choice, "readout"
                info["no_answer"] = False
                log["readout"] = scores
        info[INFO_KEY] = log
        self.capture.clear()
        self._fork = None                          # never reused by the next request
        return Generation(text, gen.tokens, finish, info)


# ---------- factory (what the run worker calls) ----------

def make_hooks(child: str, variant: str, config: PluginConfig | None = None, artefacts_root=None) -> PluginHooks:
    """The hooks of the plugin arm `<child>@<variant>` (raises when the arm needs the child's artefacts or its
    answer_turn declaration and the child has none)."""
    from . import artefacts
    config = config or load_plugin_config()
    arm = config.arm(variant)
    art = artefacts.load(child, artefacts_root) if arm.needs_artefacts else None
    return PluginHooks(child, arm, art, config.settings_delta(child, variant), config.refusal, config.engines.get(child),
                       config.answer_turns.get(child))


def identity_block(child: str, variant: str, config: PluginConfig | None = None, artefacts_root=None) -> dict:
    """The `plugin` block of a plugin arm's run identity (no torch needed)."""
    return make_hooks(child, variant, config, artefacts_root).identity()


def stream_adapter(child: str, default: str, config: PluginConfig | None = None) -> str:
    """The stream adapter a plugin arm runs on: the configured engine (AURA: aura_stream_hf) or the model's own."""
    config = config or load_plugin_config()
    return config.engines.get(child, default)
