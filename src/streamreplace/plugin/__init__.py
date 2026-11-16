"""Stream++: a training-free plugin applied to a streaming child's memory and answer turn (Sec. 6 of the paper).

  Recall   the memory keeps earlier frames within the child's native budget: once the budget is full, the newest
           k chunks plus a uniform sample of earlier ones (recall.py)
  Respond  the answer is delivered: the child's answer turn is requested, its silence marker may not open the
           answer, and a closed-choice reply without a valid option is read from the option logits at the answer
           start (respond.py)
  Refuse   parent-restoration steering, PRS (the mean parent-minus-child hidden state added on the answer turn), and
           a refusal gate (a fixed refusal when the answer-position state projects past a calibrated threshold on
           the child's harm direction) (guard.py)
  S_fix = Recall + Respond;  S_plug = S_fix + Refuse.

Modules:
  recipe.py     configs/plugin.yaml: arms (sfix, sfix_prs, record, shadow, prs, inert, hf), threshold rule, Recall and
                Respond settings, each child's answer turn (silence marker, logits path), item sets
  artefacts.py  plugin_artefacts/<child>/{vectors.npz, manifest.json}: load, verify, write
  guard.py      decoder-layer hooks (gate state, PRS), the first-step logits processor
  recall.py     the budgeted memory rule: reservoir, AURA's segment spans, the eviction index of a KV engine
  respond.py    silence bans, option readout, the benchmark-parser check
  hooks.py      AnswerTurnHooks implementation used by the stream adapters (models/*_stream.py)
  posthoc.py    the paper's S_plug: shadow arms + the threshold on the logged projection; the fixed-refusal job
  report.py     the recovery tables (S_plug against S, P and S_fix; gate and PRS contributions; fire rate), from
                shadow arms only
  calibrate/    calibrating a new child: inputs, directions (rhat, PRS), thresholds, harmless requests, freeze

Rows of a plugin arm carry info.plugin = {native: the child's own reply, guard: {proj (3 decimals), fired, ...},
readout: {option: logit} when the option readout answered} (hooks.finish; core/schema.py).

Integration with the run planner and worker (run/plan.py, run/worker.py; no torch needed until attach()):
  planner  cfg = load_plugin_config(section=config.section("plugin")) for the arm <child>@<variant>:
             make_hooks(child, variant, cfg).identity()   the `plugin` block of the run identity (recipe, settings
                                                          delta, answer turn, artefact sha256 and thresholds); it
                                                          refuses an arm whose child lacks the artefacts or the
                                                          answer_turn declaration the arm needs
             stream_adapter(child, default, cfg)          the adapter (AURA's plugin arms run on aura_stream_hf)
             cfg.settings_delta(child, variant)           the Recall / Respond delta, merged as the layer right after
                                                          the model's own settings (so it is in the identity and
                                                          --set still wins)
  worker   rebuilds hooks = make_hooks(child, variant, load_plugin_config(section=<the plan's section>)), refuses to
           run when hooks.identity() differs from the planned block, never calls hooks.settings() (the job's
           settings already hold the delta), and runs the arm's requests one at a time:
             hooks.begin(req); gen = adapter.stream(req, hooks=hooks)
           The adapter calls attach() on its first stream() with these hooks (again on the next one if it raised),
           then arm / logits rules / disarm around the answer turn and finish() last (hooks.py lists the calls)."""
from __future__ import annotations

from .hooks import PluginHooks, identity_block, make_hooks, stream_adapter
from .recipe import CLOSED_KINDS, INFO_KEY, REFUSAL, VIDEO_PREFIX, load_plugin_config

__all__ = ["CLOSED_KINDS", "INFO_KEY", "REFUSAL", "VIDEO_PREFIX", "PluginHooks", "identity_block", "load_plugin_config",
           "make_hooks", "stream_adapter"]
