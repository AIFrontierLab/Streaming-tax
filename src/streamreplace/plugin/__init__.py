"""The streaming plugin: a training-free fix applied to a streaming child's native answer turn (Sec. 6 of the paper).

  S_fix = Recall + Respond: the stream keeps its whole history (recall.py) and the child's own answer turn is
          committed: its silence marker may not open the answer, and a closed-choice reply without a valid option is
          read from the option logits at the answer start (respond.py).
  S+    = S_fix + a hidden-state gate (a fixed refusal when the answer-position state projects far enough onto the
          child's refusal direction) + parent-restoration steering, PRS (the mean parent-minus-child state added on
          the answer turn) (guard.py).

Modules:
  recipe.py     configs/plugin.yaml: arms (sfix, sfix_prs, record, inert, hf), threshold rule, Recall, each child's
                answer turn (silence marker, logits path), item sets
  artefacts.py  plugin_artefacts/<child>/{vectors.npz, manifest.json}: load, verify, write
  guard.py      decoder-layer hooks (gate state, PRS), the first-step logits processor
  recall.py     full-history settings, AURA's segment spans, the reservoir past a budget
  respond.py    silence bans, option readout, the benchmark-parser check
  hooks.py      AnswerTurnHooks implementation used by the stream adapters (models/*_stream.py)
  posthoc.py    the paper's S+: shadow arms + the threshold on the logged projection; the fixed-refusal job
  report.py     the recovery tables (loss, effect, recovered share, NI at -2, gate / PRS contributions, fire rate),
                from shadow arms only
  calibrate/    calibrating a new child: inputs, directions (rhat, PRS), thresholds, harmless requests, freeze

Rows of a plugin arm carry info.plugin = {native: the child's own reply, guard: {proj (3 decimals), fired, ...},
readout: {option: logit} when the option readout answered} (hooks.finish; core/schema.py).

Integration with the run planner and worker (run/plan.py, run/worker.py; no torch needed until attach()):
  planner  cfg = load_plugin_config(section=config.section("plugin")) for the arm <child>@<variant>:
             make_hooks(child, variant, cfg).identity()   the `plugin` block of the run identity (recipe, Recall
                                                          delta, answer turn, artefact sha256 and thresholds); it
                                                          refuses an arm whose child lacks the artefacts or the
                                                          answer_turn declaration the arm needs
             stream_adapter(child, default, cfg)          the adapter (AURA's plugin arms run on aura_stream_hf)
             cfg.recall_delta(child, variant)             the Recall delta, merged as the settings layer right after
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
