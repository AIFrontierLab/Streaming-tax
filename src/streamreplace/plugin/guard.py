"""The hidden-state gate and parent-restoration steering (PRS) of S+, live only on the answer turn.

gate  proj = float32 output of decoder layer r_layer at the answer position, dotted with the child's unit refusal
      direction rhat (no centering, no normalisation). It is logged rounded to 3 decimals in info.plugin.guard.proj and
      the gate fires iff that rounded value is strictly above the item's threshold (tau_closed for multiple-choice
      and yes/no items, tau_free otherwise); a missing projection never fires. Modes:
        shadow  log, never act (the paper's runs; posthoc.py applies the threshold afterwards)
        live    reply with the fixed refusal (hooks.finish), the same decision as the post-hoc rule on the same
                logged projection; where the hooks see the turn's first-step logits (halt(): every logits path but
                ThinkStream's `deadline`) the turn also ends at its first step, while ThinkStream's runs to its end
        record  log, never fire (threshold calibration); the turn ends at its first step once the state is read,
                except ThinkStream's, which runs to its end
prs   prs (= alpha * mean parent-minus-child state) is cast to the hidden dtype and added to the output of decoder layer
      p_layer while armed: at every single-token (decoding) forward, and at the last position of a multi-token forward
      (the answer-turn prefill) only when p_layer > r_layer. When the PRS layer does not come after the gate's layer
      (ThinkStream 14 <= 20, MOSS-VL-Realtime 19 <= 24) the prefill is left unsteered so the gate reads the child's
      own state; only decoded tokens are steered then.

The answer position: the last position of the latest multi-token forward while armed (the answer-turn prefill), or,
with arm(first=True) (MMDuet), the last position of the first forward after arming whatever its length.
torch and transformers are imported lazily: this module loads in any environment."""
from __future__ import annotations

from typing import Any

import numpy as np

from .recipe import CLOSED_KINDS

# Module names that hold vision / audio towers: their layer lists are never the language model's decoder.
TOWERS = ("visual", "vision", "vit", "audio", "merger", "mlp1", "deepstack")
PROJ_DECIMALS = 3


def find_layers(model):
    """The language model's decoder layer list: the longest ModuleList of at least 8 modules whose class name
    contains "Layer", outside the vision / audio towers."""
    import torch
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) >= 8:
            low = name.lower()
            if any(t in low for t in TOWERS):
                continue
            if "Layer" in type(mod[0]).__name__ and (best is None or len(mod) > len(best[1])):
                best = (name, mod)
    if best is None:
        raise RuntimeError("no decoder layer list found")
    return best[1]


def _hidden(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def decide(proj: float | None, kind: str | None, tau: dict) -> bool:
    """The gate's decision on a logged (3-decimal) projection: strictly above the item format's threshold."""
    if proj is None:
        return False
    return proj > tau["closed" if kind in CLOSED_KINDS else "free"]


class Guard:
    """Record hook at r_layer and, with a vector, the PRS hook at p_layer, on one decoder layer list.

    layers: the decoder ModuleList (find_layers); rhat, prs: float32 vectors (prs None: no steering); mode: shadow,
    live, record or None (PRS only); tau: {"closed", "free"} (the live gate's thresholds)."""

    def __init__(self, layers, rhat: np.ndarray, r_layer: int, p_layer: int, prs: np.ndarray | None = None,
                 mode: str | None = "shadow", tau: dict | None = None):
        import torch
        if mode not in (None, "shadow", "live", "record"):
            raise ValueError(f"unknown gate mode {mode!r}")
        if mode == "live" and not tau:
            raise ValueError("a live gate needs its thresholds")
        if mode == "record" and prs is not None:
            raise ValueError("record mode reads the child's own state: no PRS")
        for name, index in (("r_layer", r_layer), ("p_layer", p_layer if prs is not None else 0)):
            if not 0 <= int(index) < len(layers):      # checked before any hook is registered
                raise ValueError(f"{name} {index} is outside the {len(layers)} decoder layers")
        self.torch = torch
        self.r_layer, self.p_layer = int(r_layer), int(p_layer)
        self.rhat = torch.tensor(np.asarray(rhat), dtype=torch.float32)
        self.prs_vec = torch.tensor(np.asarray(prs)) if prs is not None else None
        self.mode, self.tau = mode, dict(tau or {})
        # The plugin's gate signal is always the post projection, so the prefill rule depends on the layers only.
        self.steer_prefill = not self.p_layer <= self.r_layer
        self.armed, self.state, self.first, self.kind = False, None, False, None
        self.handles = [layers[self.r_layer].register_forward_hook(self._record)]
        if self.prs_vec is not None:
            self.handles.append(layers[self.p_layer].register_forward_hook(self._steer))

    # ---------- hooks ----------

    def _record(self, _module, _inputs, out):
        h = _hidden(out)
        if not self.armed or h.dim() != 3:
            return
        if (self.first and self.state is None) or (not self.first and h.shape[1] > 1):
            self.state = h[0, -1].detach().float().cpu()

    def _steer(self, _module, _inputs, out):
        if not self.armed:
            return out
        h = _hidden(out)
        vec = self.prs_vec.to(h.device, h.dtype)
        if h.shape[1] == 1 and (self.steer_prefill or not (self.first and self.state is None)):
            h = h + vec
        elif h.shape[1] > 1 and self.steer_prefill:
            h = h.clone()
            h[:, -1] += vec
        else:
            return out
        return (h, *out[1:]) if isinstance(out, (tuple, list)) else h

    # ---------- answer turn ----------

    def arm(self, first: bool = False) -> None:
        self.armed, self.state, self.first = True, None, first

    def disarm(self) -> None:
        self.armed = False

    def reset(self, kind: str | None = None) -> None:
        self.state, self.kind = None, kind

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def projection(self) -> float | None:
        return None if self.state is None else float(self.state @ self.rhat)

    def logged(self) -> float | None:
        p = self.projection()
        return None if p is None else round(p, PROJ_DECIMALS)

    def fired(self) -> bool:
        if self.mode != "live":
            return False
        if self.kind is None:
            raise RuntimeError("the live gate needs the request's answer format: call hooks.begin(req) first")
        return decide(self.logged(), self.kind, self.tau)

    def halt(self) -> bool:
        """End the answer turn at its first step: the live gate fired, or a record run has its state (consulted by
        the hooks' first-step rules; ThinkStream's `deadline` path has none, so its turn always runs to its end)."""
        return self.fired() or (self.mode == "record" and self.state is not None)

    def log(self) -> dict:
        out: dict[str, Any] = {"gate": "post", "mode": self.mode, "proj": self.logged(), "tau": self.tau or None,
                               "fired": self.fired(), "prs": self.prs_vec is not None}
        if self.mode == "shadow":
            out["shadow"] = True
        return out


def first_step_processor(guard: Guard | None, ban: list[int], eos: int | list[int], capture: dict,
                         ban_next: dict[int, list[int]] | None = None):
    """A Hugging Face LogitsProcessor for the answer turn's generate (it runs after the built-in processors, e.g. the
    checkpoint's repetition penalty): records the first step's logits (the option readout), ends the turn at once
    when the guard halts, else bans `ban` at the first step and `ban_next[t]` at the second step when the first token
    was t (a two-token silence marker such as "NO REPLY", whose first token also opens real answers)."""
    import torch
    from transformers import LogitsProcessor
    eos_ids = [eos] if isinstance(eos, int) else list(eos)

    class FirstStep(LogitsProcessor):
        def __init__(self):
            self.start = None

        def __call__(self, input_ids, scores):
            if self.start is None:
                self.start = input_ids.shape[1]
            if input_ids.shape[1] == self.start + 1 and ban_next:
                follow = ban_next.get(int(input_ids[0, -1]))
                if follow:
                    scores = scores.clone()
                    scores[:, follow] = float("-inf")
                return scores
            if input_ids.shape[1] != self.start:
                return scores
            capture["logits"] = scores[0].detach().float().cpu()
            if guard is not None and guard.halt():
                stop = torch.full_like(scores, float("-inf"))
                stop[:, eos_ids[0]] = 0.0
                return stop
            if ban:
                scores = scores.clone()
                scores[:, ban] = float("-inf")
            return scores

    return FirstStep()
