"""Stage `directions`: the refusal direction (gate) and the parent-restoration vector (PRS) of a child.

Captures (GPU; the child's and the parent's OFFLINE adapters with the Video-SafetyBench offline settings: 16 uniform
frames, the pair's pixel budget; greedy, max_new_tokens = 1): every decoder layer's output at the last prompt position
of the first forward (the answer position) and the next-token log-probabilities there; states are cached as float16.

  refusal   r_l = mean h_l(harmful) - mean h_l(harmless) over the fold-0 TEXTS, child states; rhat_l = r_l / |r_l|;
            r_layer = the first argmax over int(0.3 L) <= l < int(0.6 L) of the fold-1 text AUC of h_l . rhat_l
  PRS       p_l = mean over all fold-0 inputs (texts and videos) of h_l(parent) - h_l(child); candidates: the layers
            snap(int(round(x (L - 1)))) for x in linspace(0.2, 0.8, 7) x alpha in (0.5, 1.0, 1.5); alpha p_l is added
            to layer l's output at the last prompt position; objective = mean over the fold-1 inputs of the
            full-vocabulary KL(parent || steered child) of the next token; the minimum wins, ties to the first
            candidate (layer ascending, then alpha)
Layers a model skips on some inputs (MOSS-VL's cross-attention layers on text-only inputs) copy the previous layer's
state and are never chosen. Parent and child must share the number of layers, the hidden size and the vocabulary.

The numpy part (everything after the captures) is a pure function of the cached states (`select`)."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

ALPHAS = (0.5, 1.0, 1.5)
DEPTH = (0.3, 0.6)
PRS_DEPTHS = (0.2, 0.8, 7)


# ---------- the numpy part ----------

def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos), np.asarray(neg)
    return float(((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()))


def refusal_direction(h0: np.ndarray, labels0) -> tuple[np.ndarray, np.ndarray]:
    """(r, rhat), each [L, d], from fold-0 states [n, L, d] and their labels (harmful / harmless / None)."""
    labels0 = np.asarray(labels0, dtype=object)
    r = h0[labels0 == "harmful"].mean(0) - h0[labels0 == "harmless"].mean(0)
    return r, r / np.linalg.norm(r, axis=1, keepdims=True)


def layer_aucs(h1: np.ndarray, labels1, rhat: np.ndarray) -> list[float]:
    labels1 = np.asarray(labels1, dtype=object)
    proj1 = np.einsum("nld,ld->nl", h1, rhat)
    return [auc(proj1[labels1 == "harmful", l], proj1[labels1 == "harmless", l]) for l in range(rhat.shape[0])]


def pick_r_layer(aucs: list[float], runnable: list[int], depth=DEPTH) -> int:
    L = len(aucs)
    return max((l for l in runnable if int(depth[0] * L) <= l < int(depth[1] * L)), key=lambda l: aucs[l])


def candidate_layers(L: int, runnable: list[int], depths=PRS_DEPTHS) -> list[int]:
    snap = lambda l: min(runnable, key=lambda k: (abs(k - l), k))       # noqa: E731 - a skipped layer -> nearest that runs
    return sorted(set(snap(int(round(x * (L - 1)))) for x in np.linspace(*depths)))


def kl_to_parent(lpp: np.ndarray, lpc: np.ndarray) -> float:
    """KL(parent || child) of one next-token distribution from log-probabilities (parent's from the float16 cache)."""
    lpp = lpp.astype(np.float32)
    return float((np.exp(lpp) * (lpp - lpc)).sum())


def pick_prs(kl: dict) -> tuple[int, float]:
    """(p_layer, alpha) of the smallest mean KL over keys "L{layer}_a{alpha}" (insertion order breaks ties)."""
    best = min((k for k in kl if k != "base"), key=kl.get)
    return int(best.split("_")[0][1:]), float(best.split("_a")[1])


# ---------- captures (torch) ----------

def skip_layers(model) -> set:
    cfg = getattr(model, "config", None)
    cfg = getattr(cfg, "text_config", None) or cfg
    return set(getattr(cfg, "cross_attention_layers", None) or ())


class Probe:
    """Hooks recording, for the first forward of a request, every decoder layer's last-position output and the LM
    head's last-position logits; and one steering hook (layer, vector) switched per request."""

    def __init__(self, model, head):
        import torch
        from ..guard import find_layers
        self.torch = torch
        self.layers = find_layers(model)
        self.skip = skip_layers(model)
        self.store, self.logits, self.record = {}, None, False
        self.steer = {"on": False, "layer": None, "vec": None}
        self.handles = [layer.register_forward_hook(self._layer_hook(i)) for i, layer in enumerate(self.layers)]
        self.handles.append(head.register_forward_hook(self._head_hook))

    def _layer_hook(self, i):
        def fn(_m, _inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            st = self.steer
            if st["on"] and st["layer"] == i:
                h = h.clone()
                vec = st["vec"].to(h.device, h.dtype)
                if h.shape[1] == 1:
                    h = h + vec
                else:
                    h[:, -1] += vec
                out = (h, *out[1:]) if isinstance(out, (tuple, list)) else h
            if self.record and i not in self.store and h.dim() == 3:
                self.store[i] = h[0, -1].float().cpu()
            return out
        return fn

    def _head_hook(self, _m, _inp, out):
        if self.record and self.logits is None:
            self.logits = out[0, -1].float().cpu()

    def capture(self, generate, req) -> tuple[np.ndarray, np.ndarray]:
        """[L, d] last-position states and [V] next-token log-probs of one request (one generated token)."""
        req = copy.copy(req)
        req.settings = copy.deepcopy(req.settings)
        req.settings["generation"]["max_new_tokens"] = 1
        self.store, self.logits, self.record = {}, None, True
        try:
            with self.torch.inference_mode():
                generate(req)
        finally:
            self.record = False
        for i in range(1, len(self.layers)):
            if i not in self.store and i in self.skip:
                self.store[i] = self.store[i - 1]
        h = self.torch.stack([self.store[i] for i in range(len(self.layers))]).numpy()
        return h.astype(np.float32), self.torch.log_softmax(self.logits, -1).numpy()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def capture_all(probe: Probe, generate, requests, keep_logp: bool, log=print):
    states, logps = [], []
    for k, req in enumerate(requests):
        h, lp = probe.capture(generate, req)
        states.append(h.astype(np.float16))
        if keep_logp:
            logps.append(lp.astype(np.float16))
        if k % 100 == 0:
            log(f"captured {k}/{len(requests)}")
    return np.stack(states), (np.stack(logps) if keep_logp else None)


def build(child_adapter, parent_adapter, f0: list, f1: list, requests_of, out: Path, log=print) -> dict:
    """The whole stage on GPU. child_adapter / parent_adapter: zero-argument callables returning a loaded offline
    adapter exposing .model, .head (the LM head module) and .generate(req); requests_of(model_role, inputs) -> the
    offline requests. The float16 state cache <out>.states.npz is reused when present. Writes <out> (npz:
    r, rhat, r_layer, p, p_layer, p_alpha, aucs) and returns the selection summary."""
    import gc
    import torch
    out = Path(out)
    cache = out.with_suffix(".states.npz")
    if cache.exists():
        z = np.load(cache)
        hc0, hc1, hp0, lpp1 = z["hc0"], z["hc1"], z["hp0"], z["lpp1"]
    else:
        adapter = child_adapter()
        probe = Probe(adapter.model, adapter.head)
        hc0, _ = capture_all(probe, adapter.generate, requests_of("child", f0), False, log)
        hc1, _ = capture_all(probe, adapter.generate, requests_of("child", f1), False, log)
        probe.remove()
        adapter.close()
        del probe, adapter
        gc.collect()
        torch.cuda.empty_cache()
        adapter = parent_adapter()
        probe = Probe(adapter.model, adapter.head)
        hp0, _ = capture_all(probe, adapter.generate, requests_of("parent", f0), False, log)
        _, lpp1 = capture_all(probe, adapter.generate, requests_of("parent", f1), True, log)
        probe.remove()
        adapter.close()
        del probe, adapter
        gc.collect()
        torch.cuda.empty_cache()
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, hc0=hc0, hc1=hc1, hp0=hp0, lpp1=lpp1)
    hc0, hc1, hp0 = (x.astype(np.float32) for x in (hc0, hc1, hp0))
    if hc0.shape[1:] != hp0.shape[1:]:
        raise ValueError(f"parent states {hp0.shape[1:]} and child states {hc0.shape[1:]} differ in layers / size")
    lab0, lab1 = [lab for *_, lab in f0], [lab for *_, lab in f1]
    L = hc0.shape[1]
    r, rhat = refusal_direction(hc0, lab0)
    aucs = layer_aucs(hc1, lab1, rhat)
    p = (hp0 - hc0).mean(0)
    adapter = child_adapter()
    probe = Probe(adapter.model, adapter.head)
    runnable = [l for l in range(L) if l not in probe.skip]
    r_layer = pick_r_layer(aucs, runnable)
    cand = candidate_layers(L, runnable)
    child_f1 = requests_of("child", f1)

    def mean_kl(layer: int, alpha: float) -> float:
        probe.steer.update(on=alpha > 0, layer=layer, vec=torch.tensor(alpha * p[layer]))
        values = []
        for req, lpp in zip(child_f1, lpp1):
            _, lpc = probe.capture(adapter.generate, req)
            values.append(kl_to_parent(lpp, lpc))
        probe.steer["on"] = False
        return float(np.mean(values))

    kl = {"base": mean_kl(cand[0], 0.0)}
    for layer in cand:
        for a in ALPHAS:
            kl[f"L{layer}_a{a}"] = mean_kl(layer, a)
            log(f"KL L{layer} a{a}: {kl[f'L{layer}_a{a}']:.4f} (base {kl['base']:.4f})")
    p_layer, p_alpha = pick_prs(kl)
    probe.remove()
    adapter.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, r=r, rhat=rhat, r_layer=r_layer, p=p, p_layer=p_layer, p_alpha=p_alpha, aucs=np.array(aucs))
    summary = {"layers": L, "hidden_size": int(hc0.shape[2]), "r_layer": r_layer, "r_auc_fold1": round(aucs[r_layer], 3),
               "p_layer": p_layer, "alpha": p_alpha, "p_norm": round(float(np.linalg.norm(p[p_layer])), 2),
               "candidates": cand, "kl": {k: round(v, 4) for k, v in kl.items()},
               "alpha_at_grid_edge": p_alpha == ALPHAS[-1], "n_fold0": len(f0), "n_fold1": len(f1)}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary
