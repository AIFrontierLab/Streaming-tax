"""LM-head wrapper shared by the offline adapters (ported unchanged from the paper's runs).

KeepLast wraps a model's LM head (torch is imported lazily, inside the model env):
- keep > 0 projects only the last `keep` positions. The logits of those positions are unchanged; a training step
  needs only the answer span, and a long video prefill only the last position. The LLaVA-OneVision / MMDuet offline
  adapter sets keep = 1 for video requests: without it the 160-frame prefill does not fit in 48 GB.
- capture=True stores the first call's last-position logits, i.e. the logits of the first generated token, for the
  option-letter readout of multiple-choice answers without an extra forward pass.
- `calls` counts forward calls (one per generated token during cached greedy decoding): adapters whose generate API
  returns only text (LiveStar, MOSS) reset it per request to report tokens and a budget stop (counted_stop)."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.schema import Request


def head_wrapper(head):
    """Wraps an LM head in a KeepLast module (see the module docstring)."""
    import torch

    class KeepLast(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.head, self.keep, self.capture, self.captured, self.calls = inner, 0, False, None, 0

        def forward(self, hidden):
            self.calls += 1
            if self.keep and hidden.shape[1] > self.keep:
                hidden = hidden[:, -self.keep:]
            out = self.head(hidden)
            if self.capture and self.captured is None:
                self.captured = out[0, -1].detach().float().cpu()
            return out

    return KeepLast(head)


def counted_stop(head, budget: int) -> dict:
    """tokens / finish from the head-call count of one generate() call (see head_wrapper)."""
    return {"tokens": head.calls, "finish": "length" if head.calls >= budget else "eos"}


def letter_logits(head, tokenizer, req: Request) -> dict | None:
    """The first generated token's logits for each option letter (multiple choice only), from a capturing head.
    Letters that do not encode to a single token are left out; values are rounded to 4 decimals."""
    if req.kind != "mcq" or head.captured is None:
        return None
    ids = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in req.choices}
    return {letter: round(float(head.captured[i[0]]), 4) for letter, i in ids.items() if len(i) == 1}
