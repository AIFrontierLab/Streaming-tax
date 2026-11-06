"""A toy language model with the Qwen module layout and a toy offline adapter, to run the real training unit
(generalization/finetune/train.py) on a CPU with torch and PEFT: LoRA injection with its placement checks, B = 0 at
the start, the AdamW loop with checkpoints, a resumed run equal bit for bit to an uninterrupted one, and the saved
adapter re-attached by the offline adapters' attach_lora() computing exactly the trained function.

Importing this module needs torch and peft (a model environment). The test module imports it behind importorskip;
`python finetune_toy.py <work folder>` (with this checkout's src/ on PYTHONPATH) runs the same checks in an
environment without pytest and prints them."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn

VOCAB, WIDTH = 64, 16
LAYERS = "model.language_model.layers"


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(WIDTH, WIDTH) for _ in range(4))

    def forward(self, h):
        return self.o_proj(torch.tanh(self.q_proj(h)) * self.k_proj(h) + self.v_proj(h))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(WIDTH, 2 * WIDTH), nn.Linear(WIDTH, 2 * WIDTH)
        self.down_proj = nn.Linear(2 * WIDTH, WIDTH)

    def forward(self, h):
        return self.down_proj(torch.sigmoid(self.gate_proj(h)) * self.up_proj(h))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.mlp = Attention(), MLP()

    def forward(self, h):
        h = h + self.self_attn(h)
        return h + self.mlp(h)


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer(), Layer()])

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)
        return h


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, WIDTH)
        self.language_model = LanguageModel()


class ToyModel(nn.Module):
    """model.language_model.layers.<i>.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj}: the Qwen family's names."""

    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.lm_head = nn.Linear(WIDTH, VOCAB)
        self.checkpointing = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.checkpointing = gradient_checkpointing_kwargs

    def forward(self, input_ids):
        return self.lm_head(self.model.language_model(self.model.embed(input_ids)))


def tokens(text: str) -> list[int]:
    return [1 + ord(c) % (VOCAB - 1) for c in text]


class ToyAdapter:
    """encode(req, answer) / train_forward(enc) as the offline adapters define them (models/base.py Trainable):
    the prompt tokens, then the answer and an end-of-turn token (0) as the supervised span."""

    def __init__(self, spec=None, settings=None):
        torch.manual_seed(1234)                       # the same "checkpoint" every time it is loaded
        self.model = ToyModel()

    def encode(self, req, answer: str) -> dict:
        supervised = tokens(answer) + [0]
        return {"inputs": {"input_ids": torch.tensor([tokens(req.prompt)[-12:] + supervised])}, "n": len(supervised)}

    def train_forward(self, enc: dict):
        return self.model(enc["inputs"]["input_ids"])[:, -(enc["n"] + 1):]


def setup(work: Path, budget: int = 12):
    """The demo benchmark registered, its training split prepared and a selection written; returns the experiment."""
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from finetune_helpers import DEMO, spec, write_prepared
    from streamreplace import paths as paths_module
    from streamreplace.benchmarks import registry
    from streamreplace.core.io import write_json
    from streamreplace.generalization.finetune import Experiment
    from streamreplace.generalization.finetune.data import _digest
    import finetune_demo
    registry.register(DEMO, "finetune_demo")
    paths = paths_module.resolve(home=work / "home", environ={})
    paths_module.set_current(paths)
    write_prepared(paths)
    exp = Experiment(spec(train={**spec()["train"], "grad_accum": 2, "save_every_steps": 1}), work / "root")
    order = [s.id for s in finetune_demo.training()][3:3 + budget]
    write_json(exp.root / "train_order.json", {DEMO: order})
    write_json(exp.root / "selection.json", {DEMO: {"train_sha256": _digest(order), "test_sha256": "t"}})
    return exp


def payload(exp, directory: Path, max_steps: int | None = None, budget: int = 12) -> dict:
    from finetune_helpers import DEMO
    from streamreplace.generalization.finetune import units
    from streamreplace.generalization.finetune.train import PAYLOAD_VERSION
    unit = dict(units(exp)[0], budget=budget)
    return {"version": PAYLOAD_VERSION, "unit": unit, "model": {"id": f"par@{DEMO}-0", "spec": {}},
            "adapter": "qwen_offline", "family": "qwen_vl", "settings": {}, "lora": exp.lora, "train": exp.train,
            "selection": str(exp.root), "dir": str(directory), "max_steps": max_steps}


def logits_on(model, ids: list[int]):
    with torch.no_grad():
        return model(torch.tensor([ids])).float()


def check_training(work: Path) -> dict:
    """Runs the checks of the module docstring; returns their results (booleans and the numbers behind them)."""
    from streamreplace.generalization.finetune import lora
    from streamreplace.generalization.finetune.train import Unit
    work = Path(work)
    exp = setup(work)
    probe = tokens("Question 3? left right") + [0]
    out = {}

    base = ToyAdapter().model
    before = logits_on(base, probe)
    wrapped = lora.inject(base, "qwen_vl", 16, 32, seed=0)
    out["lora_modules"] = len(wrapped)
    out["zero_init_unchanged"] = bool(torch.equal(before, logits_on(base, probe)))

    straight = Unit(payload(exp, work / "straight"), load_adapter=lambda model, settings: ToyAdapter())
    done = straight.run()
    pilot = Unit(payload(exp, work / "resumed", max_steps=2), load_adapter=lambda model, settings: ToyAdapter())
    stopped = pilot.run()
    resumed = Unit(payload(exp, work / "resumed"), load_adapter=lambda model, settings: ToyAdapter()).run()
    losses = lambda folder: [json.loads(l)["loss"] for l in (work / folder / "loss.jsonl").read_text().splitlines()]
    out.update(steps=done["steps"], pilot_stopped_at=stopped.get("pilot_stopped_at"),
               losses_equal=losses("straight") == losses("resumed"), loss_steps=len(losses("straight")),
               adapter_equal=done["adapter_sha256"] == resumed["adapter_sha256"],
               adapter_sha256=done["adapter_sha256"], lora_parameters=done["lora"]["lora_parameters"],
               no_checkpoint_left=not (work / "straight" / "ckpt.pt").exists(),
               done_file=(work / "straight" / "DONE.json").is_file())
    file_sha = hashlib.sha256((work / "straight" / "adapter" / "adapter_model.safetensors").read_bytes()).hexdigest()
    out["done_sha_is_the_file"] = file_sha == done["adapter_sha256"]

    # the saved adapter on a freshly loaded model, attached as the evaluation attaches it
    trained = ToyAdapter().model
    lora.inject(trained, "qwen_vl", 16, 32, seed=0)
    from safetensors.torch import load_file
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(trained, load_file(str(work / "straight" / "adapter" / "adapter_model.safetensors")))
    expected = logits_on(trained, probe)
    from streamreplace.models.lora import attach_lora       # what every offline adapter's load_lora() calls
    fresh = ToyAdapter().model
    attach_lora(fresh, LAYERS, 7, work / "straight" / "adapter")
    out["reload_equals_trained"] = bool(torch.equal(expected, logits_on(fresh, probe)))
    out["reload_changes_the_function"] = not bool(torch.equal(before, logits_on(fresh, probe)))
    return out


if __name__ == "__main__":
    print(json.dumps(check_training(Path(sys.argv[1])), indent=1))
