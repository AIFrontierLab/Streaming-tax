"""The three evaluation conditions of the paper and the generation jobs ("arms") behind them.

Every comparison is made inside one pair: an offline *parent* model and a streaming *child* initialised from it.

- P: the parent in the offline pipeline (whole video at once, question after the media).
- C: the child in the same offline pipeline (same frames, same prompt): isolates the checkpoint change.
- S: the child through its own native streaming interface, with the question asked at the end of the video:
  adds the protocol change.
Streaming tax S - P = adaptation tax (C - P) + execution tax (S - C) (Eq. 3 of the paper).

Stream++, the plugin, adds S_fix (Recall + Respond) and S_plug (S_fix + Refuse: parent-restoration steering and
the refusal gate). S_plug is not a generation job: it is assembled after the fact from the `sfix_prs` shadow rows
and the fixed-refusal rows. S_HF is AURA's stream rebuilt in Hugging Face transformers (the engine the plugin runs
on, without the plugin).

A job is named `<model>.<mode>` (P/C/S, as in the original runs) or `<model>@<variant>.<mode>` for plugin arms, and
the refusal pseudo-job is `fixed_refusal.offline`. Diagnostic controls (Sec. 5, the supplement) are ordinary jobs
of derived model ids declared in configs/experiments/*.yaml (e.g. `aura_8b_window.offline`)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class _StrEnum(str, Enum):
    """A str enum that formats as its value on every supported Python (3.11 changed f"{member}" for str mix-ins)."""

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, spec: str) -> str:
        return format(str(self.value), spec)


class Mode(_StrEnum):
    OFFLINE = "offline"
    STREAM = "stream"


class Role(_StrEnum):
    PARENT = "parent"
    CHILD = "child"


class Condition(_StrEnum):
    """Analysis label of an arm, always relative to one (parent, child) pair."""
    P = "P"
    C = "C"
    S = "S"
    S_FIX = "S_fix"
    S_PLUG = "S_plug"
    S_HF = "S_HF"


# Plugin arm variants (configs/plugin.yaml): sfix = Recall + Respond with the gate in shadow mode; sfix_prs = the same
# plus parent-restoration steering (the rows S_plug is assembled from); record = gate signal logged for threshold
# calibration; shadow = the native stream with the gate's score logged; prs = the native stream with PRS and the
# logged gate (the two arms of the gate-and-PRS ablation on the streaming benchmarks); inert = plugin attached but
# doing nothing (fidelity control); hf = AURA's HF stream rebuild (S_HF).
VARIANTS = ("sfix", "sfix_prs", "record", "shadow", "prs", "inert", "hf")
FIXED_REFUSAL = "fixed_refusal"        # model id of the refusal pseudo-job (rows = the fixed refusal for every id)

# Paired contrasts: delta = metric(b) - metric(a), so + means the later arm is better.
GAPS = {"C-P": ("P", "C"), "S-C": ("C", "S"), "S-P": ("P", "S")}
PLUGIN_CONTRASTS = {"loss": ("P", "S_fix"), "ckpt": ("P", "C"), "proto": ("C", "S_fix"), "effect": ("S_fix", "S_plug"),
                    "vsP": ("P", "S_plug"), "vsS": ("S", "S_plug"), "gate": ("S_fix", "S_fix+gate"),
                    "prs": ("S_fix", "S_fix+PRS"), "native": ("P", "S")}


@dataclass(frozen=True)
class Arm:
    """One generation job family: a model in one mode, optionally a plugin variant of its stream."""
    model: str
    mode: Mode
    variant: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "mode", Mode(self.mode))
        if not self.model or "." in self.model:
            raise ValueError(f"invalid model id {self.model!r} (must be non-empty and contain no '.')")
        if self.variant is not None and self.variant not in VARIANTS:
            raise ValueError(f"unknown arm variant {self.variant!r}; expected one of {', '.join(VARIANTS)}")
        if self.variant is None and "@" in self.model and self.model.rpartition("@")[2] in VARIANTS:
            raise ValueError(f"model id {self.model!r} ends in a plugin variant; use Arm(model, mode, variant)")

    def job_name(self) -> str:
        return f"{self.model}{'@' + self.variant if self.variant else ''}.{self.mode.value}"

    def __str__(self) -> str:
        return self.job_name()

    @classmethod
    def parse(cls, job_name: str) -> Arm:
        """Inverse of job_name(). Only a known plugin variant after the last '@' is read as a variant, so model ids
        that contain '@' for other reasons (e.g. fine-tuned ids '<model>@<dataset>-<budget>') stay whole."""
        stem, dot, mode = job_name.rpartition(".")
        if not dot:
            raise ValueError(f"job name {job_name!r} has no '.<mode>' suffix")
        model, at, variant = stem.rpartition("@")
        if at and variant in VARIANTS:
            return cls(model, Mode(mode), variant)
        return cls(stem, Mode(mode))


def condition_arm(condition: Condition | str, parent: str, child: str) -> Arm:
    """The generation arm behind an analysis condition of the pair (parent, child). S_plug has none (see module
    doc)."""
    condition = Condition(condition)
    if condition is Condition.P:
        return Arm(parent, Mode.OFFLINE)
    if condition is Condition.C:
        return Arm(child, Mode.OFFLINE)
    if condition is Condition.S:
        return Arm(child, Mode.STREAM)
    if condition is Condition.S_FIX:
        return Arm(child, Mode.STREAM, "sfix")
    if condition is Condition.S_HF:
        return Arm(child, Mode.STREAM, "hf")
    raise ValueError("S_plug is assembled after the fact from the sfix_prs shadow rows and the fixed-refusal rows; "
                     "it has no generation arm")


def conditions_for(media: str) -> tuple[Condition, ...]:
    """P/C for image benchmarks; P/C/S for video benchmarks (parents never stream, images are never streamed)."""
    if media == "video":
        return (Condition.P, Condition.C, Condition.S)
    if media == "image":
        return (Condition.P, Condition.C)
    raise ValueError(f"unknown benchmark media {media!r}")
