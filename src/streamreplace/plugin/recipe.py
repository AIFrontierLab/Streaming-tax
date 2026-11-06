"""The plugin recipe (configs/plugin.yaml): arms, threshold rule, per-child Recall settings, answer turns, engines and
item sets.

One recipe (arms, threshold rule) serves every child. Per child, the configuration declares only how the child's
stream runs: its Recall settings delta, its answer turn (silence marker and logits path, AnswerTurn) and the stream
adapter its arms run on; the per-child numbers are the artefacts (plugin_artefacts/<child>/). The paper's S_fix is
the `sfix` arm and its S+ is the `sfix_prs` arm with the gate applied afterwards (posthoc.py)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

REFUSAL = "I'm sorry, but I can't help with that request."   # the fixed reply when the gate fires
INFO_KEY = "plugin"     # a plugin arm's row info block: {native, guard: {proj, fired, ...}, readout} (core/schema.py)
VIDEO_PREFIX = "These are frames of a video.\n"              # Video-SafetyBench's prompt prefix (calibration requests)
CLOSED_KINDS = ("mcq", "yesno")                              # closed-choice items: tau_closed; all others: tau_free
CHILDREN = ("mmduet_7b", "thinkstream_3b", "mmduet2_3b", "aura_8b", "livestar_8b", "moss_vl_rt")
GATE_MODES = ("shadow", "live", "record")
RECALL = ("full", "native")
VARIANTS = ("higher", "rule", "old")                         # threshold variants of the recovery report
# Where a stream adapter hands the hooks its answer turn's logits (AnswerTurn.logits; hooks.py applies the rules).
LOGITS_PATHS = ("processor", "first_step", "opening", "deadline")
NO_REPLY = "NO REPLY"           # MMDuet2's silence: a reply reading "NO REPLY" (two tokens; respond.silence_bans)


@dataclass(frozen=True)
class AnswerTurn:
    """How a child's stream adapter runs its answer turn (configs/plugin.yaml `answer_turn`; both keys required).

    silence  the marker the turn may not open with when Respond is on: one token of the child's tokenizer (banned
             with <|im_end|>), "NO REPLY" (MMDuet2's two-token reply; respond.silence_bans; processor path only) or
             None (no marker; the only choice on the deadline path)
    logits   where the adapter hands the hooks the turn's logits: processor (a Hugging Face logits processor in the
             answer turn's generate: AURA, MMDuet2, LiveStar), first_step (the LM head; the rules act on the first
             step: MMDuet), opening (the LM head; the rules act on the opening steps until the first visible token
             after <|response|>: MOSS-VL-Realtime) or deadline (no logits: the option readout comes from the deadline
             fork, and a live or record gate cannot end the turn early: ThinkStream)"""
    silence: str | None
    logits: str

    def __post_init__(self):
        if self.logits not in LOGITS_PATHS:
            raise ValueError(f"answer_turn logits must be one of {LOGITS_PATHS}, got {self.logits!r}")
        if self.silence is not None and (not isinstance(self.silence, str) or not self.silence.strip()):
            raise ValueError(f"answer_turn silence must be a marker string or null, got {self.silence!r}")
        if self.silence is not None and self.logits == "deadline":
            raise ValueError("a deadline answer turn hands the hooks no logits: its silence marker cannot be banned")
        if self.silence == NO_REPLY and self.logits != "processor":
            raise ValueError(f"the two-step {NO_REPLY!r} ban runs in the processor path only")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ArmRecipe:
    """One plugin arm. recall: "full" applies the child's Recall settings; respond: silence bans + option readout;
    gate: None | shadow | live | record (module docstring of configs/plugin.yaml); prs: parent-restoration steering."""
    name: str
    recall: str = "native"
    respond: bool = False
    gate: str | None = None
    prs: bool = False

    def __post_init__(self):
        if self.recall not in RECALL:
            raise ValueError(f"arm {self.name}: recall must be one of {RECALL}, got {self.recall!r}")
        if self.gate not in (None, *GATE_MODES):
            raise ValueError(f"arm {self.name}: gate must be null or one of {GATE_MODES}, got {self.gate!r}")
        if self.gate == "record" and self.prs:
            raise ValueError(f"arm {self.name}: a record arm calibrates the gate on the child's own state (no PRS)")

    @property
    def needs_artefacts(self) -> bool:
        return self.gate is not None or self.prs

    @property
    def needs_turn(self) -> bool:
        """Respond (silence bans, option readout) and a gate that ends the turn (live, record) act on the answer
        turn's logits: such an arm needs the child's AnswerTurn."""
        return self.respond or self.gate in ("live", "record")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PluginConfig:
    refusal: str
    arms: dict[str, ArmRecipe]
    thresholds: dict
    unanswered_on_fire: tuple[str, ...]
    recall: dict[str, dict]
    engines: dict[str, str]
    children: tuple[str, ...]
    paper_children: tuple[str, ...]
    items: dict[str, str]
    calibration: dict
    answer_turns: dict[str, AnswerTurn] = field(default_factory=dict)
    splus_arm: str = "sfix_prs"
    root: Path | None = None
    raw: dict = field(default_factory=dict)

    def arm(self, variant: str) -> ArmRecipe:
        try:
            return self.arms[variant]
        except KeyError:
            raise ValueError(f"unknown plugin arm {variant!r}; configured: {', '.join(self.arms)}") from None

    def recall_delta(self, child: str, variant: str) -> dict:
        """The settings delta of the arm's Recall for this child ({} for native recall or a child without one)."""
        import copy
        return copy.deepcopy(self.recall.get(child, {})) if self.arm(variant).recall == "full" else {}

    def path(self, relative: str | Path) -> Path:
        p = Path(relative)
        return p if p.is_absolute() or self.root is None else self.root / p

    def items_path(self, spec: str) -> Path:
        """A named item set (configs/plugin.yaml items) or a path to a {dataset: [ids]} file."""
        return self.path(self.items[spec]) if spec in self.items else Path(spec)

    def load_items(self, spec: str) -> dict[str, list[str]]:
        data = json.loads(self.items_path(spec).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
            raise ValueError(f"{spec}: an item set is a JSON object {{dataset: [ids]}}")
        return {d: [str(i) for i in ids] for d, ids in data.items()}

    def calibration_path(self, key: str) -> Path:
        return self.path(self.calibration[key])


def repo_root() -> Path:
    """The source checkout holding configs/, paper/ and plugin_artefacts/."""
    from ..paths import project_root
    return project_root()


def answer_turns_from(data: Mapping[str, Any] | None) -> dict[str, AnswerTurn]:
    """configs/plugin.yaml `answer_turn`: {child: {silence, logits}}, both keys required (silence may be null)."""
    turns = {}
    for child, spec in (data or {}).items():
        spec = dict(spec or {})
        if set(spec) != {"silence", "logits"}:
            raise ValueError(f"configs/plugin.yaml: answer_turn.{child} needs exactly the keys silence and logits, "
                             f"got {sorted(spec)}")
        try:
            turns[child] = AnswerTurn(spec["silence"], spec["logits"])
        except ValueError as exc:
            raise ValueError(f"configs/plugin.yaml: answer_turn.{child}: {exc}") from None
    return turns


def plugin_config_from_dict(data: Mapping[str, Any], root: Path | None = None) -> PluginConfig:
    arms = {name: ArmRecipe(name, **(spec or {})) for name, spec in (data.get("arms") or {}).items()}
    if not arms:
        raise ValueError("configs/plugin.yaml: no arms")
    from ..core.protocol import VARIANTS as JOB_VARIANTS
    unknown = sorted(set(arms) - set(JOB_VARIANTS))
    if unknown:
        raise ValueError(f"configs/plugin.yaml: arm names {unknown} are not job variants ({', '.join(JOB_VARIANTS)})")
    thresholds = dict(data.get("thresholds") or {})
    if thresholds.get("variant", "higher") not in VARIANTS:
        raise ValueError(f"configs/plugin.yaml: thresholds.variant must be one of {VARIANTS}")
    splus = data.get("splus_arm", "sfix_prs")
    if splus not in arms:
        raise ValueError(f"configs/plugin.yaml: splus_arm {splus!r} is not an arm")
    return PluginConfig(
        refusal=str(data.get("refusal", REFUSAL)), arms=arms, thresholds=thresholds,
        unanswered_on_fire=tuple(data.get("unanswered_on_fire") or ()),
        recall={k: dict(v or {}) for k, v in (data.get("recall") or {}).items()},
        engines=dict(data.get("engines") or {}), children=tuple(data.get("children") or CHILDREN),
        paper_children=tuple(data.get("paper_children") or data.get("children") or CHILDREN),
        items=dict(data.get("items") or {}), calibration=dict(data.get("calibration") or {}),
        answer_turns=answer_turns_from(data.get("answer_turn")), splus_arm=splus, root=root, raw=dict(data))


def load_plugin_config(path: str | Path | None = None, section: Mapping | None = None) -> PluginConfig:
    """configs/plugin.yaml of the checkout (or `path`), or an already loaded config section (Config.section("plugin"))."""
    root = repo_root()
    if section is not None and section:
        return plugin_config_from_dict(section, root)
    import yaml
    path = Path(path) if path else root / "configs" / "plugin.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return plugin_config_from_dict(data, path.resolve().parent.parent)
