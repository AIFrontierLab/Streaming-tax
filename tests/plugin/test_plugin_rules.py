"""Plugin rules that need no torch: Recall, Respond, finish(), post-hoc S+, thresholds, artefacts, frozen lists."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from streamreplace.analysis.records import Job, make_record
from streamreplace.core.schema import Generation, Request
from streamreplace.plugin import artefacts, posthoc
from streamreplace.plugin import respond as R
from streamreplace.plugin.calibrate import thresholds as T
from streamreplace.plugin.hooks import PluginHooks, make_hooks
from streamreplace.plugin.recall import Reservoir, aura_policy, aura_spans
from streamreplace.plugin.recipe import REFUSAL, AnswerTurn, ArmRecipe, load_plugin_config, plugin_config_from_dict
from streamreplace.plugin.report import Recovery, gate_acted, recovered

CHILDREN = ["mmduet_7b", "thinkstream_3b", "mmduet2_3b", "aura_8b", "livestar_8b", "moss_vl_rt"]
LAYERS = {"mmduet_7b": (8, 19), "thinkstream_3b": (20, 14), "mmduet2_3b": (18, 21), "aura_8b": (11, 18),
          "livestar_8b": (11, 25), "moss_vl_rt": (24, 19)}
TAUS = {"mmduet_7b": (0.225, 0.225), "thinkstream_3b": (16.064, 16.064), "mmduet2_3b": (1.1485, 1.2629249999999999),
        "aura_8b": (-1.1555, -1.1555), "livestar_8b": (-0.6805, 3.6417249999999997),
        "moss_vl_rt": (5.2385, 6.433749999999999)}


def mcq(i="x", text_kind="mcq"):
    return Request(i, "", "q", kind=text_kind, choices={"A": "a", "B": "b", "C": "c", "D": "d"} if text_kind == "mcq" else {})


# ---------- Recall ----------

def test_aura_spans_official_full_and_budget():
    spans, evicted = aura_spans(95.4, "official")
    assert len(spans) == 30 and evicted == 65 and spans[-1][0] == pytest.approx(94.4)
    spans, evicted = aura_spans(95.4, "full")
    assert len(spans) == 95 and evicted == 0 and spans[0][0] == pytest.approx(0.4)
    assert aura_spans(95.4, 120) == aura_spans(95.4, "full")
    spans, evicted = aura_spans(95.4, 40, recent=30, stream_id="v")
    starts = [s for s, _ in spans]
    assert len(spans) == 40 and evicted == 55 and starts == sorted(starts)
    assert starts[-30:] == [95.4 - 95 + i for i in range(65, 95)]
    assert aura_spans(0.6, "full") == ([(0.0, 0.6)], 0)
    assert aura_policy({"stream": {"window": "full"}}) == "full" and aura_policy({}) == "official"
    assert aura_policy({"stream": {"window": "full", "budget": 40}}) == 40           # Recall bounded by a budget
    for stream in ({"window": 40}, {"budget": 40}, {"window": 40, "budget": 40}):    # AURA's own last-N window
        with pytest.raises(ValueError, match="Recall policy"):
            aura_policy({"stream": stream})


def test_reservoir_is_causal_and_deterministic():
    a, b = Reservoir(3, 2, "s"), Reservoir(3, 2, "s")
    for i in range(20):
        a.push(i)
        b.push(i)
    assert a.items() == b.items() and len(a.items()) == 5
    assert [c for _, c in a.items()][-3:] == [17, 18, 19]


# ---------- Respond ----------

def test_readout_rules():
    req, yes = mcq(), mcq("y", "yesno")
    assert R.options(req) == ["A", "B", "C", "D"] and R.options(yes) == ["Yes", "No"]
    assert R.options(Request("z", "", "q", kind="open")) == []
    for text in ["B", "B.", "(B) the man", "**B**", " C: cooking", "D"]:
        assert R.opens_with_option(text, R.options(req)), text
    for text in ["The answer is B", "E", "Based on the video, B", "", "I'm sorry"]:
        assert not R.opens_with_option(text, R.options(req)), text
    assert R.opens_with_option("Yes, he does.", ["Yes", "No"]) and R.opens_with_option("no", ["Yes", "No"])
    assert not R.opens_with_option("The boy eats.", ["Yes", "No"])
    assert R.read(np.array([0.0, 2.0, 5.0, 1.0]), {"A": 0, "B": 1, "C": 2, "D": 3}) == \
        ("C", {"A": 0.0, "B": 2.0, "C": 5.0, "D": 1.0})


class Tok:
    """A toy tokenizer: ids are positions in VOCAB."""
    VOCAB = ["A", "B", "C", "D", "Yes", "No", "NO", "no", " No", "RE", "REP", "Reply", "R", "<|im_end|>", "<|silent|>",
             "<|silence|>", "<|response|>", "hello"]

    def __len__(self):
        return len(self.VOCAB)

    def decode(self, ids):
        return "".join(self.VOCAB[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self.VOCAB.index(text)] if text in self.VOCAB else []

    def convert_tokens_to_ids(self, token):
        return self.VOCAB.index(token)


def test_silence_bans_per_declared_marker():
    """Each child's bans come from its answer_turn declaration in configs/plugin.yaml (the paper's markers)."""
    tok, v = Tok(), Tok.VOCAB
    turns = load_plugin_config().answer_turns
    assert {c: (t.silence, t.logits) for c, t in turns.items()} == {
        "mmduet_7b": (None, "first_step"), "thinkstream_3b": (None, "deadline"),
        "mmduet2_3b": ("NO REPLY", "processor"), "aura_8b": ("<|silent|>", "processor"),
        "livestar_8b": (None, "processor"), "moss_vl_rt": ("<|silence|>", "opening")}
    first, follow = R.silence_bans(turns["mmduet2_3b"].silence, tok)
    assert sorted(v[i] for i in first) == ["<|im_end|>", "NO"]                 # only the capital NO opens the ban
    assert set(follow) == {v.index("NO"), v.index("no"), v.index(" No"), v.index("No")}
    assert sorted(v[i] for i in follow[v.index("No")]) == ["RE", "REP", "Reply"]   # "R" is too short to ban
    assert R.silence_bans(turns["aura_8b"].silence, tok) == ([v.index("<|silent|>"), v.index("<|im_end|>")], None)
    assert R.silence_bans(turns["moss_vl_rt"].silence, tok) == ([v.index("<|silence|>"), v.index("<|im_end|>")], None)
    for child in ("mmduet_7b", "livestar_8b", "thinkstream_3b"):
        assert R.silence_bans(turns[child].silence, tok) == ([], None)

    class Unknown(Tok):                       # a tokenizer without the marker: None, or its unknown-token id
        unk_token_id = 17

        def convert_tokens_to_ids(self, token):
            return self.VOCAB.index(token) if token in self.VOCAB else self.unk_token_id
    for unk in (None, 17):
        Unknown.unk_token_id = unk
        with pytest.raises(ValueError, match="not a token"):
            R.silence_bans("<|quiet|>", Unknown())


def test_answer_turn_declarations_are_validated():
    base = {"arms": {"sfix": {"recall": "full", "respond": True, "gate": "shadow"}}, "splus_arm": "sfix"}
    config = plugin_config_from_dict({**base, "answer_turn": {"c": {"silence": None, "logits": "first_step"}}})
    assert config.answer_turns == {"c": AnswerTurn(None, "first_step")}
    for bad in ({"logits": "processor"}, {"silence": None}, {"silence": None, "logits": "processor", "extra": 1},
                {"silence": None, "logits": "lm_head"}, {"silence": "", "logits": "processor"},
                {"silence": "NO REPLY", "logits": "first_step"},     # the two-step ban runs in the processor only
                {"silence": "<|s|>", "logits": "deadline"}):        # no logits: nothing to ban
        with pytest.raises(ValueError, match="answer_turn"):
            plugin_config_from_dict({**base, "answer_turn": {"c": bad}})


def test_arms_acting_on_the_logits_refuse_an_undeclared_child():
    """A child without an answer_turn entry never runs Respond or a live / record gate without its bans, readout or
    halt; arms whose rules never touch the logits still run."""
    config = load_plugin_config()
    assert [a for a in config.arms if config.arm(a).needs_turn] == ["sfix", "sfix_prs", "record"]
    for variant in ("sfix", "sfix_prs", "record"):
        with pytest.raises(ValueError, match="answer_turn"):
            PluginHooks("new_child", config.arm(variant), artefact=object())
    with pytest.raises(ValueError, match="answer_turn"):
        PluginHooks("new_child", ArmRecipe("live", "native", False, "live"), artefact=object())
    inert = make_hooks("new_child", "inert", config)
    assert inert.path is None and inert.identity()["answer_turn"] is None
    assert make_hooks("aura_8b", "sfix", config).identity()["answer_turn"] == {"silence": "<|silent|>",
                                                                               "logits": "processor"}


class FakeGuard:
    def __init__(self, fired):
        self._fired = fired

    def log(self):
        return {"gate": "post", "proj": 1.0, "fired": self._fired, "prs": False}


def hooks_for(child="aura_8b", respond=True, guard=None):
    h = PluginHooks(child, ArmRecipe("sfix" if respond else "inert", "full" if respond else "native", respond),
                    turn=load_plugin_config().answer_turns[child])
    h.tokenizer, h.guard = Tok(), guard
    return h


def test_finish_refusal_readout_and_passthrough():
    req = mcq()
    out = hooks_for(guard=FakeGuard(True)).finish(req, Generation("Got it, I'll", 4, "eos", {"no_answer": False}))
    assert (out.text, out.finish, out.info["plugin"]["native"]) == (REFUSAL, "guard", "Got it, I'll")
    assert "readout" not in out.info["plugin"]
    h = hooks_for(guard=FakeGuard(False))
    h.capture["logits"] = np.array([1.0, 3.0, 0.0, 0.0] + [9.0] * 14)
    out = h.finish(req, Generation("The person is cooking.", 5, "eos", {}))
    assert (out.text, out.finish) == ("B", "readout") and out.info["plugin"]["readout"]["B"] == 3.0
    assert out.info["no_answer"] is False and out.info["plugin"]["guard"]["fired"] is False
    h.capture["logits"] = np.array([1.0, 3.0, 0.0, 0.0] + [9.0] * 14)
    assert h.finish(req, Generation("A. cooking", 3, "eos", {})).text == "A. cooking"   # holds an option: kept
    silent = hooks_for()
    silent.capture["logits"] = np.array([5.0, 3.0, 0.0, 0.0] + [9.0] * 14)
    out = silent.finish(req, Generation("", 0, "eos", {"no_answer": True}))
    assert out.text == "A" and out.info["no_answer"] is False
    open_req = Request("o", "", "q", kind="open")
    assert silent.finish(open_req, Generation("", 0, "eos", {"no_answer": True})).text == ""   # never on open items
    inert = hooks_for(respond=False)
    inert.capture["logits"] = np.array([1.0, 3.0] + [0.0] * 16)
    out = inert.finish(req, Generation("The person is cooking.", 5, "eos", {}))
    assert out.text == "The person is cooking." and out.info["plugin"] == {"native": "The person is cooking."}


class ArmableGuard:
    """A guard stand-in that keeps its armed flag and records the calls the hooks make on it."""

    def __init__(self):
        self.armed, self.calls = False, []

    def arm(self, first=False):
        self.armed = True
        self.calls.append("arm")

    def disarm(self):
        self.armed = False
        self.calls.append("disarm")

    def reset(self, kind=None):
        self.calls.append(("reset", kind))


def test_begin_disarms_a_guard_left_armed():
    """Safety net: an answer turn that ended without disarm() (an adapter error between arm and disarm) cannot carry
    an armed guard into the next request; begin() disarms it, and the next answer turn arms it again."""
    h = hooks_for(guard=ArmableGuard())
    h.begin(mcq())
    h.arm()
    assert h.armed and h.guard.armed                            # the turn ends here without disarm()
    h.begin(mcq("y", "yesno"))
    assert not h.armed and not h.guard.armed and h.guard.calls[-2:] == ["disarm", ("reset", "yesno")]
    h.arm()
    assert h.armed and h.guard.armed and h.guard.calls[-1] == "arm"


def test_finish_refuses_a_row_without_the_arms_guard_or_respond():
    """attach() never succeeded: an arm with a gate / PRS, or with Respond, must not write a valid-looking row."""
    config = load_plugin_config()
    turn = config.answer_turns["aura_8b"]
    for recipe, art in ((config.arm("sfix_prs"), artefacts.load("aura_8b")), (ArmRecipe("r", "native", True), None)):
        with pytest.raises(RuntimeError, match="attach"):
            PluginHooks("aura_8b", recipe, art, turn=turn).finish(mcq(), Generation("A", 1, "eos", {}))
    out = PluginHooks("aura_8b", config.arm("inert")).finish(mcq(), Generation("A", 1, "eos", {}))
    assert out.info["plugin"] == {"native": "A"}                # nothing attached, nothing needed


def test_eager_decode_follows_prs():
    """ThinkStream decodes its answer chunk eagerly only when the hooks steer decode steps (PRS)."""
    config = load_plugin_config()
    art = artefacts.load("thinkstream_3b")
    expected = {"sfix": False, "sfix_prs": True, "record": False, "inert": False, "hf": False}
    for arm, eager in expected.items():
        recipe = config.arm(arm)
        hooks = PluginHooks("thinkstream_3b", recipe, art if recipe.needs_artefacts else None,
                            turn=config.answer_turns["thinkstream_3b"])
        assert hooks.eager_decode is eager, arm


def test_thinkstream_readout_comes_from_the_deadline_fork_only_without_an_answer():
    class Logits(list):
        def float(self):
            return self

        def cpu(self):
            return self

    h = hooks_for("thinkstream_3b")
    calls = []
    h.deadline_readout(lambda: calls.append(1) or Logits([0.1, 2.0, 0.0, 0.0] + [0.0] * 14))
    out = h.finish(mcq(), Generation("", None, "unknown", {"no_answer": True, "failure": "Deadline exceeded"}))
    assert (out.text, out.finish, out.info["failure"]) == ("B", "readout", "Deadline exceeded")
    h.deadline_readout(lambda: calls.append(2) or Logits([9.0] * 18))
    out = h.finish(mcq(), Generation("C", 1, "eos", {"no_answer": False}))
    assert out.text == "C" and calls == [1]                     # an answered turn never runs the fork
    assert h._fork is None                                      # not reused by the next request


def test_answered_uses_the_benchmark_rule_parser(monkeypatch):
    from streamreplace.benchmarks import registry
    seen = {}

    def parse(view, text):
        seen["view"] = view
        return ("B" if "second" in text else None), "rule"

    monkeypatch.setattr(registry, "get", lambda name: SimpleNamespace(parse=parse))
    req = Request("x", "demo", "q", kind="mcq", choices={"A": "a", "B": "b"}, sample=SimpleNamespace(id="x"))
    assert R.answered(req, "the second one", ["A", "B"]) and seen["view"].id == "x"
    assert not R.answered(req, "B", ["A", "B"])                  # the benchmark's parser decides, not the regex


# ---------- post-hoc S+ ----------

def job(dataset, rows: dict, name="m@sfix_prs.stream"):
    recs = {}
    for i, (proj, correct) in rows.items():
        row = {"id": i, "pred": "A" if correct else "B", "parse": "rule", "correct": correct, "judge_pred": None,
               "judge_correct": None, "error": None, "finish": "eos",
               "info": {"plugin": {"guard": {"proj": proj}}} if proj is not None else {}}
        recs[i] = make_record(row, False)
    return Job(dataset, name, recs, len(recs), None)


def test_fired_by_format_and_strict_threshold():
    smp = {"a": SimpleNamespace(kind="mcq"), "b": SimpleNamespace(kind="mcq"), "c": SimpleNamespace(kind="open"),
           "d": SimpleNamespace(kind="open"), "e": SimpleNamespace(kind="yesno")}
    j = job("d", {"a": (0.5, True), "b": (2.0, True), "c": (2.0, True), "d": (None, True), "e": (1.0, True)})
    assert posthoc.fired(j, smp, {"closed": 1.0, "free": 3.0}) == {"b"}   # 1.0 is not above 1.0: strict
    assert posthoc.fired(j, smp, {"closed": 0.0, "free": 1.0}) == {"a", "b", "c", "e"}   # no projection: never fires


def test_assembly_takes_the_refusal_row_or_no_answer_on_understanding_sets():
    smp = {i: SimpleNamespace(kind="mcq") for i in "abc"}
    arm = job("videomme", {"a": (5.0, True), "b": (0.0, True), "c": (5.0, False)})
    refusal = job("videomme", {i: (None, True) for i in "abc"}, "fixed_refusal.offline")   # a parser credited it
    gated, ids = posthoc.assemble(arm, refusal, smp, {"closed": 1.0, "free": 1.0}, ("videomme",))
    assert ids == {"a", "c"} and gated.rows["b"] is arm.rows["b"]
    assert not gated.rows["a"].correct and not gated.rows["a"].answered and gated.rows["a"].pred is None
    vsb = job("video_safetybench", {"a": (5.0, False), "b": (0.0, False)})
    ref = job("video_safetybench", {"a": (None, True), "b": (None, True)}, "fixed_refusal.offline")
    free_form = {i: SimpleNamespace(kind="open") for i in "ab"}
    gated, _ = posthoc.assemble(vsb, ref, free_form, {"closed": 9.0, "free": 1.0}, ("videomme",))
    assert gated.rows["a"] is ref.rows["a"] and gated.rows["b"] is vsb.rows["b"] and gated.uid != vsb.uid


def test_refusal_job_rows(tmp_path):
    ids = ["1", "2"]
    assert posthoc.write_refusal_job(tmp_path, "d", ids, {"1": "p1", "2": "p2"}) == 2
    assert posthoc.write_refusal_job(tmp_path, "d", ids + ["3"], {"1": "p1", "2": "p2", "3": "p3"}) == 1
    rows = [json.loads(x) for x in (tmp_path / "d" / "fixed_refusal.offline.jsonl").read_text().splitlines()]
    assert [r["id"] for r in rows] == ["1", "2", "3"] and {r["raw"] for r in rows} == {REFUSAL}
    assert rows[0]["finish"] == "guard" and rows[0]["info"]["no_answer"] is False and rows[2]["prompt"] == "p3"
    assert set(rows[0]["info"]) == {"no_answer", "plugin"}


def test_report_reads_shadow_arms_only():
    """plugin report assembles S+ from shadow arms: an arm configured with another gate, or rows a live / record gate
    made (the refusal already in the row, finish "guard"), would enter S_fix, PRS and S+ with the gate's decision."""
    config = load_plugin_config()
    arms = {**config.raw["arms"], "sfix_prs": {**config.raw["arms"]["sfix_prs"], "gate": "live"}}
    with pytest.raises(ValueError, match="shadow"):
        Recovery(None, None, ".", ".", {}, [], {}, plugin_config_from_dict({**config.raw, "arms": arms}))

    def rec(i, finish="eos", guard=None):
        row = {"id": i, "pred": "A", "parse": "rule", "correct": True, "judge_pred": None, "judge_correct": None,
               "error": None, "finish": finish, "info": {"plugin": {"guard": guard}} if guard else {}}
        return make_record(row, False)
    shadow = {"a": rec("a", guard={"mode": "shadow", "proj": 1.0, "fired": False, "shadow": True}),
              "b": rec("b", guard={"proj": 2.0, "fired": False, "shadow": True}), "c": rec("c")}
    assert gate_acted(Job("d", "m@sfix_prs.stream", shadow, 3, None)) == []
    live = {"a": rec("a", "guard", {"mode": "live", "proj": 9.0, "fired": True}),
            "b": rec("b", guard={"mode": "live", "proj": 0.1, "fired": False}),
            "c": rec("c", guard={"mode": "record", "proj": 0.1, "fired": False}), "d": rec("d")}
    assert gate_acted(Job("d", "m@sfix_prs.stream", live, 4, None)) == ["a", "b", "c"]


def test_recovered_share_and_noninferiority():
    cell = {"loss": {"delta": -10.0, "ci95": [-15.0, -5.0]}, "effect": {"delta": 4.0, "ci95": [1.0, 7.0]}}
    recovered(cell)
    assert cell["recovered"] == pytest.approx(0.4) and cell["ni2"] is None
    cell = {"loss": {"delta": -1.0, "ci95": [-4.0, 2.0]}, "effect": {"delta": -0.5, "ci95": [-1.9, 0.9]}}
    recovered(cell)
    assert cell["recovered"] is None and cell["ni2"] is True
    cell = {"loss": {"delta": 3.0, "ci95": [1.0, 5.0]}, "effect": {"delta": -1.0, "ci95": [-3.0, 1.0]}}
    recovered(cell)
    assert cell["recovered"] is None and cell["ni2"] is False


# ---------- thresholds ----------

def test_threshold_rule():
    values = [float(v) for v in range(401)]
    assert T.tau_closed(values) == 0.975 * 400
    assert T.tau_closed([0.1234, 0.98765]) == round(float(np.quantile([0.1234, 0.98765], 0.975)), 4)
    free, q = T.tau_free(1.0, [0.0] * 100)
    assert (free, q) == (1.0, 0.0)
    free, q = T.tau_free(-1.0, list(np.linspace(0, 1, 100)))
    assert free == q == float(np.quantile(np.linspace(0, 1, 100), 0.975))
    with pytest.raises(ValueError):
        T.tau_free(0.0, [0.0] * 99)
    rows = {"d": {"1": {"info": {"plugin": {"guard": {"proj": 0.5}}}}, "2": {"error": "boom"}}}
    with pytest.raises(ValueError, match="1 threshold item"):
        T.record_projections(rows, {"d": ["1", "2"]})
    assert T.record_projections(rows, {"d": ["1"]}) == [0.5]
    rows = [{"id": "x", "guard": {"proj": 1.0}}, {"id": "x", "error": "late failure"}, {"id": "y", "guard": {"proj": 2.0}}]
    assert T.fit_projections(rows, ["x"]) == [1.0]               # a later error never replaces a good row
    assert T.guard_proj({"info": {"plugin": {"guard": {"proj": 0.25}}}, "error": None}) == 0.25
    assert T.guard_proj({"info": {"plugin": {"guard": {"proj": 0.25}}}, "error": "boom"}) is None
    assert T.guard_proj({"info": {"no_answer": True}}) is None and T.guard_proj(None) is None


# ---------- shipped artefacts and lists ----------

@pytest.mark.parametrize("child", CHILDREN)
def test_shipped_artefacts(child):
    assert artefacts.check(child) == []
    art = artefacts.load(child)
    assert (art.r_layer, art.p_layer) == LAYERS[child] and art.alpha == 1.5
    assert (art.thresholds["closed"], art.thresholds["free"]) == TAUS[child]
    assert art.steer_prefill == (child not in ("thinkstream_3b", "moss_vl_rt"))
    assert art.tau("old") == {"closed": TAUS[child][0], "free": TAUS[child][0]}
    assert art.tau("rule")["free"] == art.thresholds["free_harmless_q975"]
    strings = [v for v in json.dumps(art.manifest).split('"') if v.startswith("/")]
    assert "/" not in art.manifest["vectors"]["file"] and not strings          # no absolute paths


def test_tampered_vectors_are_refused(tmp_path):
    import shutil
    shutil.copytree(artefacts.root_dir() / "aura_8b", tmp_path / "aura_8b")
    path = tmp_path / "aura_8b" / "vectors.npz"
    data = bytearray(path.read_bytes())
    data[-40] ^= 1
    path.write_bytes(bytes(data))
    assert artefacts.check("aura_8b", tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        artefacts.load("aura_8b", tmp_path)


def test_recipe_and_frozen_lists():
    config = load_plugin_config()
    assert config.arm("sfix") == ArmRecipe("sfix", "full", True, "shadow", False)
    assert config.arm("sfix_prs").prs and config.arm("record").gate == "record" and not config.arm("inert").respond
    assert config.recall_delta("thinkstream_3b", "sfix") == {"stream": {"window": "full", "max_len": 81920}}
    assert config.recall_delta("thinkstream_3b", "inert") == {} and config.recall_delta("mmduet_7b", "sfix") == {}
    sizes = lambda name: {d: len(v) for d, v in config.load_items(name).items()}    # noqa: E731
    assert sizes("threshold") == {"videomme": 45, "tomato": 75, "timeblind": 80, "vmdt_fairness": 101}
    assert sum(sizes("pilot2").values()) == 908 and sum(sizes("full").values()) == 10140
    assert sum(sizes("dev").values()) == 451
    pilot, full = config.load_items("pilot2"), config.load_items("full")
    assert all(not set(pilot[d]) & set(full[d]) for d in pilot)
    harmless = json.loads(config.calibration_path("harmless").read_text())["items"]
    assert [sum(it["role"] == r for it in harmless) for r in ("fit", "guardrail")] == [100, 92]
    texts = json.loads(config.calibration_path("texts").read_text())
    assert len(texts["items"]) == 600 and sum(t["fold"] == 0 for t in texts["items"]) == 400
    assert "text" not in texts["items"][0]                     # indices only, no third-party text
    assert len(json.loads(config.calibration_path("videos").read_text())["videos"]) == 300


def test_plugin_verify_command(capsys):
    from streamreplace.cli.main import main
    assert main(["plugin", "verify"]) == 0
    assert "plugin verify: ok" in capsys.readouterr().out
