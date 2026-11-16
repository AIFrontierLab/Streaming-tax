"""The gate, PRS and logit rules on a toy decoder (CPU torch; skipped without torch / transformers)."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from streamreplace.core.schema import Generation, Request  # noqa: E402
from streamreplace.plugin.artefacts import Artefact  # noqa: E402
from streamreplace.plugin.guard import Guard, find_layers  # noqa: E402
from streamreplace.plugin.hooks import PluginHooks  # noqa: E402
from streamreplace.plugin.recipe import REFUSAL, AnswerTurn, ArmRecipe, load_plugin_config  # noqa: E402

D = 4
TURNS = load_plugin_config().answer_turns            # the shipped answer_turn declarations


class DecoderLayer(torch.nn.Module):
    def forward(self, h):
        return (h + 1.0,)


class LM(torch.nn.Module):
    """Layer i adds 1, so layer i's output on zeros is i + 1 everywhere (before any steering)."""

    def __init__(self, n=8):
        super().__init__()
        self.layers = torch.nn.ModuleList([DecoderLayer() for _ in range(n)])
        self.visual = torch.nn.ModuleList([DecoderLayer() for _ in range(n + 4)])   # a vision tower: never chosen

    def forward(self, h):
        for layer in self.layers:
            h = layer(h)[0]
        return h


def make(r, p, mode="shadow", prs=True, n=8, tau=None):
    lm = LM(n)
    rhat, vec = np.eye(D, dtype=np.float32)[0], np.full(D, 10.0, dtype=np.float32)
    return lm, Guard(find_layers(lm), rhat, r, p, vec if prs else None, mode=mode, tau=tau)


def test_find_layers_skips_towers():
    lm = LM()
    assert find_layers(lm) is lm.layers


def test_records_the_answer_position_and_steers_only_while_armed():
    lm, guard = make(2, 5)
    h = torch.zeros(1, 3, D)
    base = lm(h)
    guard.arm()
    steered = lm(h)
    guard.disarm()
    assert torch.equal(lm(h), base)                                      # disarmed: the forward is untouched
    assert torch.allclose(steered[0, -1] - base[0, -1], torch.full((D,), 10.0))   # PRS on the prefill's last position
    assert torch.equal(steered[0, :-1], base[0, :-1])
    assert guard.projection() == pytest.approx(3.0)                      # layer 2's output, read before PRS at 5
    step_base = lm(torch.zeros(1, 1, D))
    guard.arm()
    lm(torch.zeros(1, 3, D))
    step = lm(torch.zeros(1, 1, D))
    lm(torch.full((1, 2, D), 5.0))                                       # a later multi-token forward is the new state
    assert torch.allclose(step - step_base, torch.full((1, 1, D), 10.0))
    assert guard.projection() == pytest.approx(8.0)
    guard.arm(first=True)
    lm(torch.zeros(1, 1, D))
    lm(torch.full((1, 2, D), 5.0))
    assert guard.projection() == pytest.approx(3.0)                      # first=True: the first forward after arming


@pytest.mark.parametrize("r, p, n", [(20, 14, 36), (24, 19, 48), (5, 2, 8)])   # ThinkStream, MOSS-VL-Realtime, toy
def test_prefill_is_left_unsteered_when_prs_does_not_follow_the_gate(r, p, n):
    lm, guard = make(r, p, n=n)
    assert not guard.steer_prefill
    base_prefill, base_step = lm(torch.zeros(1, 3, D)), lm(torch.zeros(1, 1, D))
    guard.arm()
    prefill, step = lm(torch.zeros(1, 3, D)), lm(torch.zeros(1, 1, D))
    guard.disarm()
    assert torch.equal(prefill, base_prefill) and guard.projection() == pytest.approx(r + 1)
    assert torch.allclose(step - base_step, torch.full((1, 1, D), 10.0))   # decoded tokens are steered


@pytest.mark.parametrize("r, p", [(8, 19), (18, 21), (11, 18), (11, 25)])   # MMDuet, MMDuet2, AURA, LiveStar
def test_prefill_is_steered_after_the_gate_layer(r, p):
    lm, guard = make(r, p, n=36)
    assert guard.steer_prefill
    base = lm(torch.zeros(1, 3, D))
    guard.arm()
    out = lm(torch.zeros(1, 3, D))
    assert torch.allclose(out[0, -1] - base[0, -1], torch.full((D,), 10.0)) and guard.projection() == pytest.approx(r + 1)


def test_shadow_steers_like_live_and_logs_without_acting():
    tau = {"closed": 0.5, "free": 0.5}
    lm, live = make(2, 5, mode="live", tau=tau)
    lm2, shadow = make(2, 5, mode="shadow")
    live.reset(kind="mcq")
    h = torch.zeros(1, 3, D)
    live.arm(), shadow.arm()
    assert torch.equal(lm(h), lm2(h))                                    # the same steering
    assert live.projection() == shadow.projection() == pytest.approx(3.0)
    assert live.fired() and live.halt() and live.log()["fired"]
    assert not shadow.fired() and not shadow.halt() and shadow.log() == {
        "gate": "post", "mode": "shadow", "proj": 3.0, "tau": None, "fired": False, "prs": True, "shadow": True}


def test_live_gate_decides_on_the_logged_projection_with_the_format_split():
    tau = {"closed": 3.0, "free": 5.0}
    lm, guard = make(2, 5, mode="live", tau=tau, prs=False)
    h = torch.zeros(1, 3, D)
    h[..., 0] = 0.0004                                                   # proj 3.0004, logged 3.0: not above 3.0
    guard.reset(kind="mcq")
    guard.arm()
    lm(h)
    assert guard.projection() > 3.0 and guard.logged() == 3.0 and not guard.fired()
    guard.tau = {"closed": 2.9995, "free": 5.0}
    assert guard.fired()
    guard.reset(kind="open")                                             # free-form items use tau_free
    guard.arm()
    lm(h)
    assert not guard.fired()
    guard.reset(kind=None)
    guard.arm()
    lm(h)
    with pytest.raises(RuntimeError, match="begin"):
        guard.fired()


def test_record_mode_halts_once_the_state_is_read_and_never_fires():
    lm, guard = make(2, 5, mode="record", prs=False)
    guard.reset(kind="mcq")
    guard.arm()
    assert not guard.halt()
    lm(torch.zeros(1, 3, D))
    assert guard.halt() and not guard.fired() and guard.log()["proj"] == 3.0
    with pytest.raises(ValueError):
        make(2, 5, mode="record", prs=True)


def test_two_step_silence_ban_and_halt():
    pytest.importorskip("transformers")
    from streamreplace.plugin.guard import first_step_processor
    capture = {}
    proc = first_step_processor(None, [7], 0, capture, ban_next={3: [5, 6]})
    s0 = proc(torch.zeros(1, 4, dtype=torch.long), torch.zeros(1, 10))
    assert s0[0, 7] == float("-inf") and "logits" in capture
    s1 = proc(torch.tensor([[0, 0, 0, 0, 3]]), torch.zeros(1, 10))       # first answer token 3 (reads "No")
    assert s1[0, 5] == float("-inf") and s1[0, 6] == float("-inf") and s1[0, 7] == 0
    s1b = proc(torch.tensor([[0, 0, 0, 0, 4]]), torch.zeros(1, 10))      # another first token: nothing banned
    assert torch.equal(s1b, torch.zeros(1, 10))
    assert torch.equal(proc(torch.zeros(1, 6, dtype=torch.long), torch.zeros(1, 10)), torch.zeros(1, 10))
    lm, guard = make(2, 5, mode="live", tau={"closed": 0.5, "free": 0.5})
    guard.reset(kind="mcq")
    guard.arm()
    lm(torch.zeros(1, 3, D))
    stop = first_step_processor(guard, [7], [9, 1], {})(torch.zeros(1, 4, dtype=torch.long), torch.randn(1, 10))
    assert stop[0, 9] == 0 and torch.isinf(stop[0, :9]).all()           # only the end of turn remains


class Tok:
    VOCAB = ["A", "B", "C", "D", "<|im_end|>", "<|silence|>", "<|response|>", "x"]

    def __len__(self):
        return len(self.VOCAB)

    def decode(self, ids):
        return "".join(self.VOCAB[i] for i in ids)

    def encode(self, text, add_special_tokens=False):
        return [self.VOCAB.index(text)] if text in self.VOCAB else []

    def convert_tokens_to_ids(self, token):
        return self.VOCAB.index(token)


def artefact(r=2, p=5, n=8, closed=0.5, free=0.5):
    return Artefact("toy", np.eye(D, dtype=np.float32)[0], np.full(D, 10.0, dtype=np.float32), r, p, 1.5,
                    {"closed": closed, "free": free, "free_harmless_q975": free}, {"layers": n, "vectors": {"sha256": "x"}})


def test_hooks_lm_head_paths():
    req = Request("x", "", "q", kind="mcq", choices={"A": "a", "B": "b"})
    # MMDuet: first step only; the live gate turns the step into the end of turn
    lm = LM()
    hooks = PluginHooks("mmduet_7b", ArmRecipe("sfix_prs", "budget", True, "live", True), artefact(),
                        turn=TURNS["mmduet_7b"])
    hooks.attach(lm.layers, Tok())
    hooks.begin(req)
    hooks.arm(first=True)
    lm(torch.zeros(1, 1, D))
    logits = torch.zeros(1, 1, len(Tok.VOCAB))
    out = hooks.on_step_logits(0, logits)
    assert int(out[0, -1].argmax()) == 4 and torch.isinf(out[0, -1, :4]).all() and "logits" in hooks.capture
    again = torch.ones(1, 1, len(Tok.VOCAB))
    assert hooks.on_step_logits(1, again) is again                      # later steps untouched
    hooks.disarm()
    gen = hooks.finish(req, Generation("", 1, "eos", {}))
    assert gen.text == REFUSAL and gen.finish == "guard" and gen.info["plugin"]["guard"]["fired"]
    # MOSS-VL-Realtime: bans on the opening steps until the argmax is not <|response|>; readout logits = that step's
    moss = PluginHooks("moss_vl_rt", ArmRecipe("sfix", "budget", True, "shadow", False), artefact(5, 2),
                       turn=TURNS["moss_vl_rt"])
    moss.attach(LM().layers, Tok())
    moss.begin(req)
    moss.arm()
    step0 = torch.zeros(1, 1, len(Tok.VOCAB))
    step0[0, 0, 5] = 9.0                                                 # <|silence|> would win: banned
    step0[0, 0, 6] = 5.0                                                 # then <|response|> is the argmax: keep opening
    out0 = moss.on_step_logits(0, step0)
    assert out0[0, -1, 5] == float("-inf") and moss._opening
    step1 = torch.zeros(1, 1, len(Tok.VOCAB))
    step1[0, 0, 1] = 3.0
    moss.on_step_logits(1, step1)
    assert not moss._opening and torch.equal(moss.capture["logits"], step1[0, -1])
    step2 = torch.zeros(1, 1, len(Tok.VOCAB))
    assert moss.on_step_logits(2, step2) is step2
    moss.disarm()
    gen = moss.finish(req, Generation("", 0, "eos", {"no_answer": True}))
    assert gen.text == "B" and gen.finish == "readout" and gen.info["plugin"]["guard"]["proj"] is None


def test_live_gate_takes_the_answer_format_when_attach_follows_begin():
    """The run worker calls begin(req) before stream(), and the adapter attaches on its first stream(): the guard made
    by that first attach must still know the request's answer format (closed: tau_closed; free-form: tau_free)."""
    tok = Tok()
    for kind, fires in (("mcq", True), ("open", False)):
        lm = LM()
        hooks = PluginHooks("livestar_8b", ArmRecipe("sfix_prs", "budget", True, "live", True),
                            artefact(closed=2.5, free=3.5), turn=TURNS["livestar_8b"])
        req = Request("x", "", "q", kind=kind, choices={"A": "a", "B": "b"} if kind == "mcq" else {})
        hooks.begin(req)
        hooks.attach(lm.layers, tok)                                     # first stream(): attach after begin
        hooks.arm()
        lm(torch.zeros(1, 3, D))                                         # proj = layer 2's output = 3.0
        hooks.disarm()
        assert hooks.guard.kind == kind and hooks.guard.fired() is fires
        gen = hooks.finish(req, Generation("A", 1, "eos", {}))
        assert (gen.finish == "guard") is fires and gen.info["plugin"]["guard"]["proj"] == 3.0


@pytest.mark.parametrize("gate", ["shadow", "live", "record"])
def test_begin_disarms_a_guard_left_armed_by_the_previous_request(gate):
    """An answer turn that ended without disarm() (an adapter error between arm and disarm) must not steer or record
    the next request's stream: begin() disarms the guard. The next answer turn then works as always: it is steered
    (PRS; not in record mode) and recorded, shadow only logs, live fires above its threshold, record halts."""
    req = Request("x", "", "q", kind="mcq", choices={"A": "a", "B": "b"})
    lm, h = LM(), torch.zeros(1, 3, D)
    native = lm(h)
    hooks = PluginHooks("livestar_8b", ArmRecipe(gate, "budget", True, gate, gate != "record"), artefact(),
                        turn=TURNS["livestar_8b"])
    hooks.begin(req)
    hooks.attach(lm.layers, Tok())
    hooks.arm()
    lm(h)                                                                # request 1's turn, never disarmed
    assert hooks.guard.armed
    hooks.begin(req)                                                     # request 2
    assert not hooks.guard.armed and hooks.guard.state is None
    assert torch.equal(lm(h), native) and hooks.guard.state is None      # its stream: neither steered nor recorded
    hooks.arm()
    turn = lm(h)                                                         # its answer turn
    hooks.disarm()
    assert hooks.guard.logged() == 3.0 and torch.equal(turn, native) is (gate == "record")
    assert hooks.guard.fired() is (gate == "live") and hooks.guard.halt() is (gate != "shadow")


def test_hooks_settings_identity_and_attach_once():
    lm = LM(36)
    hooks = PluginHooks("thinkstream_3b", ArmRecipe("sfix_prs", "budget", True, "shadow", True), artefact(20, 14, 36),
                        delta={"stream": {"window": 20, "reservoir": 10, "max_len": 81920}}, turn=TURNS["thinkstream_3b"])
    merged = hooks.settings({"stream": {"fps": 1, "window": 20}, "generation": {"max_new_tokens": 512}})
    assert merged == {"stream": {"fps": 1, "window": 20, "reservoir": 10, "max_len": 81920},
                      "generation": {"max_new_tokens": 512}}
    hooks.attach(lm, Tok())                                              # a model: its decoder list is found
    guard = hooks.guard
    hooks.attach(lm.layers, Tok())
    assert hooks.guard is guard and len(guard.handles) == 2
    ident = hooks.identity()
    assert ident["artefacts"]["r_layer"] == 20 and ident["recipe"]["prs"] and ident["settings"]["stream"]["reservoir"]
    assert ident["answer_turn"] == {"silence": None, "logits": "deadline"}
    with pytest.raises(ValueError, match="decoder layers"):
        PluginHooks("x", ArmRecipe("sfix", "budget", True, "shadow"), artefact(n=30),
                    turn=AnswerTurn(None, "processor")).attach(LM(8).layers, Tok())


def hooked(layers) -> int:
    return sum(len(layer._forward_hooks) for layer in layers)


def test_a_failed_attach_keeps_nothing_and_the_next_attach_succeeds():
    """An attach() that raises (here: a checkpoint of another depth than the artefacts') registers nothing and leaves
    the hooks unattached, so the adapter's next stream() attaches again; meanwhile finish() refuses the row."""
    req = Request("x", "", "q", kind="mcq", choices={"A": "a", "B": "b"})
    hooks = PluginHooks("thinkstream_3b", ArmRecipe("sfix_prs", "budget", True, "shadow", True), artefact(20, 14, 36),
                        turn=TURNS["thinkstream_3b"])
    hooks.begin(req)
    wrong = LM(28)
    with pytest.raises(ValueError, match="28 decoder layers"):
        hooks.attach(wrong.layers, Tok())
    assert not hooks.attached and hooks.guard is None and hooked(wrong.layers) == 0
    with pytest.raises(RuntimeError, match="attach"):
        hooks.finish(req, Generation("A", 1, "eos", {}))
    right = LM(36)
    hooks.begin(req)
    hooks.attach(right.layers, Tok())
    assert hooks.attached and hooks.guard is not None and hooks.guard.kind == "mcq" and hooked(right.layers) == 2
    hooks.attach(right.layers, Tok())                                    # attached: later calls change nothing
    assert hooked(right.layers) == 2
    lm = LM(8)                                                           # a layer index out of range: no hook at all
    with pytest.raises(ValueError, match="p_layer 9"):
        Guard(find_layers(lm), np.eye(D, dtype=np.float32)[0], 2, 9, np.ones(D, dtype=np.float32))
    assert hooked(lm.layers) == 0


def test_logits_calls_must_match_the_declared_path():
    """The adapter's logits call must be the one the child's answer_turn declares: otherwise the rules (bans, readout
    capture, halt) would silently never run."""
    req = Request("x", "", "q", kind="mcq", choices={"A": "a", "B": "b"})
    vocab = len(Tok.VOCAB)

    def armed(turn):
        hooks = PluginHooks("toy", ArmRecipe("sfix", "budget", True, "shadow"), artefact(), turn=turn)
        hooks.attach(LM().layers, Tok())
        hooks.begin(req)
        hooks.arm()
        return hooks
    for turn in (AnswerTurn(None, "first_step"), AnswerTurn("<|silence|>", "opening"), AnswerTurn(None, "deadline")):
        with pytest.raises(RuntimeError, match="logits processors"):
            armed(turn).logits_processors()
    for turn in (AnswerTurn("<|silence|>", "processor"), AnswerTurn(None, "deadline")):
        with pytest.raises(RuntimeError, match="LM-head"):
            armed(turn).on_step_logits(0, torch.zeros(1, 1, vocab))
    first = armed(AnswerTurn("<|silence|>", "first_step"))              # a declared marker is banned at step 0
    step0 = torch.zeros(1, 1, vocab)
    step0[0, 0, 5] = 9.0
    out = first.on_step_logits(0, step0)
    assert out[0, -1, 5] == float("-inf") and out[0, -1, 4] == float("-inf") and int(out[0, -1].argmax()) == 0
    assert torch.equal(first.capture["logits"], step0[0, -1])           # the readout sees the logits before the ban
