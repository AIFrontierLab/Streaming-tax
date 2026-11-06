"""The six stream adapters driven with the real plugin hooks (plugin/hooks.py with configs/plugin.yaml and the shipped
artefacts) on CPU stand-ins of their models (plugin_fakes.py; skipped without torch, and the Hugging Face generate
paths without transformers). As the run worker does, each request is hooks.begin(req), then
adapter.stream(req, hooks). Checked per adapter: the hooks fire in the protocol's order around the answer turn only;
the gate logs the projection of that turn's answer position; PRS reaches that turn only (the prefill's last
position only when the PRS layer follows the gate's layer); the Respond rules act through the adapter's own logits
path (silence bans, the option readout, a live gate's halt); an inert arm reproduces the native row."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import plugin_fakes as F  # noqa: E402

from streamreplace.config.settings import merge  # noqa: E402
from streamreplace.plugin import artefacts, recall  # noqa: E402
from streamreplace.plugin.hooks import PluginHooks, make_hooks, stream_adapter  # noqa: E402
from streamreplace.plugin.recipe import REFUSAL, ArmRecipe, load_plugin_config  # noqa: E402

T = F.ID
SENTENCE = [{T[" The"]: 50, T["B"]: 30}, {T[" man"]: 50}, {T[" walks"]: 50}, {T["."]: 50}, {F.EOS: 50}]
NO_REPLY = [{T["NO"]: 50}, {T[" REPLY"]: 50}, {F.EOS: 50}]
THINK_ANSWER = [{T["</think>"]: 50}, {T["<response>"]: 50}, {T["B"]: 50}, {F.EOS: 50}]


def run(adapter, hooks, req):
    """What the run worker does for one request of a plugin arm."""
    hooks.begin(req)
    return adapter.stream(req, hooks=hooks)


def plugin_arm(child, variant, timeline, lm):
    """The arm's real hooks (spied) and the child's artefacts, with the PRS probe on the toy decoder."""
    art = artefacts.load(child)
    hooks = timeline.spy(make_hooks(child, variant))
    timeline.probe(lm, art.p_layer, art.prs)
    return hooks, art


def live_arm(child, timeline, lm, closed, free):
    """S+ with a live gate at the given thresholds (the shipped vectors and layers)."""
    art = artefacts.load(child)
    art.thresholds = {"closed": closed, "free": free, "free_harmless_q975": free}
    hooks = timeline.spy(PluginHooks(child, ArmRecipe("sfix_prs", "full", True, "live", True), art,
                                     turn=load_plugin_config().answer_turns[child]))
    timeline.probe(lm, art.p_layer, art.prs)
    return hooks


def proj(lm, prompt, art) -> float:
    """The projection the gate must log: decoder layer r_layer's output at the answer prompt's last position."""
    return round(float(lm.state(prompt, art.r_layer) @ torch.tensor(art.rhat)), 3)


def answer_turn(prompt_length, decode_steps, steer_prefill):
    """The decoder forwards of an armed answer turn under PRS: the prefill (its last position steered only when the
    PRS layer follows the gate's layer), then one steered forward per decoded token."""
    return [(prompt_length, (prompt_length - 1,) if steer_prefill else ())] + [(1, (0,))] * decode_steps


# ---------- AURA (transformers rebuild): HF generate with the hooks' logits processors ----------

def test_aura_hf_answer_turn(monkeypatch):
    pytest.importorskip("transformers")
    lm, tl = F.toy_for("aura_8b"), F.Timeline()
    hooks, art = plugin_arm("aura_8b", "sfix_prs", tl, lm)
    script = [{T["<|silent|>"]: 50, T["C"]: 40}, {F.EOS: 50}]                    # natively the turn stays silent
    adapter = F.aura_hf(monkeypatch, lm, script)
    gen = run(adapter, hooks, F.request())
    assert tl.calls() == ["begin", "attach", "logits_processors", "arm", "disarm", "finish"]
    assert (gen.text, gen.finish, gen.info["no_answer"]) == ("C", "eos", False)    # <|silent|> banned at step 1
    block = gen.info["plugin"]
    assert block["native"] == "C" and "readout" not in block
    assert block["guard"]["proj"] == proj(lm, adapter.processor.answer_prompt, art) and not block["guard"]["fired"]
    prompt = adapter.processor.answer_prompt.shape[1]
    assert tl.forwards(inside=True) == answer_turn(prompt, 1, steer_prefill=True) and not tl.forwards(inside=False)
    native = F.aura_hf(monkeypatch, F.toy_for("aura_8b"), script).stream(F.request())
    assert native.text == "" and native.info["no_answer"] and "plugin" not in native.info


def test_aura_hf_live_gate_and_record_mode_end_the_turn_at_its_first_step(monkeypatch):
    pytest.importorskip("transformers")
    script = [{T["C"]: 50}, {F.EOS: 50}]
    prompt = F.aura_prompt(3)                                                       # a 3.4-s video: 3 segments
    lm, tl = F.toy_for("aura_8b"), F.Timeline()
    expected = proj(lm, prompt, artefacts.load("aura_8b"))
    hooks = live_arm("aura_8b", tl, lm, closed=expected - 0.5, free=expected + 0.5)
    adapter = F.aura_hf(monkeypatch, lm, script)
    gen = run(adapter, hooks, F.request())                                           # closed choice: fires
    assert (gen.text, gen.finish, gen.info["plugin"]["native"]) == (REFUSAL, "guard", "")
    assert gen.info["plugin"]["guard"]["fired"] and tl.forwards(inside=True) == [(prompt.shape[1], (6,))]
    gen = run(adapter, hooks, F.request(kind="open", sample_id="q2"))                # free-form: below tau_free
    assert (gen.text, gen.finish) == ("C", "eos") and not gen.info["plugin"]["guard"]["fired"]
    lm, tl = F.toy_for("aura_8b"), F.Timeline()
    hooks, _ = plugin_arm("aura_8b", "record", tl, lm)
    gen = run(F.aura_hf(monkeypatch, lm, script), hooks, F.request(kind="open"))
    guard = gen.info["plugin"]["guard"]
    assert (guard["mode"], guard["proj"], guard["fired"], guard["prs"]) == ("record", expected, False, False)
    assert gen.text == "" and tl.forwards(inside=True) == [(prompt.shape[1], ())]  # state read, turn ended


def test_aura_plugin_arms_cut_every_segment(monkeypatch):
    """The Recall delta of AURA's plugin arms (stream.window: full) makes aura_stream_hf cut all floor(d) one-second
    segments, the same as plugin.recall.aura_spans(d, "full"); arms without Recall keep the official last 30."""
    pytest.importorskip("transformers")
    config = load_plugin_config()
    assert stream_adapter("aura_8b", "aura_stream", config) == "aura_stream_hf"
    native = {"generation": {"max_new_tokens": 8}, "stream": {"fps": 1}}
    for variant in config.arms:
        settings = merge(native, config.recall_delta("aura_8b", variant))
        adapter = F.aura_hf(monkeypatch, F.toy_for("aura_8b"), [{T["C"]: 50}, {F.EOS: 50}], frames=954,
                            settings=settings)
        plan = adapter.schedule(F.request(settings=settings))
        if config.arm(variant).recall == "full":
            assert settings["stream"]["window"] == "full"
            spans, evicted = recall.aura_spans(95.4, "full")
            assert [tuple(s) for s in plan["spans"]] == spans and (plan["segments"], plan["evicted"]) == (95, evicted)
            hooks = make_hooks("aura_8b", variant)
            gen = run(adapter, hooks, F.request(settings=settings))
            assert adapter.processor.videos == 95 and gen.info["segments"] == 95 and gen.info["evicted"] == 0
            assert gen.info["window_start"] == round(95.4 - 95, 2)
        else:
            spans, _ = recall.aura_spans(95.4, "official")
            assert plan["segments"] == 30 and [f"{s:.2f}" for s, _ in plan["spans"]] == [f"{s:.2f}" for s, _ in spans]


def test_aura_recall_budget_keeps_a_reservoir(monkeypatch):
    """Recall bounded by a budget (stream.budget with stream.window full; no shipped arm sets one): aura_stream_hf
    keeps plugin.recall.aura_spans(d, B), the last 30 segments plus a hash-priority reservoir of older ones."""
    pytest.importorskip("transformers")
    settings = {"generation": {"max_new_tokens": 8}, "stream": {"fps": 1, "window": "full", "budget": 40}}
    adapter = F.aura_hf(monkeypatch, F.toy_for("aura_8b"), [{T["C"]: 50}, {F.EOS: 50}], frames=954, settings=settings)
    plan = adapter.schedule(F.request(settings=settings))
    spans, evicted = recall.aura_spans(95.4, 40, stream_id="v.mp4")
    assert [tuple(s) for s in plan["spans"]] == spans
    assert (plan["segments"], plan["evicted"]) == (40, evicted) == (40, 55)
    assert plan["budget"] == 40 and plan["question_segment"] == 39 and plan["window_start"] == round(spans[0][0], 2)
    assert [s for s, _ in spans][-30:] == [95.4 - 95 + i for i in range(65, 95)]   # the last 30, then the reservoir
    unbounded = dict(settings, stream={"fps": 1, "window": "full"})
    full = adapter.schedule(F.request(settings=unbounded))
    assert full["segments"] == 95 and "budget" not in full and full["digest"] != plan["digest"]
    gen = run(adapter, make_hooks("aura_8b", "sfix"), F.request(settings=settings))
    assert adapter.processor.videos == 40 and (gen.info["segments"], gen.info["evicted"]) == (40, 55)
    with pytest.raises(ValueError, match="budget"):                              # a budget bounds Recall only
        adapter.schedule(F.request(settings=dict(settings, stream={"fps": 1, "budget": 40})))


# ---------- MMDuet2: the final turn's generate ----------

def test_mmduet2_answer_turn_and_two_step_silence_ban(monkeypatch):
    pytest.importorskip("transformers")
    final = [{T["NO"]: 50, T["No"]: 40}, {T[" REPLY"]: 50, T[","]: 40}, {F.EOS: 50}]   # natively "NO REPLY"
    lm, tl = F.toy_for("mmduet2_3b"), F.Timeline()
    hooks, art = plugin_arm("mmduet2_3b", "sfix_prs", tl, lm)
    adapter = F.mmduet2(monkeypatch, lm, [NO_REPLY, NO_REPLY, final])             # 5 frames: turns of 2, 2, 1
    gen = run(adapter, hooks, F.request(kind="yesno"))
    assert tl.calls() == ["begin", "attach", "logits_processors", "arm", "disarm", "finish"]
    assert (gen.text, gen.info["no_answer"], gen.info["turns"], gen.info["replies_before_question"]) == \
        ("No,", False, 3, 0)                                   # "NO" banned first, then "REPLY" after "No"
    block = gen.info["plugin"]
    assert block["guard"]["proj"] == proj(lm, adapter.client.answer_prompt, art) and "readout" not in block
    prompt = adapter.client.answer_prompt.shape[1]
    assert tl.forwards(inside=True) == answer_turn(prompt, 2, steer_prefill=True)
    assert tl.forwards(inside=False) == [(4, ()), (1, ()), (1, ())] * 2          # two earlier turns, untouched
    native = F.mmduet2(monkeypatch, F.toy_for("mmduet2_3b"), [NO_REPLY, NO_REPLY, final]).stream(
        F.request(kind="yesno"))
    assert native.text == "NO REPLY" and native.info["no_answer"]


# ---------- LiveStar: the answer chat() on a shared session ----------

def test_livestar_answer_turn_readout_and_shared_session(monkeypatch):
    pytest.importorskip("transformers")
    lm, tl = F.toy_for("livestar_8b"), F.Timeline()
    hooks, art = plugin_arm("livestar_8b", "sfix_prs", tl, lm)
    adapter = F.livestar(monkeypatch, lm, [SENTENCE, SENTENCE])
    gen = run(adapter, hooks, F.request())
    assert tl.calls() == ["begin", "attach", "logits_processors", "arm", "disarm", "finish"]
    assert (gen.text, gen.finish) == ("B", "readout")            # a sentence to a letter question: read out
    block = gen.info["plugin"]
    assert block["native"] == "The man walks." and max(block["readout"], key=block["readout"].get) == "B"
    assert block["guard"]["proj"] == proj(lm, adapter.model.answer_prompts[0], art)
    prompt = adapter.model.answer_prompts[0].shape[1]
    assert tl.forwards(inside=True) == answer_turn(prompt, 4, steer_prefill=True)
    assert tl.forwards(inside=False) == [(3, ())] * 2                           # the narration, untouched
    second = run(adapter, hooks, F.request(sample_id="q2"))                    # same video: same session
    assert F.FakeSession.created == 1 and second.info["session_shared"] and second.text == "B"
    assert tl.calls()[6:] == ["begin", "logits_processors", "arm", "disarm", "finish"]       # attached once


# ---------- MMDuet: the first _generate_response after the question, LM-head logits ----------

def test_mmduet_answer_turn(monkeypatch):
    lm, tl = F.toy_for("mmduet_7b"), F.Timeline()
    hooks, art = plugin_arm("mmduet_7b", "sfix_prs", tl, lm)
    adapter = F.mmduet(monkeypatch, lm, fire={1}, scripts=[[{F.EOS: 50}], [{T["B"]: 50}, {F.EOS: 50}]])
    gen = run(adapter, hooks, F.request())
    assert tl.calls() == ["begin", "attach", "arm", "on_step_logits", "disarm", "finish"]
    assert gen.text == "B" and gen.info["forced"]
    assert gen.info["plugin"]["guard"]["proj"] == proj(lm, F.ids(*F.GEN_PROMPT), art)    # first forward after arm
    assert tl.forwards(inside=True) == answer_turn(len(F.GEN_PROMPT), 1, steer_prefill=True)
    assert tl.forwards(inside=False) == [(2, ()), (2, ()), (3, ()), (2, ()), (3, ()), (2, ())]   # frames, a reply
    #                                     before the question, the question, the last frame: all untouched


def test_mmduet_live_gate_forces_the_end_of_turn_at_the_first_step(monkeypatch):
    art = artefacts.load("mmduet_7b")
    lm, tl = F.toy_for("mmduet_7b"), F.Timeline()
    expected = proj(lm, F.ids(*F.GEN_PROMPT), art)
    hooks = live_arm("mmduet_7b", tl, lm, closed=expected - 0.5, free=expected + 0.5)
    adapter = F.mmduet(monkeypatch, lm, scripts=[[{T["B"]: 50}, {F.EOS: 50}]] * 2)
    gen = run(adapter, hooks, F.request())
    assert (gen.text, gen.finish, gen.info["plugin"]["native"]) == (REFUSAL, "guard", "")
    assert tl.forwards(inside=True) == [(3, (2,))]                             # eos forced: one forward only
    gen = run(adapter, hooks, F.request(kind="open", sample_id="q2"))
    assert gen.text == "B" and not gen.info["plugin"]["guard"]["fired"]


def test_mmduet_failed_attach_installs_no_call_site_and_the_next_request_attaches(monkeypatch):
    """hooks.attach() raising on the first request (a checkpoint of another depth than the artefacts') leaves the
    adapter without call sites and the hooks unattached; the next request attaches, and the call sites exist once."""
    lm = F.toy_for("mmduet_7b")
    hooks = make_hooks("mmduet_7b", "sfix_prs")
    layers = hooks.artefact.manifest["layers"]
    hooks.artefact.manifest = dict(hooks.artefact.manifest, layers=layers + 1)
    adapter = F.mmduet(monkeypatch, lm, scripts=[[{T["B"]: 50}, {F.EOS: 50}]] * 2)
    native = adapter.infer._generate_response
    with pytest.raises(ValueError, match="decoder layers"):
        run(adapter, hooks, F.request())
    assert adapter._attached is None and adapter.infer._generate_response == native and not lm.lm_head._forward_hooks
    assert not hooks.attached and not any(layer._forward_hooks for layer in lm.layers)
    hooks.artefact.manifest = dict(hooks.artefact.manifest, layers=layers)
    gen = run(adapter, hooks, F.request(sample_id="q2"))
    assert gen.text == "B" and gen.info["plugin"]["guard"]["proj"] is not None and hooks.guard is not None
    assert len(lm.lm_head._forward_hooks) == 1 and sum(len(layer._forward_hooks) for layer in lm.layers) == 2


# ---------- MOSS-VL-Realtime: armed when the question drains, LM-head logits on the opening steps ----------

def test_moss_answer_turn_opening_bans_and_readout(monkeypatch):
    script = [{T["<|silence|>"]: 50, T["<|response|>"]: 40}, {T["<|silence|>"]: 50, T[" The"]: 45, T["B"]: 30},
              {T["<|silence|>"]: 50}]
    lm, tl = F.toy_for("moss_vl_rt"), F.Timeline()
    hooks, art = plugin_arm("moss_vl_rt", "sfix_prs", tl, lm)
    adapter = F.moss(monkeypatch, lm, script)
    gen = run(adapter, hooks, F.request())
    assert tl.calls() == ["begin", "attach", "arm", "on_step_logits", "disarm", "finish"]
    assert (gen.text, gen.finish, gen.info["plugin"]["native"]) == ("B", "readout", "The")
    assert gen.info["plugin"]["guard"]["proj"] == proj(lm, adapter.model.answer_prompt, art)
    assert tl.forwards(inside=True) == answer_turn(3, 2, steer_prefill=False)   # PRS layer 19 <= gate layer 24
    assert tl.forwards(inside=False) == [(2, ())] * 2                           # the frames before the question
    native = F.moss(monkeypatch, F.toy_for("moss_vl_rt"), script).stream(F.request())
    assert native.text == "" and native.info["no_answer"]                     # it opens with <|silence|>


# ---------- ThinkStream: the deadline chunk's generate, eager decode under PRS, the deadline fork ----------

def test_thinkstream_answer_turn_decodes_eagerly_under_prs(monkeypatch):
    for variant, eager in (("sfix_prs", True), ("sfix", False)):
        lm, tl = F.toy_for("thinkstream_3b"), F.Timeline()
        hooks, art = plugin_arm("thinkstream_3b", variant, tl, lm)
        adapter, engines = F.thinkstream(monkeypatch, lm, THINK_ANSWER)
        gen = run(adapter, hooks, F.request())
        assert tl.calls() == ["begin", "attach", "arm", "disarm", "finish"]
        assert gen.text == "B" and not gen.info["no_answer"]
        assert gen.info["plugin"]["guard"]["proj"] == proj(lm, F.chunk_ids(2, True), art)
        prompt = F.chunk_ids(2, True).shape[1]
        # the prefill is never steered (PRS layer 14 <= gate layer 20); decode steps only when run eagerly
        assert tl.forwards(inside=True) == answer_turn(prompt, 3 if eager else 0, steer_prefill=False)
        assert engines[0].model.calls == (3 if eager else 0)
        assert tl.forwards(inside=False) == [(F.chunk_ids(0, False).shape[1], ()), (F.chunk_ids(1, False).shape[1], ())]


def test_thinkstream_deadline_readout_from_the_unsteered_fork(monkeypatch):
    lm, tl = F.toy_for("thinkstream_3b"), F.Timeline()
    hooks, art = plugin_arm("thinkstream_3b", "sfix_prs", tl, lm)
    adapter, _ = F.thinkstream(monkeypatch, lm, [{F.EOS: 50}], deadline=True)   # the thought ends before </think>
    forks = []

    def fork(turn):
        forks.append(turn)
        lm.lm_head.script = [{T["C"]: 30}]
        with torch.inference_mode():
            return lm(torch.cat([turn.chunk["input_ids"], F.ids("<think>", "</think>", "<response>")], dim=1))[0, -1]
    adapter._fork = fork
    gen = run(adapter, hooks, F.request())
    assert tl.calls() == ["begin", "attach", "arm", "disarm", "deadline_readout", "finish"]
    assert (gen.text, gen.finish, gen.info["no_answer"]) == ("C", "readout", False) and "Deadline" in gen.info["failure"]
    prompt = F.chunk_ids(2, True).shape[1]
    assert tl.forwards(inside=False)[-1] == (prompt + 3, ())                   # the fork: after disarm, unsteered
    expected = proj(lm, F.chunk_ids(2, True), art)                             # and never read by the gate
    assert gen.info["plugin"]["guard"]["proj"] == expected == round(hooks.guard.projection(), 3)
    gen = run(adapter, hooks, F.request(kind="open", sample_id="q2"))          # open questions: no readout
    assert gen.text == "" and gen.info["no_answer"] and len(forks) == 1


# ---------- inert arms: the native row ----------

def build(child, monkeypatch, lm):
    if child == "aura_8b":
        return F.aura_hf(monkeypatch, lm, [{T["C"]: 40}, {F.EOS: 50}])
    if child == "mmduet2_3b":
        return F.mmduet2(monkeypatch, lm, [NO_REPLY, NO_REPLY, [{T["B"]: 50}, {F.EOS: 50}]])
    if child == "livestar_8b":
        return F.livestar(monkeypatch, lm, [SENTENCE])
    if child == "mmduet_7b":
        return F.mmduet(monkeypatch, lm, fire={1}, scripts=[[{F.EOS: 50}], [{T["B"]: 50}, {F.EOS: 50}]])
    if child == "moss_vl_rt":
        return F.moss(monkeypatch, lm, [{T["<|response|>"]: 50}, {T["B"]: 50}, {T["<|silence|>"]: 50}])
    return F.thinkstream(monkeypatch, lm, THINK_ANSWER)[0]


@pytest.mark.parametrize("child", ["aura_8b", "mmduet2_3b", "livestar_8b", "mmduet_7b", "moss_vl_rt", "thinkstream_3b"])
def test_an_inert_arm_reproduces_the_native_row(monkeypatch, child):
    """The plugin attached with nothing switched on (arm `inert`; AURA also `hf`) changes nothing but adds
    info.plugin = {native}."""
    if child in ("aura_8b", "mmduet2_3b", "livestar_8b"):
        pytest.importorskip("transformers")
    native = build(child, monkeypatch, F.toy_for(child)).stream(F.request())
    for variant in ("inert", "hf") if child == "aura_8b" else ("inert",):
        tl = F.Timeline()
        gen = run(build(child, monkeypatch, F.toy_for(child)), tl.spy(make_hooks(child, variant)), F.request())
        assert {"begin", "attach", "arm", "disarm", "finish"} <= set(tl.calls())
        assert (gen.text, gen.tokens, gen.finish) == (native.text, native.tokens, native.finish)
        info = {k: v for k, v in gen.info.items() if k not in ("plugin", "session_shared")}
        assert info == native.info and gen.info["plugin"] == {"native": native.text}
