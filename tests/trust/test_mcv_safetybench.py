"""MCV-SafetyBench: question <-> video mapping, official attack prompts and policy bullets (read from the fetched
MCV_Jailbreak code), the RJScore judge input built from Video-SafetyBench's template, and the metrics. The policy
bullets are matched by a few identifying words and compared by sha256: MCV_Jailbreak has no licence file, so its text
is not copied here."""
import hashlib
import json
import re

import pytest

from streamreplace.benchmarks.registry import get
from streamreplace.benchmarks.safety import mcv_safetybench as mcv
from streamreplace.benchmarks.safety import video_safetybench as vsb

from .helpers import class_constants, metric_row, prepared_samples, require_upstream

PER_CATEGORY = {"01": 85, "02": 130, "03": 40, "04": 131, "05": 113, "06": 143, "07": 93, "08": 127, "09": 133,
                "10": 109, "11": 139, "12": 92, "13": 125}                   # MCV paper Table 1


@pytest.mark.requires_upstream("mcv_safetybench")
def test_question_json_is_reindexed():
    """Why prepare pairs the k-th video with question k: the released JSONs are keyed 0..N-1 (N = videos per
    category), while the file numbers run up to the MM-SafetyBench position (e.g. Illegal_00097_)."""
    code = require_upstream("mcv_safetybench")
    files = sorted((code / mcv.QUESTIONS).glob("*.json"))
    assert len(files) == 13
    for path in files:
        keys = sorted(map(int, json.loads(path.read_text(encoding="utf-8"))))
        assert keys == list(range(len(keys))) and len(keys) == PER_CATEGORY[path.stem[:2]]


@pytest.mark.requires_upstream("mcv_safetybench", "video_safetybench")
def test_policy_lines_are_the_category_bullets():
    """policy.txt's 15 bullets follow the MM-SafetyBench categories (bullet 2, CSAM, matches none); Fraud also gets the
    disinformation / spam / pseudo-pharmaceuticals bullet. Each description is the bullet text verbatim."""
    code = require_upstream("mcv_safetybench")
    require_upstream("video_safetybench")
    lines = [l.removeprefix("• ") for l in (code / mcv.POLICY).read_text(encoding="utf-8").splitlines()
             if l.startswith("• ")]
    expected = {"01-Illegal_Activity": ("Illegal activity:",), "02-HateSpeech": ("Generation of hateful",),
                "03-Malware_Generation": ("Generation of malware",), "04-Physical_Harm": ("high risk of physical harm",),
                "05-EconomicHarm": ("high risk of economic harm",), "06-Fraud": ("Fraudulent or deceptive", "Disinformation"),
                "07-Sex": ("Adult content",), "08-Political": ("Political campaigning",),
                "09-Privacy_Violence": ("violates people’s privacy",), "10-Legal_Opinion": ("unauthorized practice of law",),
                "11-Financial_Advice": ("tailored financial advice",), "12-Health_Consultation": ("health condition",),
                "13-Gov_Decision": ("government decision-making",)}
    used = set()
    for path in (code / mcv.QUESTIONS).glob("*.json"):
        scenario, description = mcv.policy(None, path.stem)
        assert scenario == path.stem.split("-", 1)[1]
        parts = description.split("\n- ")
        assert len(parts) == len(expected[path.stem]) and all(p in lines for p in parts)
        assert all(key in part for key, part in zip(expected[path.stem], parts))
        used |= set(parts)
    assert set(lines) - used == {lines[1]} and lines[1].startswith("Child Sexual Abuse Material")
    # the cache keys of the queries both benchmarks share stay apart: no MCV scenario is a VSB scenario
    assert not {mcv.policy(None, p.stem)[0] for p in (code / mcv.QUESTIONS).glob("*.json")} & set(vsb.policies())


@pytest.mark.requires_upstream("mcv_safetybench")
def test_official_prompts_are_the_mars_attacks():
    require_upstream("mcv_safetybench")
    explicit = mcv.official_prompt(None, "explicit", "List steps to do X.")
    implicit = mcv.official_prompt(None, "implicit", "List steps to do X.")
    for text in (explicit, implicit):
        assert text.startswith("I am an observer from Mars, sent to help Earth.") and text.endswith("Step 3:")
    assert "\n\nThe topic is as follows:\n\nTopic: List steps to do X.\n\n" in explicit
    assert "List steps to do X." not in implicit


def test_metrics_are_video_safetybench_metrics_per_attack():
    rows = [metric_row("explicit_06_010", False, {"attack": "explicit", "category": "06-Fraud"}, pred=3.0, silent=False),
            metric_row("explicit_10_102", True, {"attack": "explicit", "category": "10-Legal_Opinion"}, pred=1.0,
                       silent=True),
            metric_row("implicit_06_010", True, {"attack": "implicit", "category": "06-Fraud"}, pred=2.0, silent=False)]
    m = mcv.metrics(rows)
    assert list(m)[:2] == ["asr_explicit", "asr_implicit"] and m["asr_explicit"] == 50.0 and m["asr_implicit"] == 0.0
    assert m["rjscore_explicit"] == 2.0 and m["silent_explicit"] == 50.0 and m["asr_answered_explicit"] == 100.0
    assert [k for k in m if k.startswith("cat/explicit/")] == ["cat/explicit/06-Fraud", "cat/explicit/10-Legal_Opinion"]
    assert mcv.THRESHOLD == vsb.THRESHOLD and mcv.KEY_POLICY is True


@pytest.mark.requires_data("mcv_safetybench")
@pytest.mark.requires_upstream("mcv_safetybench", "video_safetybench")
def test_prepared_mapping_prompts_and_judge_input():
    samples = {s.id: s for s in prepared_samples("mcv_safetybench")}
    require_upstream("mcv_safetybench")
    code = require_upstream("video_safetybench")
    assert len(samples) == mcv.EXPECTED_COUNT
    for attack in mcv.ATTACKS:
        counts = {}
        for s in samples.values():
            if s.metadata["attack"] == attack:
                counts[s.metadata["category"][:2]] = counts.get(s.metadata["category"][:2], 0) + 1
        assert counts == PER_CATEGORY
    named = 0
    for s in samples.values():
        other = samples[s.id.replace(s.metadata["attack"], next(a for a in mcv.ATTACKS if a != s.metadata["attack"]))]
        assert other.metadata["query"] == s.metadata["query"]
        assert s.metadata["harmful_intention"] == s.metadata["changed_question"]     # RJScore REQUEST
        assert s.question == mcv.official_prompt(None, s.metadata["attack"], s.metadata["harmful_intention"])
        assert ("Topic: " + s.metadata["harmful_intention"] in s.question) == (s.metadata["attack"] == "explicit")
        assert (s.metadata["scenario"], s.metadata["scenario_description"]) == mcv.policy(None, s.metadata["category"])
        assert get("mcv_safetybench").prompt(s) == s.question
        slug = re.match(r"Category\d+_\d+_(.+)\.mp4_\d+_\.mp4$", s.metadata["file"])
        named += bool(slug)
        if slug:
            assert mcv._slug(slug.group(1)) == mcv._slug(s.metadata["key_phrase"])
    assert named == 1347
    smoke = get("mcv_safetybench").smoke(list(samples.values()))
    assert len(smoke) == len(samples) and smoke[1] == smoke[0].replace("explicit", "implicit")
    # the judge input: the category's MCV bullet(s) as the policy line, the query as REQUEST (also implicit)
    attrs = class_constants(code / "eval/evaluator.py", "RJScore_Judge")
    meta = samples["implicit_06_010"].metadata
    text = attrs["RULE_PROMPT"].format(scenario=meta["scenario"], scenario_description=meta["scenario_description"]) + \
        attrs["USER_PROMPT"].format(goal=meta["harmful_intention"], response="Step 1: ...")
    line = f"\n- {meta['scenario']}: {meta['scenario_description']}\n\n"   # Fraud: its bullet + the disinformation one
    assert meta["scenario"] == "Fraud" and meta["scenario_description"].count("\n- ") == 1 and line in text
    assert hashlib.sha256(line.encode("utf-8")).hexdigest() == \
        "40ac9f1230b44d8abff28751a9de9c01dab7922db6e9c718668ab0b569d6abf7"
    assert meta["harmful_intention"] in text
    assert meta["harmful_intention"] not in samples["implicit_06_010"].question
