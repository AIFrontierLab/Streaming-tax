"""The paper's prompt-robustness specification and frozen ids (configs/generalization/robustness.yaml,
paper/frozen/robustness/), and, with the prepared Video-MME and MMBench data, the regeneration of the paper's items
and deterministic edits; with the fetched PromptBench source, the inlined StressTest sentence."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import re

import pytest

from streamreplace import paths
from streamreplace.data.samples import samples_file
from streamreplace.generalization.robustness import edits, instruction, make, perturb

REPO = Path(__file__).resolve().parents[3]
FROZEN = REPO / "paper" / "frozen" / "robustness"
DETERMINISTIC = ["typo", "stress", "irrelevant", "ins_typo", "ins_para", "ins_irrelevant", "ins300_typo",
                 "ins300_para", "ins300_irrelevant"]
REASONS = re.compile(r"^(judged not equivalent|rules: [a-z ,]+)$")


@pytest.fixture(scope="module")
def spec():
    return make.load_spec(REPO / "configs")


def frozen(name: str) -> dict:
    return json.loads((FROZEN / name).read_text(encoding="utf-8"))


def test_paper_specification(spec):
    assert spec["seed"] == 0 and spec["datasets"] == {"videomme": 300, "mmbench": 300}
    assert spec["pairs"] == ["p1", "p2", "p3", "p4", "p5"]
    assert make.families(spec) == {"typo": ["typo"], "para": ["para"], "distraction": ["stress", "irrelevant"]}
    assert make.instruction_conditions(spec) == ["ins_typo", "ins_para", "ins_irrelevant", "ins300_typo",
                                                 "ins300_para", "ins300_irrelevant"]
    assert make.stem_conditions(spec) == ["clean", "typo", "para", "stress", "irrelevant"]


def test_instruction_rewrites_pass_the_paraphrase_rule_checks(spec):
    """The hand-written instruction rewrites (ins_para, reused by ins300_para) pass the rule checks every accepted
    stem paraphrase passed (real rewording, numbers, length) and keep the demand (the letter, directly)."""
    official = instruction.instruction_span("mmbench")
    texts = spec["conditions"]["ins_para"]["texts"]
    assert texts == spec["conditions"]["ins300_para"]["texts"] and len(set(texts)) == 8
    for text in texts:
        assert perturb.paraphrase_problems(official, text, "", spec["paraphrase"]["min_change"]) == []
        assert "letter" in text and "direct" in text.lower()


def test_frozen_items_are_ids_only_and_consistent(spec):
    items = frozen("items.json")
    assert items["seed"] == spec["seed"] and {d: len(v) for d, v in items["subset"].items()} == {"videomme": 300,
                                                                                                 "mmbench": 1118}
    for dataset, ids in items["subset"].items():
        assert ids == sorted(ids) and items["ids_sha256"][dataset] == hashlib.sha256("\n".join(ids).encode()).hexdigest()
        assert all(re.fullmatch(r"\d{3}-\d|\d+", i) for i in ids)
    assert set(items["nested"]) == set(make.instruction_conditions(spec))
    for condition, subset in items["nested"].items():
        assert set(subset["mmbench"]) <= set(items["subset"]["mmbench"])
        assert len(subset["mmbench"]) == (373 if condition.startswith("ins_") else 1118)
    assert len({int(i) % 1_000_000 for i in items["nested"]["ins_typo"]["mmbench"]}) == 100


def test_frozen_edit_digests_hold_no_benchmark_text(spec):
    digests = frozen("edits.json")
    assert digests["file"] == edits.OVERRIDES_FILE and digests["seed"] == spec["seed"]
    assert list(digests["conditions"]) == [c for c in spec["conditions"] if c != "clean"]
    for condition, entry in digests["conditions"].items():
        assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
        assert entry["rows"] == (373 if condition.startswith("ins_") else 1118 if condition.startswith("ins300")
                                 else 1418)
        for units in entry["skipped"].values():
            assert all(REASONS.match(reason) for reason in units.values())
    para = digests["conditions"]["para"]["skipped"]
    assert len(para["videomme"]) == 6 and len(para["mmbench"]) == 15          # 294 + 285 units remain
    assert all(not entry["skipped"] for c, entry in digests["conditions"].items() if c != "para")


def is_prepared(dataset: str) -> bool:
    return samples_file(dataset).is_file()


@pytest.mark.requires_data("videomme", "mmbench")
def test_paper_items_and_deterministic_edits_regenerate(spec, tmp_path):
    """From the prepared data: the item selection gives the frozen ids, and the deterministic edits (every condition
    but the paraphrases) give override files byte-identical to the paper's (sha256)."""
    if not (is_prepared("videomme") and is_prepared("mmbench")):
        pytest.skip("Video-MME and MMBench are not prepared under the data root")
    exp = make.Experiment(spec, tmp_path / "robustness", {}, paths.current())
    make.make_subset(exp)
    assert make.compare_items(exp, frozen("items.json")) == []
    edits.make_edits(exp, only=DETERMINISTIC)
    result = edits.compare_digests(exp, {c: v for c, v in frozen("edits.json")["conditions"].items()
                                         if c in DETERMINISTIC})
    assert result == {c: "identical" for c in DETERMINISTIC}


@pytest.mark.requires_data("videomme", "mmbench")
@pytest.mark.parametrize("dataset", ["videomme", "mmbench"])
def test_edited_spans_occur_once_in_every_prepared_prompt(dataset):
    """The override hook needs each stem (and MMBench's instruction) exactly once in the rendered prompt: the
    instruction, options and hint must not repeat it."""
    if not is_prepared(dataset):
        pytest.skip(f"{dataset} is not prepared under the data root")
    from streamreplace.benchmarks import registry
    from streamreplace.data.samples import load_samples
    bench, unit = registry.get(dataset), make.DATASETS[dataset]
    bad = []
    for s in load_samples(dataset):
        prompt = bench.prompt(bench.view(s))
        spans = [unit.stem(s)] + ([unit.instruction()] if unit.instruction else [])
        bad += [s.id for span in spans if prompt.count(span) != 1]
    assert not bad, f"{len(bad)} samples, e.g. {bad[:5]}"


@pytest.mark.requires_upstream("promptbench")
def test_inlined_stress_sentence_equals_promptbench():
    """perturb.STRESS_TEST is the third StressTestTransformation text of the pinned PromptBench source."""
    from streamreplace.models.upstream import upstream_path
    source = upstream_path("promptbench/promptbench/prompt_attack/transformations.py", require=False)
    if not source.is_file():
        pytest.skip("PromptBench is not fetched (streamreplace fetch-upstream promptbench)")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "StressTestTransformation")
    texts = next(sub.value for sub in ast.walk(node) if isinstance(sub, ast.Assign)
                 and isinstance(sub.targets[0], ast.Name) and sub.targets[0].id == "texts")

    def literal(element):
        if isinstance(element, ast.BinOp) and isinstance(element.op, ast.Mult):      # " and true is true "*5
            return ast.literal_eval(element.left) * ast.literal_eval(element.right)
        return ast.literal_eval(element)
    assert [literal(e) for e in texts.elts][2] == perturb.STRESS_TEST
