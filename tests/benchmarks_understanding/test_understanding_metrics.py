"""metrics() of the understanding benchmarks on small synthetic row sets. Expected dicts (values AND key order: the
first key is the headline the paper reports) are what the code of the paper's runs returned for the same rows."""
from __future__ import annotations

import pytest

from streamreplace.benchmarks.registry import get


def row(i, correct, kind="mcq", **meta):
    return {"id": i, "correct": correct, "kind": kind, "metadata": meta}


def items(d: dict) -> list:
    return list(d.items())


def test_mme_scores_acc_plus_acc_plus_per_subtask():
    rows = [row("a#1", True, "yesno", image="a", category="existence"),
            row("a#2", True, "yesno", image="a", category="existence"),
            row("b#1", True, "yesno", image="b", category="existence"),
            row("b#2", False, "yesno", image="b", category="existence"),
            row("c#1", False, "yesno", image="c", category="code_reasoning"),
            row("c#2", False, "yesno", image="c", category="code_reasoning"),
            row("d#1", True, "yesno", image="d", category="count")]       # an image with one question: no acc+
    assert items(get("mme").metrics(rows)) == [
        ("perception", 225.0), ("cognition", 0.0), ("task/code_reasoning", 0.0), ("task/count", 100.0),
        ("task/existence", 125.0)]


def test_mmbench_circular_accuracy_needs_every_rotation():
    rows = [row("1", True, question=1, rotation=0, rotations=2, l2="x"),
            row("1000001", True, question=1, rotation=1, rotations=2, l2="x"),
            row("2", True, question=2, rotation=0, rotations=3, l2="y"),
            row("1000002", False, question=2, rotation=1, rotations=3, l2="y"),
            row("2000002", True, question=2, rotation=2, rotations=3, l2="y"),
            row("3", False, question=3, rotation=0, rotations=2, l2="y")]      # incomplete question: not counted
    assert items(get("mmbench").metrics(rows)) == [
        ("circular_acc", 50.0), ("vanilla_acc", 66.67), ("questions", 2), ("l2/x", 100.0), ("l2/y", 0.0)]


def test_timeblind_scores_complete_instances_only():
    rows = [row(str(k), True, "yesno" if k < 2 else "mcq", instance=0, role=k, type="yes_no") for k in range(4)]
    rows += [row(str(4 + k), k != 2, "mcq", instance=1, role=k, type="multiple_choice") for k in range(4)]
    rows += [row(str(8 + k), True, "mcq", instance=2, role=k, type="multiple_choice") for k in range(3)]
    assert items(get("timeblind").metrics(rows)) == [
        ("I_Acc", 50.0), ("Q_Acc", 75.0), ("V_Acc", 75.0), ("Acc", 87.5), ("instances", 2),
        ("type/multiple_choice", 85.71), ("type/yes_no", 100.0)]
    assert get("timeblind").metrics([]) == {"I_Acc": None}


def test_dynamath_average_and_worst_case():
    rows = [row(f"s1v{v}", True, "numeric", seed="1", subject="algebra") for v in range(10)]
    rows += [row(f"s2v{v}", v != 3, "mcq", seed="2", subject="geometry") for v in range(10)]
    rows += [row("s3v0", True, "text", seed="3", subject="algebra")]               # incomplete seed: not in worst
    assert items(get("dynamath").metrics(rows)) == [
        ("avg_acc", 95.24), ("worst_acc", 50.0), ("complete_seeds", 2), ("kind/mcq", 90.0), ("kind/numeric", 100.0),
        ("kind/text", 100.0), ("subject/algebra", 100.0), ("subject/geometry", 90.0)]


def test_videomme_accuracy_over_all_questions_and_official_ratio():
    """acc divides by all questions; acc_official by the answered ones (upstream's ratio). Rows without a video
    path skip the duration split (len/...)."""
    rows = [dict(row("1", True, duration="short", task="T1", domain="D1"), answered=True),
            dict(row("2", True, duration="short", task="T1", domain="D2"), answered=True),
            dict(row("3", False, duration="short", task="T2", domain="D1"), answered=True),
            dict(row("4", False, duration="short", task="T2", domain="D1"), answered=False)]
    assert items(get("videomme").metrics(rows)) == [
        ("acc", 50.0), ("acc_official", 66.67), ("answered", 75.0), ("duration/short", 50.0), ("task/T1", 100.0),
        ("task/T2", 0.0), ("domain/D1", 33.33), ("domain/D2", 100.0)]


@pytest.mark.parametrize("name, rows, expected", [
    ("tomato", [row("count:a", True, reasoning="count", demonstration="human"),
                row("count:b", False, reasoning="count", demonstration="object"),
                row("rotation:c", True, reasoning="rotation", demonstration="human")],
     [("acc", 66.67), ("reasoning/count", 50.0), ("reasoning/rotation", 100.0), ("demo/human", 100.0),
      ("demo/object", 0.0)]),
    ("mathvista", [row("1", True, task="geometry"), row("2", False, task=None)],
     [("acc", 50.0), ("task/None", 0.0), ("task/geometry", 100.0)]),
    ("mmmu", [row("1", True, subject="Art"), row("2", False, subject="Math"), row("3", True, subject="Math")],
     [("acc", 66.67), ("subject/Art", 100.0), ("subject/Math", 50.0)]),
    ("scienceqa", [row("1", True, subject="Art"), row("2", False, subject="Math"), row("3", True, subject="Math")],
     [("acc", 66.67), ("subject/Art", 100.0), ("subject/Math", 50.0)]),
    ("mmstar", [row("1", True, category="colors"), row("2", False, category="shapes")],
     [("acc", 50.0), ("cat/colors", 100.0), ("cat/shapes", 0.0)]),
    ("visulogic", [row("1", True, category="colors"), row("2", False, category="shapes")],
     [("acc", 50.0), ("cat/colors", 100.0), ("cat/shapes", 0.0)]),
    ("videoreasonbench", [row("1", True, "open", task="cup", dim="order_state"),
                          row("2", False, "open", task="hrd", dim="order_state")],
     [("acc", 50.0), ("task/cup", 100.0), ("task/hrd", 0.0), ("dim/order_state", 50.0)]),
])
def test_accuracy_metrics(name, rows, expected):
    assert items(get(name).metrics(rows)) == expected
