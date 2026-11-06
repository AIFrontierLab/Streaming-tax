"""Sample selection of the fine-tuning experiment (`finetune prepare`): the stratified test subset, the training order
with nested budgets, the leakage filters, write-once files, the paper's frozen lists and the control questions."""
from dataclasses import replace
import json

import pytest
from PIL import Image

import finetune_demo
from finetune_helpers import DEMO, write_prepared

from streamreplace.core.io import read_json, write_json
from streamreplace.core.schema import Sample
from streamreplace.generalization.finetune import data


def test_stratified_subset_keeps_shares_and_is_deterministic():
    samples = [Sample(f"{m}{i}", "q", "A", "mcq", {"A": "a", "B": "b"}, metadata={"source_map": m})
               for m, n in (("x", 600), ("y", 300), ("z", 100)) for i in range(n)]
    chosen = data.stratified(samples, 100, ("source_map",), seed=0)
    assert len(chosen) == 100 and sum(c.startswith("x") for c in chosen) == 60 and sum(c.startswith("z") for c in chosen) == 10
    assert chosen == data.stratified(list(reversed(samples)), 100, ("source_map",), seed=0)
    assert data.stratified(samples[:50], 100, (), 0) == sorted(s.id for s in samples[:50])


def test_largest_remainder_gives_the_leftover_seats_to_the_largest_remainders():
    samples = [Sample(f"{m}{i}", "q", "A", "open", metadata={"k": m}) for m, n in (("a", 5), ("b", 3), ("c", 2))
               for i in range(n)]
    chosen = data.stratified(samples, 3, ("k",), seed=0)          # quotas 1.5, 0.9, 0.6 -> 1, 1, 1
    assert sorted(c[0] for c in chosen) == ["a", "b", "c"]


def test_training_order_interleaves_and_prefixes_nest():
    samples = [Sample(f"{p}{i}", "q", "a", "open", metadata={"subset": p}) for p in ("action", "scenery") for i in range(50)]
    order = data.ordered(samples, "subset", seed=0)
    assert [s.metadata["subset"] for s in order[:6]] == ["action", "scenery"] * 3
    assert order == data.ordered(list(reversed(samples)), "subset", seed=0)
    plain = data.ordered(samples, None, seed=0)
    assert [s.id for s in plain] == sorted((s.id for s in samples), key=lambda i: data._key(0, i))


def test_prepare_writes_the_three_files_once(exp):
    record = data.prepare(exp, DEMO, workers=1)
    assert record["test"] == 24 and record["test_subset"] == 12 and record["train"] == 60
    assert record["dropped_group"] == 3 and record["pool"] == 57 and record["budget"] == 12
    subset, order = read_json(exp.root / "subset.json")[DEMO], read_json(exp.root / "train_order.json")[DEMO]
    assert len(subset) == 12 and subset == sorted(subset)
    assert sum(int(i[1:]) < 16 for i in subset) == 8               # category shares: 16 of 24 in k1
    assert len(order) == 12 and not {f"t{i:03d}" for i in range(3)} & set(order)   # case c0 is a test case
    assert [finetune_demo.training()[int(i[1:])].metadata["subset"] for i in order[:4]] == ["action", "scenery"] * 2
    selection = read_json(exp.root / "selection.json")[DEMO]
    assert selection["test_sha256"] == data._digest(subset) and selection["train_sha256"] == data._digest(order)
    assert data.selection(exp.root, DEMO) == (order, selection)
    assert data.prepare(exp, DEMO, workers=1)["train_sha256"] == selection["train_sha256"]      # re-run: verified


def test_a_changed_selection_is_refused_and_redo_only_before_use(exp, demo):
    data.prepare(exp, DEMO, workers=1)
    renamed = [replace(s, id="u" + s.id[1:]) for s in finetune_demo.training(80)]   # another training pool
    write_prepared(demo, train=renamed)
    with pytest.raises(ValueError, match="fixed once written"):
        data.prepare(exp, DEMO, workers=1)
    (exp.root / DEMO).mkdir()
    (exp.root / DEMO / f"par@{DEMO}-0.offline.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="already use"):
        data.prepare(exp, DEMO, workers=1, redo=True)
    (exp.root / DEMO / f"par@{DEMO}-0.offline.jsonl").unlink()
    record = data.prepare(exp, DEMO, workers=1, redo=True)
    assert record["train"] == 80 and list((exp.root / "_old").glob("*/selection_ftdemo.json"))


def test_a_tampered_training_order_is_detected(exp):
    data.prepare(exp, DEMO, workers=1)
    orders = read_json(exp.root / "train_order.json")
    orders[DEMO] = list(reversed(orders[DEMO]))
    write_json(exp.root / "train_order.json", orders)
    with pytest.raises(ValueError, match="does not match"):
        data.selection(exp.root, DEMO)


def test_too_few_training_samples_are_refused(exp, demo):
    write_prepared(demo, train=finetune_demo.training(10))
    with pytest.raises(ValueError, match="only 7 training samples"):
        data.prepare(exp, DEMO, workers=1)


def frames(folder, name, values):
    from streamreplace.media.video_meta import write_frame_list
    paths = []
    for i, value in enumerate(values):
        path = folder / "frames" / f"{name}_{i}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.effect_noise((64, 40), 60) if value is None else Image.new("L", (64, 40), value)
        if value is not None:                          # a gradient so the frame has structure (unit-norm thumbnail)
            image = Image.linear_gradient("L").resize((64, 40)).point(lambda p, v=value: (p + v) % 256)
        image.convert("RGB").save(path)
        paths.append(str(path))
    return str(write_frame_list(folder / "media" / f"{name}.frames.json", paths, 1.0))


def test_near_duplicate_training_frames_are_dropped(exp, demo, monkeypatch):
    """CHECK_NEAR_DUPLICATES: a training sample with a frame correlating >= 0.95 with a test frame is dropped."""
    folder = demo.dataset_dir(DEMO)
    test = [Sample(s.id, s.question, s.answer, s.kind, s.choices, video=frames(folder, f"test{n}", [n * 40]),
                   metadata=s.metadata) for n, s in enumerate(finetune_demo.questions(4))]
    train = [Sample(s.id, s.question, s.answer, s.kind, s.choices, metadata=s.metadata,
                    video=frames(folder, f"train{n}", [0 if n == 4 else None])) for n, s in enumerate(finetune_demo.training(20))]
    write_prepared(demo, test=test, train=train)
    monkeypatch.setattr(finetune_demo, "CHECK_NEAR_DUPLICATES", True)
    record = data.prepare(exp, DEMO, workers=1)
    assert record["dropped_near_duplicate"] == 1 and "t004" not in read_json(exp.root / "train_order.json")[DEMO]
    assert record["dropped_group"] == 3 and record["budget"] == 12


def test_the_paper_lists_can_be_used_and_compared(exp, tmp_path):
    frozen = tmp_path / "frozen"
    lists = {"subset": {DEMO: ["q00"]}, "train_order": {DEMO: ["t003"]},
             "selection": {DEMO: {"train_sha256": data._digest(["t003"]), "test_sha256": data._digest(["q00"])}}}
    for name, value in lists.items():
        write_json(frozen / f"{name}.json", value)
    written = data.use_frozen(exp.root, [DEMO], frozen)
    assert len(written) == 3 and data.compare_frozen(exp.root, [DEMO], frozen) == []
    assert data.subset_ids(exp.root, DEMO) == ["q00"] and data.selection(exp.root, DEMO)[0] == ["t003"]
    write_json(frozen / "subset.json", {DEMO: ["q01"]})
    assert data.compare_frozen(exp.root, [DEMO], frozen) == [f"subset.json: {DEMO} differs from the paper's "
                                                              "(paper/frozen/finetune/subset.json)"]
    with pytest.raises(ValueError, match="differs from the paper's selection"):
        data.use_frozen(exp.root, [DEMO], frozen)
    with pytest.raises(ValueError, match="has no"):
        data.use_frozen(exp.root, ["other"], frozen)


def test_the_shipped_paper_lists_are_complete():
    frozen = data.frozen_dir()
    subset, order = read_json(frozen / "subset.json"), read_json(frozen / "train_order.json")
    selection = read_json(frozen / "selection.json")
    assert {d: len(v) for d, v in subset.items()} == {"lingoqa": 500, "openuav_qa": 2000, "eyepcr": 2000}
    assert {d: len(v) for d, v in order.items()} == {"lingoqa": 5000, "openuav_qa": 5000, "eyepcr": 5000}
    for dataset in subset:
        assert selection[dataset]["test_sha256"] == data._digest(subset[dataset])
        assert selection[dataset]["train_sha256"] == data._digest(order[dataset])
    controls = read_json(frozen / data.CONTROLS_FILE)
    assert {d: len(v) for d, v in controls["blank"].items()} == {"openuav_qa": 500, "lingoqa": 500, "eyepcr": 500}
    assert all(set(ids) <= set(subset[d]) for d, ids in controls["blank"].items())
    assert controls["images"] == {"openuav_qa": subset["openuav_qa"]}
    text = "".join((frozen / f).read_text(encoding="utf-8") for f in ("subset.json", "train_order.json", "controls.json"))
    assert "?" not in text                              # ids only, no question text


def test_control_questions_follow_the_stratified_rule(exp):
    data.prepare(exp, DEMO, workers=1)
    blank = data.control_subset(exp, DEMO, 4)
    test = [s for s in finetune_demo.questions() if s.id in set(data.subset_ids(exp.root, DEMO))]
    assert blank == data.stratified(test, 4, ("category",), 0) and set(blank) <= set(data.subset_ids(exp.root, DEMO))
    assert data.control_subset(exp, DEMO, None) == data.subset_ids(exp.root, DEMO)
    path = data.write_subset(exp.folder("blank"), {DEMO: blank})
    assert json.loads(path.read_text()) == {DEMO: blank}
    with pytest.raises(ValueError, match="differs from the recorded ids"):
        data.write_subset(exp.folder("blank"), {DEMO: blank[:2]})


def test_control_folders_are_compared_with_the_papers_questions(exp, tmp_path):
    frozen = tmp_path / "frozen"
    write_json(frozen / data.CONTROLS_FILE, {"blank": {DEMO: ["q00", "q01"]}, "images": {DEMO: ["q00"]}})
    assert data.compare_controls(exp, frozen) == []                # no control folder written yet
    data.write_subset(exp.folder("blank"), {DEMO: ["q00", "q02"]})
    assert data.compare_controls(exp, frozen) == [f"blank/subset.json: {DEMO} differs from the paper's control "
                                                  f"questions (paper/frozen/finetune/{data.CONTROLS_FILE})"]


def test_a_missing_subset_is_explained(exp):
    with pytest.raises(ValueError, match="run: streamreplace finetune prepare"):
        data.subset_ids(exp.root, DEMO)
