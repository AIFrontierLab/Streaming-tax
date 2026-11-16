"""Renderers: display conventions, Table 2, master tables, figure data, absolute-score and robustness tables."""
from __future__ import annotations

import pytest

from streamreplace.analysis.render import eval_tables, fig_gaps, gaps_json, latex, master, table_gaps
from streamreplace.analysis.specs import load_profile


def test_gaps_summary_of_the_papers_rows_keeps_its_wording():
    """GAPS.md's summary derives its counts from the rows; for the paper's contrasts they give its Holm families."""
    import json
    from streamreplace.analysis.render.gaps_md import summary_lines
    from streamreplace.analysis.specs import repo_root
    rows = json.loads((repo_root() / "paper" / "expected" / "contrasts.json").read_text(encoding="utf-8"))["rows"]
    lines = summary_lines(rows)
    for line, gap in zip(lines[2:5], ("C-P", "S-C", "S-P")):
        assert line.startswith(f"- Video {gap} (Video-MME, TOMATO, TimeBlind I_Acc x 6): ")
    assert lines[-1].startswith("- Trust, broad families (safety parent 51, safety S-C 21, VMDT parent 89, VMDT S-C "
                                "41): ")


def test_display_conventions():
    assert latex.absolute(24.65) == "24.6" and latex.absolute(75.35) == "75.3"   # format() on the binary value
    assert latex.absolute(0.04) == "0.0" and latex.absolute(-0.04) == "0.0" and latex.absolute(None) == "--"
    assert latex.delta(-0.04) == "+0.0" and latex.delta(-2.25) == "-2.2" and latex.delta(3.0) == "+3.0"
    assert latex.master_delta(-0.04) == "+0.0" and latex.master_delta(-0.06) == "-0.1"
    assert latex.round4(1.234567) == 1.2346 and latex.round4(None) is None


def test_tex_cells_reads_rows_and_skips_rules():
    text = latex.tex_table("lr", [latex.tex_row(["A", "B"])],
                           ["\\multicolumn{2}{l}{group} \\\\", latex.tex_row(["x", "$+1.0$"]), "\\midrule",
                            latex.tex_row(["y", "--"])], "caption", "tab:x")
    assert latex.tex_cells(text) == [["x", "$+1.0$"], ["y", "--"]]


def cp_row(child, delta, p_broad=0.5, tost3=False, flags=(), dataset="mathvista", metric="acc", gap="C-P"):
    return {"dataset": dataset, "metric": metric, "child": child, "gap": gap, "delta": delta, "delta_pp": delta,
            "ci95_pp": [delta - 1, delta + 1], "ci90_pp": [delta - 0.8, delta + 0.8], "p": p_broad,
            "p_holm": p_broad, "p_holm_broad": p_broad, "holm_family": "f", "holm_m": 1, "holm_broad_family": "f",
            "holm_broad_m": 1, "tost3": tost3, "a": 50.0, "b": 50.0 + delta, "n": 100, "coverage_a": 100.0,
            "coverage_b": 99.0, "coverage_gap": -1.0, "flags": list(flags), "headline": True, "reading": "",
            "verdict": "inconclusive", "flag_notes": []}


def test_table2_cells_shading_and_qualifications():
    assert table_gaps.cell({"delta": -2.2, "holm_sig": True, "flags": []}) == \
        "\\cellcolor[HTML]{F4DCD9}$\\mathbf{-2.2}$"
    assert table_gaps.cell({"delta": 0.04, "holm_sig": False, "flags": []}) == "$+0.0$"
    assert table_gaps.cell({"delta": -3.9, "holm_sig": True, "flags": ["invalid"]}) == \
        "\\textcolor[HTML]{8A939B}{$-3.9^{\\ddagger}$}"
    assert table_gaps.cell({"delta": 1.0, "flags": ["non_answer", "artifact"]}).count("\\dagger") == 1
    assert table_gaps.cell(None) == "--"


def test_gaps_json_and_table2_from_contrast_rows(tmp_path):
    children = ["a_kid", "b_kid"]
    rows = [cp_row("a_kid", -2.2, p_broad=0.01), cp_row("b_kid", 0.5, tost3=True)]
    robust = tmp_path / "contrasts.csv"
    robust.write_text("pair,child,family,readout,n,value,low,high,p_holm,equivalent\n"
                      "p1,a_kid,typo,acc,600,0.7,-1.0,2.3,1.0,True\n"
                      "p1,a_kid,para,acc,579,0.71,-1.2,2.6,1.0,True\n"
                      "p1,a_kid,distraction,acc,600,0.12,-1.6,1.9,1.0,True\n"
                      "p1,a_kid,typo,prob,600,0.1,-1.0,2.3,1.0,True\n")
    data = gaps_json.build(rows, children, robustness=robust)
    cells = {(c["panel"], c["column"], c["child"]): c for c in data["cells"]}
    assert cells[("offline", "mathvista", "a_kid")]["holm_sig"] is True
    assert cells[("offline", "robust", "a_kid")]["delta"] == 0.12 and cells[("offline", "robust", "a_kid")]["family"] \
        == "distraction"
    assert cells[("offline", "robust", "b_kid")]["delta"] is None
    text = table_gaps.latex(data, {"a_kid": "A", "b_kid": "B"})
    rows_tex = latex.tex_cells(text)
    assert rows_tex[0][0] == "A" and rows_tex[0][2] == "\\cellcolor[HTML]{F4DCD9}$\\mathbf{-2.2}$"
    assert rows_tex[0][-3:] == ["$+0.7$", "$+0.7$", "$+0.1$"]
    assert rows_tex[1][-3:] == ["--", "--", "--"]
    assert len(table_gaps.flat(data)) == 2 * len(table_gaps.COLUMNS)


def test_master_tables_marks():
    r = cp_row("a_kid", -0.04, tost3=True)
    assert master.tex_cell(r) == "$+0.0^{\\approx}$"
    assert master.tex_cell(cp_row("a_kid", -2.2, p_broad=0.01, tost3=True)) == "$\\mathbf{-2.2}$"
    r2 = {**cp_row("a_kid", 1.0), "coverage_gap": 5.0}
    assert master.tex_cell(r2) == "$+1.0$" and master.tex_cell(r2, coverage_marks=True) == "$+1.0^{c}$"
    text = master.latex([cp_row("a_kid", 1.0)], "C-P", ["a_kid"], {"a_kid": "A"})
    assert "tab:master-cp" in text and latex.tex_cells(text) == [["MathVista", "$+1.0$"]]
    md, table = master.markdown([cp_row("a_kid", 1.0)], "C-P", ["a_kid"], {"a_kid": "A"})
    assert "| mathvista · acc | +1.0 |" in md and table.startswith("dataset,metric,child")


def test_figure_data_prints_daggers():
    rows = [cp_row(c, d, dataset="videomme", gap=g, flags=f) for c, d, g, f in
            [("a", -3.0, "C-P", []), ("b", 1.0, "C-P", ["invalid"]), ("a", -5.0, "S-P", ["forced"]),
             ("b", 0.04, "S-P", [])]]
    data = fig_gaps.build(rows, ["a", "b"])
    shown = fig_gaps.printed(data)
    assert shown[("offline", "b", "videomme")] == "$+1.0^{\\ddagger}$"
    assert shown[("stream", "a", "videomme")] == "$-5.0^{\\dagger}$"
    assert shown[("stream", "b", "videomme")] == "$+0.0$" and shown[("offline", "mean", "videomme")] == "$-1.0$"


def metrics_row(dataset, model, mode, value, done=10, expected=10, errors=0, name=None):
    return {"dataset": dataset, "model": model, "mode": mode, "metrics": {name or "acc": value}, "done": done,
            "expected": expected, "errors": errors}


def test_eval_tables_absolute_scores_and_incomplete_coverage():
    profile = load_profile()                                    # the paper profile: roster order, names, daggers
    rows = [metrics_row("videomme", "llava_ov_7b", "offline", 69.33), metrics_row("videomme", "mmduet_7b", "stream",
                                                                                  66.0),
            metrics_row("timeblind", "mmduet_7b", "stream", 2.0, done=2392, expected=2400, errors=8, name="I_Acc"),
            metrics_row("video_safetybench", "aura_8b", "stream", 75.35, name="asr_benign")]
    out = eval_tables.build(rows, profile)
    video = latex.tex_cells(out["eval_video"][0])
    assert video[0][0] == "MMDuet-7B$^\\dagger$" and video[0][1] == "69.3" and video[0][3] == "66.0"
    assert video[0][9] == "2.0$^\\ddagger$"
    assert "eight MMDuet TimeBlind generation errors leave 2,392/2,400 completed queries; this entry has" \
        in out["eval_video"][0]
    safety = {r[0] + r[1]: r for r in latex.tex_cells(out["eval_safety"][0])}
    assert safety["AURA-8BS"][4] == "75.3"
    text, csv_rows, fields = out["eval_image"]
    assert len(csv_rows) == len(profile.models) and "mathvista/acc" in fields


def test_robustness_tables_from_the_robustness_report():
    profile = load_profile()
    family = lambda v, n=300: {"n": n, "delta_rob_pp": v, "delta_rob_ci95": [v - 2, v + 2]}  # noqa: E731
    document = {
        "robustness_by_benchmark": [
            {"dataset": "videomme", "child": "mmduet_7b", "clean_parent": 68.666, "clean_child": 68.0,
             "typo": family(0.6667), "para": family(0.34, 294), "distraction": family(-0.6667)},
            {"dataset": "mmbench", "child": "mmduet_7b", "clean_parent": 82.666, "clean_child": 81.333,
             "typo": family(0.6667), "para": family(1.1, 285), "distraction": family(0.8)}],
        "instruction_robustness": [
            {"conditions": ["ins300_typo"], "units": 300, "rows": []},
            {"conditions": ["ins_typo", "ins_para", "ins_irrelevant"], "units": 100, "rows": [
                {"child": "mmduet_7b",
                 **{k: {"n": n, "delta_rob_pp": 0.0, "delta_rob_ci95": [0.0, 0.0], "p_holm": 1.0,
                        "instruction_parent": 80.0, "instruction_child": 80.0}
                    for k, n in (("typo", 100), ("para", 95), ("irrelevant", 100))}}]}]}
    out = eval_tables.robustness(document, profile)
    rob = latex.tex_cells(out["eval_robustness"])
    assert rob[0] == ["MMDuet-7B", "Video-MME", "68.7", "68.0", "+0.7 [-1.3, +2.7]", "+0.3 [-1.7, +2.3]",
                      "-0.7 [-2.7, +1.3]"]
    assert "(294 Video-MME and 285 MMBench units)" in out["eval_robustness"]
    ins = latex.tex_cells(out["eval_instruction"])
    assert ins[0][:4] == ["MMDuet-7B", "80.0", "80.0", "+0.0 [+0.0, +0.0]"]
    assert "comparison uses 95" in out["eval_instruction"] and "All 3 instruction-level" in out["eval_instruction"]


def test_finetuning_tables_from_the_finetuning_report():
    profile = load_profile()
    acc = lambda base: {"acc@0": base, "acc@1k": base + 10.04, "acc@2k": base + 12.0, "acc@5k": None}  # noqa: E731
    datasets = ("lingoqa", "eyepcr", "openuav_qa")
    accuracy = {d: {eval_tables.finetune_id(m, profile): acc(40.0) for m in profile.models} for d in datasets}
    for d in datasets:
        accuracy[d]["mmduet_7b"] = acc(99.0)           # the unmerged id is not what MMDuet was fine-tuned as
    document = {
        "finetuning_accuracy": accuracy,
        "finetuning_primary": [{"dataset": d, "pair": "p", "child": eval_tables.finetune_id(c, profile),
                                "DeltaGain": "+1.80", "CI": "[-2.33, +5.93]", "p": "0.407", "Holm": "1"}
                               for d in ("lingoqa", "eyepcr") for c in profile.children],
        "finetuning_max_common_budget": [{"dataset": "lingoqa", "n_test": 500}, {"dataset": "eyepcr", "n_test": 2000}]}
    out = eval_tables.finetuning(document, profile)
    rows = latex.tex_cells(out["eval_finetuning"])
    assert len(rows) == len(profile.models)
    assert rows[0] == ["LLaVA-OneVision-7B", "P"] + ["40.0", "50.0", "52.0", "--"] * 3
    assert rows[1] == ["MMDuet-7B", "C"] + ["40.0", "50.0", "52.0", "--"] * 3
    text = out["eval_finetuning"]
    assert r"\multicolumn{4}{c}{LingoQA ($n=500$)}" in text and r"\multicolumn{4}{c}{EyePCR ($n=2{,}000$)}" in text
    assert r"\multicolumn{4}{c}{OpenUAV-QA}" in text                    # no size recorded: none printed
    gains = latex.tex_cells(out["eval_finetuning_gains"])
    assert gains[0] == ["MMDuet-7B"] + ["+1.80", "[-2.33, +5.93]", "1"] * 2 and len(gains) == len(profile.children)
    assert ("all 12 child--dataset comparisons in LingoQA and EyePCR; OpenUAV-QA is excluded."
            in out["eval_finetuning_gains"])
    assert eval_tables.finetuning({}, profile) == {}


def test_common_response_table(tmp_path):
    path = tmp_path / "common_response.csv"
    path.write_text("dataset,split,child,mode,n,parent_rate,child_rate,p\n"
                    "video_safetybench,harmful,mmduet_7b,offline,97,26.804,6.1856,1.0967e-05\n")
    rows = latex.tex_cells(eval_tables.common_response(path, load_profile()))
    assert rows[0] == ["MMDuet-7B", "97", "26.8", "6.2", "1.1e-05"]
    assert rows[1][1:] == ["--"] * 4
