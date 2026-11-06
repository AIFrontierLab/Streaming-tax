"""report -> stats -> tables on the synthetic pair, and the CLI wiring (analysis/pipeline.py, cli/cmd_*.py)."""
from __future__ import annotations

import json

import pytest

from analysis_fakes import N, write_job
from streamreplace.analysis import pipeline


def test_report_writes_metrics_results_and_pairs(synthetic):
    root, cat, prof = synthetic
    files = pipeline.report(root, prof, cat)
    jobs = {(j["dataset"], j["model"], j["mode"]): j for j in json.loads((root / "metrics.json").read_text())["jobs"]}
    assert len(jobs) == 8
    assert jobs[("img", "par", "offline")]["metrics"]["acc"] == 80.0
    assert jobs[("img", "kid", "offline")]["truncated"] == 5
    stream = jobs[("vid", "kid", "stream")]
    assert (stream["no_answer"], stream["forced"], stream["done"], stream["expected"]) == (2, 1, N, N)
    assert jobs[("saf", "kid", "stream")]["errors"] == 1 and jobs[("saf", "kid", "stream")]["done"] == N - 1
    text = (root / "RESULTS.md").read_text()
    assert text.startswith("# Results (full)") and "## p: parent `par`" in text
    assert "| img | acc | 80.0 | 75.0 (-5.0) |" in text                        # parent, child (offline), stream
    assert (root / "pairs.csv").read_text().startswith("pair,dataset,id,parent,candidate")
    assert set(files) == {"metrics", "results", "pairs"}


def test_stats_writes_contrasts_csv_and_gaps_md(synthetic):
    root, cat, prof = synthetic
    doc, files = pipeline.stats(root, prof, cat, overlay=False)
    assert len(doc["rows"]) == 7 and doc["overlay"] is None
    assert doc["checks"]["official_mismatches"] == []
    assert doc["summary"]["eq1_max_residual"] < 1e-9
    assert (root / "analysis" / "no-overlay" / "contrasts.json").is_file()
    csv_text = (root / "analysis" / "no-overlay" / "contrasts.csv").read_text().splitlines()
    assert csv_text[0].startswith("dataset,metric,family") and len(csv_text) == 8
    md = (root / "analysis" / "no-overlay" / "GAPS.md").read_text()
    assert "## Eq. 1 gap table" in md and "| Kid | vid | acc |" in md
    flags = {(r["dataset"], r["gap"]): r["flags"] for r in doc["rows"]}
    assert flags[("img", "C-P")] == ["invalid"]


def test_stats_leaves_out_a_job_without_counted_rows(synthetic, capsys):
    """Rows waiting for scoring (here every row of the video stream, as before `score --judge`) do not stop `stats`:
    the contrasts that need the job are left out like a missing job's and listed under checks.uncounted."""
    root, cat, prof = synthetic
    scores = root / "vid" / "kid.stream.scores.jsonl"
    rows = [json.loads(line) for line in scores.read_text().splitlines()]
    scores.write_text("".join(json.dumps({**r, "pred": None, "parse": None, "correct": None}) + "\n" for r in rows))
    doc, files = pipeline.stats(root, prof, cat, overlay=False)
    assert [(r["dataset"], r["gap"]) for r in doc["rows"]] == [("img", "C-P"), ("vid", "C-P"), ("saf", "C-P"),
                                                               ("saf", "S-C"), ("saf", "S-P")]
    left_out = doc["checks"]["uncounted"]
    assert len(left_out) == 2 and left_out[0].startswith("vid acc kid S-C: kid.offline (40 counted rows) and "
                                                         "kid.stream (0) share no counted row")
    assert "| Kid | vid | acc |" in (root / "analysis" / "no-overlay" / "GAPS.md").read_text()
    from streamreplace.cli import cmd_stats
    assert "checks.uncounted" in cmd_stats.__doc__


def test_gaps_summary_counts_follow_the_roster(synthetic):
    from streamreplace.analysis.render.gaps_md import summary_lines
    root, cat, prof = synthetic
    doc, _ = pipeline.stats(root, prof, cat, overlay=False)
    lines = summary_lines(doc["rows"])
    assert lines[2].startswith("- Video C-P (vid acc x 1): ")
    assert lines[-1].startswith("- Trust, broad families (safety parent 2, safety S-C 1): ")
    md = (root / "analysis" / "no-overlay" / "GAPS.md").read_text()
    assert "Holm over the 1 headline contrasts" in md
    image_only = [r for r in doc["rows"] if r["dataset"] == "img"]
    assert summary_lines(image_only)[0] == "- Eq. 1 additivity: no metric with all three gaps."


def test_written_tables_are_checked_cell_by_cell(synthetic, tmp_path):
    root, cat, prof = synthetic
    pipeline.report(root, prof, cat)
    pipeline.stats(root, prof, cat, overlay=False, out=root / "analysis")
    pipeline.tables(root, prof, paper=True)
    from streamreplace.analysis.render.fig_gaps import printed
    from streamreplace.analysis.render.latex import tex_cells
    cells = tex_cells((root / "tables" / "eval_video.tex").read_text())
    shown = printed(json.loads((root / "tables" / "gaps_figure.json").read_text()))
    figure = {"/".join(k): {"printed": v} for k, v in shown.items()}
    expected = tmp_path / "tables.json"
    expected.write_text(json.dumps({"tables": {"eval_video": cells, "eval_robustness": [["x"]]}, "figure": figure}))
    found = pipeline.check_tables(root / "tables", expected)
    assert found["differences"] == [] and found["missing"] == ["eval_robustness"]
    assert found["cells"] == sum(map(len, cells)) + len(figure) and len(cells) == 1
    changed = [list(row) for row in cells]
    changed[0][-1] = "$+9.9$"
    expected.write_text(json.dumps({"tables": {"eval_video": changed}}))
    assert pipeline.check_tables(root / "tables", expected)["differences"] == [
        f"eval_video / {cells[0][0]} / S: {cells[0][-1]} vs $+9.9$"]          # row label / column header


def test_stats_with_the_rerun_overlay_needs_the_rerun_rows(synthetic, tmp_path):
    root, cat, prof = synthetic
    (tmp_path / "capped.json").write_text(json.dumps({"pairs": {"p": {"img": {"par": {"cap": 64, "ids": ["img-004"]},
                                                                               "kid": {"cap": 64, "ids": []}}}}}))
    with pytest.raises(FileNotFoundError):
        pipeline.stats(root, prof, cat)
    for job in ("par.offline", "kid.offline"):
        write_job(root / "rerun_capped" / "p", "img", job, lambda k: True)
    doc, _ = pipeline.stats(root, prof, cat)
    img = doc["rows"][0]
    assert doc["overlay"]["name"] == "truncation_rerun" and doc["overlay"]["rows"] == "rerun_capped"
    assert img["a"] == pytest.approx(80.0) and img["b"] == pytest.approx(77.5)    # img-004 now right for the child
    assert [m["model"] for m in doc["merged_truncation"]] == ["kid", "par"]


def test_tables_render_from_the_written_files(synthetic):
    root, cat, prof = synthetic
    pipeline.report(root, prof, cat)
    pipeline.stats(root, prof, cat, overlay=False, out=root / "analysis")
    result = pipeline.tables(root, prof, paper=True)
    written = {name for name in result["files"]}
    assert {"gaps.tex", "gaps.json", "gaps_figure.json", "master_cp.tex", "master_sp.tex", "eval_image.tex",
            "eval_video.csv"} <= written
    assert "eval_robustness, eval_instruction" in result["skipped"]
    master = (root / "tables" / "master_cp.tex").read_text()
    assert "tab:master-cp" in master
    gaps = json.loads((root / "tables" / "gaps.json").read_text())
    assert gaps["children"] == ["kid"]


def test_tables_without_inputs_says_what_to_run(tmp_path, synthetic):
    _, _, prof = synthetic
    result = pipeline.tables(tmp_path / "empty", prof)
    assert result["files"] == {} and "streamreplace stats" in " ".join(result["skipped"].values())


def test_cli_commands_are_registered():
    from streamreplace.cli.main import build_parser
    parser = build_parser()
    for argv in (["report"], ["stats", "--no-overlay"], ["tables", "--paper"], ["rerun-capped", "select"],
                 ["rerun-capped", "run", "--dry-run"], ["rerun-capped", "report"]):
        args = parser.parse_args(argv)
        assert callable(args.func)


def test_profile_from_the_loaded_configuration(tmp_config):
    """The CLI's profile: configs/<dir>/analysis.yaml (with local overrides) and the parents from pairs.yaml;
    relative profile paths resolve against the configuration folder's parent."""
    from streamreplace.analysis.specs import profile_from_config
    section = {"roster": {"children": ["child"]}, "engine": {"reps": 100},
               "metrics": [{"dataset": "demo", "name": "acc", "kind": "acc", "family": "video"}],
               "overlays": {"truncation_rerun": {"pairs": ["p"], "datasets": ["demo"], "ids": "frozen/capped.json"}}}
    config = tmp_config(sections={"analysis": section}, check_registry=False)
    profile = profile_from_config(config)
    assert profile.parents == {"child": "parent"} and profile.reps == 100 and profile.pair_of("child") == "p"
    assert profile.path("frozen/capped.json") == config.config_dir.resolve().parent / "frozen" / "capped.json"
    with pytest.raises(ValueError):
        profile_from_config(tmp_config(check_registry=False))         # no analysis.yaml: no metrics


def test_duration_split_is_skipped_when_its_metric_is_not_in_the_profile(synthetic):
    root, cat, prof = synthetic
    from streamreplace.analysis.arms import Arms
    from streamreplace.analysis.contrasts import duration_split
    prof.duration_split = {"dataset": "videomme", "metric": "acc", "reference": "par.offline", "bins": [[0, 60]]}
    assert duration_split(prof, Arms(root, cat, prof), cat) == []
