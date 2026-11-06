"""`streamreplace fetch-upstream` and `streamreplace download` from the command line (no network: listings, offline
plans, verification and refusals only)."""
from __future__ import annotations

import json

import pytest

from streamreplace import paths
from streamreplace.cli.main import main


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    for var in ("STREAMREPLACE_HOME", "STREAMREPLACE_PATHS", "STREAMREPLACE_UPSTREAM", "STREAMREPLACE_DATA",
                "STREAMREPLACE_CHECKPOINTS"):
        monkeypatch.delenv(var, raising=False)
    paths.reset()
    yield
    paths.reset()


def test_both_commands_are_registered(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "fetch-upstream" in out and "download" in out


def test_fetch_upstream_list(capsys):
    assert main(["fetch-upstream", "--list"]) == 0
    out = capsys.readouterr().out
    assert "videoreasonbench" in out and "reference" in out and "(+1 patched)" in out


def test_verify_only_on_an_empty_root_fails(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "fetch-upstream", "--verify-only", "tomato", "--json"]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["root"] == str(tmp_path / "upstream")
    assert result["results"][0]["state"] == "missing"


def test_reference_and_unknown_names_are_rejected(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "fetch-upstream", "aura"]) == 2
    assert "provenance only" in capsys.readouterr().err
    assert main(["--home", str(tmp_path), "fetch-upstream", "nope"]) == 2


def test_fetch_from_a_local_checkout(tmp_path, capsys):
    from streamreplace.provisioning import upstream
    entry = upstream.load_manifest()["timeblind"]
    source = tmp_path / "src"
    (source / "timeblind").mkdir(parents=True)
    # the pinned files are not shipped: a wrong copy must be refused, nothing must be written
    for path in entry.files:
        (source / "timeblind" / path).write_text("not the upstream file\n")
    assert main(["--home", str(tmp_path / "h"), "fetch-upstream", "timeblind", "--from", str(source)]) == 1
    assert not (tmp_path / "h" / "upstream" / "timeblind").exists()


def test_download_list_and_offline_plans(tmp_path, capsys):
    assert main(["download", "list"]) == 0
    out = capsys.readouterr().out
    assert "Qwen/Qwen3.8-27B-FP8" in out and "lingoqa" in out and "[needs --accept-terms]" in out
    assert main(["--home", str(tmp_path), "download", "models", "--dry-run", "--offline", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert len(result["results"]) == 12
    assert all(r["target"].startswith(str(tmp_path / "checkpoints")) for r in result["results"])
    assert main(["--home", str(tmp_path), "download", "judges", "--dry-run", "--offline"]) == 0
    assert "5/5 checkpoints planned" in capsys.readouterr().out
    assert main(["--home", str(tmp_path), "download", "datasets", "paper-core", "--dry-run", "--offline"]) == 0
    assert "15/15 datasets planned" in capsys.readouterr().out


def test_download_refusals_and_flag_errors(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "download", "datasets", "lingoqa"]) == 1
    assert "--accept-terms lingoqa" in capsys.readouterr().err
    assert main(["--home", str(tmp_path), "download", "datasets", "openuav_qa"]) == 1
    assert "--allow-unpinned" in capsys.readouterr().err
    assert main(["--home", str(tmp_path), "download", "models", "--offline"]) == 2
    assert not (tmp_path / "data").exists() and not (tmp_path / "checkpoints").exists()


def test_download_verify_only_reports_missing_data(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "download", "datasets", "mmstar", "--verify-only"]) == 1
    assert "does not exist" in capsys.readouterr().out
    assert main(["--home", str(tmp_path), "download", "models", "qwen25vl_3b", "--verify-only"]) == 1
    assert "not downloaded" in capsys.readouterr().out
