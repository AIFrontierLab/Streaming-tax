"""Environment definitions, the checked patch of the aura environment, and `streamreplace env` with a fake uv and
fake interpreters (nothing is installed)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess

import pytest

from streamreplace.cli.main import main
from streamreplace.models.patches import qwen25_rope
from streamreplace.provisioning import envs

ENVS = {"core": "3.12.14", "modern": "3.12.14", "llava": "3.10.14", "livestar": "3.10.14", "mmduet2": "3.10.14",
        "aura": "3.11.16", "judge": "3.12.14"}
OPTIONAL = {"lingo"}          # the Lingo-Judge environment of the fine-tuning experiment (envs/lingo, when present)


def test_every_environment_is_defined_and_pinned():
    defined = envs.definitions()
    assert {name: d.python for name, d in defined.items() if name in ENVS} == ENVS
    assert set(defined) - set(ENVS) <= OPTIONAL
    for name, definition in defined.items():
        assert definition.requirements.is_file(), name
        text = definition.lock.read_text()
        packages, options = envs.parse_lock(text)                  # raises on an unpinned line
        assert packages and "-e" not in [o.split()[0] for o in options], name
        assert "file://" not in text and not re.search(r"(^|\s)/[A-Za-z]", text), name   # no local paths
    assert envs.names()[0] == envs.FFMPEG


def test_locks_agree_with_what_the_code_checks():
    lock = {name: d.packages() for name, d in envs.definitions().items()}
    assert lock["modern"]["transformers"] == "==" + qwen25_rope.TESTED_TRANSFORMERS
    assert lock["modern"]["qwen-vl-utils"] == "==0.0.14" and lock["modern"]["torchcodec"] == "==0.7.0"
    assert lock["core"]["numpy"] == "==2.5.3"
    for name in ("core", "modern", "llava", "livestar", "mmduet2", "aura"):
        assert lock[name]["decord"] == "==0.6.0" and lock[name]["av"] == "==13.1.0", name
    (patch,) = envs.patches(env="aura")
    assert lock["aura"]["transformers"] == "==" + patch.package.split("==")[1]
    for name, version in (("modern", "2.8.3"), ("livestar", "2.7.4.post1"), ("mmduet2", "2.6.3")):
        assert lock[name]["flash-attn"].startswith("@ https://github.com/Dao-AILab/flash-attention/releases/"), name
        assert f"flash_attn-{version}+" in lock[name]["flash-attn"], name
    assert "flash-attn" not in lock["llava"]                         # the paper's LLaVA runs used sdpa


def test_the_judge_environment_has_what_the_judges_import():
    """vLLM serves the extraction judge; HarmBench's evaluation utilities import spaCy (with en_core_web_sm) and
    datasketch, VideoReasonBench's import google-genai; the RJScore judge loads with accelerate."""
    lock = envs.definitions()["judge"].packages()
    for name in ("vllm", "torch", "transformers", "accelerate", "openai", "google-genai"):
        assert name in lock, name
    assert lock["spacy"].startswith("==3.8.") and lock["datasketch"] == "==1.6.5"
    assert "/en_core_web_sm-3.8.0/" in lock["en-core-web-sm"]
    assert {"spacy", "datasketch", "en_core_web_sm", "google.genai", "accelerate"} <= set(envs.PROBES["judge"])


def test_parse_lock_options_and_urls():
    packages, options = envs.parse_lock("# comment\n--extra-index-url https://x/simple\nFoo_Bar==1.0\n"
                                        "baz @ https://x/baz-1.whl  # 1.0\n")
    assert packages == {"foo-bar": "==1.0", "baz": "@ https://x/baz-1.whl"} and options[0].startswith("--extra")
    with pytest.raises(ValueError):
        envs.parse_lock("numpy>=1\n")


def synthetic_original(diff: str) -> tuple[str, str]:
    """(an original file the diff applies to, the expected result): the hunks' lines at their stated positions,
    filler lines elsewhere."""
    _, hunks = envs._hunks(diff)
    size = max(start + length for start, length, _ in hunks) + 3
    before = [f"filler {i}\n" for i in range(1, size + 1)]
    after = list(before)
    shift = 0
    for start, _, body in hunks:
        old = [line[1:] for line in body if line[0] in " -"]
        new = [line[1:] for line in body if line[0] in " +"]
        before[start - 1:start - 1 + len(old)] = old
        after[start - 1 + shift:start - 1 + shift + len(old)] = new
        shift += len(new) - len(old)
    return "".join(before), "".join(after)


def test_the_aura_patch_changes_exactly_the_two_lines_of_the_readme():
    (patch,) = envs.patches(env="aura")
    diff = patch.file.read_text()
    target, hunks = envs._hunks(diff)
    assert target.endswith(patch.target) and patch.target == "transformers/models/qwen3_vl/video_processing_qwen3_vl.py"
    removed = [line[1:].strip() for _, _, body in hunks for line in body if line[0] == "-"]
    added = [line[1:].strip() for _, _, body in hunks for line in body if line[0] == "+"]
    assert removed == ['raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")',
                       "min_frames = 4"]
    assert added == ["num_frames = temporal_factor", "min_frames = 2"]
    before, after = synthetic_original(diff)
    assert envs.apply_unified_diff(before, diff) == after


def make_patch(tmp_path: Path, folder: Path) -> tuple[envs.Patch, Path, str, str]:
    (aura,) = envs.patches(env="aura")
    diff = aura.file.read_text()
    before, after = synthetic_original(diff)
    site = tmp_path / "site-packages"
    (site / Path(aura.target).parent).mkdir(parents=True)
    (site / aura.target).write_text(before)
    patch = envs.Patch("demo", "demo", aura.file, aura.target, hashlib.sha256(before.encode()).hexdigest(),
                       hashlib.sha256(after.encode()).hexdigest())
    return patch, site, before, after


def test_apply_patch_checks_the_file_before_and_after(tmp_path):
    patch, site, before, after = make_patch(tmp_path, tmp_path)
    assert envs.patch_status(patch, site) == "unpatched"
    assert envs.apply_patch(patch, site) == "applied" and (site / patch.target).read_text() == after
    assert envs.patch_status(patch, site) == "applied" and envs.apply_patch(patch, site) == "already applied"
    (site / patch.target).write_text(before.replace("filler 1\n", "changed\n"))
    assert envs.patch_status(patch, site) == "unexpected"
    with pytest.raises(envs.PatchError):
        envs.apply_patch(patch, site)
    assert (site / patch.target).read_text().startswith("changed")   # untouched


def test_apply_unified_diff_refuses_context_mismatches():
    diff = "--- a/x.py\n+++ b/x.py\n@@ -2,2 +2,2 @@\n context\n-old\n+new\n"
    assert envs.apply_unified_diff("first\ncontext\nold\nlast\n", diff) == "first\ncontext\nnew\nlast\n"
    with pytest.raises(envs.PatchError, match="line 3"):
        envs.apply_unified_diff("first\ncontext\nOLD\nlast\n", diff)


class FakeRun:
    """uv and an environment's interpreter, faked: uv venv creates bin/python; interpreter calls answer with JSON."""
    def __init__(self, site: Path, installed: dict, version: str = "3.10.14"):
        self.site, self.installed, self.version, self.commands = site, installed, version, []

    def __call__(self, command, **kwargs):
        self.commands.append(command)
        if command[0] == "uv":
            if command[1] == "venv":
                python = Path(command[-1]) / "bin" / "python"
                python.parent.mkdir(parents=True, exist_ok=True)
                python.write_text("")
            return subprocess.CompletedProcess(command, 0, "", "")
        code = command[command.index("-c") + 1]
        if "purelib" in code:
            out = {"purelib": str(self.site)}
        elif "platform" in code:
            out = {"version": self.version}
        elif "distributions" in code:
            out = self.installed
        else:
            out = {module: {"ok": module != "broken"} for module in command[command.index("-c") + 2:]}
        return subprocess.CompletedProcess(command, 0, "noise\n" + json.dumps(out) + "\n", "")


@pytest.fixture
def demo(tmp_path):
    """A definitions folder with one environment `demo` and one patch for it."""
    folder = tmp_path / "defs"
    (folder / "demo").mkdir(parents=True)
    (folder / "demo" / "requirements.lock").write_text("# demo\nnumpy==1.26.4\nflash_attn @ https://x/fa.whl\n")
    (folder / "demo" / "requirements.in").write_text("numpy\n")
    (folder / "demo" / "python-version").write_text("3.10.14\n")
    patch, site, before, after = make_patch(tmp_path, folder)
    (folder / "patches").mkdir()
    (folder / "patches" / "demo.patch").write_text(patch.file.read_text())
    (folder / "patches" / "manifest.yaml").write_text(json.dumps({"patches": [{
        "name": "demo-patch", "env": "demo", "file": "demo.patch", "target": patch.target,
        "sha256_before": patch.sha256_before, "sha256_after": patch.sha256_after}]}))
    installed = {"numpy": {"version": "1.26.4", "url": None},
                 "flash-attn": {"version": "2.6.3", "url": "https://x/fa.whl"},
                 "streamreplace": {"version": "0.1.0", "url": "file:///repo", "editable": True}}
    return folder, site, installed, after


def test_build_runs_uv_then_applies_and_records_the_patch(tmp_path, demo):
    folder, site, installed, after = demo
    run = FakeRun(site, installed)
    record = envs.build("demo", tmp_path / "envs", uv="uv", run=run, folder=folder, log=lambda line: None)
    venv, sync = run.commands[0], run.commands[1]
    assert venv[:2] == ["uv", "venv"] and venv[venv.index("--python") + 1] == "3.10.14"
    assert sync[:3] == ["uv", "pip", "sync"] and sync[-1] == str(folder / "demo" / "requirements.lock")
    assert (site / "transformers/models/qwen3_vl/video_processing_qwen3_vl.py").read_text() == after
    assert record["patches"] == [{"name": "demo-patch", "status": "applied", "sha256": record["patches"][0]["sha256"]}]
    stored = json.loads((tmp_path / "envs" / "demo" / envs.ENV_FILE).read_text())
    assert stored["lock_sha256"] == hashlib.sha256((folder / "demo" / "requirements.lock").read_bytes()).hexdigest()
    rerun = FakeRun(site, installed)                                  # an existing environment is synced, not recreated
    envs.build("demo", tmp_path / "envs", uv="uv", run=rerun, folder=folder, log=lambda line: None)
    assert [c[1] for c in rerun.commands if c[0] == "uv"] == ["pip"]


def test_plans_of_the_real_environments(tmp_path):
    judge = envs.plan("judge", tmp_path, uv="uv")
    assert judge.commands[-1][-3:-1] == ["--index-strategy", "unsafe-best-match"]
    core = envs.plan("core", tmp_path, uv="uv")
    assert core.commands[-1][:6] == ["uv", "pip", "install", "--python", str(tmp_path / "core/bin/python"), "--no-deps"]
    assert [p.name for p in envs.plan("aura", tmp_path).patches] == ["aura-transformers-qwen3vl-video"]
    ffmpeg = envs.plan(envs.FFMPEG, tmp_path, conda="micromamba")
    assert ffmpeg.commands == [["micromamba", "create", "-y", "-p", str(tmp_path / "ffmpeg"), "-c", "conda-forge",
                                envs.FFMPEG_SPEC]]


def test_check_compares_the_environment_with_its_lock(tmp_path, demo):
    folder, site, installed, _ = demo
    run = FakeRun(site, installed)
    envs.build("demo", tmp_path / "envs", uv="uv", run=run, folder=folder, log=lambda line: None)
    report = envs.check("demo", tmp_path / "envs", run=FakeRun(site, installed), probes=False, folder=folder)
    assert report["ok"] and report["patches"] == {"demo-patch": "applied"} and not report["warnings"]
    changed = {**installed, "numpy": {"version": "2.0.0", "url": None}, "extra-pkg": {"version": "1", "url": None}}
    report = envs.check("demo", tmp_path / "envs", run=FakeRun(site, changed, version="3.10.9"), probes=False,
                        folder=folder)
    assert not report["ok"] and report["packages"]["different"] == [["numpy", "==1.26.4", "==2.0.0"]]
    assert report["packages"]["extra"] == ["extra-pkg"] and any("3.10.9" in w for w in report["warnings"])
    wrong = envs.check("demo", tmp_path / "envs", run=FakeRun(site, installed, version="3.12.1"), probes=False,
                       folder=folder)
    assert any("Python 3.12.1" in p for p in wrong["problems"])
    missing = envs.check("demo", tmp_path / "elsewhere", folder=folder)
    assert not missing["ok"] and "does not exist" in missing["problems"][0]


def test_cli_lists_and_plans_without_installing(tmp_path, capsys):
    assert main(["--home", str(tmp_path), "env", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in listed["envs"]] == ["ffmpeg", *sorted(envs.definitions())]
    assert set(ENVS) <= {row["name"] for row in listed["envs"]}
    assert not any(row["built"] for row in listed["envs"])
    assert main(["--home", str(tmp_path), "env", "build", "all", "--dry-run", "--uv", "uv", "--conda", "conda"]) == 0
    out = capsys.readouterr().out
    assert f"$ uv pip sync --python {tmp_path / 'envs/modern/bin/python'}" in out
    assert "patch aura-transformers-qwen3vl-video" in out and "conda create -y -p" in out
    assert not (tmp_path / "envs").exists()
    assert main(["--home", str(tmp_path), "env", "check", "llava", "--no-probe"]) == 1
