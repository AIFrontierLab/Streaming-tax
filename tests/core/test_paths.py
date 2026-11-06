"""Root resolution (paths.py): defaults under STREAMREPLACE_HOME, the four sources in priority order, per-root
overrides, per-checkpoint and per-benchmark folder maps, and the environment handed to subprocesses."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import yaml

from streamreplace import paths
from streamreplace.config import CheckpointRef


def test_defaults_live_under_the_workspace_of_the_checkout(tmp_path):
    p = paths.resolve(environ={}, project=tmp_path)
    home = (tmp_path / "workspace").resolve()
    assert p.home == home and p.sources["home"] == "default"
    for root in paths.ROOTS:
        assert p.root(root) == home / root and p.sources[root] == "home"
    assert p.paths_file is None and p.as_dict()["upstream"] == str(home / "upstream")
    with pytest.raises(ValueError, match="unknown root"):
        p.root("models")


def test_sources_in_priority_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    project, files = tmp_path / "project", tmp_path / "files"
    files.mkdir()
    paths_file = files / "roots.yaml"
    paths_file.write_text(yaml.safe_dump({"paths": {"home": "file-home", "cache": "file-cache"}}))
    block = {"home": "block-home", "data": "block-data", "envs": "block-envs"}
    env = {"STREAMREPLACE_HOME": "env-home", "STREAMREPLACE_CACHE": "env-cache"}

    p = paths.resolve(config_paths=block, environ={}, project=project)
    assert (p.home, p.data) == ((project / "block-home").resolve(), (project / "block-data").resolve())
    p = paths.resolve(config_paths=block, paths_file=paths_file, environ={}, project=project)
    assert p.home == (files / "file-home").resolve() and p.cache == (files / "file-cache").resolve()
    assert p.data == (project / "block-data").resolve()            # explicit roots survive a later home
    assert p.outputs == p.home / "outputs" and p.sources["data"] == "config paths block"
    p = paths.resolve(config_paths=block, paths_file=paths_file, environ=env, project=project)
    assert p.home == (tmp_path / "env-home").resolve() and p.cache == (tmp_path / "env-cache").resolve()
    assert p.sources["home"] == "environment" and p.envs == (project / "block-envs").resolve()
    p = paths.resolve(home="flag-home", outputs="flag-out", config_paths=block, paths_file=paths_file,
                      environ=env, project=project)
    assert p.home == (tmp_path / "flag-home").resolve() and p.outputs == (tmp_path / "flag-out").resolve()
    assert p.sources["outputs"] == "command line" and p.upstream == p.home / "upstream"


def test_paths_file_from_the_environment_and_folder_maps(tmp_path):
    project = tmp_path / "project"
    paths_file = tmp_path / "roots.yaml"
    paths_file.write_text(yaml.safe_dump({"data": "/srv/data", "dataset_dirs": {"videomme": "vm"},
                                          "checkpoint_dirs": {"Qwen/Qwen3-VL-8B-Instruct": "q3"}}))
    block = {"dataset_dirs": {"videomme": "old", "tomato": "t"}}
    p = paths.resolve(config_paths=block, environ={"STREAMREPLACE_PATHS": str(paths_file)}, project=project)
    assert p.paths_file == paths_file and p.sources["data"] == "paths file (STREAMREPLACE_PATHS)"
    assert p.data == Path("/srv/data")
    assert p.dataset_dir("videomme") == (tmp_path / "vm").resolve()             # the later layer wins per entry
    assert p.dataset_dir("tomato") == (project / "t").resolve()
    assert p.dataset_dir("mme") == p.data / "mme"
    ref = "Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert p.checkpoint_dir(ref) == (tmp_path / "q3").resolve()
    assert p.checkpoint_dir(CheckpointRef.parse(ref)) == (tmp_path / "q3").resolve()
    assert p.checkpoint_dir("aurateam/AURA@1") == p.checkpoints / "aurateam" / "AURA"


@pytest.mark.parametrize("content, message", [
    ({"modelz": "x"}, "unknown path keys"),
    ({"dataset_dirs": ["a"]}, "must map names to folders"),
    (["data"], "must be a mapping"),
])
def test_invalid_paths_files_are_refused(tmp_path, content, message):
    paths_file = tmp_path / "roots.yaml"
    paths_file.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match=message):
        paths.resolve(paths_file=paths_file, environ={}, project=tmp_path)


def test_subprocess_environment_reproduces_the_roots(tmp_path):
    paths_file = tmp_path / "roots.yaml"
    paths_file.write_text(yaml.safe_dump({"dataset_dirs": {"videomme": "vm"}}))
    p = paths.resolve(home=tmp_path / "h", paths_file=paths_file, config_paths={"data": "d"}, environ={},
                      project=tmp_path)
    again = paths.resolve(environ=p.environ(), project=tmp_path / "elsewhere")
    assert again.as_dict() == p.as_dict() and again.dataset_dir("videomme") == p.dataset_dir("videomme")
    assert p.env_python("local") == Path(sys.executable)
    assert p.env_python("modern") == p.envs / "modern" / "bin" / "python"


def test_current_paths_follow_set_current_and_reset(tmp_path, monkeypatch):
    for name in list(os.environ):
        if name.startswith("STREAMREPLACE_"):
            monkeypatch.delenv(name)
    monkeypatch.setattr(paths, "local_config_paths", lambda config_dir=None: None)
    monkeypatch.setenv("STREAMREPLACE_HOME", str(tmp_path / "env-home"))
    paths.reset()
    assert paths.home() == (tmp_path / "env-home").resolve()
    mine = paths.resolve(home=tmp_path / "mine", environ={})
    paths.set_current(mine)
    assert paths.current() is mine and paths.outputs_dir() == mine.outputs and paths.upstream_root() == mine.upstream
    assert paths.dataset_dir("mme") == mine.data / "mme" and paths.env_python("x") == mine.envs / "x/bin/python"
    paths.reset()
    assert paths.data_dir() == (tmp_path / "env-home" / "data").resolve()


def test_local_config_paths_block(tmp_path):
    assert paths.local_config_paths(tmp_path) is None
    (tmp_path / "local.yaml").write_text("paths: {home: /scratch/ws}\nsuites: {}\n")
    assert paths.local_config_paths(tmp_path) == {"home": "/scratch/ws"}
