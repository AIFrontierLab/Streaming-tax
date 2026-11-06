"""Which judge scores which benchmark, the judge configuration (configs/judges.yaml) and judge construction.

A benchmark's JUDGE policy names its judge (POLICY_JUDGE): the extraction judge serves the FALLBACK and ALWAYS
policies, the HarmBench classifiers CLASSIFIER, the RJScore judge RJSCORE and the Lingo-Judge LINGO. NONE needs no
judge. configs/judges.yaml has one entry per judge (see that file for the fields); `load_specs()` reads it into
`JudgeSpec`s and `build(name)` returns the configured judge object (the classes are listed in judges/base.py).

Every judge class takes (spec: JudgeSpec, paths: Paths | None) and derives from `ConfiguredJudge`, which provides
the parts all judges share: the config hash that names the cache file, the cache itself, checkpoint resolution and the
launch of the judge process in the judge's own environment (judges/entry.py)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..core.io import fingerprint
from .base import JUDGES, load_judge_class
from .cache import JudgeCache

# Benchmark JUDGE policy (benchmarks/base.JudgePolicy values) -> judge name.
POLICY_JUDGE = {"fallback": "extract", "always": "extract", "classifier": "harmbench_cls", "rjscore": "rjscore",
                "lingo": "lingo"}


def judge_name(bench) -> str | None:
    """The judge a benchmark's rows need, or None (policy NONE)."""
    policy = getattr(bench, "JUDGE", None)
    policy = getattr(policy, "value", policy)
    return POLICY_JUDGE.get(str(policy).lower()) if policy else None


@dataclass(frozen=True)
class JudgeSpec:
    """One entry of configs/judges.yaml. checkpoint(s) are "<HF repo>@<revision>" references (paths.checkpoint_dir
    resolves the local folder); every other key is judge-specific and kept in `options`."""
    name: str
    env: str = "judge"
    backend: str = ""
    checkpoint: str | None = None
    checkpoints: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, name: str, value: Mapping | None) -> JudgeSpec:
        if not isinstance(value, Mapping):
            raise ValueError(f"judges.{name}: expected a mapping")
        value = dict(value)
        checkpoints = value.pop("checkpoints", None) or {}
        if not isinstance(checkpoints, Mapping):
            raise ValueError(f"judges.{name}.checkpoints: expected a mapping of role -> repo@revision")
        spec = cls(name=name, env=str(value.pop("env", "judge")), backend=str(value.pop("backend", "")),
                   checkpoint=value.pop("checkpoint", None), checkpoints=dict(checkpoints), options=value)
        for ref in ([spec.checkpoint] if spec.checkpoint else []) + list(spec.checkpoints.values()):
            if not isinstance(ref, str) or not ref:
                raise ValueError(f"judges.{name}: invalid checkpoint reference {ref!r}")
        return spec

    def option(self, *keys: str, default=None):
        """A nested option, e.g. spec.option("server", "port", default=8031)."""
        value: Any = self.options
        for key in keys:
            if not isinstance(value, Mapping) or key not in value:
                return default
            value = value[key]
        return value


def read_section(config_dir: Path | None = None) -> dict:
    """configs/judges.yaml with configs/local.yaml's `judges:` overrides merged in."""
    from ..config.load import load_config
    return load_config(config_dir, check_registry=False).section("judges")


def load_specs(section: Mapping | None = None) -> dict[str, JudgeSpec]:
    section = read_section() if section is None else section
    unknown = sorted(set(section) - set(JUDGES))
    if unknown:
        raise ValueError(f"configs/judges.yaml: unknown judges {unknown}; known: {', '.join(JUDGES)}")
    return {name: JudgeSpec.parse(name, value) for name, value in section.items()}


def build(name: str, spec: JudgeSpec | None = None, paths=None, section: Mapping | None = None):
    """The configured judge object for `name` (spec from configs/judges.yaml unless given)."""
    if spec is None:
        specs = load_specs(section)
        if name not in specs:
            raise ValueError(f"judge {name!r} is not configured in configs/judges.yaml")
        spec = specs[name]
    try:
        cls = load_judge_class(name)
    except ImportError as exc:
        raise ValueError(f"judge {name!r} is not available: {exc}") from exc
    return cls(spec, paths)


class ConfiguredJudge:
    """Shared parts of the judges (see the module docstring). Subclasses set NAME, ENV and PROMPT_VERSION and define
    identity(), requests(), apply(), run() and self_test()."""
    NAME = ""
    ENV = "judge"
    PROMPT_VERSION = 1           # bump when a prompt or the parsing of a judge reply changes (renames the cache)
    SELFTEST_IN_RUN = False      # True: run(reqs, cache, gpus, selftest=True) self-tests before judging
    DEFAULT_GPUS: tuple[int, ...] = (0,)     # the judge's GPUs when `score` is given none (--judge-gpus)

    def __init__(self, spec: JudgeSpec, paths=None):
        if spec.name != self.NAME:
            raise ValueError(f"{type(self).__name__} got the spec of judge {spec.name!r}")
        self.spec = spec
        self._paths = paths

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.config_hash()})"

    @property
    def paths(self):
        if self._paths is not None:
            return self._paths
        from .. import paths
        return paths.current()

    @property
    def env(self) -> str:
        return self.spec.env or self.ENV

    def identity(self) -> dict:
        """What the judge's answers depend on (never GPUs, ports or paths)."""
        raise NotImplementedError

    def config_hash(self) -> str:
        return fingerprint(self.identity())

    def cache(self, outputs: Path | None = None) -> JudgeCache:
        return JudgeCache.at(Path(outputs) if outputs is not None else self.paths.outputs, self.NAME,
                             self.config_hash())

    def checkpoint_dir(self, ref: str | None = None) -> Path:
        ref = ref or self.spec.checkpoint
        if not ref:
            raise ValueError(f"judge {self.NAME!r}: no checkpoint configured (configs/judges.yaml)")
        return self.paths.checkpoint_dir(ref)

    def launch(self, requests: list[dict], cache: JudgeCache, gpus: list[int], spec: dict | None = None,
               extra_env: Mapping[str, str] | None = None, tag: str = "") -> None:
        """Runs `python -m streamreplace.judges.entry` in the judge's environment on these requests; the process
        appends its answers to the cache file."""
        from .entry import launch
        launch(self.NAME, requests, cache.path, spec or {}, gpus, env=self.env, paths=self.paths,
               extra_env=extra_env, tag=tag or self.config_hash())
        cache.refresh()

    def self_test(self, gpus: list[int]) -> None:
        """Default: nothing to check."""
        return None
