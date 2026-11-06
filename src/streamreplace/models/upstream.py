"""Where fetched third-party code lives, and how adapters and scorers reach it.

Third-party code (the streaming models' runtimes, the benchmarks' official scorers) is not shipped with this
repository. `streamreplace fetch-upstream` clones each repository at the commit pinned in manifests/upstream.yaml
into `<upstream root>/<name>` (default $STREAMREPLACE_HOME/upstream; override with STREAMREPLACE_UPSTREAM). This module
only resolves those directories; fetching and hash verification belong to the provisioning step.

Adapters import upstream packages inside `upstream.use("llava", "mmduet2/demo")`: the listed directories are put at
the front of sys.path for the duration of the block, so imports resolve to the pinned copies, and sys.path is
restored afterwards (modules imported inside the block stay usable)."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sys
from typing import Iterator

from .. import paths


class UpstreamMissing(FileNotFoundError):
    """A required upstream repository (or a file in it) has not been fetched."""


class Upstream:
    """The upstream resolver handed to every adapter (see models/base.py). root=None follows paths.current()."""

    def __init__(self, root: Path | None = None):
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else paths.current().upstream

    def path(self, relative: str, require: bool = True) -> Path:
        """`<root>/<relative>`, where relative = "<name>/<path inside that repository>"."""
        path = self.root / relative
        if require and not path.exists():
            raise UpstreamMissing(f"third-party path {relative!r} is missing ({path}); run: streamreplace "
                                  f"fetch-upstream {relative.split('/', 1)[0]}")
        return path

    def available(self, name: str) -> bool:
        return (self.root / name).is_dir()

    @contextmanager
    def use(self, *entries: str) -> Iterator[list[Path]]:
        """Puts `<root>/<entry>` for each entry at the front of sys.path (first entry first) for the block."""
        added = [str(self.path(entry)) for entry in entries]
        for path in reversed(added):
            sys.path.insert(0, path)
        try:
            yield [Path(p) for p in added]
        finally:
            for path in added:
                if path in sys.path:
                    sys.path.remove(path)


def upstream_root() -> Path:
    return Upstream().root


def upstream_path(relative: str, require: bool = True) -> Path:
    return Upstream().path(relative, require)


def available(name: str) -> bool:
    return Upstream().available(name)


def use_upstream(*entries: str):
    """Context manager: Upstream().use(*entries)."""
    return Upstream().use(*entries)
