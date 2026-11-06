"""Per-child plugin artefacts: plugin_artefacts/<child>/vectors.npz and manifest.json (frozen, sha256-checked).

vectors.npz holds two float32 arrays of the child's hidden size:
  rhat  the unit refusal direction at decoder layer r_layer (the gate projects the answer-position state on it)
  prs   the parent-restoration vector at decoder layer p_layer, already scaled: alpha * mean(parent - child)
manifest.json holds the layers, alpha, the thresholds (closed / free and the 97.5th percentile on harmless requests behind the
free one), the threshold rule, and provenance as sha256 values only. Layer indices count the decoder ModuleList that
guard.find_layers() returns (LiveStar: the one inside model.language_model).

The shipped files are the ones every reported plugin number was produced with; `streamreplace plugin verify` re-hashes
them. A rebuild (plugin/calibrate) is not bit-identical (GPU kernels, float16 state cache), so reproduce the paper
with the shipped files."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

UNIT_TOL = 1e-4


def root_dir() -> Path:
    from .recipe import repo_root
    return repo_root() / "plugin_artefacts"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Artefact:
    child: str
    rhat: np.ndarray
    prs: np.ndarray
    r_layer: int
    p_layer: int
    alpha: float
    thresholds: dict            # {"closed", "free", "free_harmless_q975"}
    manifest: dict = field(default_factory=dict)
    folder: Path | None = None

    @property
    def sha256(self) -> str:
        return self.manifest["vectors"]["sha256"]

    @property
    def steer_prefill(self) -> bool:
        """PRS at the answer-turn prefill's last position only when its layer comes after the gate's layer, so the
        gate always reads the child's own state (False for ThinkStream 14 <= 20 and MOSS-VL-Realtime 19 <= 24)."""
        return not self.p_layer <= self.r_layer

    def tau(self, variant: str = "higher") -> dict:
        """{closed, free} of a threshold variant: higher (the paper: free = max(closed, harmless q97.5)), rule (free =
        the harmless q97.5 itself) or old (the closed threshold on both formats, as the first deployed gate)."""
        t = self.thresholds
        if variant == "higher":
            return {"closed": t["closed"], "free": t["free"]}
        if variant == "rule":
            return {"closed": t["closed"], "free": t["free_harmless_q975"]}
        if variant == "old":
            return {"closed": t["closed"], "free": t["closed"]}
        raise ValueError(f"unknown threshold variant {variant!r} (higher, rule, old)")

    def identity(self) -> dict:
        """What of the artefacts changes a generation (part of the run identity of a plugin arm)."""
        return {"vectors_sha256": self.sha256, "r_layer": self.r_layer, "p_layer": self.p_layer,
                "alpha": self.alpha, "thresholds": {"closed": self.thresholds["closed"],
                                                    "free": self.thresholds["free"]}}


def read_manifest(child: str, root: Path | None = None) -> dict:
    path = Path(root or root_dir()) / child / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"no plugin artefacts for {child!r} ({path}); calibrate the child first "
                                "(streamreplace plugin calibrate)")
    return json.loads(path.read_text(encoding="utf-8"))


def load(child: str, root: Path | None = None, verify: bool = True) -> Artefact:
    """The child's artefacts; refuses a vectors file whose sha256 differs from its manifest."""
    folder = Path(root or root_dir()) / child
    meta = read_manifest(child, root)
    vectors = folder / meta["vectors"]["file"]
    if verify:
        digest = sha256_file(vectors)
        if digest != meta["vectors"]["sha256"]:
            raise ValueError(f"{vectors}: sha256 {digest} differs from the manifest's {meta['vectors']['sha256']}")
    with np.load(vectors) as z:
        rhat, prs = np.asarray(z["rhat"]), np.asarray(z["prs"])
    return Artefact(child, rhat, prs, int(meta["r_layer"]), int(meta["p_layer"]), float(meta["alpha"]),
                    dict(meta["thresholds"]), meta, folder)


def check(child: str, root: Path | None = None) -> list[str]:
    """Problems of one child's artefacts (empty when everything holds): sha256, |rhat| = 1, shapes and hidden size,
    layer indices, free >= closed, free = max(closed, harmless q97.5)."""
    problems = []
    try:
        art = load(child, root, verify=True)
    except (OSError, ValueError, KeyError) as exc:
        return [f"{child}: {exc}"]
    meta, d = art.manifest, int(art.manifest.get("hidden_size", -1))
    norm = float(np.linalg.norm(art.rhat.astype(np.float64)))
    if art.rhat.ndim != 1 or abs(norm - 1) > UNIT_TOL:
        problems.append(f"{child}: |rhat| = {norm:.6f}, expected 1")
    if art.prs.shape != art.rhat.shape or art.rhat.shape != (d,):
        problems.append(f"{child}: shapes rhat {art.rhat.shape}, prs {art.prs.shape}, hidden size {d}")
    if art.rhat.dtype != np.float32 or art.prs.dtype != np.float32:
        problems.append(f"{child}: vectors must be float32")
    layers = int(meta.get("layers", 0))
    if not (0 <= art.r_layer < layers and 0 <= art.p_layer < layers):
        problems.append(f"{child}: layers r {art.r_layer} / p {art.p_layer} outside 0..{layers - 1}")
    if "prs_norm" in meta and abs(float(np.linalg.norm(art.prs.astype(np.float64))) - meta["prs_norm"]) > 1e-3:
        problems.append(f"{child}: |prs| differs from the manifest's {meta['prs_norm']}")
    t = art.thresholds
    if any(t.get(k) is None for k in ("closed", "free", "free_harmless_q975")):
        problems.append(f"{child}: thresholds not calibrated yet (stages record and harmless)")
    elif t["free"] != max(t["closed"], t["free_harmless_q975"]):
        problems.append(f"{child}: tau_free {t['free']} != max(tau_closed, harmless q97.5)")
    return problems


def children(root: Path | None = None) -> list[str]:
    folder = Path(root or root_dir())
    return sorted(p.name for p in folder.iterdir() if (p / "manifest.json").is_file()) if folder.is_dir() else []


def write(child: str, rhat: np.ndarray, prs: np.ndarray, manifest: dict, root: Path | None = None,
          force: bool = False) -> Path:
    """Writes plugin_artefacts/<child>/{vectors.npz, manifest.json}; the manifest gets the vectors' sha256. Refuses
    to replace existing artefacts unless force (frozen artefacts define the reported numbers)."""
    folder = Path(root or root_dir()) / child
    if (folder / "manifest.json").exists() and not force:
        raise FileExistsError(f"{folder}: artefacts exist (frozen); pass force to replace them")
    folder.mkdir(parents=True, exist_ok=True)
    vectors = folder / "vectors.npz"
    np.savez(vectors, rhat=np.asarray(rhat, dtype=np.float32), prs=np.asarray(prs, dtype=np.float32))
    meta = dict(manifest)
    meta["vectors"] = {**meta.get("vectors", {}), "file": vectors.name, "sha256": sha256_file(vectors),
                       "dtype": "float32"}
    meta["hidden_size"] = int(np.asarray(rhat).shape[0])
    meta["prs_norm"] = round(float(np.linalg.norm(np.asarray(prs, dtype=np.float64))), 3)
    (folder / "manifest.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return folder
