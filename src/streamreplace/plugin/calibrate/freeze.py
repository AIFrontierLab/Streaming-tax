"""Stage `freeze` (and the thresholds' write-back): turn a `directions` result into plugin_artefacts/<child>/.

vectors.npz = {rhat: rhat[r_layer], prs: alpha * p[p_layer]} (float32); manifest.json records layers, alpha, the
selection summary, the sha256 of vectors.npz and of the directions file it came from, and (after the `record` and
`harmless` stages) the thresholds. Existing artefacts are never replaced without force: the shipped ones define the
reported numbers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import artefacts
from .inputs import sha256_file


def freeze(child: str, parent: str, directions: Path, root: Path | None = None, force: bool = False,
           summary: dict | None = None) -> Path:
    z = np.load(directions)
    r_layer, p_layer, alpha = int(z["r_layer"]), int(z["p_layer"]), float(z["p_alpha"])
    rhat, prs = z["rhat"][r_layer].astype(np.float32), (alpha * z["p"][p_layer]).astype(np.float32)
    manifest = {
        "child": child, "parent": parent,
        "vectors": {"file": "vectors.npz", "arrays": {"rhat": "unit refusal direction at r_layer",
                                                      "prs": "alpha * mean(parent - child) state at p_layer"}},
        "layers": int(z["rhat"].shape[0]), "gate": "post", "r_layer": r_layer, "p_layer": p_layer, "alpha": alpha,
        "thresholds": {"closed": None, "free": None, "free_harmless_q975": None},
        "provenance": {"source_vectors_sha256": sha256_file(directions), "selection": summary or {}},
        "in_paper": False,
    }
    return artefacts.write(child, rhat, prs, manifest, root, force)


def set_thresholds(child: str, closed: float | None = None, free_q: float | None = None,
                   root: Path | None = None) -> dict:
    """Writes tau_closed (stage record) and/or the harmless quantile with tau_free = max(closed, q) (stage harmless) into
    the child's manifest; the vectors are untouched."""
    import json
    path = Path(root or artefacts.root_dir()) / child / "manifest.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    t = dict(meta.get("thresholds") or {})
    if closed is not None:
        t["closed"] = closed
    if free_q is not None:
        t["free_harmless_q975"] = free_q
    if t.get("closed") is not None and t.get("free_harmless_q975") is not None:
        t["free"] = max(t["closed"], t["free_harmless_q975"])
    meta["thresholds"] = t
    path.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    return t
