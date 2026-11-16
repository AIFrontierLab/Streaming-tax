"""Every C-P, S-C and S-P contrast of the paper, with Holm families, TOST / non-inferiority marks and verdicts.

Design (fixed before the results were read; the margins are stated as post hoc relative to the original runs):
  items        the ids both arms counted (all formal jobs are complete); stream silence and errors are wrong for
               understanding; errors are safe and silence is judged by RJScore for safety (records.py)
  orientation  delta = b - a for higher-is-better metrics; ASR and the VMDT location accuracies (lower is better)
               use the safe / non-leak share, so delta = ASR_a - ASR_b: + always means the child (or the child's
               stream run) is better
  clusters     Video-MME / TOMATO / VideoReasonBench: video; TimeBlind I_Acc: instance (4 queries); DynaMath avg:
               seed (10 variants), worst: seed; MMBench circular: question (all rotations); MME: image within
               category (strata); other image sets, HarmBench, VSB, MCV (per split): the item; VMDT: the official
               clusters (vmdt.py, 20,000 draws, also the 90% interval)
  p            exact McNemar for single-item binary clusters, else cluster sign-flip (10,000 flips); VMDT: the
               official bootstrap p
  families     image-headline (7 headline metrics x 6 children = 42), image-secondary (DynaMath worst, MME
               cognition, VisuLogic), video/<gap> (Video-MME, TOMATO, TimeBlind I_Acc x 6 = 18 per gap),
               vrb/<gap>, and per trust benchmark x split (HarmBench C-P; VSB, MCV per split and VMDT per metric
               pooling C-P, S-C, S-P = 18). Main text ("broad"): safety parent 54, safety S-C 24, VMDT parent 96,
               VMDT S-C 48; every significance mark of the paper uses the broad families
  readings     'equivalent ±3' only when TOST at ±3 points passes, 'decline' / 'increase' only when the declared-
               family Holm p < 0.05
  replacement  per child x metric x mode (C, S vs P): replaceable when the lower 95% bound > -5 points and the
               response coverage is within 2 points of the parent's; inferior when the upper bound < -5 points
  not reported a stream arm that answers none of a set's requests has no ASR and no privacy score there (silence
               alone would give the best value), and a VMDT fairness arm whose replies leave a component without a
               rated item has no fairness score. Their S-C and S-P contrasts are left out of the rows and of the
               Holm families and listed under `unreported` (MMDuet on VMDT privacy, MCV implicit and three fairness
               scores; MMDuet2 on both MCV splits)
MME deltas are also given in points of the maximum (perception /2000 -> /20, cognition /800 -> /8).
S-P = (C-P) + (S-C) holds exactly at the point estimates (same items, same arms)."""
from __future__ import annotations

import numpy as np

from .arms import Arms
from .records import Catalog, Job, metric_rows
from .specs import GAPS, Metric, Profile
from .stats import equivalent, holm, noninferior, paired_contrast, units
from .vmdt import Engine

ROW_KEYS = ("dataset", "metric", "family", "headline", "lower_is_better", "scale", "child", "gap", "a_job", "b_job",
            "n", "coverage_a", "coverage_b")


# ---------- units ----------

def scope(metric: Metric, job: Job, catalog: Catalog) -> list[str]:
    """The job's ids the metric is computed on (its split, if any)."""
    if metric.split is None:
        return sorted(job.rows)
    smp = catalog.samples(metric.dataset)
    return sorted(i for i in job.rows if catalog.stratum(metric.dataset, smp[i]) == metric.split)


def _cluster(metric: Metric, sample, i: str) -> str:
    if metric.cluster is None:
        return i
    return sample.video if metric.cluster == "video" else str(sample.metadata[metric.cluster])


def build_units(metric: Metric, ids: list[str], a: Job, b: Job, catalog: Catalog):
    """Cluster sufficient statistics for the metric on the common ids (see stats.Units). Clusters enter in
    first-appearance order over the sorted ids; that order fixes which bootstrap weight each cluster gets."""
    smp = catalog.samples(metric.dataset)
    if metric.kind in ("acc", "safe"):
        table: dict = {}
        for i in ids:
            key = ("all", _cluster(metric, smp[i], i))
            na, nb, n = table.get(key, (0, 0, 0))
            table[key] = (na + a.rows[i].correct, nb + b.rows[i].correct, n + 1)
        return units(table)
    if metric.kind == "group":
        groups: dict = {}
        for i in ids:
            groups.setdefault(str(smp[i].metadata[metric.group]), []).append(i)
        table = {}
        for g, members in groups.items():
            size = metric.size if isinstance(metric.size, int) else smp[members[0]].metadata[metric.size]
            complete = len(members) == size
            table[("all", g)] = (int(complete and all(a.rows[i].correct for i in members)),
                                 int(complete and all(b.rows[i].correct for i in members)), int(complete))
        return units(table)
    if metric.kind == "mme":
        cognition = catalog.bench(metric.dataset).COGNITION
        cats = sorted({smp[i].metadata["category"] for i in ids
                       if (smp[i].metadata["category"] in cognition) == (metric.name == "cognition")})
        images: dict = {}
        for i in ids:
            cat = smp[i].metadata["category"]
            if cat in cats:
                images.setdefault((cat, smp[i].metadata["image"]), []).append(i)
        table = {}
        for (cat, image), members in images.items():
            j, width = cats.index(cat), 2 * len(cats)
            num_a, num_b, den = np.zeros(width), np.zeros(width), np.zeros(width)
            ra, rb = [a.rows[i].correct for i in members], [b.rows[i].correct for i in members]
            num_a[2 * j], num_b[2 * j], den[2 * j] = sum(ra), sum(rb), len(members)
            complete = len(members) == 2                       # acc+ counts images with both questions (official)
            num_a[2 * j + 1], num_b[2 * j + 1], den[2 * j + 1] = complete and all(ra), complete and all(rb), complete
            table[(cat, image)] = (num_a, num_b, den)
        return units(table, weight=np.ones(2 * len(cats)))
    raise ValueError(metric.kind)


def coverage(metric: Metric, job: Job, ids: list[str]) -> float | None:
    """Response coverage on the metric's items: an extractable answer (understanding, VMDT fairness) or a non-silent,
    non-error reply (safety, where a refusal is a valid response; VMDT privacy, where no extraction can be a
    refusal)."""
    if not ids:
        return None
    if metric.coverage_rule == "responded":
        ok = [not (job.rows[i].silent or job.rows[i].error) for i in ids]
    else:
        ok = [job.rows[i].answered for i in ids]
    return 100 * sum(ok) / len(ids)


def no_reply(metric: Metric, job: Job, ids: list[str]) -> bool:
    """The no-reply rule: a stream job that replied to none of the metric's items has no score on a metric whose
    coverage is "a non-silent reply" (ASR, VMDT privacy): silence alone would give the best value."""
    return (job.mode == "stream" and metric.coverage_rule == "responded" and bool(ids)
            and coverage(metric, job, ids) == 0)


# ---------- contrasts ----------

class Contrasts:
    """Computes contrast rows for one profile on one set of arms."""

    def __init__(self, profile: Profile, arms: Arms, catalog: Catalog):
        self.profile, self.arms, self.catalog = profile, arms, catalog
        self.vmdt = Engine(catalog, profile.vmdt_draws, profile.seed)
        self.uncounted: list[str] = []        # contrasts left out: both jobs exist, but no counted row in common
        self.unreported: list[dict] = []      # contrasts left out: the stream arm has no score (module docstring)

    def pair(self, metric: Metric, a: Job, b: Job) -> dict:
        """The contrast of two arbitrary jobs (also used by the plugin report): a, b, delta, intervals, p, n, coverage."""
        ids = sorted(set(scope(metric, a, self.catalog)) & set(scope(metric, b, self.catalog)))
        if metric.kind == "vmdt":
            result = self.vmdt.contrast(metric, a, b)
        else:
            result = paired_contrast(build_units(metric, ids, a, b, self.catalog), self.profile.reps,
                                     self.profile.seed, scale=metric.scale)
            if metric.lower_is_better:                         # report ASR; delta (safe share) is ASR_a - ASR_b
                result.update(a=100 - result["a"], b=100 - result["b"])
        return {"n": len(ids), "coverage_a": coverage(metric, a, ids), "coverage_b": coverage(metric, b, ids),
                **result}

    def row(self, metric: Metric, child: str, gap: str) -> dict | None:
        """One contrast row, or None when the gap needs a mode the metric lacks or an arm is missing. Jobs whose
        rows are not scored yet (no counted row the two arms share: e.g. safety rows waiting for `score --judge`)
        are left out like missing ones and listed in `uncounted`."""
        parent = self.profile.parents[child]
        sides = {"P": (parent, "offline"), "C": (child, "offline"), "S": (child, "stream")}
        (ma, mode_a), (mb, mode_b) = (sides[s] for s in GAPS[gap])
        if "stream" in (mode_a, mode_b) and "stream" not in metric.modes:
            return None
        a, b = self.arms.job(metric.dataset, ma, mode_a, child), self.arms.job(metric.dataset, mb, mode_b, child)
        if a is None or b is None:
            return None
        if not set(scope(metric, a, self.catalog)) & set(scope(metric, b, self.catalog)):
            self.uncounted.append(f"{metric.dataset} {metric.name} {child} {gap}: {ma}.{mode_a} ({len(a.rows)} "
                                  f"counted rows) and {mb}.{mode_b} ({len(b.rows)}) share no counted row; score "
                                  "them first (`streamreplace score`, with --judge for judge-scored benchmarks)")
            return None
        skip = {"dataset": metric.dataset, "metric": metric.name, "child": child, "gap": gap}
        ids = sorted(set(scope(metric, a, self.catalog)) & set(scope(metric, b, self.catalog)))
        if no_reply(metric, b, ids):
            self.unreported.append({**skip, "reason": f"{mb}.{mode_b} replied to none of the {len(ids)} requests"})
            return None
        try:
            result = self.pair(metric, a, b)
        except (IndexError, ZeroDivisionError, ValueError) as exc:
            if metric.kind != "vmdt" or mode_b != "stream":
                raise
            self.unreported.append({**skip, "reason": f"{mb}.{mode_b}: no defined score ({type(exc).__name__})"})
            return None
        if result["delta"] != result["delta"]:              # NaN: a VMDT component without a rated item
            self.unreported.append({**skip, "reason": f"{mb}.{mode_b}: no defined score"})
            return None
        head = {"dataset": metric.dataset, "metric": metric.name, "family": metric.family, "headline": metric.headline,
                "lower_is_better": metric.lower_is_better, "scale": metric.scale, "child": child, "gap": gap,
                "a_job": f"{ma}.{mode_a}", "b_job": f"{mb}.{mode_b}"}
        return {**head, **result}

    def compute(self) -> list[dict]:
        """All rows in the order metrics x children x gaps, annotated."""
        out = []
        for metric in self.profile.metrics:
            for child in self.profile.children:
                for gap in GAPS:
                    row = self.row(metric, child, gap)
                    if row is not None:
                        out.append(row)
        return annotate(out, self.profile)


# ---------- Holm families, margins, readings, verdicts ----------

def family_of(row: dict, profile: Profile) -> str:
    if row["family"] in profile.holm.get("split_by_gap", ()):
        return f"{row['family']}/{row['gap']}"
    return row["family"]


def broad_family_of(row: dict, profile: Profile) -> str:
    """Main-text families: contrasts involving the parent (C-P, S-P) vs S-C, per broad group of datasets;
    other metrics keep their declared family."""
    kind = "parent" if row["gap"] in ("C-P", "S-P") else "S-C"
    for name, datasets in profile.holm.get("broad", {}).items():
        if row["dataset"] in datasets:
            return f"{name}/{kind}"
    return family_of(row, profile)


def _holm_into(rows: list[dict], family, prefix: str):
    groups: dict = {}
    for i, r in enumerate(rows):
        groups.setdefault(family(r), {})[i] = r["p"]
    for name, ps in groups.items():
        for i in ps:
            rows[i].update({f"p_{prefix}": None, f"{prefix}_family": name,
                            f"{prefix}_m": sum(p is not None for p in ps.values())})
        for i, p in holm(ps).items():
            rows[i][f"p_{prefix}"] = p


def reading_of(tost: bool, p_holm: float | None, delta: float, margin: float = 3, alpha: float = 0.05) -> str:
    """'equivalent ±3' when TOST at ±3 points passes, 'decline' / 'increase' when the Holm p is below 0.05 (both
    can hold), else 'inconclusive'."""
    reading = []
    if tost:
        reading.append(f"equivalent ±{margin:g}")
    if p_holm is not None and p_holm < alpha:
        reading.append("decline" if delta < 0 else "increase")
    return "; ".join(reading) or "inconclusive"


def annotate(rows: list[dict], profile: Profile) -> list[dict]:
    """Holm within the declared and the broad families, TOST / NI flags, coverage gap, reading and verdict."""
    m = profile.margins
    alpha = profile.holm.get("alpha", 0.05)
    _holm_into(rows, lambda r: family_of(r, profile), "holm")
    _holm_into(rows, lambda r: broad_family_of(r, profile), "holm_broad")
    for r in rows:
        ci90, ci95 = r.get("ci90_pp"), r.get("ci95_pp")
        r.update({f"tost{t:g}": equivalent(ci90, t) for t in profile.tost_margins})
        r.update({f"ni{t:g}": noninferior(ci95, t) for t in profile.ni_margins})
        r["coverage_gap"] = (None if r["coverage_a"] is None or r["coverage_b"] is None
                             else r["coverage_b"] - r["coverage_a"])
        tost_key = f"tost{m.get('reading_tost', 3):g}"
        r["reading"] = reading_of(r[tost_key], r.get("p_holm"), r["delta"], m.get("reading_tost", 3), alpha)
        if r["gap"] in ("C-P", "S-P"):
            ni = r[f"ni{m.get('verdict_ni', 5):g}"]
            if ni and abs(r["coverage_gap"] or 0) <= m.get("coverage_tolerance", 2):
                verdict = "replaceable"
            elif ni:
                verdict = "non-inferior, coverage differs"
            elif ci95 and ci95[1] < -m.get("inferior", 5):
                verdict = "inferior"
            else:
                verdict = "inconclusive"
            r["verdict"] = verdict
    return rows


# ---------- checks and summaries ----------

def official_metrics(job: Job, catalog: Catalog) -> dict:
    """The benchmark's own metrics on the job's records (as RESULTS.md, final scoring)."""
    return catalog.metrics(job.dataset, metric_rows(job, catalog))


def check_official(rows: list[dict], arms: Arms, catalog: Catalog, profile: Profile) -> list[str]:
    """Arm values must equal the benchmark's own metrics to rounding; returns the mismatches."""
    by_name = {(m.dataset, m.name): m for m in profile.metrics}
    tolerance = profile.check_official
    bad, seen = [], {}
    for r in rows:
        metric = by_name[(r["dataset"], r["metric"])]
        for side in ("a", "b"):
            model, mode = r[f"{side}_job"].rsplit(".", 1)
            job = arms.job(r["dataset"], model, mode, r["child"])
            key = (r["dataset"], job.uid)
            if key not in seen:
                seen[key] = official_metrics(job, catalog)
            official = seen[key].get(metric.official_key)
            limit = tolerance.get(r["dataset"], tolerance.get("tolerance", 0.011))   # official values are rounded
            if official is not None and abs(official - r[side]) > limit:
                bad.append(f"{r['dataset']} {r['metric']} {model}.{mode} ({r['child']}): {r[side]:.3f} vs official "
                           f"{official}")
    return bad


def sig(r: dict) -> bool:
    return r.get("p_holm") is not None and r["p_holm"] < 0.05


def summary(rows: list[dict]) -> dict:
    """Counts behind the paper's text (declared families; 'trust_broad' with the main-text families)."""
    head = [r for r in rows if r["family"] == "image-headline"]
    out = {"image_headline": len(head), "image_tost3": sum(bool(r["tost3"]) for r in head),
           "image_tost2": sum(bool(r["tost2"]) for r in head),
           "image_holm_decline": sum(sig(r) and r["delta"] < 0 for r in head),
           "image_holm_increase": sum(sig(r) and r["delta"] > 0 for r in head)}
    for gap in GAPS:
        fam = [r for r in rows if r["family"] == "video" and r["gap"] == gap]
        out[f"video_{gap}"] = {"tests": len(fam), "holm_decline": sum(sig(r) and r["delta"] < 0 for r in fam),
                               "holm_increase": sum(sig(r) and r["delta"] > 0 for r in fam),
                               "tost3": sum(bool(r["tost3"]) for r in fam)}
    verdicts: dict = {}
    for r in rows:
        if r.get("verdict") and r["headline"]:
            mode = "C" if r["gap"] == "C-P" else "S"
            verdicts.setdefault(mode, {}).setdefault(r["verdict"], 0)
            verdicts[mode][r["verdict"]] += 1
    out["replacement"] = verdicts
    idx = {(r["dataset"], r["metric"], r["child"], r["gap"]): r for r in rows}
    residuals = [abs(idx[k[:3] + ("S-P",)]["delta"] - idx[k[:3] + ("C-P",)]["delta"] - idx[k[:3] + ("S-C",)]["delta"])
                 for k in idx if k[3] == "C-P" and k[:3] + ("S-P",) in idx and k[:3] + ("S-C",) in idx]
    out["eq1_max_residual"] = max(residuals) if residuals else None       # S-P = (C-P) + (S-C) on the same items
    trust = [r for r in rows if r["family"] not in ("image-headline", "image-secondary", "video", "vrb")]
    out["trust"] = {"tests": len(trust), "holm_significant": sum(sig(r) for r in trust),
                    "raw_significant": sum(r["p"] is not None and r["p"] < 0.05 for r in trust)}
    out["trust_lost_by_holm"] = [f"{r['dataset']} {r['metric']} {r['child']} {r['gap']} (p {r['p']:.3g} -> {r['p_holm']:.3g})"
                                 for r in trust if r["p"] is not None and r["p"] < 0.05 <= r["p_holm"]]
    out["trust_broad"] = {"holm_significant": sum(r["p_holm_broad"] is not None and r["p_holm_broad"] < 0.05 for r in trust)}
    out["trust_lost_by_broad_holm_only"] = [
        f"{r['dataset']} {r['metric']} {r['child']} {r['gap']} (Holm {r['p_holm']:.3g} -> broad {r['p_holm_broad']:.3g})"
        for r in trust if r["p_holm"] is not None and r["p_holm"] < 0.05 <= r["p_holm_broad"]]
    return out


def duration_split(profile: Profile, arms: Arms, catalog: Catalog) -> list[dict]:
    """Video-MME S - C per video-duration bin (seconds from the reference job's info.duration); descriptive."""
    spec = profile.duration_split
    if not spec:
        return []
    metric = next((m for m in profile.metrics if (m.dataset, m.name) == (spec["dataset"], spec["metric"])), None)
    if metric is None:                                  # a profile without that metric: nothing to split
        return []
    model, mode = spec["reference"].rsplit(".", 1)
    ref = arms.formal(spec["dataset"], model, mode)
    if ref is None:
        return []
    duration = {i: r.info.get("duration") for i, r in ref.rows.items()}
    out = []
    for child in profile.children:
        a, b = arms.formal(spec["dataset"], child, "offline"), arms.formal(spec["dataset"], child, "stream")
        if a is None or b is None:
            continue
        for lo, hi in spec["bins"]:
            ids = sorted(i for i in set(a.rows) & set(b.rows) if duration.get(i) is not None and lo <= duration[i] < hi)
            if not ids:                                 # nothing counted in this bin yet (unscored rows)
                continue
            r = paired_contrast(build_units(metric, ids, a, b, catalog), profile.reps, profile.seed)
            out.append({"child": child, "bin": f"[{lo}, {hi}) s", "n": len(ids), **r})
    return out

