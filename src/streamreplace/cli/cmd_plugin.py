"""`streamreplace plugin run|assemble|report|calibrate|verify`: the streaming plugin (docs/plugin.md).

run        the plugin arms <child>@<arm>.stream on an item set: writes <outputs>/subset.json, then runs `run` with
           this invocation's global options (--config, --home, --paths, --config-dir); --print prints that command
           instead of running it (the subset.json is still written, since the printed command needs it)
assemble   the fixed-refusal pseudo-job fixed_refusal.offline on the item set (score it with `streamreplace score`)
report     the recovery tables: S+ assembled post hoc from the shadow arms, against P, C and S (report.json,
           REPORT.md, figure.json under <outputs>/plugin), with the analysis profile of the loaded configuration
calibrate  a new child: stages directions, freeze, record, harmless (docs/plugin.md, "Calibrating a new child").
           The two GPU stages load models: directions the child's and the parent's offline pipelines, harmless the
           child's stream. They run in that model environment's interpreter, started the way `run` starts its
           workers (run/envs.py), on one GPU slot (--gpus, default the first of run.gpus)
verify     the shipped artefacts (sha256, |rhat| = 1, shapes, thresholds), and that the item lists parse

The gate of the plugin arms runs in shadow mode as in the paper. A deployment that should refuse during generation
sets the arm's gate to live in an extra config file (--config FILE with `plugin: {arms: {sfix_prs: {gate: live}}}`):
a gate that fires replies with the fixed refusal, and where the hooks see the answer turn's first step (every child
but ThinkStream) the turn also ends there; ThinkStream's turn runs to its end and its reply is then replaced.
`plugin report` reads shadow arms only: it refuses an arm whose gate is not shadow, and rows a live or record gate
made.

Paper reproduction on the second pilot's items (Fig. plugin):
  streamreplace --outputs OUT plugin run --children mmduet_7b,... --arms sfix,sfix_prs --items pilot2 --gpus 0
  streamreplace --outputs OUT plugin assemble --items pilot2
  streamreplace --outputs OUT score --judge
  streamreplace --outputs OUT plugin report --items pilot2 --formal FORMAL_OUTPUTS"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from . import command_line, context, csv_list, global_options, gpu_list

ITEMS_HELP = "item set: a name under `items` in configs/plugin.yaml (pilot2, full, threshold, dev) or a JSON file " \
             "{dataset: [ids]} (default: pilot2)"
WORKER = "STREAMREPLACE_PLUGIN_WORKER"      # set in a calibration stage's model-environment process


def register(subparsers) -> None:
    parser = subparsers.add_parser("plugin", help="the streaming plugin: run, assemble, report, calibrate, verify",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actions = parser.add_subparsers(dest="plugin_action", required=True)

    run = actions.add_parser("run", help="run plugin arms on an item set",
                             description="Writes <outputs>/subset.json (the item set) and runs the arms "
                                         "<child>@<arm>.stream through `streamreplace run`, with the global options "
                                         "of this invocation.")
    run.add_argument("--children", type=csv_list, help="children (default: the paper's five)")
    run.add_argument("--arms", type=csv_list, default=["sfix", "sfix_prs"], help="arms (default: sfix,sfix_prs)")
    run.add_argument("--items", default="pilot2", help=ITEMS_HELP)
    run.add_argument("--gpus", type=gpu_list, help="GPU slots, one worker each, e.g. 0,0,1 (default: run.gpus in configs)")
    run.add_argument("--print", dest="print_only", action="store_true",
                     help="print the run command instead of running it (<outputs>/subset.json is still written: the "
                          "command reads it)")
    run.add_argument("extra", nargs=argparse.REMAINDER, help="further `run` options after --, e.g. -- --dry-run")
    run.set_defaults(func=_run)

    assemble = actions.add_parser("assemble", help="write the fixed-refusal job on an item set",
                                  description="Writes fixed_refusal.offline (the refusal the gate gives when it "
                                              "fires) for every item of the set, with the prompts of the arms' rows.")
    assemble.add_argument("--items", default="pilot2", help=ITEMS_HELP)
    assemble.add_argument("--formal", type=Path, help="outputs folder with formal rows to take prompts from")
    assemble.set_defaults(func=_assemble)

    report = actions.add_parser("report", help="recovery tables (S+ assembled post hoc)",
                                description="S+ assembled post hoc from the shadow arms (threshold variants), "
                                            "against P, C and S of the same items: report.json, REPORT.md, "
                                            "figure.json.")
    report.add_argument("--items", default="pilot2", help=ITEMS_HELP)
    report.add_argument("--children", type=csv_list, help="children (default: the paper's five)")
    report.add_argument("--variant", default="all", choices=("all", "higher", "rule", "old"),
                        help="threshold variant (default: all three; the paper's is higher)")
    report.add_argument("--formal", type=Path, help="outputs folder of the formal P/C/S jobs (default: the outputs root)")
    report.add_argument("--profile", metavar="FILE",
                        help="analysis profile YAML instead of configs/analysis.yaml (metrics, names)")
    report.add_argument("--out", type=Path, help="folder for report.json, REPORT.md, figure.json (default <outputs>/plugin)")
    report.add_argument("--interim", action="store_true", help="write the tables even if rows are missing")
    report.set_defaults(func=_report)

    cal = actions.add_parser("calibrate", help="calibrate the plugin for a child",
                             description="Stages, in order: directions (GPU: refusal direction and parent-restoration "
                                         "vector), freeze (writes plugin_artefacts/<child>/), record (tau_closed from "
                                         "the record arm's rows), harmless (GPU: tau_free). The GPU stages run in the "
                                         "model environment's interpreter.")
    cal.add_argument("child", help="the child's model id (configs/models.yaml or a --config file)")
    cal.add_argument("--parent", help="the child's offline parent (default: from configs/pairs.yaml)")
    cal.add_argument("--stage", required=True, choices=("directions", "freeze", "record", "harmless"))
    cal.add_argument("--advbench", type=Path, help="AdvBench harmful_behaviors.csv (stage directions)")
    cal.add_argument("--alpaca", type=Path, help="Stanford Alpaca alpaca_data.json (stage directions)")
    cal.add_argument("--clips", type=Path, help="folder of the benign clips (stage harmless; cut when missing)")
    cal.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary for cutting benign clips")
    cal.add_argument("--work", type=Path, help="calibration work folder (default <outputs>/plugin_calibration/<child>)")
    cal.add_argument("--gpus", type=gpu_list,
                     help="GPU slot of the model process of stages directions and harmless (default: the first of "
                          "run.gpus in configs)")
    cal.add_argument("--force", action="store_true", help="replace existing artefacts (stage freeze)")
    cal.set_defaults(func=_calibrate)

    verify = actions.add_parser("verify", help="check the shipped artefacts; check that the item lists parse",
                                description="Per child: the artefacts' sha256, |rhat| = 1, array shapes and "
                                            "thresholds against plugin_artefacts/<child>/manifest.json. Then every "
                                            "item list and calibration list of configs/plugin.yaml must parse as "
                                            "JSON.")
    verify.add_argument("--children", type=csv_list, help="children (default: every configured child)")
    verify.set_defaults(func=_verify)


def _config(args):
    from ..plugin.recipe import load_plugin_config
    ctx = context(args)
    try:
        section = ctx.config.section("plugin")
    except (ValueError, OSError):                     # an incomplete configuration still lets `plugin verify` work
        section = None
    return load_plugin_config(section=section) if section else load_plugin_config()


def _children(args, config) -> list[str]:
    children = args.children or list(config.paper_children)
    unknown = sorted(set(children) - set(config.children))
    if unknown:
        raise ValueError(f"no plugin configuration for {unknown}")
    return children


def _write_subset(outputs: Path, items: dict) -> Path:
    path = outputs / "subset.json"
    if path.exists():
        have = json.loads(path.read_text(encoding="utf-8"))
        if have != items:
            raise ValueError(f"{path} exists with other items: use another --outputs folder per item set")
    else:
        outputs.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, indent=1) + "\n", encoding="utf-8")
    return path


def run_argv(args, models: list[str], benchmarks: list[str]) -> list[str]:
    """The `run` command line of the arms: this invocation's global options (the inner run must see the same
    configuration files, e.g. a live gate or a new child, and the same workspace), then the arms."""
    argv = [*global_options(args, outputs=True), "run", "--models", ",".join(models), "--protocols", "S",
            "--benchmarks", ",".join(benchmarks)]
    if args.gpus:
        argv += ["--gpus", ",".join(map(str, args.gpus))]
    return argv + [a for a in args.extra if a != "--"]


def _run(args) -> int:
    config = _config(args)
    ctx = context(args)
    for arm in args.arms:
        config.arm(arm)
    items = config.load_items(args.items)
    _write_subset(ctx.paths.outputs, items)
    models = [f"{c}@{a}" for c in _children(args, config) for a in args.arms]
    argv = run_argv(args, models, list(items))
    print(command_line(argv))
    if args.print_only:
        return 0
    from .main import main
    return main(argv)


def _assemble(args) -> int:
    from ..plugin import posthoc
    config = _config(args)
    ctx = context(args)
    outputs = ctx.paths.outputs
    items = config.load_items(args.items)
    for dataset, ids in items.items():
        folder = outputs / dataset
        sources = sorted(folder.glob("*@*.stream.jsonl")) if folder.is_dir() else []
        if args.formal:
            sources += sorted((args.formal / dataset).glob("*.jsonl"))
        sources = [p for p in sources if not p.name.endswith(".scores.jsonl")]
        prompts = posthoc.prompts_from(sources, set(ids))
        added = posthoc.write_refusal_job(outputs, dataset, ids, prompts, config.refusal)
        print(f"{dataset}: {added} fixed-refusal rows added ({len(ids)} items)")
    print("next: " + command_line([*global_options(args, outputs=True), "score", "--judge"]))
    return 0


def _report(args) -> int:
    from ..analysis.records import Catalog
    from ..analysis.specs import profile_from_config
    from ..core.io import write_json
    from ..plugin import artefacts
    from ..plugin.report import Recovery, figure_data, markdown
    config = _config(args)
    ctx = context(args)
    children = _children(args, config)
    items = config.load_items(args.items)
    profile = profile_from_config(ctx.config, args.profile)
    missing = [c for c in children if c not in profile.parents]
    if missing:
        raise ValueError(f"no parent for {missing}: the pair of a child belongs in configs/pairs.yaml or an extra "
                         "--config file (pairs: {<pair>: {parent: ..., children: [...]}})")
    arms = ("sfix", config.splus_arm)
    rec = Recovery(profile, Catalog(), args.formal or ctx.paths.outputs, ctx.paths.outputs, items, children,
                   {c: artefacts.load(c) for c in children}, config, arms)
    variants = ("higher", "rule", "old") if args.variant == "all" else (args.variant,)
    result = rec.compute(variants)
    for note in rec.notes:
        print("note:", note, file=sys.stderr)
    if rec.problems:
        print("incomplete inputs:\n  " + "\n  ".join(rec.problems), file=sys.stderr)
        if not args.interim:
            return 2
    out = args.out or ctx.paths.outputs / "plugin"
    write_json(out / "report.json", result)
    (out / "REPORT.md").write_text(markdown(result, {c: profile.name_of(c) for c in children},
                                            f"Plugin recovery ({args.items})"), encoding="utf-8")
    write_json(out / "figure.json", figure_data(result, variants[0]))
    print(f"wrote {out}/report.json, REPORT.md, figure.json ({len(result['cells'])} cells)")
    return 0


# ---------- calibrate ----------

def stage_block(config, stage: str, child: str, parent: str | None):
    """The adapter block whose environment runs a GPU stage: directions captures the child's and then the parent's
    offline pipeline in one process (both must use the same environment), harmless runs the child's stream (its
    plugin arms keep the stream's environment, also when configs/plugin.yaml `engines` swaps the adapter)."""
    spec = config.model(child)
    if stage == "harmless":
        return spec.adapter("stream")
    block = spec.adapter("offline")
    if parent is not None:
        other = config.model(parent).adapter("offline")
        if other.env != block.env:
            raise ValueError(f"stage directions loads {child} and {parent} in one process, but their offline "
                             f"environments differ ({block.env}, {other.env})")
    return block


def stage_command(args, ctx, block, child: str, parent: str | None, work: Path) -> tuple[list[str], dict]:
    """(command, extra environment variables) that run this calibration stage in the model environment: its
    interpreter with `-m streamreplace`, this invocation's global options and the stage's options (paths made
    absolute), and the variables of a run worker (run/envs.py: this checkout's src/ on PYTHONPATH, HF_HUB_OFFLINE,
    thread caps, the FFmpeg libraries for models marked ffmpeg, the GPU slot)."""
    from ..run.envs import default_threads, python_for, worker_vars
    argv = [*global_options(args, outputs=True), "plugin", "calibrate", child, "--stage", args.stage,
            "--work", os.path.abspath(work)]
    if parent is not None:
        argv += ["--parent", parent]
    for flag, value in (("--advbench", args.advbench), ("--alpaca", args.alpaca), ("--clips", args.clips)):
        if value is not None:
            argv += [flag, os.path.abspath(value)]
    ffmpeg = str(args.ffmpeg)
    argv += ["--ffmpeg", os.path.abspath(ffmpeg) if os.sep in ffmpeg else ffmpeg]
    gpu = (args.gpus or list(ctx.config.run.get("gpus") or [0]))[0]
    variables = worker_vars(ctx.paths, block.env, gpu, default_threads(1), block.ffmpeg)
    variables[WORKER] = "1"
    return [str(python_for(ctx.paths, block.env)), "-m", "streamreplace", *argv], variables


def _in_model_env(args, ctx, child: str, parent: str | None, work: Path) -> int | None:
    """Runs a GPU stage again in its model environment and returns the exit code; None when this process already is
    that stage's process (or the model runs in this interpreter: env "local")."""
    if os.environ.get(WORKER) == "1":
        return None
    block = stage_block(ctx.config, args.stage, child, parent)
    if block.env == "local":
        return None
    from ..paths import project_root
    command, variables = stage_command(args, ctx, block, child, parent, work)
    if not Path(command[0]).exists():
        raise ValueError(f"stage {args.stage} of {child} runs in the {block.env!r} environment, which is not built "
                         f"({command[0]}): streamreplace env build {block.env}")
    print(f"stage {args.stage}: {block.env} environment, GPU {variables['CUDA_VISIBLE_DEVICES']}\n  "
          + command_line(command[1:], shlex.quote(command[0])), flush=True)
    return subprocess.call(command, env={**os.environ, **variables}, cwd=str(project_root()))


def _calibrate(args) -> int:
    from ..core.io import read_latest
    from ..plugin.calibrate import freeze, thresholds as T
    config = _config(args)
    ctx = context(args)
    child = args.child
    work = args.work or ctx.paths.outputs / "plugin_calibration" / child
    parent = args.parent or next((p.parent for p in ctx.config.pairs.values() if child in p.children), None)
    if args.stage == "directions":
        if not (args.advbench and args.alpaca):
            raise ValueError("stage directions needs --advbench and --alpaca (your own copies; docs/plugin.md)")
        if parent is None:
            raise ValueError(f"no parent for {child}: pass --parent or add the pair to configs/pairs.yaml")
    if args.stage in ("directions", "harmless"):
        code = _in_model_env(args, ctx, child, parent if args.stage == "directions" else None, work)
        if code is not None:
            return code
    if args.stage == "directions":
        return _directions(args, config, ctx, child, parent, work)
    if args.stage == "freeze":
        path = freeze.freeze(child, parent, work / "directions.npz", force=args.force,
                             summary=json.loads((work / "directions.json").read_text()))
        print(f"wrote {path}")
        return 0
    if args.stage == "record":
        items = config.load_items("threshold")
        rows = {d: read_latest(ctx.paths.outputs / d / f"{child}@record.stream.jsonl") for d in items}
        closed = T.tau_closed(T.record_projections(rows, items))
        print(f"{child}: tau_closed {closed} on {sum(map(len, items.values()))} threshold items")
        print(freeze.set_thresholds(child, closed=closed))
        return 0
    return _harmless(args, config, ctx, child, work)


def _directions(args, config, ctx, child, parent, work) -> int:
    from ..models.registry import create
    from ..models.upstream import Upstream
    from ..plugin.calibrate import directions, inputs
    from ..data.samples import load_samples
    texts = inputs.texts_from_frozen(config.calibration_path("texts"), args.advbench, args.alpaca)
    videos = inputs.calibration_videos(load_samples("videomme"))
    problems = inputs.check_videos(videos, config.calibration_path("videos"))
    if problems:
        raise ValueError("; ".join(problems))
    f0, f1 = inputs.calib_inputs(texts, videos, config.calibration.get("fold0_videos", 96),
                                 config.calibration.get("fold1_videos", 48))
    bench = config.calibration.get("capture_benchmark", "video_safetybench")
    root = ctx.paths.dataset_dir("videomme")

    def adapter(model):
        spec = ctx.config.model(model)
        return lambda: create(spec.offline.adapter, spec, ctx.config.settings(bench, model, "offline"), Upstream())

    def media(video):
        return None if video is None else str(video if Path(video).is_absolute() else root / video)

    def requests_of(role, items):
        model = child if role == "child" else parent
        settings = ctx.config.settings(bench, model, "offline")
        return [inputs.request((k, t, media(v), lab), settings) for k, t, v, lab in items]

    summary = directions.build(adapter(child), adapter(parent), f0, f1, requests_of, work / "directions.npz")
    print(json.dumps(summary, indent=1))
    return 0


def _harmless(args, config, ctx, child, work) -> int:
    from ..core.io import read_jsonl
    from ..plugin import artefacts
    from ..plugin.calibrate import harmless, freeze, thresholds as T
    items = [it for it in harmless.load(config.calibration_path("harmless")) if it["role"] == "fit"]
    out = work / f"{child}@sfix.harmless.jsonl"
    if not out.exists() or len({r["id"] for r in read_jsonl(out) if not r.get("error")}) < len(items):
        from ..models.registry import create
        from ..models.upstream import Upstream
        from ..plugin.hooks import make_hooks, stream_adapter
        if not args.clips:
            raise ValueError("stage harmless needs --clips (the benign clips; cut from your Video-MME copy when missing)")
        from ..data.samples import load_samples
        root = ctx.paths.dataset_dir("videomme")
        sources = {s.metadata["video"]: (Path(s.video) if Path(s.video).is_absolute() else root / s.video)
                   for s in load_samples("videomme")}
        harmless.cut(items, sources.__getitem__, args.clips, args.ffmpeg)
        hooks = make_hooks(child, "sfix", config)
        spec = ctx.config.model(child)
        settings = hooks.settings(ctx.config.settings(config.calibration.get("harmless_benchmark", "video_safetybench"),
                                                      child, "stream"))
        adapter = create(stream_adapter(child, spec.stream.adapter, config), spec, settings, Upstream())
        errors = harmless.run(adapter, hooks, harmless.requests(items, args.clips, settings), out)
        adapter.close()
        if errors:
            print(f"{errors} harmless-request requests failed; run the stage again", file=sys.stderr)
            return 1
    closed = artefacts.read_manifest(child)["thresholds"].get("closed")
    if closed is None:
        raise ValueError(f"{child}: run stage record first (tau_closed)")
    free, q = T.tau_free(closed, T.fit_projections(read_jsonl(out), [it["id"] for it in items]))
    print(f"{child}: harmless q97.5 {q}, tau_free {free}")
    print(freeze.set_thresholds(child, free_q=q))
    return 0


def _verify(args) -> int:
    from ..plugin import artefacts
    config = _config(args)
    children = args.children or list(config.children)
    problems = []
    for child in children:
        found = artefacts.check(child)
        problems += found
        if not found:
            art = artefacts.load(child)
            print(f"{child}: ok  sha256 {art.sha256[:12]}  r_layer {art.r_layer}  p_layer {art.p_layer}  "
                  f"alpha {art.alpha}  tau closed {art.thresholds['closed']} / free {art.thresholds['free']}")
    for name in list(config.items) + ["texts", "videos", "harmless"]:
        path = config.items_path(name) if name in config.items else config.calibration_path(name)
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path}: {exc}")
    for p in problems:
        print("PROBLEM:", p, file=sys.stderr)
    print("plugin verify:", "ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0
