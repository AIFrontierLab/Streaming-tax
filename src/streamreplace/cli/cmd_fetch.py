"""`streamreplace fetch-upstream`: the third-party code the adapters and scorers use, at the pinned commits.

    streamreplace fetch-upstream [NAME ...]          fetch (default: all) into the upstream root and verify
    streamreplace fetch-upstream --verify-only       check what is there against the pinned blobs (offline)
    streamreplace fetch-upstream --list              what manifests/upstream.yaml pins

NAMEs are entries of manifests/upstream.yaml, `all`, or a group (`core`: P/C/S evaluation and scoring,
`generalization`: prompt robustness and fine-tuning). Files go to <upstream root>/<name>/<path in the repository>
(upstream root: --paths / STREAMREPLACE_UPSTREAM, default $STREAMREPLACE_HOME/upstream). Every file is checked
against its git blob SHA-1; a tree that already verifies is kept unless --force. Reference-only entries (AURA,
MOSS-VL) document provenance and are never fetched. Exit code 1 when anything is missing or does not verify."""
from __future__ import annotations

import argparse
import json
import sys

from . import context


def register(subparsers) -> None:
    parser = subparsers.add_parser("fetch-upstream", help="download the pinned third-party code and verify it",
                                   description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", metavar="NAME", help="upstream entries, groups (core, generalization) or "
                        "all (default: all)")
    parser.add_argument("--verify-only", action="store_true", help="only check the fetched files (no network)")
    parser.add_argument("--list", action="store_true", help="list the pinned repositories and exit")
    parser.add_argument("--force", action="store_true", help="download again even when the tree verifies")
    parser.add_argument("--method", choices=("auto", "raw", "tarball"), default="auto",
                        help="raw: one request per file; tarball: one archive per repository; auto: raw, then the "
                             "tarball for a repository whose raw download fails (default)")
    parser.add_argument("--from", dest="source", metavar="DIR",
                        help="copy from existing checkouts DIR/<name>/ instead of downloading (still verified)")
    parser.add_argument("--jobs", type=int, default=8, help="parallel downloads per repository (default 8)")
    parser.add_argument("--manifest", help="manifest file (default: manifests/upstream.yaml)")
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    parser.set_defaults(func=run)


def _list(manifest) -> int:
    print(f"{'name':18s} {'group':14s} {'files':>5s}  {'commit':10s} {'licence':28s} repository")
    for entry in manifest.values():
        group = entry.group if entry.fetch else "reference"
        patched = f" (+{len(entry.patches)} patched)" if entry.patches else ""
        print(f"{entry.name:18s} {group:14s} {len(entry.files):5d}  {entry.commit[:10]} {entry.license[:28]:28s} "
              f"{entry.repo}{patched}")
    return 0


def run(args) -> int:
    from ..provisioning import upstream
    manifest = upstream.load_manifest(args.manifest)
    if args.list:
        return _list(manifest)
    try:
        entries = upstream.select(manifest, args.names)
    except upstream.ManifestError as exc:
        print(f"streamreplace fetch-upstream: {exc}", file=sys.stderr)
        return 2
    root = context(args).paths.upstream
    log = (lambda message: print(message, file=sys.stderr)) if args.json else print
    if not args.json:
        print(f"upstream root: {root}")
    if args.verify_only:
        statuses = {e.name: upstream.verify(e, root) for e in entries}
        for status in statuses.values():
            status.action = status.action or "verified"
            if not args.json:
                print(f"{status.name:18s} {status.summary()}")
    else:
        method = "local" if args.source else args.method
        statuses = upstream.fetch_all(entries, root, method=method, force=args.force, jobs=args.jobs,
                                      source=args.source, log=log)
    bad = [s.name for s in statuses.values() if not s.ok]
    if args.json:
        print(json.dumps({"root": str(root), "results": [s.as_dict() for s in statuses.values()]}, indent=1))
    else:
        total = sum(len(e.files) for e in entries)
        print(f"{len(entries) - len(bad)}/{len(entries)} repositories verified ({total} pinned files)"
              + (f"; not ok: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0
