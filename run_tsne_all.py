"""
Batch runner for ``tsne_visualization.py``.

Runs the t-SNE visualization sequentially over every per-mouse folder found
under a dataset root, forwarding *all* of the underlying script's options
unchanged so a batch run is identical to running it by hand in each folder --
just repeated for every mouse.

Mouse folders
-------------
A dataset root holds one folder per mouse, named like ``1lc`` / ``2mp`` (a
number then the ``lc``/``mp`` cohort tag); each holds a
``Cluster_detail_results.csv`` and a ``session_1_out.mat`` (see the sibling
Cluster_Comparison/dataset_config.py for the same convention). Folders are
discovered with the regex --pattern (default ``(\\d+)(lc|mp)$``, case-
insensitive) matched against the folder name; the captured ``<n><cohort>`` is
the mouse id used in logs. Only folders that actually contain the details CSV
are kept, so stray directories are skipped rather than failing mid-run.

Options are preserved
---------------------
Anything this wrapper does not recognize is forwarded verbatim to
tsne_visualization.py, so every flag it accepts works here too
(``--color-by``, ``--weekly``, ``--per-week-embedding``, ``--palette``,
rendering flags, ...). The wrapper injects ``--data-root <mouse folder>`` last
(so it wins) for each run; the .mat / details / temporal names keep the
underlying script's defaults unless you forward your own.

Save / load embedding (the expensive step) per mouse
----------------------------------------------------
  * Forward ``--save-embedding`` and each mouse's embedding is cached in its
    own ``<folder>\\graphs\\TSNE\\embedding_multiscale.csv``.
  * Pass ``--reuse-embedding`` (a wrapper flag) and any mouse that already has
    that cached CSV is re-plotted from it via ``--load-embedding`` -- so a
    second pass (e.g. to re-color) skips the similarity-load / PCA / openTSNE
    for mice already done. Mice without a cache compute normally.
Do NOT forward a bare ``--load-embedding <path>``: it is one fixed file and
would be (mis)applied to every mouse. Use ``--reuse-embedding`` instead.

Examples
--------
    # Global cluster plot for every mouse, caching each embedding:
    python run_tsne_all.py --dataset-root C:\\mitopark_tsne \\
        --save-embedding --weekly

    # Re-color everything by the TBA gradient, reusing the cached embeddings:
    python run_tsne_all.py --dataset-root C:\\mitopark_tsne \\
        --reuse-embedding --color-by tba --weekly

    # Just the lc cohort, separate per-week embeddings, see the commands first:
    python run_tsne_all.py --dataset-root C:\\mitopark_tsne \\
        --pattern "(\\d+)lc$" --per-week-embedding --save-weekly-embeddings \\
        --skip-global --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SCRIPT = os.path.join(HERE, "tsne_visualization.py")
DEFAULT_PATTERN = r"(\d+)(lc|mp)$"
# Mirror tsne_visualization.py's defaults so we can locate a cached embedding
# and skip folders that obviously lack the inputs.
DETAILS_NAME = "Cluster_detail_results.csv"
EMBEDDING_REL = os.path.join("graphs", "TSNE", "embedding_multiscale.csv")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run tsne_visualization.py over every mouse folder in a "
                    "dataset root. Unknown options are forwarded to it verbatim.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Any option not listed above is passed straight through to "
               "tsne_visualization.py (e.g. --color-by, --weekly, --palette).",
    )
    p.add_argument("--dataset-root", required=True,
                   help="Directory holding the per-mouse folders (1lc, 2mp, ...).")
    p.add_argument("--pattern", default=DEFAULT_PATTERN,
                   help="Case-insensitive regex a folder name must match to be "
                        "treated as a mouse. The full match is the mouse id.")
    p.add_argument("--only", nargs="+", default=None, metavar="MOUSE",
                   help="Restrict to these folder names (e.g. 1lc 2mp).")
    p.add_argument("--details-name", default=DETAILS_NAME,
                   help="Details CSV that must exist for a folder to be run; "
                        "also forwarded to the underlying script.")
    p.add_argument("--reuse-embedding", action="store_true",
                   help="For any mouse with a cached embedding_multiscale.csv, "
                        "add --load-embedding <that file> so it re-plots without "
                        "recomputing. Mice without a cache compute normally.")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter used to run the visualization script.")
    p.add_argument("--script", default=DEFAULT_SCRIPT,
                   help="Path to tsne_visualization.py.")
    p.add_argument("--continue-on-error", dest="continue_on_error",
                   action="store_true", default=True,
                   help="Keep going after a mouse fails (default).")
    p.add_argument("--stop-on-error", dest="continue_on_error",
                   action="store_false",
                   help="Abort the whole batch on the first failure.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the command for each mouse without running it.")
    return p.parse_known_args(argv)


def discover_mice(root, pattern, only, details_name):
    """Return [(mouse_id, folder_path), ...] sorted by cohort then number.

    A folder qualifies if its name matches `pattern` (case-insensitive) and it
    contains `details_name`. `only`, if given, restricts to those folder names.
    """
    rx = re.compile(pattern, re.IGNORECASE)
    only = set(only) if only else None
    found = []
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            continue
        if only is not None and name not in only:
            continue
        m = rx.search(name)
        if not m:
            continue
        if not os.path.isfile(os.path.join(folder, details_name)):
            print(f"[skip] {name}: no {details_name}")
            continue
        found.append((m.group(0), folder))
    # Sort by cohort (lc/mp) then numeric id so the run reads 1lc, 2lc, ..., 1mp.
    def key(item):
        mm = re.search(r"(\d+)(\D*)$", item[0])
        num = int(mm.group(1)) if mm else 0
        cohort = mm.group(2).lower() if mm else item[0]
        return (cohort, num)
    return sorted(found, key=key)


def output_root_for(folder, forwarded):
    """Where the underlying script will write for this folder, honoring a
    forwarded --output-root (else its default of <folder>\\graphs)."""
    if "--output-root" in forwarded:
        i = forwarded.index("--output-root")
        if i + 1 < len(forwarded):
            return forwarded[i + 1]
    return os.path.join(folder, "graphs")


def build_command(args, folder, forwarded):
    """Assemble the subprocess command for one mouse folder."""
    cmd = [args.python, args.script, *forwarded]
    if args.reuse_embedding and "--load-embedding" not in forwarded:
        cached = os.path.join(folder, EMBEDDING_REL)
        if os.path.isfile(cached):
            cmd += ["--load-embedding", cached]
    # Inject the per-mouse data root and details name LAST so they override any
    # value the user forwarded.
    cmd += ["--data-root", folder, "--details-name", args.details_name]
    return cmd


def main(argv=None):
    args, forwarded = parse_args(argv)

    if not os.path.isdir(args.dataset_root):
        sys.exit(f"[error] dataset root not found: {args.dataset_root}")
    if not os.path.isfile(args.script):
        sys.exit(f"[error] visualization script not found: {args.script}")
    if "--load-embedding" in forwarded and not args.reuse_embedding:
        print("[warn] a forwarded --load-embedding is a single fixed file and "
              "will be applied to EVERY mouse. Use --reuse-embedding for "
              "per-mouse cached embeddings.")

    mice = discover_mice(args.dataset_root, args.pattern, args.only, args.details_name)
    if not mice:
        sys.exit(f"[error] no mouse folders matching /{args.pattern}/i with a "
                 f"{args.details_name} under {args.dataset_root}")

    print(f"[..] {len(mice)} mouse folder(s): {', '.join(m for m, _ in mice)}")
    results = []  # (mouse, returncode)
    batch_t0 = time.time()
    for idx, (mouse, folder) in enumerate(mice, 1):
        cmd = build_command(args, folder, forwarded)
        header = f"[{idx}/{len(mice)}] mouse {mouse}  ({folder})"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        print("  " + " ".join(_quote(c) for c in cmd))
        if args.dry_run:
            results.append((mouse, None))
            continue
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        dt = time.time() - t0
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        print(f"[{status}] mouse {mouse} in {dt:.1f}s")
        results.append((mouse, rc))
        if rc != 0 and not args.continue_on_error:
            sys.exit(f"[error] mouse {mouse} failed (exit {rc}); stopping "
                     f"(--continue-on-error to keep going).")

    # Summary
    print("\n" + "=" * 40)
    print("Batch summary" + (" (dry run)" if args.dry_run else
                             f" in {time.time() - batch_t0:.1f}s"))
    print("=" * 40)
    failures = 0
    for mouse, rc in results:
        if args.dry_run:
            mark = "planned"
        elif rc == 0:
            mark = "ok"
        else:
            mark = f"FAILED (exit {rc})"
            failures += 1
        print(f"  {mouse:<10} {mark}")
    if failures:
        sys.exit(f"[done] {failures}/{len(results)} mouse folder(s) failed.")
    print("[done]")


def _quote(s):
    """Minimal shell-ish quoting for the echoed command (display only)."""
    return f'"{s}"' if (" " in s or not s) else s


if __name__ == "__main__":
    main()
