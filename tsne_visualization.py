"""
Standalone t-SNE visualization for accelerometer cluster data.

Extracted from ``visualizations_pipeline.ipynb`` (the "TSNE" section) and turned
into a command-line tool.

It reads a MATLAB ``session_*_out.mat`` similarity matrix plus a
``Cluster_detail_results.csv`` (which supplies per-point cluster labels and the
recording folder/week each point belongs to), reduces the similarity matrix with
PCA, runs openTSNE, and colors the embedding.

Color masks (--color-by)
------------------------
  cluster        cluster index -> glasbey palette (categorical, default)
  temporal_class early/mid/late/sustained/... (categorical)
  tba_class      high/low total-body-accel class (categorical)
  tba            total-body-accel value (continuous gradient)
The last three come from ``Cluster_detail_results_temporal.csv`` (same rows as
the main details file, with extra columns). It is attached automatically when
present.

Three things it can produce:
  1. A single global embedding.
  2. Per-week views of that *same* global embedding (subset after embedding),
     one PNG per week plus an animated GIF. (--weekly)
  3. A *separate* embedding computed independently for each week (subset the
     similarity matrix before embedding), one PNG per week plus a GIF, with an
     option to save each week's embedding to CSV. (--per-week-embedding)

The embedding step is the expensive part, so the global embedding can be saved
to CSV (--save-embedding) and reloaded later (--load-embedding) to re-plot
without recomputing.

Example
-------
    python tsne_visualization.py \
        --data-root  C:\\mitopark_tsne\\1lc \
        --output-root C:\\mitopark_tsne\\1lc\\graphs \
        --palette glasbey \
        --color-by cluster \
        --save-embedding \
        --weekly

    # Re-color a saved embedding by TBA gradient (no expensive recompute):
    python tsne_visualization.py --load-embedding <path>\\embedding_multiscale.csv \
        --color-by tba --weekly

    # A fresh embedding per week, saving each one:
    python tsne_visualization.py --per-week-embedding --save-weekly-embeddings --skip-global
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

import h5py
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA

import openTSNE


# --------------------------------------------------------------------------- #
# Palettes
# --------------------------------------------------------------------------- #
# glasbey variants are generated on demand (distinct colors sized to the number
# of categories); the plotly names are fixed qualitative palettes that get cycled.
GLASBEY_PALETTES = {
    "glasbey": dict(),
    "glasbey_light": dict(lightness_bounds=(50, 90)),
    "glasbey_dark": dict(lightness_bounds=(10, 55)),
    "glasbey_cb": dict(colorblind_safe=True),
}
PLOTLY_PALETTES = [
    "Plotly", "D3", "G10", "Dark24", "Light24", "Alphabet",
    "Bold", "Safe", "Set1", "Pastel", "Antique", "Prism",
]
PALETTE_CHOICES = list(GLASBEY_PALETTES) + PLOTLY_PALETTES


def palette_colors(palette, n):
    """Return n hex colors from the named palette (glasbey generated, plotly cycled)."""
    if palette in GLASBEY_PALETTES:
        import glasbey  # imported lazily so plotly-only palettes don't need it
        return list(glasbey.create_palette(palette_size=n, **GLASBEY_PALETTES[palette]))
    base = getattr(px.colors.qualitative, palette)
    return [base[i % len(base)] for i in range(n)]


# --------------------------------------------------------------------------- #
# Color masks
# --------------------------------------------------------------------------- #
COLOR_CHOICES = ["cluster", "temporal_class", "tba_class", "tba"]
TEMPORAL_COLUMNS = {"temporal_class", "tba_class", "tba"}
COLOR_LABELS = {
    "cluster": "Cluster",
    "temporal_class": "Temporal class",
    "tba_class": "TBA class",
    "tba": "TBA",
}
# Hand-picked colors for the categorical temporal masks (falls back to the chosen
# palette for any category not listed here). NaN/missing -> "unknown" -> gray.
PREFERRED_COLORS = {
    "temporal_class": {
        "early": "#4575b4", "mid": "#74add1", "late": "#f46d43",
        "sustained": "#d73027", "uncategorized": "#bdbdbd", "unknown": "#7a7a7a",
    },
    "tba_class": {"low": "#5c95db", "high": "#e83a3a", "unknown": "#7a7a7a"},
}
PREFERRED_ORDER = {
    "temporal_class": ["early", "mid", "late", "sustained", "uncategorized", "unknown"],
    "tba_class": ["low", "high", "unknown"],
}
# De-emphasized categories drawn first so they sit UNDERNEATH the meaningful
# points (plotly draws earlier categories at the bottom of the z-order).
BACKGROUND_CATEGORIES = {"uncategorized", "unknown"}


class ColorSpec:
    """Everything needed to color a scatter, computed once over all points so the
    global plot and every weekly frame share the same mask."""

    def __init__(self, color_by, values, mode, discrete_map=None,
                 category_order=None, colorscale=None, crange=None):
        self.color_by = color_by
        self.values = values          # full-length array aligned to details rows
        self.mode = mode              # "categorical" | "continuous"
        self.discrete_map = discrete_map
        self.category_order = category_order
        self.colorscale = colorscale
        self.crange = crange          # (min, max) for continuous
        self.label = COLOR_LABELS[color_by]


def build_coloring(details, color_by, palette, gradient_colorscale):
    if color_by in TEMPORAL_COLUMNS and color_by not in details.columns:
        sys.exit(f"[error] --color-by {color_by} needs the '{color_by}' column; "
                 f"provide Cluster_detail_results_temporal.csv (see --temporal-name).")

    if color_by == "tba":  # continuous gradient
        values = pd.to_numeric(details["tba"], errors="coerce").to_numpy(dtype=float)
        crange = (float(np.nanmin(values)), float(np.nanmax(values)))
        return ColorSpec(color_by, values, "continuous",
                         colorscale=gradient_colorscale, crange=crange)

    if color_by == "cluster":
        values = details["ClusterLabel"].to_numpy()
        names = sorted(set(values), key=lambda s: int(s))
        colors = palette_colors(palette, len(names))
        discrete_map = {n: colors[i] for i, n in enumerate(names)}
        order = names
    else:  # temporal_class / tba_class
        values = details[color_by].astype("string").fillna("unknown").to_numpy()
        present = list(pd.unique(values))
        pref_order = PREFERRED_ORDER[color_by]
        semantic = ([c for c in pref_order if c in present]
                    + [c for c in present if c not in pref_order])
        # Draw background categories first (bottom) so meaningful points sit on top.
        order = ([c for c in semantic if c in BACKGROUND_CATEGORIES]
                 + [c for c in semantic if c not in BACKGROUND_CATEGORIES])
        fallback = palette_colors(palette, len(order))
        preferred = PREFERRED_COLORS[color_by]
        discrete_map = {name: preferred.get(name, fallback[i]) for i, name in enumerate(order)}

    return ColorSpec(color_by, values, "categorical",
                     discrete_map=discrete_map, category_order=order)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_details(details_path, temporal_path=None):
    """Load Cluster_detail_results.csv and derive per-point cluster/week labels.

    Rows are aligned 1:1 with the similarity-matrix rows. Adds:
      * ``ClusterLabel`` - cluster id as a string
      * ``Week`` / ``WeekSort`` - week label from Folder_Name (forward-filled) + sort key
    If a temporal CSV is available and row-aligned, its temporal_class/tba/tba_class
    columns are attached (unless already present).
    """
    df = pd.read_csv(details_path)
    if "ClusterIdx" not in df.columns:
        raise KeyError(f"'ClusterIdx' column not found in {details_path}")

    df["ClusterLabel"] = df["ClusterIdx"].astype(int).astype(str)

    folder = df.get("Folder_Name")
    if folder is None:
        raise KeyError(f"'Folder_Name' column not found in {details_path}")
    # Forward-fill occasional gaps so every point inherits a week (matches the
    # notebook's combined_matrix convention).
    df["Week"] = folder.ffill().astype("string")
    df["WeekSort"] = df["Week"].map(week_sort_key)

    if not TEMPORAL_COLUMNS.issubset(df.columns) and temporal_path and os.path.exists(temporal_path):
        _attach_temporal(df, temporal_path)
    return df


def _attach_temporal(df, temporal_path):
    """Attach temporal_class/tba/tba_class from a row-aligned temporal CSV."""
    tdf = pd.read_csv(temporal_path)
    if len(tdf) != len(df):
        print(f"[warn] {os.path.basename(temporal_path)} has {len(tdf)} rows vs "
              f"{len(df)}; skipping temporal columns.")
        return
    if "ClusterIdx" in tdf.columns and not np.array_equal(
            tdf["ClusterIdx"].to_numpy(), df["ClusterIdx"].to_numpy()):
        print(f"[warn] {os.path.basename(temporal_path)} ClusterIdx does not align; "
              f"skipping temporal columns.")
        return
    for col in TEMPORAL_COLUMNS:
        if col in tdf.columns:
            df[col] = tdf[col].to_numpy()
    print(f"[ok] attached temporal columns from {os.path.basename(temporal_path)}")


def week_sort_key(label):
    """Numeric-ish sort key from a folder/week label like 'w10' or 'w24_ldopa'.

    'w10' -> (10, '') ; 'w24_saline' -> (24, 'saline') ; unknown -> (inf, str).
    """
    if not isinstance(label, str):
        return (float("inf"), "")
    m = re.search(r"(?:week|w)\s*(\d+)", label, re.IGNORECASE)
    if not m:
        return (float("inf"), label)
    suffix = label[m.end():].lstrip("_ ").lower()
    return (int(m.group(1)), suffix)


def ordered_weeks(details, only=None):
    """Return week labels in chronological order, optionally filtered to `only`."""
    weeks = [w for w in details["Week"].dropna().unique()]
    weeks.sort(key=week_sort_key)
    if only:
        only = set(only)
        missing = only - set(weeks)
        if missing:
            print(f"[warn] requested weeks not present and skipped: {sorted(missing)}")
        weeks = [w for w in weeks if w in only]
    return weeks


# --------------------------------------------------------------------------- #
# Embedding
# --------------------------------------------------------------------------- #
def pca_reduce(matrix, n_components, seed):
    n_components = min(n_components, matrix.shape[0], matrix.shape[1])
    return PCA(n_components=n_components, random_state=seed).fit_transform(matrix)


def valid_perplexities(perplexities, n_points):
    """Keep only perplexities small enough for the sample size (openTSNE needs
    3*perplexity < n_points); fall back to a single sane value if none fit."""
    good = [p for p in perplexities if 3 * p < n_points]
    if not good:
        fallback = max(2, min(30, (n_points - 1) // 3))
        print(f"[warn] no requested perplexity fits {n_points} points; "
              f"using perplexity={fallback}")
        return [fallback]
    if good != list(perplexities):
        print(f"[warn] clamped perplexities {list(perplexities)} -> {good} "
              f"for {n_points} points")
    return good


def compute_embedding(reduced, perplexities, metric, n_jobs, seed):
    """Run multiscale openTSNE on a PCA-reduced matrix. Returns an (N, 2) array."""
    perplexities = valid_perplexities(perplexities, reduced.shape[0])
    t0 = time.time()
    affinities = openTSNE.affinity.Multiscale(
        reduced,
        perplexities=perplexities,
        metric=metric,
        n_jobs=n_jobs,
        random_state=seed,
    )
    init = openTSNE.initialization.pca(reduced, random_state=42)
    embedding = openTSNE.TSNE(n_jobs=n_jobs).fit(
        affinities=affinities,
        initialization=init,
    )
    print(f"[time] openTSNE fit: {time.time() - t0:.1f}s "
          f"({reduced.shape[0]} points, perplexities={perplexities})")
    return np.asarray(embedding)


# --------------------------------------------------------------------------- #
# Plotting / GIF
# --------------------------------------------------------------------------- #
def sanitize(name):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(name)).strip("_")


def plot_embedding(embedding, values, spec, title, out_path, args, show=False):
    """Scatter an (N, 2) embedding colored per `spec` and write it to `out_path`."""
    kwargs = dict(
        x=embedding[:, 0],
        y=embedding[:, 1],
        color=values,
        title=title,
        labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": spec.label},
    )
    if spec.mode == "categorical":
        kwargs["color_discrete_map"] = spec.discrete_map
        kwargs["category_orders"] = {"color": spec.category_order}
    else:
        kwargs["color_continuous_scale"] = spec.colorscale
        kwargs["range_color"] = spec.crange

    fig = px.scatter(**kwargs)
    fig.update_traces(marker=dict(size=args.marker_size))
    fig.update_layout(width=args.width, height=args.height)
    fig.write_image(out_path)
    if show:
        fig.show()
    return out_path


def build_gif(png_paths, gif_path, fps):
    """Assemble PNG frames into a looping GIF at the given fps (needs >=2 frames).

    All frames are quantized to a single shared palette built from every frame,
    so the GIF has one global color table. Without that, Pillow gives each RGB
    frame its own local palette and many viewers (Windows Photos, some browsers)
    render only the first frame -- the GIF looks static even though the frames
    are there.
    """
    if len(png_paths) < 2:
        print(f"[warn] only {len(png_paths)} frame(s); skipping GIF {gif_path}")
        return None
    from PIL import Image

    rgb = [Image.open(p).convert("RGB") for p in png_paths]
    # One palette that covers colors from every frame (stack them, then quantize).
    montage = Image.fromarray(np.concatenate([np.asarray(im) for im in rgb], axis=0))
    shared_palette = montage.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    frames = [im.quantize(palette=shared_palette, dither=Image.Dither.NONE) for im in rgb]

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=[int(round(1000 / fps))] * len(frames),
        loop=0,
        disposal=2,       # clear each frame before drawing the next
        optimize=False,   # don't let the optimizer merge/drop frames
    )
    print(f"[ok] wrote GIF ({len(frames)} frames @ {fps} fps): {gif_path}")
    return gif_path


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def run_weekly_subset(embedding, details, spec, out_root, args):
    """Mode: subset the ALREADY-computed global embedding to each week and plot."""
    sub_dir = os.path.join(out_root, f"weekly_by_week_{args.color_by}")
    os.makedirs(sub_dir, exist_ok=True)
    weeks = ordered_weeks(details, args.weeks)
    frames = []
    for week in weeks:
        mask = (details["Week"] == week).to_numpy()
        n = int(mask.sum())
        if n == 0:
            continue
        title = f"openTSNE by {spec.label} - Week {week} (n={n})"
        out_path = os.path.join(sub_dir, f"tsne_week_{sanitize(week)}.png")
        plot_embedding(embedding[mask], spec.values[mask], spec, title, out_path,
                       args, show=args.show)
        print(f"[ok] week {week}: {out_path}")
        frames.append(out_path)
    build_gif(frames, os.path.join(sub_dir, f"tsne_weekly_{args.color_by}.gif"), args.gif_fps)


def run_per_week_embedding(mat_path, details, spec, out_root, args):
    """Mode: compute a SEPARATE embedding per week (subset the similarity matrix
    to that week's rows/cols before embedding)."""
    sub_dir = os.path.join(out_root, f"weekly_embeddings_{args.color_by}")
    os.makedirs(sub_dir, exist_ok=True)
    weeks = ordered_weeks(details, args.weeks)
    frames = []
    with h5py.File(mat_path, "r") as f:
        sim = f["Clusters"]["sim"]
        for week in weeks:
            mask = (details["Week"] == week).to_numpy()
            n = int(mask.sum())
            if n < args.min_week_points:
                print(f"[skip] week {week}: {n} points < min-week-points "
                      f"({args.min_week_points})")
                continue
            print(f"[..] week {week}: reading {n}x{n} similarity submatrix")
            # h5py fancy-indexes one axis at a time: pull the week's rows, then
            # subset the columns in-memory (submatrix is small).
            rows = sim[mask, :]
            sub = rows[:, mask].astype(np.dtype(args.dtype))
            del rows

            reduced = pca_reduce(sub, args.pca_components, args.random_state)
            del sub
            embedding = compute_embedding(
                reduced, args.perplexities, args.metric, args.n_jobs, args.random_state,
            )

            if args.save_weekly_embeddings:
                csv_path = os.path.join(sub_dir, f"embedding_week_{sanitize(week)}.csv")
                pd.DataFrame(embedding, columns=["tsne_1", "tsne_2"]).to_csv(csv_path, index=False)
                print(f"[ok] saved weekly embedding: {csv_path}")

            title = f"openTSNE by {spec.label} - Week {week} embedding (n={n})"
            out_path = os.path.join(sub_dir, f"tsne_week_{sanitize(week)}.png")
            plot_embedding(embedding, spec.values[mask], spec, title, out_path,
                           args, show=args.show)
            print(f"[ok] week {week}: {out_path}")
            frames.append(out_path)
    build_gif(frames, os.path.join(sub_dir, f"tsne_weekly_embeddings_{args.color_by}.gif"),
              args.gif_fps)


def run_global(mat_path, details, spec, out_root, args):
    """Compute (or load) the single global embedding and plot it."""
    if args.load_embedding:
        print(f"[..] loading embedding: {args.load_embedding}")
        embedding = pd.read_csv(args.load_embedding).to_numpy()
        if embedding.shape[0] != len(details):
            raise ValueError(
                f"loaded embedding has {embedding.shape[0]} rows but details has "
                f"{len(details)}; they must match row-for-row")
        embedding = embedding[:, :2]
    else:
        print(f"[..] loading similarity matrix ({args.dtype}) from {mat_path}")
        t0 = time.time()
        with h5py.File(mat_path, "r") as f:
            sim = f["Clusters"]["sim"]
            matrix = sim.astype(np.dtype(args.dtype))[:]
        print(f"[time] loaded {matrix.shape} in {time.time() - t0:.1f}s")

        reduced = pca_reduce(matrix, args.pca_components, args.random_state)
        del matrix
        if args.save_reduced:
            rp = os.path.join(out_root, "similarity_matrix_reduced.csv")
            pd.DataFrame(reduced).to_csv(rp, index=False)
            print(f"[ok] saved PCA-reduced matrix: {rp}")

        embedding = compute_embedding(
            reduced, args.perplexities, args.metric, args.n_jobs, args.random_state,
        )

    if args.save_embedding and not args.load_embedding:
        ep = os.path.join(out_root, "embedding_multiscale.csv")
        pd.DataFrame(embedding, columns=["tsne_1", "tsne_2"]).to_csv(ep, index=False)
        print(f"[ok] saved embedding: {ep}")

    out_path = os.path.join(out_root, f"tsne_by_{args.color_by}.png")
    plot_embedding(embedding, spec.values, spec,
                   f"openTSNE Multiscale colored by {spec.label}",
                   out_path, args, show=args.show)
    print(f"[ok] global plot: {out_path}")
    return embedding


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate t-SNE visualizations of accelerometer cluster data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--data-root", default=r"C:\mitopark_tsne\1lc",
                   help="Directory containing the .mat and details CSV(s).")
    p.add_argument("--output-root", default=None,
                   help="Directory for outputs (default: <data-root>\\graphs). "
                        "Results go in a 'TSNE' subfolder.")
    p.add_argument("--mat-name", default="session_1_out.mat",
                   help="Similarity-matrix .mat filename inside --data-root.")
    p.add_argument("--details-name", default="Cluster_detail_results.csv",
                   help="Cluster detail results CSV inside --data-root.")
    p.add_argument("--temporal-name", default="Cluster_detail_results_temporal.csv",
                   help="CSV with temporal_class/tba/tba_class columns (row-aligned "
                        "to the details CSV); attached automatically when present.")

    # Embedding save / reuse
    p.add_argument("--save-embedding", action="store_true",
                   help="Save the global embedding to embedding_multiscale.csv.")
    p.add_argument("--load-embedding", default=None,
                   help="Load a previously-saved global embedding CSV and skip the "
                        "expensive similarity-load/PCA/openTSNE steps.")
    p.add_argument("--save-reduced", action="store_true",
                   help="Also save the PCA-reduced similarity matrix.")

    # Color
    p.add_argument("--color-by", default="cluster", choices=COLOR_CHOICES,
                   help="Which color mask to use: cluster index (glasbey), "
                        "temporal_class, tba_class, or a tba gradient.")
    p.add_argument("--palette", default="glasbey", choices=PALETTE_CHOICES,
                   help="Categorical palette (used for cluster mode and as a "
                        "fallback for unlisted temporal categories).")
    p.add_argument("--gradient-colorscale", default="Viridis",
                   help="Continuous colorscale for --color-by tba (any plotly "
                        "colorscale name, e.g. Viridis, Turbo, Inferno).")

    # Weekly modes
    p.add_argument("--weekly", action="store_true",
                   help="Subset the global embedding to each week, plot each, and "
                        "save a GIF.")
    p.add_argument("--per-week-embedding", action="store_true",
                   help="Compute a separate embedding for each week (subset the "
                        "similarity matrix first), plot each, and save a GIF.")
    p.add_argument("--save-weekly-embeddings", action="store_true",
                   help="With --per-week-embedding, save each week's embedding CSV.")
    p.add_argument("--weeks", nargs="+", default=None,
                   help="Restrict weekly modes to these week labels (e.g. w8 w9).")
    p.add_argument("--min-week-points", type=int, default=100,
                   help="Skip weeks with fewer points in --per-week-embedding mode.")
    p.add_argument("--skip-global", action="store_true",
                   help="Skip the global embedding/plot entirely (e.g. when you only "
                        "want --per-week-embedding).")
    p.add_argument("--gif-fps", type=float, default=2.0, help="Frames per second for GIFs.")

    # Embedding / TSNE params
    p.add_argument("--pca-components", type=int, default=50,
                   help="PCA components before openTSNE.")
    p.add_argument("--perplexities", type=int, nargs="+", default=[50, 500],
                   help="Multiscale perplexities.")
    p.add_argument("--metric", default="cosine", help="Affinity metric.")
    p.add_argument("--random-state", type=int, default=3, help="Random seed.")
    p.add_argument("--n-jobs", type=int, default=-1, help="Parallel jobs (-1 = all cores).")
    p.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                   help="dtype for the similarity matrix in memory "
                        "(float32 halves RAM; the full matrix can be very large).")

    # Rendering
    p.add_argument("--width", type=int, default=1200, help="Figure width (px).")
    p.add_argument("--height", type=int, default=1000, help="Figure height (px).")
    p.add_argument("--marker-size", type=int, default=5, help="Scatter marker size.")
    p.add_argument("--show", action="store_true", help="Also open figures interactively.")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    mat_path = os.path.join(args.data_root, args.mat_name)
    details_path = os.path.join(args.data_root, args.details_name)
    temporal_path = os.path.join(args.data_root, args.temporal_name)
    output_root = args.output_root or os.path.join(args.data_root, "graphs")
    out_dir = os.path.join(output_root, "TSNE")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(details_path):
        sys.exit(f"[error] details CSV not found: {details_path}")

    print(f"[..] loading cluster details: {details_path}")
    details = load_details(details_path, temporal_path)
    print(f"[ok] {len(details)} points across {details['Week'].nunique()} weeks")

    # Shared color mask, computed once over all points.
    spec = build_coloring(details, args.color_by, args.palette, args.gradient_colorscale)
    if spec.mode == "categorical":
        print(f"[ok] color-by '{args.color_by}' -> {len(spec.category_order)} categories "
              f"(palette '{args.palette}')")
    else:
        print(f"[ok] color-by '{args.color_by}' -> gradient '{args.gradient_colorscale}' "
              f"range {spec.crange[0]:.3g}..{spec.crange[1]:.3g}")

    need_mat = (not args.skip_global and not args.load_embedding) or args.per_week_embedding
    if need_mat and not os.path.exists(mat_path):
        sys.exit(f"[error] similarity .mat not found: {mat_path}")

    embedding = None
    if not args.skip_global:
        embedding = run_global(mat_path, details, spec, out_dir, args)

    if args.weekly:
        if embedding is None:
            sys.exit("[error] --weekly needs a global embedding; drop --skip-global "
                     "or pass --load-embedding.")
        run_weekly_subset(embedding, details, spec, out_dir, args)

    if args.per_week_embedding:
        run_per_week_embedding(mat_path, details, spec, out_dir, args)

    print("[done]")


if __name__ == "__main__":
    main()
