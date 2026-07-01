#!/usr/bin/env python3
"""
cluster_spatial_map.py
======================
Map behavior clusters onto the arena field-of-view for a 3D-arena miniscope session.

For each cluster event (from `session_X_out.mat`, table `Clusters.idx_ts`, OR from a
detail CSV via --detail-csv) it finds the mouse's (x, y) position at that moment (from
the `mouse_centroid_*.csv` tracking file), then produces:
  1. cluster_positions.csv        - merged table: cluster_id, rel_time_s, x, y, match_gap_s
  2. cluster_map_combined.png     - all clusters as colored dots over the arena
  3. cluster_map_faceted.png      - one panel per cluster (best for spotting hot-spots)
  4. cluster_accel_map.png        - events colored by acceleration + colorbar
                                    (only with --features-csv / a 'features' file)
  5. cluster_density_map.png      - events colored by local position density + colorbar
  6. (optional) photo overlays    - the same, but on top of a real arena video frame

--------------------------------------------------------------------------------------
REQUIREMENTS (install once):
    pip install h5py numpy matplotlib pillow scipy
--------------------------------------------------------------------------------------
TYPICAL USAGE (Windows examples):

  # 1) Plain version (data-derived arena background) -- no video frame needed
  python cluster_spatial_map.py ^
      --mat "Y:\\...\\session_1_out.mat" ^
      --centroid "C:\\...\\mouse_centroid_2025-12-12T10_02_57.csv" ^
      --outdir "C:\\Users\\me\\Desktop\\session1_out"

  # 1b) Cluster events from a detail CSV instead of a .mat (columns:
  #     Timestamp, ClusterIdx, Folder_Name). Optionally keep only one folder.
  python cluster_spatial_map.py ^
      --detail-csv "C:\\...\\cluster_detail.csv" ^
      --folder_subset "session_1" ^
      --centroid "...\\mouse_centroid_...csv" ^
      --outdir "...\\session1_out"

  # 1c) Point at one folder and let the script find everything in it. Files are
  #     matched by case-insensitive name substring (see --indir help below):
  #       *Cluster_detail_results* -> detail-csv   *session_1_out* -> mat
  #       *centroid*               -> centroid     *Capture*       -> frame
  python cluster_spatial_map.py ^
      --indir "C:\\Users\\me\\Desktop\\session1" ^
      --folder_subset "session_1" ^
      --outdir "...\\session1_out"

  # 2) Overlay on a real arena frame saved as PNG (auto-aligns using time-0 position)
  python cluster_spatial_map.py ^
      --mat "...\\session_1_out.mat" ^
      --centroid "...\\mouse_centroid_...csv" ^
      --frame "C:\\Users\\me\\Desktop\\Capture.PNG" ^
      --outdir "...\\session1_out"

  # 3) Same, but set the frame->tracking mapping by hand if auto-align looks off
  python cluster_spatial_map.py ... --frame Capture.PNG --offset-x 297 --offset-y -12 --scale 1.0

--------------------------------------------------------------------------------------
NOTES / ASSUMPTIONS
  * idx_ts is a MATLAB `table` saved in a v7.3 (HDF5) .mat. We decode it from the
    file's MCOS subsystem. Columns are auto-identified: the integer/small-range column
    is the cluster id, the monotonically increasing column is the timestamp.
  * --detail-csv is an alternative source holding the same content as Clusters.idx_ts
    with a header row: Timestamp, ClusterIdx, Folder_Name. Use --folder_subset to keep
    only the rows whose Folder_Name matches a given string.
  * Alignment between the two clocks assumes BOTH recordings start together: each is
    converted to relative time (first sample = 0) and every cluster event is matched to
    the nearest centroid sample. The script prints the match gap so you can sanity check.
  * With --indir you may drop in several '{folder}--mouse_centroid_*.csv' files. Each is
    bound to the matching Folder_Name in the detail/features CSV, aligned on its own
    clock, and all folders are then drawn on the same figures. Folders with no centroid
    (or centroids with no CSV rows) are skipped with a note.
  * The density map colors each event by how many datapoints fall within
    --density-radius pixels of it (an absolute count). For colors to mean the same
    thing across sessions, run every session with the SAME --density-radius and the
    SAME --density-vmax (and the same arena pixel scale). Without --density-vmax the
    scale is auto-fit to that one session and is not comparable.
  * Pixel origin is TOP-LEFT (y increases downward), matching how a video frame displays.
  * Photo auto-alignment assumes the tracking video is a 1:1 (unscaled) crop of the
    camera frame, and locates the mouse at time 0 by finding the darkest compact blob.
    Verify cluster_map_alignment_check.png; nudge --offset-x/--offset-y if needed.
"""

import os, csv, argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------
# 1. Decode the Clusters.idx_ts MATLAB table from a v7.3 .mat file
# ----------------------------------------------------------------------------------
def load_cluster_events(mat_path):
    """Return (cluster_id [int N], timestamp [float N]) from Clusters.idx_ts.

    idx_ts is stored as a MATLAB `table` object inside the MCOS subsystem. We scan the
    subsystem for a cell of two equal-length numeric vectors and identify which is the
    cluster id (integer valued, small number of unique values) and which is the
    timestamp (strictly increasing).
    """
    with h5py.File(mat_path, "r") as f:
        if "#subsystem#" not in f or "MCOS" not in f["#subsystem#"]:
            raise RuntimeError("No MCOS subsystem found - is idx_ts really a MATLAB table?")
        mcos = np.array(f["#subsystem#"]["MCOS"]).ravel()

        candidates = []  # (len, colA, colB)
        for ref in mcos:
            try:
                obj = f[ref]
            except Exception:
                continue
            if isinstance(obj, h5py.Group):
                continue
            arr = np.array(obj)
            # we want an object/reference array holding exactly 2 numeric vectors
            if arr.dtype != object or arr.size != 2:
                continue
            cols = []
            for r in arr.ravel():
                try:
                    d = np.array(f[r]).ravel().astype(float)
                except Exception:
                    cols = None
                    break
                cols.append(d)
            if not cols or len(cols) != 2:
                continue
            if len(cols[0]) != len(cols[1]) or len(cols[0]) < 5:
                continue
            candidates.append((len(cols[0]), cols[0], cols[1]))

        if not candidates:
            raise RuntimeError("Could not locate the idx_ts data columns in the MCOS subsystem.")

        # pick the longest candidate (the event table) -- idx_details is shorter / 3-col
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, a, b = candidates[0]

        # identify which column is which
        def looks_like_time(v):
            return np.all(np.diff(v) >= -1e-9)  # (weakly) increasing
        def looks_like_id(v):
            return np.allclose(v, np.round(v)) and (np.max(v) - np.min(v)) < 1000

        if looks_like_time(b) and not looks_like_time(a):
            cid, ts = a, b
        elif looks_like_time(a) and not looks_like_time(b):
            cid, ts = b, a
        elif looks_like_id(a) and not looks_like_id(b):
            cid, ts = a, b
        else:
            cid, ts = b, a  # fallback: assume (id, time) order
        return np.round(cid).astype(int), ts.astype(float)


# ----------------------------------------------------------------------------------
# 1b. Load cluster events from a detail CSV (alternative to the .mat source)
#     Columns (header required): Timestamp, ClusterIdx, Folder_Name
# ----------------------------------------------------------------------------------
def load_cluster_events_csv(csv_path, folder_subset=None):
    """Return (cluster_id [int N], timestamp [float N]) from a detail CSV.

    The CSV must have a header row with columns Timestamp, ClusterIdx and
    Folder_Name (this holds the same content as Clusters.idx_ts). Column lookup
    is case-insensitive and tolerant of surrounding whitespace.

    If `folder_subset` is given, only the rows whose Folder_Name equals that exact
    string (after stripping whitespace) are kept. Events are returned sorted by
    timestamp so the downstream relative-time alignment behaves as expected.
    """
    cid, ts = [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        # map normalized name -> actual header text
        fieldmap = {name.strip().lower(): name for name in fieldnames}
        try:
            ts_key = fieldmap["timestamp"]
            id_key = fieldmap["clusteridx"]
        except KeyError:
            raise RuntimeError(
                "--detail-csv must have a header with columns 'Timestamp' and "
                "'ClusterIdx' (found: %s)" % (fieldnames,))
        folder_key = fieldmap.get("folder_name")

        if folder_subset is not None and folder_key is None:
            raise RuntimeError(
                "--folder_subset was given but the CSV has no 'Folder_Name' column.")

        for row in reader:
            if folder_subset is not None:
                if (row.get(folder_key) or "").strip() != folder_subset:
                    continue
            try:
                t = float(row[ts_key])
                c = float(row[id_key])
            except (ValueError, TypeError, KeyError):
                continue  # skip blank / malformed rows
            ts.append(t)
            cid.append(c)

    if not ts:
        if folder_subset is not None:
            raise RuntimeError(
                "No rows in --detail-csv matched --folder_subset '%s'." % folder_subset)
        raise RuntimeError("No usable rows found in --detail-csv.")

    ts = np.array(ts, dtype=float)
    cid = np.round(np.array(cid, dtype=float)).astype(int)
    order = np.argsort(ts, kind="stable")
    return cid[order], ts[order]


# ----------------------------------------------------------------------------------
# 1c. Load a features CSV (alternative cluster source that also carries acceleration)
#     Columns (header required): Timestamp, Cluster, Folder_Name, TotAccelBA
#     Same role as the detail CSV, but the id column is 'Cluster' (not 'ClusterIdx')
#     and there is an extra 'TotAccelBA' per-event acceleration value.
# ----------------------------------------------------------------------------------
def load_feature_events_csv(csv_path, folder_subset=None):
    """Return (cluster_id [int N], timestamp [float N], accel [float N]).

    Behaves like load_cluster_events_csv but reads the 'Cluster' id column and an
    extra 'TotAccelBA' column. Rows missing a parseable acceleration keep their
    cluster/timestamp (so the cluster maps are unchanged) and get accel = NaN, so
    they are simply skipped on the acceleration-colored map. Sorted by timestamp.
    """
    cid, ts, acc = [], [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        fieldmap = {name.strip().lower(): name for name in fieldnames}
        try:
            ts_key = fieldmap["timestamp"]
            id_key = fieldmap["cluster"]
            acc_key = fieldmap["totaccelba"]
        except KeyError:
            raise RuntimeError(
                "--features-csv must have a header with columns 'Timestamp', "
                "'Cluster' and 'TotAccelBA' (found: %s)" % (fieldnames,))
        folder_key = fieldmap.get("folder_name")

        if folder_subset is not None and folder_key is None:
            raise RuntimeError(
                "--folder_subset was given but the CSV has no 'Folder_Name' column.")

        for row in reader:
            if folder_subset is not None:
                if (row.get(folder_key) or "").strip() != folder_subset:
                    continue
            try:
                t = float(row[ts_key])
                c = float(row[id_key])
            except (ValueError, TypeError, KeyError):
                continue  # need a valid event to place a point at all
            try:
                a = float(row[acc_key])
            except (ValueError, TypeError, KeyError):
                a = np.nan  # missing accel -> point still counts for cluster maps
            ts.append(t); cid.append(c); acc.append(a)

    if not ts:
        if folder_subset is not None:
            raise RuntimeError(
                "No rows in --features-csv matched --folder_subset '%s'." % folder_subset)
        raise RuntimeError("No usable rows found in --features-csv.")

    ts = np.array(ts, dtype=float)
    cid = np.round(np.array(cid, dtype=float)).astype(int)
    acc = np.array(acc, dtype=float)
    order = np.argsort(ts, kind="stable")
    return cid[order], ts[order], acc[order]
# ----------------------------------------------------------------------------------
def load_centroid(csv_path):
    t, x, y = [], [], []
    with open(csv_path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or len(row) < 6:
                continue
            try:
                h, m, s, ms = int(row[0]), int(row[1]), int(row[2]), int(row[3])
                xx, yy = float(row[4]), float(row[5])
            except ValueError:
                continue  # skip header / bad lines
            t.append(h * 3600 + m * 60 + s + ms / 1000.0)
            x.append(xx); y.append(yy)
    return np.array(t), np.array(x), np.array(y)


# ----------------------------------------------------------------------------------
# 3. Match each cluster event to the nearest centroid sample (relative-time alignment)
# ----------------------------------------------------------------------------------
def align_events(ts, ct, cx, cy):
    ts_rel = ts - ts[0]
    ct_rel = ct - ct[0]
    idx = np.clip(np.searchsorted(ct_rel, ts_rel), 0, len(ct_rel) - 1)
    left = np.clip(idx - 1, 0, len(ct_rel) - 1)
    choose_left = np.abs(ct_rel[left] - ts_rel) < np.abs(ct_rel[idx] - ts_rel)
    match = np.where(choose_left, left, idx)
    gap = np.abs(ct_rel[match] - ts_rel)
    return ts_rel, cx[match], cy[match], gap


# ----------------------------------------------------------------------------------
# 4. Colors for up to ~60 clusters
# ----------------------------------------------------------------------------------
def cluster_colors(uids):
    base = (list(plt.cm.tab20.colors) + list(plt.cm.tab20b.colors) + list(plt.cm.tab20c.colors))
    while len(base) < len(uids):
        base += base
    return {c: base[i] for i, c in enumerate(uids)}


# ----------------------------------------------------------------------------------
# 5a. Plain figures: data-derived arena background (occupancy of all positions)
# ----------------------------------------------------------------------------------
def plot_plain(cluster_id, ex, ey, cx, cy, outdir):
    uids = np.arange(1, cluster_id.max() + 1)
    col = cluster_colors(uids)
    W, H = cx.max(), cy.max()
    hist, _, _ = np.histogram2d(cx, cy, bins=120, range=[[0, W], [0, H]])
    occ = np.log1p(hist.T)

    # combined
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.imshow(occ, extent=[0, W, 0, H], origin="lower", cmap="Greys",
              alpha=0.85, aspect="equal", vmax=occ.max() * 0.9)
    for c in uids:
        sel = cluster_id == c
        ax.scatter(ex[sel], ey[sel], s=14, color=col[c], alpha=0.7, edgecolors="none", label=str(c))
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    ax.set_title("Behavior clusters over arena (grey = mouse occupancy)\nTop-left origin")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left",
              ncol=2, fontsize=7, markerscale=1.4)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "cluster_map_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    _facet(uids, cluster_id, ex, ey, col, outdir, "cluster_map_faceted.png",
           bg=("occ", occ, W, H))


# ----------------------------------------------------------------------------------
# 5b. Photo figures: overlay on a real arena frame
# ----------------------------------------------------------------------------------
def plot_photo(cluster_id, ex, ey, cx, cy, frame_path, outdir, ox, oy, scale):
    from PIL import Image
    img = np.array(Image.open(frame_path).convert("RGB"))
    H, W = img.shape[:2]
    mx = lambda v: v * scale + ox
    my = lambda v: v * scale + oy
    uids = np.arange(1, cluster_id.max() + 1)
    col = cluster_colors(uids)

    # alignment check
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    ax.plot(mx(cx), my(cy), color="cyan", lw=0.25, alpha=0.5)
    ax.scatter([mx(cx[0])], [my(cy[0])], s=300, marker="*", color="red",
               edgecolor="k", zorder=5, label="tracked t0")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title("Alignment check: trajectory (cyan) + tracked time-0 (red star)")
    ax.legend(loc="lower right")
    fig.savefig(os.path.join(outdir, "cluster_map_alignment_check.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)

    # combined overlay
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img)
    for c in uids:
        sel = cluster_id == c
        ax.scatter(mx(ex[sel]), my(ey[sel]), s=12, color=col[c], alpha=0.75,
                   edgecolors="none", label=str(c))
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    ax.set_title("Behavior clusters over arena (%d events, %d clusters)" % (len(ex), len(uids)))
    ax.legend(title="Cluster", bbox_to_anchor=(1.005, 1), loc="upper left",
              ncol=2, fontsize=7, markerscale=1.5)
    fig.savefig(os.path.join(outdir, "cluster_photo_overlay.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    _facet(uids, cluster_id, ex, ey, col, outdir, "cluster_photo_faceted.png",
           bg=("img", img, W, H), mapxy=(mx, my))


def _facet(uids, cluster_id, ex, ey, col, outdir, fname, bg, mapxy=None):
    n = len(uids); ncol = 6; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.2 * nrow))
    axes = np.array(axes).ravel()
    mx = mapxy[0] if mapxy else (lambda v: v)
    my = mapxy[1] if mapxy else (lambda v: v)
    kind, data, W, H = bg
    for k, c in enumerate(uids):
        a = axes[k]; sel = cluster_id == c
        if kind == "img":
            a.imshow(data)
        else:
            a.imshow(data, extent=[0, W, 0, H], origin="lower", cmap="Greys",
                     alpha=0.5, aspect="equal", vmax=data.max() * 0.9)
        a.scatter(mx(ex[sel]), my(ey[sel]), s=7, color=col[c], alpha=0.85, edgecolors="none")
        a.set_xlim(0, W); a.set_ylim(H, 0); a.set_xticks([]); a.set_yticks([])
        a.set_title("Cl %d (n=%d)" % (c, sel.sum()), fontsize=8)
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle("Per-cluster locations over arena", y=1.0, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------------
# 5c. Shared "colored scatter over the arena + colorbar" used by the acceleration
#     map and the position-density map. Uses the real frame as background when one
#     is supplied, otherwise the data-derived occupancy map.
# ----------------------------------------------------------------------------------
def _scatter_colored(values, ex, ey, cx, cy, outdir, cmap, clabel, title_tmpl,
                     base_name, frame_path=None, ox=0.0, oy=0.0, scale=1.0,
                     robust=True, vmin=None, vmax=None):
    finite = np.isfinite(values)
    vals = values[finite]
    if vals.size == 0:
        print("  NOTE: no finite values for %s; skipping." % base_name)
        return
    # explicit fixed limits win (needed for cross-session comparability); else
    # robust percentile clipping; else plain min/max
    if vmin is None or vmax is None:
        if robust:  # clip to 2nd/98th pct so one spike does not flatten the scale
            lo, hi = np.percentile(vals, [2, 98])
            if lo == hi:
                lo, hi = vals.min(), vals.max()
        else:
            lo, hi = vals.min(), vals.max()
        vmin = lo if vmin is None else vmin
        vmax = hi if vmax is None else vmax

    if frame_path:
        from PIL import Image
        img = np.array(Image.open(frame_path).convert("RGB"))
        H, W = img.shape[:2]
        px = ex[finite] * scale + ox
        py = ey[finite] * scale + oy
        fname, figsize = base_name + "_overlay.png", (12, 8)
    else:
        W, H = cx.max(), cy.max()
        hist, _, _ = np.histogram2d(cx, cy, bins=120, range=[[0, W], [0, H]])
        img = np.log1p(hist.T)
        px, py = ex[finite], ey[finite]
        fname, figsize = base_name + "_map.png", (11, 10)

    fig, ax = plt.subplots(figsize=figsize)
    if frame_path:
        ax.imshow(img); ax.axis("off")
    else:
        ax.imshow(img, extent=[0, W, 0, H], origin="lower", cmap="Greys",
                  alpha=0.85, aspect="equal", vmax=img.max() * 0.9)
        ax.set_xlabel("X (px)"); ax.set_ylabel("Y (px)")
    sc = ax.scatter(px, py, c=vals, s=16, cmap=cmap, vmin=vmin, vmax=vmax,
                    alpha=0.85, edgecolors="none")
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title(title_tmpl % int(finite.sum()))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(clabel)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_accel(accel, ex, ey, cx, cy, outdir, frame_path=None, ox=0.0, oy=0.0, scale=1.0):
    """Acceleration map: events colored by TotAccelBA, with a colorbar."""
    _scatter_colored(
        accel, ex, ey, cx, cy, outdir, cmap="viridis", clabel="TotAccelBA",
        title_tmpl="Behavior events over arena, colored by acceleration\n"
                   "%d events (TotAccelBA)",
        base_name="cluster_accel", frame_path=frame_path, ox=ox, oy=oy,
        scale=scale, robust=True)


# ----------------------------------------------------------------------------------
# 5d. Position-density map: color each event by how many datapoints lie within a
#     fixed radius (in pixels). Because the radius is fixed and the count is an
#     absolute number (not normalized to the session), the same color means the
#     same density across sessions -- as long as you use the SAME --density-radius
#     and the SAME --density-vmax for every session, on the same arena pixel scale.
# ----------------------------------------------------------------------------------
def _point_density(x, y, radius):
    """Per-point density = number of datapoints within `radius` pixels (counting the
    point itself). Fixed radius + absolute count => values are directly comparable
    across sessions. Uses a KD-tree when available, else a fixed-size grid count."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x)
    if n == 0:
        return np.array([])
    pts = np.column_stack([x, y])
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        try:
            counts = tree.query_ball_point(pts, radius, return_length=True)
        except TypeError:  # older scipy without return_length
            counts = np.array([len(tree.query_ball_point(p, radius)) for p in pts])
        return np.asarray(counts, dtype=float)
    except Exception:
        # fallback: count points in fixed pixel-size cells anchored at the origin,
        # so the binning (and thus the counts) is identical across sessions
        cell = max(radius, 1.0)
        ix = np.floor(x / cell).astype(int)
        iy = np.floor(y / cell).astype(int)
        from collections import Counter
        cnt = Counter(zip(ix.tolist(), iy.tolist()))
        return np.array([cnt[(a, b)] for a, b in zip(ix.tolist(), iy.tolist())], float)


def plot_density(ex, ey, cx, cy, outdir, frame_path=None, ox=0.0, oy=0.0, scale=1.0,
                 radius=30.0, vmin=None, vmax=None):
    """Position-density map: each event colored by the number of datapoints within
    `radius` px. Pass a fixed `vmax` (and matching `radius`) across sessions for
    comparable colors."""
    dens = _point_density(ex, ey, radius)
    if vmax is None:
        print("  NOTE: density colors are auto-scaled to THIS session and are not "
              "comparable across sessions. Set --density-vmax (and the same "
              "--density-radius) for every session to fix the scale.")
        lo, hi, robust = None, None, True
    else:
        lo = vmin if vmin is not None else 0.0
        hi, robust = vmax, False
    _scatter_colored(
        dens, ex, ey, cx, cy, outdir, cmap="inferno",
        clabel="datapoints within %g px" % radius,
        title_tmpl="Behavior events over arena, colored by position density\n"
                   "%d events",
        base_name="cluster_density", frame_path=frame_path, ox=ox, oy=oy,
        scale=scale, robust=robust, vmin=lo, vmax=hi)


# ----------------------------------------------------------------------------------
# 6. Auto-detect the time-0 frame->tracking offset (darkest compact blob = mouse)
# ----------------------------------------------------------------------------------
def auto_offset(frame_path, t0x, t0y, scale, ignore_bottom=70):
    """Find the mouse blob in the frame, return (offset_x, offset_y) so that
    pixel = tracking*scale + offset places the tracked t0 onto the detected blob."""
    from PIL import Image
    a = np.array(Image.open(frame_path).convert("L")).astype(float)
    H, W = a.shape
    b = a.copy()
    b[H - ignore_bottom:, :] = 255  # ignore media-player control bar
    thr = np.percentile(b, 1.5)     # darkest ~1.5% of pixels
    ys, xs = np.where(b < thr)
    if len(xs) == 0:
        raise RuntimeError("No dark blob found for auto-alignment; set --offset-x/--offset-y manually.")
    # densest dark region on a coarse grid, then refine
    gh = np.zeros((H // 20 + 1, W // 20 + 1))
    for x, y in zip(xs, ys):
        gh[y // 20, x // 20] += 1
    gy, gx = np.unravel_index(np.argmax(gh), gh.shape)
    cxc, cyc = gx * 20 + 10, gy * 20 + 10
    sel = (np.abs(xs - cxc) < 60) & (np.abs(ys - cyc) < 60)
    bx, by = xs[sel].mean(), ys[sel].mean()
    return bx - t0x * scale, by - t0y * scale


# ----------------------------------------------------------------------------------
# 7. Auto-discover the input files inside a single folder (--indir)
#    Files are assigned to roles by case-insensitive filename substring:
#       "cluster_detail_results" -> detail_csv      "session_1_out" -> mat
#       "centroid"               -> centroid        "capture"       -> frame
# ----------------------------------------------------------------------------------
def resolve_indir(indir):
    """Scan `indir` and return a dict with keys detail_csv, features_csv, mat,
    frame (each a single path or None) and centroid (a list of paths, possibly
    empty). Matching is case-insensitive substring on the file name. Multiple
    centroid files are allowed; the other roles must each match at most one file.
    """
    if not os.path.isdir(indir):
        raise RuntimeError("--indir is not a directory: %s" % indir)

    roles = {  # role -> (substring, [matching paths])
        "detail_csv":   ("cluster_detail_results", []),
        "features_csv": ("features", []),
        "mat":          ("session_1_out", []),
        "centroid":     ("centroid", []),
        "frame":        ("capture", []),
    }
    for fname in sorted(os.listdir(indir)):
        full = os.path.join(indir, fname)
        if not os.path.isfile(full):
            continue
        low = fname.lower()
        for needle, hits in roles.values():
            if needle in low:
                hits.append(full)

    found = {}
    for role, (needle, hits) in roles.items():
        if role == "centroid":
            found[role] = list(hits)  # multiple centroid files are allowed
            continue
        if len(hits) > 1:
            raise RuntimeError(
                "--indir matched %d files for role '%s' (substring '%s'): %s. "
                "Remove the extras or pass that file explicitly instead."
                % (len(hits), role, needle, [os.path.basename(h) for h in hits]))
        found[role] = hits[0] if hits else None
    return found


# ----------------------------------------------------------------------------------
# 7b. Helpers + assembler: load events, match each Folder_Name to its own centroid
#     file, and concatenate every folder's positions so all plots share one image.
# ----------------------------------------------------------------------------------
def _folder_from_centroid_name(fname):
    """Pull the folder label out of a '{folder}--mouse_centroid_{date}.csv' name.
    Returns the text before '--mouse_centroid_' (case-insensitive), or None if the
    marker is absent (a plain single centroid)."""
    low = fname.lower()
    marker = "--mouse_centroid_"
    if marker in low:
        return fname[:low.index(marker)].strip()
    return None


def _csv_folder_values(csv_path):
    """Return the set of distinct Folder_Name values (stripped strings) present in
    a CSV, or None if the file has no Folder_Name column."""
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fmap = {n.strip().lower(): n for n in (reader.fieldnames or [])}
        key = fmap.get("folder_name")
        if key is None:
            return None
        vals = set()
        for row in reader:
            v = (row.get(key) or "").strip()
            if v:
                vals.add(v)
        return vals


def assemble_positions(source_kind, source_path, centroid_paths, folder_subset):
    """Load cluster events, match each to a centroid position, and return combined
    arrays: (cluster_id, ts_rel, ex, ey, gap, accel_or_None, cx, cy, folder_labels).

    Multiple centroid files are supported. A file named
    '{folder}--mouse_centroid_*.csv' is bound to that folder, and only the events
    whose Folder_Name equals that folder are matched against it, each aligned on its
    own clock (first sample = 0). Every folder's results are concatenated so all
    downstream figures plot them on the same axes. A single centroid file without
    the '--mouse_centroid_' marker is treated as one global file matched against all
    events (the original behavior). `folder_labels` is an array naming the source
    folder per event in per-folder mode, else None.
    """
    has_accel = (source_kind == "features")

    def _load_events(fsub):
        if source_kind == "features":
            return load_feature_events_csv(source_path, fsub)
        elif source_kind == "detail":
            cid, ts = load_cluster_events_csv(source_path, fsub)
            return cid, ts, None
        else:
            cid, ts = load_cluster_events(source_path)
            return cid, ts, None

    # split centroid files into folder-bound and plain (no marker)
    foldered, plain = {}, []
    for cp in centroid_paths:
        fn = _folder_from_centroid_name(os.path.basename(cp))
        if fn is None:
            plain.append(cp)
        else:
            foldered[fn] = cp

    # ---- .mat source: no Folder_Name to split on, so one centroid only ----
    if source_kind == "mat":
        if len(centroid_paths) != 1:
            raise SystemExit(
                "ERROR: a .mat source needs exactly one centroid file (it has no "
                "Folder_Name to split on); got %d." % len(centroid_paths))
        cid, ts, _ = _load_events(None)
        ct, cx, cy = load_centroid(centroid_paths[0])
        ts_rel, ex, ey, gap = align_events(ts, ct, cx, cy)
        return cid, ts_rel, ex, ey, gap, None, cx, cy, None

    # ---- one global centroid (no folder marker): legacy single-file behavior ----
    if not foldered:
        if len(plain) != 1:
            raise SystemExit(
                "ERROR: expected a single centroid file, or folder-named "
                "'{folder}--mouse_centroid_*.csv' files; got %d centroid file(s)."
                % len(plain))
        cid, ts, acc = _load_events(folder_subset)
        ct, cx, cy = load_centroid(plain[0])
        ts_rel, ex, ey, gap = align_events(ts, ct, cx, cy)
        return cid, ts_rel, ex, ey, gap, acc, cx, cy, None

    # ---- multiple folder-named centroids: match each Folder_Name to its file ----
    if plain:
        print("  NOTE: ignoring centroid file(s) without a "
              "'{folder}--mouse_centroid_' name: %s"
              % [os.path.basename(p) for p in plain])

    available = set(foldered.keys())
    csv_folders = _csv_folder_values(source_path)

    if folder_subset is not None:
        if folder_subset not in available:
            raise SystemExit(
                "ERROR: --folder_subset '%s' has no matching "
                "'{folder}--mouse_centroid_*.csv' file in --indir." % folder_subset)
        folders = [folder_subset]
    else:
        if csv_folders is not None:
            usable = available & csv_folders
            no_centroid = sorted(csv_folders - available)
            no_events = sorted(available - csv_folders)
            if no_centroid:
                print("  NOTE: %d folder(s) in the CSV have no centroid file and are "
                      "skipped: %s" % (len(no_centroid), no_centroid))
            if no_events:
                print("  NOTE: centroid file(s) for folder(s) absent from the CSV are "
                      "skipped: %s" % no_events)
            folders = sorted(usable, key=lambda s: (len(s), s))
        else:
            folders = sorted(available, key=lambda s: (len(s), s))

    if not folders:
        raise SystemExit(
            "ERROR: no folder has BOTH events in the CSV and a matching centroid file.")

    parts = {k: [] for k in
             ("cid", "tsr", "ex", "ey", "gap", "cx", "cy", "fl")}
    acc_parts = [] if has_accel else None
    for fn in folders:
        cid, ts, acc = _load_events(fn)
        ct, cx, cy = load_centroid(foldered[fn])
        ts_rel, ex, ey, gap = align_events(ts, ct, cx, cy)
        ev_span = float(ts_rel.max()) if len(ts_rel) else 0.0
        ct_span = float(ct[-1] - ct[0]) if len(ct) else 0.0
        print("  folder %-8s: %4d events (0..%.1fs)  vs  %s (%d samples, 0..%.1fs)  "
              "gap med %.3f s, max %.3f s"
              % (fn, len(ts), ev_span, os.path.basename(foldered[fn]),
                 len(ct), ct_span, np.median(gap), gap.max()))
        if ct_span > 0 and ev_span > ct_span * 1.5:
            print("    ^ events run well past this centroid's duration. Likely a "
                  "truncated centroid, a units/rate mismatch between the two clocks, "
                  "or events that start before the centroid recording.")
        parts["cid"].append(cid); parts["tsr"].append(ts_rel)
        parts["ex"].append(ex); parts["ey"].append(ey); parts["gap"].append(gap)
        parts["cx"].append(cx); parts["cy"].append(cy)
        parts["fl"].append(np.array([fn] * len(cid), dtype=object))
        if has_accel:
            acc_parts.append(acc)

    cat = lambda key: np.concatenate(parts[key])
    accel = np.concatenate(acc_parts) if has_accel else None
    return (cat("cid"), cat("tsr"), cat("ex"), cat("ey"), cat("gap"),
            accel, cat("cx"), cat("cy"), cat("fl"))


# ----------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Map behavior clusters onto the arena FOV.")
    # exactly one cluster-event source is required: --mat, --detail-csv, or --indir
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--mat", help="session_X_out.mat (contains Clusters.idx_ts)")
    src.add_argument("--detail-csv", dest="detail_csv",
                     help="CSV with columns Timestamp, ClusterIdx, Folder_Name "
                          "(same content as Clusters.idx_ts); use in place of --mat")
    src.add_argument("--features-csv", dest="features_csv",
                     help="like --detail-csv but the id column is 'Cluster' (not "
                          "'ClusterIdx') and it adds a 'TotAccelBA' column. Enables the "
                          "acceleration-colored map. Use in place of --mat/--detail-csv.")
    src.add_argument("--indir",
                     help="folder holding all inputs; files are auto-assigned by "
                          "case-insensitive name substring: 'Cluster_detail_results'"
                          " -> detail-csv, 'features' -> features-csv, 'session_1_out'"
                          " -> mat, 'centroid' -> centroid, 'Capture' -> frame. "
                          "Multiple centroid files named "
                          "'{folder}--mouse_centroid_*.csv' are allowed: each is "
                          "matched to the CSV rows whose Folder_Name equals {folder}, "
                          "and all folders are plotted together. Replaces the explicit "
                          "source/--centroid/--frame flags.")
    p.add_argument("--folder_subset", default=None,
                   help="with the detail CSV: keep only rows whose Folder_Name equals this string")
    p.add_argument("--centroid", default=None,
                   help="mouse_centroid_*.csv tracking file (required unless --indir is given)")
    p.add_argument("--outdir", default=".", help="output folder (created if missing)")
    p.add_argument("--frame", default=None, help="optional arena video frame (PNG/JPG) for photo overlay")
    p.add_argument("--offset-x", type=float, default=None, help="frame->tracking x offset (skip auto-align)")
    p.add_argument("--offset-y", type=float, default=None, help="frame->tracking y offset (skip auto-align)")
    p.add_argument("--scale", type=float, default=1.0, help="frame->tracking scale (default 1.0)")
    p.add_argument("--density-radius", type=float, default=30.0, dest="density_radius",
                   help="radius in pixels for the position-density count (default 30). "
                        "Use the SAME value for every session you want to compare.")
    p.add_argument("--density-vmax", type=float, default=None, dest="density_vmax",
                   help="fixed upper color limit for the density map so colors match "
                        "across sessions (e.g. --density-vmax 40). Default: auto (per "
                        "session, NOT comparable).")
    p.add_argument("--density-vmin", type=float, default=0.0, dest="density_vmin",
                   help="fixed lower color limit for the density map (default 0; used "
                        "only when --density-vmax is set).")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- resolve input files: from --indir (auto-discovered) or explicit flags ----
    if args.indir:
        f = resolve_indir(args.indir)
        detail_csv, features_csv, mat_path = f["detail_csv"], f["features_csv"], f["mat"]
        centroid_paths, frame_path = f["centroid"], f["frame"]
        sources = [s for s in (detail_csv, features_csv, mat_path) if s]
        if len(sources) > 1:
            raise SystemExit(
                "ERROR: --indir contains more than one cluster source "
                "(detail-csv / features-csv / mat); the source is ambiguous. Keep "
                "only one, or pass --detail-csv / --features-csv / --mat explicitly.")
        if not sources:
            raise SystemExit(
                "ERROR: --indir has no cluster source file (none containing "
                "'Cluster_detail_results', 'features' or 'session_1_out').")
        if not centroid_paths:
            raise SystemExit("ERROR: --indir has no file containing 'centroid'.")
        print("indir    : %s" % os.path.abspath(args.indir))
        for label, path in (("detail-csv", detail_csv), ("features-csv", features_csv),
                            ("mat", mat_path), ("frame", frame_path)):
            if path:
                print("           %-12s -> %s" % (label, os.path.basename(path)))
        for path in centroid_paths:
            print("           %-12s -> %s" % ("centroid", os.path.basename(path)))
    else:
        detail_csv, features_csv, mat_path = args.detail_csv, args.features_csv, args.mat
        frame_path = args.frame
        centroid_paths = [args.centroid] if args.centroid else []
        if not centroid_paths:
            raise SystemExit("ERROR: --centroid is required unless --indir is given.")

    if args.folder_subset is not None and mat_path:
        print("  NOTE: --folder_subset is only used with the detail/features CSV; ignoring it.")

    if features_csv:
        source_kind, source_path = "features", features_csv
        print("source   : features CSV")
    elif detail_csv:
        source_kind, source_path = "detail", detail_csv
        print("source   : detail CSV")
    else:
        source_kind, source_path = "mat", mat_path
        print("source   : MATLAB .mat (Clusters.idx_ts)")

    # load events, match each to its centroid (per-folder if multiple centroids),
    # and concatenate so every figure shows all folders together
    (cluster_id, ts_rel, ex, ey, gap, accel,
     cx, cy, folder_labels) = assemble_positions(
        source_kind, source_path, centroid_paths, args.folder_subset)

    print("clusters : %d events, %d unique ids" % (len(cluster_id), len(np.unique(cluster_id))))
    print("centroid : %d samples across %d file(s)" % (len(cx), len(centroid_paths)))
    print("match gap: median %.3f s, max %.3f s  (small = good time alignment)"
          % (np.median(gap), gap.max()))
    if gap.max() > 1.0:
        print("  WARNING: large match gaps - some recordings may not start together.")

    # merged CSV (adds 'folder' in per-folder mode, 'accel' when TotAccelBA present)
    with open(os.path.join(args.outdir, "cluster_positions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["cluster_id", "rel_time_s", "x", "y", "match_gap_s"]
        if folder_labels is not None:
            header.append("folder")
        if accel is not None:
            header.append("accel")
        w.writerow(header)
        for i in range(len(ts_rel)):
            row = [cluster_id[i], round(ts_rel[i], 4), round(ex[i], 3),
                   round(ey[i], 3), round(gap[i], 4)]
            if folder_labels is not None:
                row.append(folder_labels[i])
            if accel is not None:
                row.append("" if not np.isfinite(accel[i]) else round(float(accel[i]), 4))
            w.writerow(row)

    # plain figures (always)
    plot_plain(cluster_id, ex, ey, cx, cy, args.outdir)

    # position-density map on data-derived background (always)
    plot_density(ex, ey, cx, cy, args.outdir, radius=args.density_radius,
                 vmin=args.density_vmin, vmax=args.density_vmax)

    # acceleration-colored map on data-derived background (features CSV only)
    if accel is not None:
        plot_accel(accel, ex, ey, cx, cy, args.outdir)

    # photo figures (optional)
    if frame_path:
        if args.offset_x is not None and args.offset_y is not None:
            ox, oy = args.offset_x, args.offset_y
            print("photo    : using manual offset (%.1f, %.1f), scale %.3f" % (ox, oy, args.scale))
        else:
            ox, oy = auto_offset(frame_path, cx[0], cy[0], args.scale)
            print("photo    : auto offset (%.1f, %.1f), scale %.3f  [verify alignment_check.png]"
                  % (ox, oy, args.scale))
        plot_photo(cluster_id, ex, ey, cx, cy, frame_path, args.outdir, ox, oy, args.scale)
        plot_density(ex, ey, cx, cy, args.outdir, frame_path, ox, oy, args.scale,
                     radius=args.density_radius, vmin=args.density_vmin,
                     vmax=args.density_vmax)
        if accel is not None:
            plot_accel(accel, ex, ey, cx, cy, args.outdir, frame_path, ox, oy, args.scale)

    # per-cluster summary
    print("\nper-cluster summary (id, n, mean x, mean y, std x, std y):")
    for c in np.arange(1, cluster_id.max() + 1):
        sel = cluster_id == c
        if sel.sum() == 0:
            continue
        print("  %2d  n=%4d  mean=(%.0f,%.0f)  std=(%.0f,%.0f)"
              % (c, sel.sum(), ex[sel].mean(), ey[sel].mean(), ex[sel].std(), ey[sel].std()))
    print("\nDone. Outputs written to:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
