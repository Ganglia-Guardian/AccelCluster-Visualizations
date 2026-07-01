# cluster_spatial_map.py

Maps behavior clusters onto the arena field-of-view for a 3D-arena miniscope session. Each cluster event is matched to the mouse's (x, y) position at that moment, and a set of spatial figures is produced.

---

## Requirements

```bash
pip install h5py numpy matplotlib pillow scipy
```

---

## Outputs

Every run writes to `--outdir` (created if it does not exist):

| File | Description |
|------|-------------|
| `cluster_positions.csv` | Merged table: `cluster_id`, `rel_time_s`, `x`, `y`, `match_gap_s` (+ `folder` and/or `accel` columns when applicable) |
| `cluster_map_combined.png` | All clusters as colored dots over a data-derived arena background |
| `cluster_map_faceted.png` | One panel per cluster — good for spotting spatial hot-spots |
| `cluster_density_map.png` | Events colored by local position density |
| `cluster_accel_map.png` | Events colored by acceleration (`TotAccelBA`) — only produced with `--features-csv` |
| `cluster_map_alignment_check.png` | Trajectory + time-0 star over the video frame — only with `--frame` |
| `cluster_photo_overlay.png` | Combined cluster overlay on the real arena frame — only with `--frame` |
| `cluster_photo_faceted.png` | Per-cluster facets on the real arena frame — only with `--frame` |
| `cluster_density_*_overlay.png` | Density map over the real arena frame — only with `--frame` |
| `cluster_accel_*_overlay.png` | Acceleration map over the real arena frame — only with `--frame` + `--features-csv` |

---

## Input file formats

### Cluster event source (one of three options)

**Option A — MATLAB `.mat` file** (`--mat`)  
A v7.3 (HDF5) `.mat` containing a MATLAB `table` called `Clusters.idx_ts` with two columns: a cluster ID (integer, small range) and a timestamp (monotonically increasing). The script auto-identifies which column is which.

**Option B — detail CSV** (`--detail-csv`)  
A CSV with a header row. Required columns (case-insensitive):

| Column | Content |
|--------|---------|
| `Timestamp` | Event timestamp (numeric) |
| `ClusterIdx` | Cluster ID (integer) |
| `Folder_Name` | Session/folder label (used with `--folder_subset` or multiple centroids) |

**Option C — features CSV** (`--features-csv`)  
Same as the detail CSV but the ID column is named `Cluster` (not `ClusterIdx`) and there is an extra `TotAccelBA` column carrying per-event acceleration. Using this source also produces the acceleration-colored map.

---

### Centroid tracking file (`--centroid`)

A headerless CSV with at least 6 columns per row:

```
hour, minute, second, millisecond, x, y
```

Rows that cannot be parsed are silently skipped.

---

### Arena video frame (`--frame`, optional)

Any PNG or JPG still frame from the recording. Used as the background for photo-overlay figures. When provided, the script also runs auto-alignment to map tracking coordinates onto the frame.

---

## Usage

### Simplest: `.mat` source, no photo overlay

```bash
python cluster_spatial_map.py \
    --mat "path/to/session_1_out.mat" \
    --centroid "path/to/mouse_centroid_2025-12-12T10_02_57.csv" \
    --outdir "path/to/output_folder"
```

---

### Detail CSV instead of `.mat`

```bash
python cluster_spatial_map.py \
    --detail-csv "path/to/cluster_detail.csv" \
    --folder_subset "session_1" \
    --centroid "path/to/mouse_centroid.csv" \
    --outdir "path/to/output_folder"
```

`--folder_subset` keeps only the rows in the CSV whose `Folder_Name` matches that exact string. Omit it to use all rows.

---

### Features CSV (adds acceleration map)

```bash
python cluster_spatial_map.py \
    --features-csv "path/to/features.csv" \
    --folder_subset "session_1" \
    --centroid "path/to/mouse_centroid.csv" \
    --outdir "path/to/output_folder"
```

---

### Auto-discover all inputs from a single folder (`--indir`)

Drop all input files into one folder and let the script assign them by filename substring (case-insensitive):

| Substring in filename | Assigned role |
|-----------------------|---------------|
| `Cluster_detail_results` | detail CSV |
| `features` | features CSV |
| `session_1_out` | `.mat` file |
| `centroid` | centroid CSV |
| `Capture` | arena frame |

```bash
python cluster_spatial_map.py \
    --indir "path/to/session_folder" \
    --folder_subset "session_1" \
    --outdir "path/to/output_folder"
```

Only one cluster source (detail CSV, features CSV, or `.mat`) may be present in `--indir`; if more than one matches, the script exits with an error.

---

### Multiple sessions / multiple centroid files

Name each centroid file `{folder}--mouse_centroid_{date}.csv` where `{folder}` exactly matches a `Folder_Name` value in the CSV. Place them all in the `--indir` folder. The script pairs each folder's events with its matching centroid, aligns each on its own clock, and combines all folders into unified output figures.

```
session_folder/
    Cluster_detail_results.csv      # Folder_Name column has "ses1", "ses2", ...
    ses1--mouse_centroid_2025-01-01.csv
    ses2--mouse_centroid_2025-01-02.csv
    Capture.PNG
```

```bash
python cluster_spatial_map.py \
    --indir "path/to/session_folder" \
    --outdir "path/to/output_folder"
```

---

### With a photo overlay (auto-alignment)

```bash
python cluster_spatial_map.py \
    --mat "path/to/session_1_out.mat" \
    --centroid "path/to/mouse_centroid.csv" \
    --frame "path/to/Capture.PNG" \
    --outdir "path/to/output_folder"
```

Auto-alignment finds the darkest compact blob in the frame (the mouse at time 0) and computes the offset between tracking coordinates and frame pixels. Check `cluster_map_alignment_check.png` to verify — the cyan trajectory should lie inside the arena and the red star should sit on the mouse.

---

### With a photo overlay (manual alignment)

If auto-alignment looks off, override it:

```bash
python cluster_spatial_map.py \
    --mat "path/to/session_1_out.mat" \
    --centroid "path/to/mouse_centroid.csv" \
    --frame "path/to/Capture.PNG" \
    --offset-x 297 \
    --offset-y -12 \
    --scale 1.0 \
    --outdir "path/to/output_folder"
```

`frame_pixel = tracking_coord × scale + offset`. Adjust `--offset-x` / `--offset-y` until the alignment check image looks correct.

---

## All arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--mat` | one of three† | — | Path to `session_X_out.mat` |
| `--detail-csv` | one of three† | — | Path to detail CSV (`Timestamp`, `ClusterIdx`, `Folder_Name`) |
| `--features-csv` | one of three† | — | Path to features CSV (adds `TotAccelBA` acceleration map) |
| `--indir` | one of three† | — | Folder holding all inputs; files are auto-assigned by filename |
| `--centroid` | yes (unless `--indir`) | — | Path to `mouse_centroid_*.csv` tracking file |
| `--outdir` | no | `.` | Output folder (created if missing) |
| `--folder_subset` | no | — | Keep only CSV rows whose `Folder_Name` equals this string |
| `--frame` | no | — | Arena video frame (PNG/JPG) for photo overlays |
| `--offset-x` | no | auto | Frame→tracking x offset; skips auto-alignment when set (requires `--offset-y`) |
| `--offset-y` | no | auto | Frame→tracking y offset |
| `--scale` | no | `1.0` | Frame→tracking scale factor |
| `--density-radius` | no | `30.0` | Radius in pixels for position-density count. **Use the same value for every session you want to compare.** |
| `--density-vmax` | no | auto | Fixed upper color limit for the density map. Set the same value across sessions for comparable colors. Without this, the scale is auto-fit per session and is not cross-session comparable. |
| `--density-vmin` | no | `0.0` | Fixed lower color limit for the density map (only used when `--density-vmax` is set) |

†Exactly one of `--mat`, `--detail-csv`, `--features-csv`, or `--indir` is required.

---

## Cross-session comparability

The density map colors each event by how many other events fall within `--density-radius` pixels. Because this is an absolute count (not normalized), the same color represents the same density across sessions — **but only if you use the same `--density-radius` and `--density-vmax` for every session, and the same arena pixel scale.**

Without `--density-vmax`, the color scale is auto-fit to each session individually and the maps cannot be meaningfully compared.

---

## Coordinate system

Pixel origin is **top-left**, with y increasing downward, matching how a video frame displays. All output figures use the same convention.

---

## Time alignment

Both the cluster event timestamps and the centroid timestamps are converted to relative time (first sample = 0 s). Each cluster event is then matched to the nearest centroid sample. The script prints the median and maximum match gap at runtime — a small gap (< 0.1 s) means the two recordings started together. A warning is printed if any gap exceeds 1.0 s.

---

## Windows path note

On Windows, wrap all paths in double quotes and use `^` for line continuation:

```bat
python cluster_spatial_map.py ^
    --mat "Y:\data\session_1_out.mat" ^
    --centroid "C:\tracking\mouse_centroid_2025-12-12T10_02_57.csv" ^
    --outdir "C:\Users\me\Desktop\session1_out"
```
