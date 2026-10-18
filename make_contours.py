"""Generate contour-line JSON (replay-world-meters) + preview PNG from a War Thunder level.

JSON layout:
{
  "meta": {"crs": "replay-world-meters", "bounds": {...}, ...},
  "contours": [{"elev": <meters>, "lines": [[x0,y0,x1,y1,...], ...]}],
  "coastlines": [...], "water": [...]
}
"""
import argparse
import json
import math
import os
import sys

import numpy as np
from skimage import measure

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wt_maps import load_map_heightmap
from wt_hmap import HeightmapUnavailable
from wt_paths import levels_dir

LEVELS_DIR = None  # resolved from WT_GAME_DIR (or --levels-dir) in main()
NICE_STEPS = [0.5, 1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]


def choose_interval(span, target_levels=24):
    """Pick the nicest step giving a level count close to target."""
    best = None
    for step in NICE_STEPS:
        n = span / step
        if n < 1:
            continue
        score = abs(math.log(max(n, 1e-9) / target_levels))
        if best is None or score < best[0]:
            best = (score, step)
    return float(best[1]) if best else NICE_STEPS[-1]


def rdp(points, eps):
    """Iterative Ramer-Douglas-Peucker (points: Nx2 float array)."""
    n = len(points)
    if n < 3 or eps <= 0:
        return points
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = points[i0], points[i1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        seg_len = math.hypot(dx, dy)
        sub = points[i0 + 1:i1]
        if seg_len < 1e-12:
            d = np.hypot(sub[:, 0] - p0[0], sub[:, 1] - p0[1])
        else:
            d = np.abs(dx * (sub[:, 1] - p0[1]) - dy * (sub[:, 0] - p0[0])) / seg_len
        imax = int(np.argmax(d))
        if d[imax] > eps:
            idx = i0 + 1 + imax
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return points[keep]


def sample_grid(heights, info, max_dim, flip_y=False):
    h, w = heights.shape
    step = max(1, int(math.ceil(max(h, w) / float(max_dim))))
    hs = heights[::step, ::step].astype(np.float32)
    if flip_y:
        hs = hs[::-1, :]
    nrows = hs.shape[0]
    cell = info.cell_size * step
    xs = info.ofs_x + (np.arange(hs.shape[1], dtype=np.float64) * step + step * 0.5) * info.cell_size
    rows = np.arange(nrows, dtype=np.float64)
    if flip_y:
        rows = (nrows - 1) - rows
    ys = info.ofs_y + (rows * step + step * 0.5) * info.cell_size
    return hs, xs, ys, step, cell


def build_contours(heights, info, max_dim=1024, target_levels=24, flip_y=False,
                   water_level=None):
    """Trace contour lines of the height grid.

    Levels at or below `water_level` are skipped on purpose: the sea floor is clamped and
    flat, so contouring it yields degenerate polylines that look like they jump between
    unrelated places (and the map overlay has no use for bathymetry).
    """
    hs, xs, ys, step, cell = sample_grid(heights, info, max_dim, flip_y)
    lo, hi = float(np.min(hs)), float(np.max(hs))
    interval = choose_interval(hi - lo, target_levels)
    tol = max(interval * 0.02, cell * 0.35)
    first = math.ceil(lo / interval) * interval
    levels = np.arange(first, hi, interval)

    contours = []
    skipped_underwater = 0
    for lv in levels:
        if lv <= lo or lv >= hi:
            continue
        if water_level is not None and lv <= water_level:
            skipped_underwater += 1
            continue
        lines = []
        for s in measure.find_contours(hs, float(lv)):
            if len(s) < 4:
                continue
            px = xs[0] + s[:, 1] * cell
            py = ys[0] + s[:, 0] * cell
            pts = np.stack([px, py], axis=1)
            simp = rdp(pts, tol)
            if len(simp) < 2:
                continue
            lines.append([round(float(v), 1) for p in simp for v in p])
        if lines:
            contours.append({"elev": round(float(lv), 2), "lines": lines})
    return contours, dict(interval=interval, step=step, tol=tol, lo=lo, hi=hi,
                          xs=xs, ys=ys, cell=cell,
                          skippedUnderwaterLevels=skipped_underwater)


def water_polygons(heights, info, level, max_dim=1024, flip_y=False):
    hs, xs, ys, step, cell = sample_grid(heights, info, max_dim, flip_y)
    polys = []
    for s in measure.find_contours(hs, float(level)):
        if len(s) < 6:
            continue
        px = xs[0] + s[:, 1] * cell
        py = ys[0] + s[:, 0] * cell
        pts = np.stack([px, py], axis=1)
        simp = rdp(pts, max(cell * 0.5, 1.0))
        if len(simp) >= 3:
            polys.append([round(float(v), 1) for p in simp for v in p])
    return polys


def render_preview(heights, info, contours, out_png, title, water_level=None, flip_y=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hs, xs, ys, step, cell = sample_grid(heights, info, 1500, flip_y)
    gy, gx = np.gradient(hs)
    az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    shade = np.clip(np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - aspect), 0, 1)

    fig, ax = plt.subplots(figsize=(11, 11), dpi=110)
    ax.imshow(shade, cmap="gray", origin="lower",
              extent=[xs[0], xs[-1], ys[0], ys[-1]], vmin=0, vmax=1)
    for i, c in enumerate(contours):
        major = (i % 5 == 0)
        color = "#00c8ff" if major else "#123a52"
        for line in c["lines"]:
            a = np.asarray(line, dtype=float).reshape(-1, 2)
            ax.plot(a[:, 0], a[:, 1], color=color, lw=1.3 if major else 0.55,
                    alpha=0.95 if major else 0.75, solid_capstyle="round")
    if water_level is not None:
        ax.contour(xs, ys, hs, levels=[water_level], colors="#1f6feb", linewidths=1.6)
    ax.set_title(f"{title}   {info.width}x{info.height}px  cell {info.cell_size:g}m  "
                 f"contour {contours[1]['elev'] - contours[0]['elev']:g}m" if len(contours) > 1 else title)
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y (Z) (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map", help="level name, e.g. avg_berlin.bin")
    ap.add_argument("--levels-dir", default=LEVELS_DIR)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    ap.add_argument("--max-dim", type=int, default=1024)
    ap.add_argument("--levels", type=int, default=24)
    ap.add_argument("--water-level", type=float, default=None)
    ap.add_argument("--flip-y", action="store_true",
                    help="treat file row 0 as max world Y (mirror the map vertically)")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    args.levels_dir = args.levels_dir or levels_dir()

    name = args.map if args.map.endswith(".bin") else args.map + ".bin"
    path = os.path.join(args.levels_dir, name)
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(name)[0]

    try:
        heights, info = load_map_heightmap(path, verbose=False, max_dim=args.max_dim)
    except HeightmapUnavailable as e:
        print(f"SKIP {base}: no heightmap source ({e.reason})", file=sys.stderr)
        print(f"  {e.detail}", file=sys.stderr)
        return 2
    print(info)
    print("heights: min=%.2f max=%.2f mean=%.2f m" % (heights.min(), heights.max(), heights.mean()))
    water_level = args.water_level
    if water_level is None and heights.min() < 0.0 < heights.max():
        water_level = 0.0
    contours, meta = build_contours(heights, info, args.max_dim, args.levels, args.flip_y,
                                   water_level=water_level)
    np.savez_compressed(os.path.join(args.out_dir, base + "_heightmap.npz"), h=heights)

    h, w = heights.shape
    # Bounds must cover the *sampled* grid, whose points sit at pixel centres and can
    # reach half a sample-cell past the last sample. Derive them from the sampling meta
    # so every emitted coordinate is guaranteed inside [minX,maxX] x [minY,maxY].
    xs, ys, s_cell = meta["xs"], meta["ys"], meta["cell"]
    bounds = {
        "minX": float(xs[0] - 0.5 * s_cell),
        "maxX": float(xs[-1] + 0.5 * s_cell),
        "minY": float(ys[0] - 0.5 * s_cell),
        "maxY": float(ys[-1] + 0.5 * s_cell),
    }

    source = getattr(info, "source", "hm2")
    if source == "hm2":
        note = ("x = world X (east), y = world north/Z, meters. Heights decoded from the level's "
                "HM2 block (Dagor CompressedHeightmap) - exact.")
    else:
        note = ("x = world X (east), y = world north/Z, meters. Level has no HM2 heightmap (legacy "
                "format); heights rebuilt from the lmap LandMesh triangle mesh (TIN). Verified "
                "against HM2 on 90 levels that carry both: median absolute error 0.14 m, worst "
                "2.39 m.")

    doc = {
        "meta": {
            "crs": "replay-world-meters",
            "map": base,
            "source": source,
            "bounds": {k: round(v, 2) for k, v in bounds.items()},
            "widthPx": w, "heightPx": h,
            "cellSizeM": info.cell_size,
            "heightMin": round(float(heights.min()), 2),
            "heightMax": round(float(heights.max()), 2),
            "contourIntervalM": meta["interval"],
            "sampleStep": meta["step"],
            "simplifyTolM": round(meta["tol"], 2),
            "originX": info.ofs_x, "originY": info.ofs_y,
            "yFlipped": bool(args.flip_y),
            "waterLevel": water_level,
            "note": note + (" Row 0 of the heightmap is at originY"
                            + (" (y axis mirrored)" if args.flip_y else "") + "."),
        },
        "contours": contours,
    }
    if water_level is not None:
        water = water_polygons(heights, info, water_level, args.max_dim, args.flip_y)
        doc["water"] = water
        # coastline = same water/land boundary, kept as open polylines for map drawing
        doc["coastlines"] = water

    out_json = os.path.join(args.out_dir, base + "_contours.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    nlines = sum(len(c["lines"]) for c in contours)
    npts = sum(len(l) // 2 for c in contours for l in c["lines"])
    print(f"contours: {len(contours)} levels, {nlines} polylines, {npts} points -> {out_json} "
          f"({os.path.getsize(out_json)/1e6:.2f} MB)")

    if not args.no_png:
        out_png = os.path.join(args.out_dir, base + "_contours.png")
        render_preview(heights, info, contours, out_png, base, water_level, args.flip_y)
        print("preview ->", out_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
