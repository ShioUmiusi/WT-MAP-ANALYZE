"""Land / water region per level, encoded for JSON consumers.

The app only reads JSON, so instead of shipping a PNG mask the region is written twice:

  "land.rle"   run-length encoded bitmap, row-major from (0,0), alternating 1 / 0.
               No topology, no holes, no boundary problem - decode with one loop.
  "landRings"  the same region as closed rings in world metres, to be filled with the
               **even-odd** rule. Holes are just more rings, so nothing has to be
               classified as an outer ring or a hole.

The rings are closed *by construction*: the bitmap is padded with one row/column of water
before contours are taken, so no curve is ever clipped by the array edge. Building the
rings by stitching the level's coastline polylines together is what goes wrong, because
those polylines come back open wherever they reach the border.

Grid: the heightmap's own resolution, downsampled by an integer factor to at most
`--max-dim` (default 2048). `bounds` is the same rectangle the base-map texture uses, so
sampling by pixel centre lines up with `out/textures/<map>_map.png`.

    python make_landmask_json.py --all
    python make_landmask_json.py air_denmark avg_berlin
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DST = os.path.join(OUT, "landmask")
MIN_RING_AREA_CELLS = 4.0     # drop rings smaller than (2 cells)^2
RDP_EPS_CELLS = 0.5


def water_level_of(meta):
    wl = meta.get("waterLevel")
    if wl is not None:
        return float(wl)
    lo, hi = meta.get("heightMin"), meta.get("heightMax")
    if lo is not None and hi is not None and lo < 0 < hi:
        return 0.0
    return None


def downsample_bool(land, max_dim):
    """Integer-factor box downsample, thresholded back to bool (keeps thin features)."""
    hh, ww = land.shape
    f = max(1, int(np.ceil(max(hh, ww) / float(max_dim))))
    if f == 1:
        return land, 1
    h2, w2 = (hh // f) * f, (ww // f) * f
    a = land[:h2, :w2].reshape(h2 // f, f, w2 // f, f).mean(axis=(1, 3))
    return a >= 0.5, f


def rle_of(land):
    flat = land.reshape(-1).astype(np.uint8)
    change = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate([[0], change, [len(flat)]])
    return int(flat[0]), (bounds[1:] - bounds[:-1]).astype(int).tolist()


def rdp(points, eps):
    n = len(points)
    if n < 3 or eps <= 0:
        return points
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    st = [(0, n - 1)]
    while st:
        i0, i1 = st.pop()
        if i1 <= i0 + 1:
            continue
        p0, p1 = points[i0], points[i1]
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        seg = float(np.hypot(dx, dy))
        sub = points[i0 + 1:i1]
        d = (np.hypot(sub[:, 0] - p0[0], sub[:, 1] - p0[1]) if seg < 1e-12
             else np.abs(dx * (sub[:, 1] - p0[1]) - dy * (sub[:, 0] - p0[0])) / seg)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i0 + 1 + k] = True
            st.append((i0, i0 + 1 + k))
            st.append((i0 + 1 + k, i1))
    return points[keep]


def shoelace(px, py):
    return 0.5 * float(np.dot(px, np.roll(py, -1)) - np.dot(py, np.roll(px, -1)))


def build(stem, max_dim=2048, write=True):
    npz = os.path.join(OUT, f"{stem}_heightmap.npz")
    cj = os.path.join(OUT, f"{stem}_contours.json")
    if not (os.path.exists(npz) and os.path.exists(cj)):
        return None
    h = np.load(npz)["h"].astype(np.float64)
    meta = json.load(open(cj, encoding="utf-8"))["meta"]
    wl = water_level_of(meta)
    b = meta["bounds"]

    land_full = np.ones_like(h, bool) if wl is None else (h > wl)
    land, factor = downsample_bool(land_full, max_dim)
    hh, ww = land.shape
    cellx = (b["maxX"] - b["minX"]) / ww
    celly = (b["maxY"] - b["minY"]) / hh

    starts, runs = rle_of(land)

    padded = np.pad(land.astype(np.uint8), 1, constant_values=0)
    rings = []
    epts = 0
    for r in measure.find_contours(padded.astype(np.float32), 0.5):
        if len(r) < 4:
            continue
        px = b["minX"] + (r[:, 1] - 1 + 0.5) * cellx
        py = b["maxY"] - (r[:, 0] - 1 + 0.5) * celly
        # rings that follow the padded border sit half a cell outside the map; clamping
        # keeps every emitted coordinate inside `bounds` (the contract the app checks)
        px = np.clip(px, b["minX"], b["maxX"])
        py = np.clip(py, b["minY"], b["maxY"])
        if abs(shoelace(px, py)) < MIN_RING_AREA_CELLS * cellx * celly:
            continue
        simp = rdp(np.stack([px, py], 1), RDP_EPS_CELLS * max(cellx, celly))
        # round first, then clamp: rounding to 0.1 m can otherwise push a border point
        # a hair outside `bounds`, which the app's own validator would reject
        simp = np.round(simp, 1)
        simp[:, 0] = np.clip(simp[:, 0], b["minX"], b["maxX"])
        simp[:, 1] = np.clip(simp[:, 1], b["minY"], b["maxY"])
        # drop consecutive duplicates introduced by the clamp
        keep = np.ones(len(simp), bool)
        for i in range(1, len(simp)):
            if np.hypot(*(simp[i] - simp[i - 1])) < 1e-9:
                keep[i] = False
        simp = simp[keep]
        if len(simp) < 3:
            continue
        rings.append([float(v) for p in simp for v in p])
        epts += len(simp)

    doc = {
        "schema": "wt-landmask/1",
        "map": stem,
        "crs": "replay-world-meters",
        "bounds": b,
        "waterLevel": wl,
        "landFraction": round(float(land.mean()), 6),
        "land": {
            "encoding": "rle-rows",
            "width": ww, "height": hh,
            "rowZero": "maxY",
            "startsWith": starts,
            "runs": runs,
        },
        "landRings": {"fillRule": "evenodd", "rings": rings, "pointCount": epts},
        "gridNote": (f"mask grid {ww}x{hh}"
                     + (f", downsampled {factor}x from the heightmap" if factor > 1 else "")
                     + "; pixel centres map by worldX = minX+(px+0.5)*(maxX-minX)/width, "
                       "worldY = maxY-(py+0.5)*(maxY-minY)/height"),
        "waterLevelNote": ("water is h > waterLevel counted as land; levels equal to the "
                           "waterline count as water"),
    }
    if wl is None:
        doc["waterLevelNote"] = ("this level has no heights below zero, so the whole map is "
                                 "land; rivers above sea level are NOT water by this criterion")
    if write:
        os.makedirs(DST, exist_ok=True)
        p = os.path.join(DST, f"{stem}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
        size = os.path.getsize(p)
    else:
        size = len(json.dumps(doc, separators=(",", ":")))
    return dict(stem=stem, w=ww, h__=hh, land=float(land.mean()), rings=len(rings),
                pts=epts, runs=len(runs), wl=wl, size=size, factor=factor)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-dim", type=int, default=2048)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    stems = args.maps or sorted(os.path.basename(p)[:-len("_contours.json")]
                                for p in glob.glob(os.path.join(OUT, "*_contours.json")))
    rows = []
    for s in stems:
        r = build(s, max_dim=args.max_dim, write=not args.no_write)
        if r is None:
            print(f"{s}: no heightmap/contours, skipped")
            continue
        rows.append(r)
        print(f"{s:26s} {r['w']:5d}x{r['h__']:<5d} land {100*r['land']:5.1f}%  "
              f"rings {r['rings']:5d}/{r['pts']:6d}pt  runs {r['runs']:6d}  "
              f"{r['size']/1024:7.1f} KB  wl {r['wl']}")
    if rows and not args.no_write:
        idx = {
            "schema": "wt-landmask/1",
            "crs": "replay-world-meters",
            "note": ("One file per level. `land.rle` is the primary form (decode row-major, "
                     "1/0 alternating, from (0,0)); `landRings` is the same region as closed "
                     "rings for even-odd fill. Both use the level's `bounds`, so they line up "
                     "with the base-map texture by pixel centre."),
            "maps": {r["stem"]: {"file": f"{r['stem']}.json",
                                 "grid": [r["w"], r["h__"]],
                                 "landFraction": round(r["land"], 6),
                                 "waterLevel": r["wl"],
                                 "rings": r["rings"],
                                 "bytes": r["size"]} for r in rows},
        }
        with open(os.path.join(DST, "index.json"), "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=1, sort_keys=True)
        tot = sum(r["size"] for r in rows)
        print(f"\nwrote {len(rows)} files, {tot/1e6:.1f} MB -> {DST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
