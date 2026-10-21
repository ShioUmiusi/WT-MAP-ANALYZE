"""Build land/water masks so the app can shade land and keep water transparent.

Two products per level, both aligned to the exported base-map texture grid:

  out/landmask/<map>_land.png   8-bit alpha-style mask, 255 = land, 0 = water.
                                Same width/height and pixel convention as
                                out/textures/<map>_map.png (row 0 = world maxY), so it can be
                                used directly as an alpha channel / multiply layer.

  out/landmask/<map>_rings.json closed rings in world metres, to be filled with the
                                **even-odd** rule (holes are separate rings, so no need to
                                work out which ring is a hole).

Rings: `measure.find_contours` clips curves at the array edge, so a coastline that
reaches the border comes back as an open polyline and any ring assembly has to guess how
to stitch it. The mask is padded with one pixel of water first, so every ring is closed
by construction, and the vector form is a by-product of the same raster.

    python make_landmask.py --all
    python make_landmask.py air_denmark avg_berlin
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from PIL import Image
from skimage import measure

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
MASK_DIR = os.path.join(OUT, "landmask")
TEX_DIR = os.path.join(OUT, "textures")


def water_level_of(meta):
    wl = meta.get("waterLevel")
    if wl is not None:
        return float(wl)
    lo, hi = meta.get("heightMin"), meta.get("heightMax")
    if lo is not None and hi is not None and lo < 0 < hi:
        return 0.0
    return None


def texture_for(stem, tex_index):
    """The base-map texture this mask is aligned to (same file the app already loads)."""
    e = tex_index.get(stem)
    if not e:
        return None, None
    for key in ("map", "tankmap", "planemap", "airmap"):
        if key in e["textures"]:
            t = e["textures"][key]
            return key, t
    return None, None


def rdp(points, eps):
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
        seg = float(np.hypot(dx, dy))
        sub = points[i0 + 1:i1]
        if seg < 1e-12:
            d = np.hypot(sub[:, 0] - p0[0], sub[:, 1] - p0[1])
        else:
            d = np.abs(dx * (sub[:, 1] - p0[1]) - dy * (sub[:, 0] - p0[0])) / seg
        imax = int(np.argmax(d))
        if d[imax] > eps:
            idx = i0 + 1 + imax
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return points[keep]


def build(stem, tex_index, write=True, eps_px=0.75):
    npz = os.path.join(OUT, f"{stem}_heightmap.npz")
    cj = os.path.join(OUT, f"{stem}_contours.json")
    if not (os.path.exists(npz) and os.path.exists(cj)):
        return None
    h = np.load(npz)["h"].astype(np.float64)
    meta = json.load(open(cj, encoding="utf-8"))["meta"]
    wl = water_level_of(meta)
    b = meta["bounds"]

    if wl is None:
        land = np.ones_like(h, dtype=bool)
    else:
        land = h > wl

    # 1) raster mask on the base-map texture grid
    key, t = texture_for(stem, tex_index)
    if t is None:
        return dict(stem=stem, skipped="no texture")
    tw, th = int(t["width"]), int(t["height"])
    landf = land.astype(np.float32)
    im = Image.fromarray(landf, mode="F").resize((tw, th), Image.BILINEAR)
    alpha = np.clip(np.asarray(im, np.float32), 0.0, 1.0)
    png = os.path.join(MASK_DIR, f"{stem}_land.png")
    if write:
        os.makedirs(MASK_DIR, exist_ok=True)
        Image.fromarray((alpha * 255).astype(np.uint8), mode="L").save(png, optimize=True)

    # 2) closed rings: pad with water so nothing is clipped by the array edge
    padded = np.pad(land.astype(np.uint8), 1, mode="constant", constant_values=0)
    cell_x = (b["maxX"] - b["minX"]) / h.shape[1]
    cell_y = (b["maxY"] - b["minY"]) / h.shape[0]
    rings = []
    npts = 0
    for s in measure.find_contours(padded.astype(np.float32), 0.5):
        if len(s) < 8:
            continue
        # padded pixel (row, col) -> un-padded pixel centre -> world metres
        px = b["minX"] + (s[:, 1] - 1 + 0.5) * cell_x
        py = b["maxY"] - (s[:, 0] - 1 + 0.5) * cell_y
        pts = np.stack([px, py], axis=1)
        simp = rdp(pts, eps_px * max(cell_x, cell_y))
        if len(simp) < 3:
            continue
        rings.append([round(float(v), 1) for p in simp for v in p])
        npts += len(simp)

    doc = {
        "map": stem,
        "crs": "replay-world-meters",
        "bounds": b,
        "waterLevel": wl,
        "landFraction": round(float(land.mean()), 5),
        "mask": {
            "png": os.path.basename(png),
            "width": tw, "height": th,
            "alignedTo": f"out/textures/{t['png']}",
            "textureKey": key,
            "values": "255 = land, 0 = water; row 0 = world maxY (same as the base map)",
        },
        "rings": {
            "fillRule": "evenodd",
            "rings": rings,
            "pointCount": npts,
        },
        "note": ("Water is h <= waterLevel on the level's heightmap. Rivers and lakes that sit "
                 "above that level are treated as land."),
    }
    if write:
        with open(os.path.join(MASK_DIR, f"{stem}_rings.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"))
    return dict(stem=stem, w=int(tw), h=int(th), land=float(land.mean()),
                rings=len(rings), pts=npts, wl=wl, png=png if write else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()
    tex_index = json.load(open(os.path.join(TEX_DIR, "index.json"), encoding="utf-8"))["maps"]
    stems = args.maps
    if args.all or not stems:
        stems = sorted(os.path.basename(p)[:-len("_contours.json")]
                       for p in glob.glob(os.path.join(OUT, "*_contours.json")))
    rows = []
    for s in stems:
        r = build(s, tex_index, write=not args.no_write)
        if r is None:
            print(f"{s}: no heightmap/contours, skipped")
            continue
        if "skipped" in r:
            print(f"{s}: {r['skipped']}")
            continue
        rows.append(r)
        print(f"{s:28s} {r['w']}x{r['h']}  land {100*r['land']:5.1f}%  "
              f"rings {r['rings']:4d} ({r['pts']} pts)  waterLevel {r['wl']}")
    if rows and not args.no_write:
        idx = {
            "schema": "wt-landmask/1",
            "crs": "replay-world-meters",
            "note": ("One mask per level, aligned to the base-map texture the app already "
                     "loads: same width/height, same pixel convention (row 0 = world maxY). "
                     "Use the PNG as the land alpha, or fill `rings` with the even-odd rule."),
            "maps": {r["stem"]: {"png": f"{r['stem']}_land.png",
                                 "rings": f"{r['stem']}_rings.json",
                                 "width": r["w"], "height": r["h"],
                                 "landFraction": round(r["land"], 5),
                                 "waterLevel": r["wl"]} for r in rows},
        }
        with open(os.path.join(MASK_DIR, "index.json"), "w", encoding="utf-8") as f:
            json.dump(idx, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"\nwrote {len(rows)} masks -> {MASK_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
