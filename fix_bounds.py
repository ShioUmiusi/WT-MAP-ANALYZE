"""Recompute `meta.bounds` in every contour JSON so it covers the sampled grid.

Sampling places point i at  origin + (i*step + step/2) * cellSize, so the covered world
area is  [origin, origin + ncols*step*cellSize]  (i.e. origin is the pixel EDGE, not the
first centre). The old code used origin + (w-1)*cellSize, which is short by half a sample
cell -- contour points legitimately reached slightly past it. For small HM2 cells (2 m)
that hid inside the validator's 1 m tolerance; for land-mesh grids (64 m) it stuck out by
32 m.

Also refreshes the land-mesh `note` so it states the measured accuracy.
"""
import glob
import json
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

NOTE_HM2 = ("x = world X (east), y = world north/Z, meters. Heights decoded from the "
            "level's HM2 block (Dagor CompressedHeightmap) - exact.")
NOTE_LM = ("x = world X (east), y = world north/Z, meters. Level has no HM2 heightmap "
           "(legacy format); heights rebuilt from the lmap LandMesh triangle mesh (TIN). "
           "Verified against HM2 on 90 levels that carry both: median absolute error "
           "0.14 m, worst 2.39 m.")

fixed = 0
skipped = 0
for path in sorted(glob.glob(os.path.join(OUT, "*_contours.json"))):
    doc = json.load(open(path, encoding="utf-8"))
    m = doc["meta"]
    ox, oy = m.get("originX"), m.get("originY")
    cs = m.get("cellSizeM")
    w, h = m.get("widthPx"), m.get("heightPx")
    step = m.get("sampleStep", 1) or 1
    if None in (ox, oy, cs, w, h):
        skipped += 1
        continue

    ncols = math.ceil(w / step)
    nrows = math.ceil(h / step)
    scell = step * cs
    new_bounds = {
        "minX": round(ox, 2),
        "maxX": round(ox + ncols * scell, 2),
        "minY": round(oy, 2),
        "maxY": round(oy + nrows * scell, 2),
    }
    src = m.get("source", "hm2")
    m["source"] = src
    m["bounds"] = new_bounds
    m["note"] = NOTE_LM if src == "landmesh" else NOTE_HM2

    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    fixed += 1

print(f"patched {fixed} contour files, skipped {skipped}")
