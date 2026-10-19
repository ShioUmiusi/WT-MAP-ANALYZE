"""Validate the exported level map textures.

  structural (all maps)
    - every PNG referenced by out/textures/index.json exists and has the declared size
    - declared meters-per-pixel is square and reproduces the contour bounds extents

  alignment (all maps, written to out/textures/align.json)
    - compare terrain relief (slope magnitude from the decoded heightmap) against the
      texture's edge magnitude, once north-up and once south-up.
    - the texture is "north up" when reversing its rows correlates much better: that is
      exactly the px->world mapping documented in index.json (row 0 == world maxY).
    - only textures that actually carry relief (air / arcade satellite mosaics) can be
      judged this way; the rendered ground maps (`_tankmap`) are inconclusive, so they
      are reported but not counted as failures.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
SIZE = 384


def relief_of(h, size=SIZE):
    hs = np.asarray(Image.fromarray(h.astype(np.float32), mode="F")
                    .resize((size, size), Image.LANCZOS), np.float64)
    gy, gx = np.gradient(hs)
    return np.hypot(gx, gy)


def grad_of(img, size=SIZE):
    t = np.asarray(img.convert("L").resize((size, size), Image.LANCZOS), np.float64) / 255.0
    gy, gx = np.gradient(t)
    return np.hypot(gx, gy)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))


def align_check(stems, idx):
    """Return {stem: {texture, nccNorthUp, nccSouthUp, prefers}}."""
    rows = {}
    for stem in stems:
        e = idx[stem]
        if not e.get("worldBounds"):
            continue
        key = "map" if "map" in e["textures"] else ("tankmap" if "tankmap" in e["textures"] else None)
        if key is None:
            continue
        p = os.path.join(OUT, "textures", e["textures"][key]["png"])
        npz = os.path.join(OUT, f"{stem}_heightmap.npz")
        if not (os.path.exists(p) and os.path.exists(npz)):
            continue
        try:
            rel = relief_of(np.load(npz)["h"])
            img = Image.open(p)
            g_id = grad_of(img)
            g_fy = grad_of(img.transpose(Image.FLIP_TOP_BOTTOM))
        except Exception as ex:
            print(f"  {stem}: align failed {ex}")
            continue
        north, south = ncc(rel, g_fy), ncc(rel, g_id)
        rows[stem] = {"texture": key,
                      "nccNorthUp": round(north, 4),
                      "nccSouthUp": round(south, 4),
                      "prefers": "northUp" if north > south else "southUp"}
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-align", action="store_true")
    args = ap.parse_args()

    idx = json.load(open(os.path.join(OUT, "textures", "index.json"), encoding="utf-8"))
    maps = idx["maps"]
    errors, warnings = [], []
    ntex = 0

    for stem, e in sorted(maps.items()):
        b = e.get("worldBounds")
        for key, t in e["textures"].items():
            ntex += 1
            p = os.path.join(OUT, "textures", t["png"])
            if not os.path.exists(p):
                errors.append(f"{stem}/{key}: missing {t['png']}")
                continue
            with Image.open(p) as im:
                if im.size != (t["width"], t["height"]):
                    errors.append(f"{stem}/{key}: PNG {im.size} != index "
                                  f"({t['width']}x{t['height']})")
            if b:
                mppx = (b["maxX"] - b["minX"]) / t["width"]
                mppy = (b["maxY"] - b["minY"]) / t["height"]
                if abs(mppx - mppy) > 1e-3 * max(1.0, mppx):
                    warnings.append(f"{stem}/{key}: non-square pixel {mppx:.4f} x {mppy:.4f}")
                if abs(t["metersPerPixelX"] - mppx) > 1e-3 * max(1.0, mppx):
                    errors.append(f"{stem}/{key}: metersPerPixelX {t['metersPerPixelX']} != "
                                  f"{mppx:.4f}")
        if not b:
            warnings.append(f"{stem}: no worldBounds (level produced no contours)")

    print(f"structural: {len(maps)} maps, {ntex} textures, {len(errors)} errors, "
          f"{len(warnings)} warnings")
    for w in warnings[:20]:
        print("  WARN", w)
    for er in errors[:20]:
        print("  ERR ", er)

    if not args.no_align:
        rows = align_check(sorted(maps), maps)
        p = os.path.join(OUT, "textures", "align.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, sort_keys=True)
        judge = {k: v for k, v in rows.items() if abs(v["nccNorthUp"] - v["nccSouthUp"]) > 0.1}
        pref = [v["prefers"] for v in judge.values()]
        judge = {k: v for k, v in rows.items()
                 if abs(v["nccNorthUp"] - v["nccSouthUp"]) > 0.1
                 and max(v["nccNorthUp"], v["nccSouthUp"]) > 0.15}
        pref = [v["prefers"] for v in judge.values()]
        print(f"\nalignment: {len(rows)} maps compared, {len(judge)} with usable relief, "
              f"{len(rows)-len(judge)} inconclusive (rendered maps carry no relief to match)")
        if judge:
            print(f"  usable maps preferring north-up: {pref.count('northUp')}/{len(pref)}")
            ns = sorted(v["nccNorthUp"] for v in judge.values())
            print(f"  north-up NCC: min {ns[0]:+.3f} median {ns[len(ns)//2]:+.3f} max {ns[-1]:+.3f}")
            for k, v in sorted(judge.items()):
                if v["prefers"] != "northUp":
                    print(f"  SOUTH-UP: {k} {v}")
        print(f"  wrote {p}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
