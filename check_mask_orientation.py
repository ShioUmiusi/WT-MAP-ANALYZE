"""Orientation check for the land masks, using the base-map texture as an independent source.

On satellite and rendered base maps the sea is dark blue. If a mask is stored upside down
its 'water' pixels land on the texture's land and vice versa, so the agreement with the
texture's own water colour collapses. Land fraction alone cannot catch that - it is the
same either way.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
LM = os.path.join(OUT, "landmask")
TEX = os.path.join(OUT, "textures")


def decode(doc):
    l = doc["land"]
    w, h = l["width"], l["height"]
    flat = np.empty(w * h, np.uint8)
    pos, val = 0, int(l["startsWith"])
    for run in l["runs"]:
        flat[pos:pos + run] = val
        pos += run
        val ^= 1
    return flat.reshape(h, w).astype(bool)


def tex_water_mask(stem, size):
    ti = json.load(open(os.path.join(TEX, "index.json"), encoding="utf-8"))["maps"].get(stem)
    if not ti:
        return None
    for k in ("map", "tankmap", "planemap", "airmap"):
        if k in ti["textures"]:
            im = Image.open(os.path.join(TEX, ti["textures"][k]["png"])).convert("RGB")
            a = np.asarray(im.resize((size, size), Image.BILINEAR), np.int32)
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            # blue-dominant and not too bright: catches sea on both satellite and
            # hand-rendered base maps
            return (b > r + 8) & (b >= g) & (b < 170)
    return None


def main():
    stems = sys.argv[1:] or sorted(
        os.path.basename(p)[:-5] for p in os.listdir(LM)
        if p.endswith(".json") and p != "index.json")
    rows = []
    for s in stems:
        p = os.path.join(LM, f"{s}.json")
        if not os.path.exists(p):
            continue
        doc = json.load(open(p, encoding="utf-8"))
        if doc.get("waterLevel") is None:
            continue
        land = decode(doc)
        hh, ww = land.shape
        tw = tex_water_mask(s, 256)
        if tw is None:
            continue
        # resample the mask to the texture comparison grid
        m = np.asarray(Image.fromarray((land * 255).astype(np.uint8), "L")
                       .resize((256, 256), Image.BILINEAR), np.float64) / 255.0
        land_b = m > 0.5
        water_b = ~land_b
        # agreement: texture water should sit where the mask says water
        a = float((tw == water_b).mean())
        # also the flipped reading, to show what a wrong orientation would score
        land_f = np.asarray(Image.fromarray((land[::-1, :] * 255).astype(np.uint8), "L")
                            .resize((256, 256), Image.BILINEAR), np.float64) / 255.0 > 0.5
        af = float((tw == (~land_f)).mean())
        rows.append((a, af, s, float(doc["landFraction"]), float(tw.mean())))
    rows.sort()
    print(f"{'map':26s} {'agree':>7s} {'ifFlipped':>10s} {'landFrac':>9s} {'texWater':>9s}")
    for a, af, s, lf, twm in rows:
        flag = "" if a >= af else "   <-- flipped version fits better"
        print(f"{s:26s} {100*a:6.1f}% {100*af:9.1f}% {100*lf:8.1f}% {100*twm:8.1f}%{flag}")
    if rows:
        ag = np.array([r[0] for r in rows])
        fl = np.array([r[1] for r in rows])
        print(f"\nmedian agreement {100*np.median(ag):.1f}%   "
              f"median if-flipped {100*np.median(fl):.1f}%   "
              f"correct orientation wins {int((ag > fl).sum())}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
