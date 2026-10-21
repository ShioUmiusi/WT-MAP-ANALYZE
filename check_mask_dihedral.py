"""Which orientation does the stored mask actually have, compared with the base map?

Tests all 8 dihedral transforms of the decoded mask against the texture's water colour and
reports the agreement for each. The identity should win if `rowZero: maxY` is honest.
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
N = 256

OPS = {
    "identity": lambda a: a,
    "flipY": lambda a: a[::-1, :],
    "flipX": lambda a: a[:, ::-1],
    "rot180": lambda a: a[::-1, ::-1],
    "transpose": lambda a: a.T,
    "rot90": lambda a: np.rot90(a, 1),
    "rot270": lambda a: np.rot90(a, 3),
    "antitranspose": lambda a: np.rot90(a, 2).T,
}


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


def tex_water(stem):
    ti = json.load(open(os.path.join(TEX, "index.json"), encoding="utf-8"))["maps"].get(stem)
    if not ti:
        return None
    for k in ("map", "tankmap", "planemap", "airmap"):
        if k in ti["textures"]:
            im = Image.open(os.path.join(TEX, ti["textures"][k]["png"])).convert("RGB")
            a = np.asarray(im.resize((N, N), Image.BILINEAR), np.int32)
            r, g, b = a[..., 0], a[..., 1], a[..., 2]
            return (b > r + 8) & (b >= g) & (b < 170)
    return None


for stem in sys.argv[1:] or ["air_denmark", "malta", "sicily", "guadalcanal"]:
    doc = json.load(open(os.path.join(LM, f"{stem}.json"), encoding="utf-8"))
    m = decode(doc)
    tw = tex_water(stem)
    if tw is None:
        print(stem, "no texture")
        continue
    print(f"\n{stem}: land {100*doc['landFraction']:.1f}%  texture-water {100*tw.mean():.1f}%")
    scores = {}
    for name, op in OPS.items():
        a = op(m)
        if a.shape != (N, N):
            a = np.asarray(Image.fromarray((a * 255).astype(np.uint8), "L")
                           .resize((N, N), Image.BILINEAR), np.float64) > 0.5
        else:
            a = np.asarray(Image.fromarray((a * 255).astype(np.uint8), "L")
                           .resize((N, N), Image.BILINEAR), np.float64) > 0.5
        water = ~a
        scores[name] = float((tw == water).mean())
    for name, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        mark = "  <== best" if v == max(scores.values()) else ""
        print(f"   {name:14s} {100*v:6.1f}%{mark}")
