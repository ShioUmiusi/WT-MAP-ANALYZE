"""Validate the JSON land masks and render a proof that they are directly usable.

Checks
  1. RLE decodes to the same land fraction as `h > waterLevel` on the heightmap.
  2. the bitmap is the north-up flip of `h > waterLevel` (row 0 = maxY), i.e. the
     declared `rowZero` matches the data; a south-up grid fails here.
  3. every ring is closed and lies inside `bounds`.
  4. the rings and the RLE describe the same region (rasterise the rings with the
     even-odd rule and compare against the bitmap).

Renders `out/landmask/_proof/<map>_proof.png`: the decoded mask used as alpha over a
checkerboard, with the base-map texture showing through on the land side when
`out/textures/` exists.
"""
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
LM = os.path.join(OUT, "landmask")
TEX = os.path.join(OUT, "textures")
PROOF = os.path.join(OUT, "landmask", "_proof")
os.makedirs(PROOF, exist_ok=True)


def decode(doc):
    l = doc["land"]
    w, h = l["width"], l["height"]
    flat = np.empty(w * h, np.uint8)
    pos, val = 0, int(l["startsWith"])
    for run in l["runs"]:
        flat[pos:pos + run] = val
        pos += run
        val ^= 1
    assert pos == w * h, f"RLE covers {pos} of {w*h} pixels"
    return flat.reshape(h, w).astype(bool)


def rasterise_rings(doc, w, h):
    """Fill the rings with the even-odd rule at mask resolution; returns a bool array."""
    im = Image.new("1", (w, h), 0)
    b = doc["bounds"]
    dr = ImageDraw.Draw(im)
    for ring in doc["landRings"]["rings"]:
        a = np.asarray(ring, float).reshape(-1, 2)
        pts = [((x - b["minX"]) / (b["maxX"] - b["minX"]) * w,
                (b["maxY"] - y) / (b["maxY"] - b["minY"]) * h) for x, y in a]
        # PIL cannot do even-odd directly, so emulate: draw each ring into its own
        # bitmap and XOR them together
        one = Image.new("1", (w, h), 0)
        ImageDraw.Draw(one).polygon(pts, fill=1)
        im = Image.fromarray(np.asarray(im, np.uint8) ^ np.asarray(one, np.uint8))
    return np.asarray(im).astype(bool)


def main():
    stems = sys.argv[1:] or sorted(
        os.path.basename(p)[:-5] for p in os.listdir(LM)
        if p.endswith(".json") and p != "index.json")
    ok = True
    for stem in stems:
        p = os.path.join(LM, f"{stem}.json")
        if not os.path.exists(p):
            continue
        doc = json.load(open(p, encoding="utf-8"))
        b = doc["bounds"]
        land = decode(doc)
        hh, ww = land.shape

        # 1. compare with the heightmap at full resolution
        npz = os.path.join(OUT, f"{stem}_heightmap.npz")
        ref = None
        orient = None
        if os.path.exists(npz):
            h = np.load(npz)["h"]
            wl = doc["waterLevel"]
            ref_arr = np.ones_like(h, bool) if wl is None else (h > wl)
            ref = float(ref_arr.mean())
            # The JSON grid is north-up (row 0 = maxY); the heightmap is south-up
            # (row 0 = minY). Rebuild the same grid the generator uses and compare
            # against it flipped, so a wrong row order cannot pass on fractions alone.
            f = max(1, int(np.ceil(max(h.shape) / 2048.0)))
            if f > 1:
                hh2, ww2 = h.shape[0] // f, h.shape[1] // f
                ref_arr = (ref_arr[:hh2 * f, :ww2 * f]
                           .reshape(hh2, f, ww2, f).mean(axis=(1, 3)) >= 0.5)
            ref_up = ref_arr[::-1]
            if ref_up.shape == land.shape:
                orient = float((ref_up == land).mean())
        frac = float(land.mean())
        d_frac = None if ref is None else abs(frac - ref)

        # 2. rings closed and inside bounds
        unclosed = 0
        outside = 0
        for ring in doc["landRings"]["rings"]:
            a = np.asarray(ring, float).reshape(-1, 2)
            if np.hypot(*(a[0] - a[-1])) > 1e-6:
                unclosed += 1
            if (a[:, 0].min() < b["minX"] - 1e-6 or a[:, 0].max() > b["maxX"] + 1e-6
                    or a[:, 1].min() < b["minY"] - 1e-6 or a[:, 1].max() > b["maxY"] + 1e-6):
                outside += 1

        # 3. rings vs bitmap agreement
        ring_mask = rasterise_rings(doc, ww, hh)
        agree = float((ring_mask == land).mean())

        status = "OK"
        if d_frac is not None and d_frac > 0.01:
            status = "LAND-FRACTION MISMATCH"
            ok = False
        if orient is not None and orient < 0.995:
            status = "ORIENTATION MISMATCH"
            ok = False
        if unclosed or outside:
            status = "RING PROBLEM"
            ok = False
        if agree < 0.97:
            status = "RINGS != BITMAP"
            ok = False
        print(f"{stem:26s} {ww}x{hh}  land {100*frac:5.1f}% "
              f"(ref {100*ref if ref is not None else float('nan'):5.1f}%)  "
              f"rings closed {len(doc['landRings']['rings'])-unclosed}/"
              f"{len(doc['landRings']['rings'])}  outside {outside}  "
              f"rings==bitmap {100*agree:5.1f}%  "
              f"north-up {100*orient if orient is not None else float('nan'):5.1f}%  {status}")

        # proof render
        size = 720
        base = None
        tex_idx_path = os.path.join(TEX, "index.json")
        if os.path.exists(tex_idx_path):
            ti = json.load(open(tex_idx_path, encoding="utf-8"))["maps"].get(stem)
            if ti:
                for k in ("map", "tankmap", "planemap", "airmap"):
                    if k in ti["textures"]:
                        base = Image.open(os.path.join(TEX, ti["textures"][k]["png"])).convert("RGB")
                        base = base.resize((size, size), Image.LANCZOS)
                        break
        mask = Image.fromarray((land * 255).astype(np.uint8), "L").resize((size, size), Image.BILINEAR)
        checker = Image.new("RGB", (size, size), (235, 235, 235))
        dc = ImageDraw.Draw(checker)
        for y in range(0, size, 24):
            for x in range(0, size, 24):
                if (x // 24 + y // 24) % 2:
                    dc.rectangle([x, y, x + 23, y + 23], fill=(200, 200, 200))
        out = checker.copy()
        if base is not None:
            out.paste(base, (0, 0), mask)          # land = base map, water = checkerboard
        else:
            out.paste(Image.new("RGB", (size, size), (210, 120, 90)), (0, 0), mask)
        ImageDraw.Draw(out).text((8, 8), f"{stem}: land shaded, water transparent (from JSON)",
                                 fill=(0, 0, 0))
        q = os.path.join(PROOF, f"{stem}_proof.png")
        out.save(q)
    print("\nall checks passed" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
