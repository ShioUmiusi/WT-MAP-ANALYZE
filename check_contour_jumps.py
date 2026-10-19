"""Detect contour polylines that contain a jump between disconnected fragments.

Two independent tests per polyline:

1. **off-isoline distance**. Sample the heightmap along each segment and convert the
   height error into a distance using the local slope: ``d ~= |h - level| / |grad h|``.
   A straight RDP segment that really follows the isoline has ``d`` of the order of the
   simplification tolerance; a segment that bridges two unrelated fragments has ``d``
   of hundreds of metres even in steep terrain.

2. **topology**. The sampled grid has no holes, so every contour must be either closed
   or terminate on the array border. A polyline whose two ends are both strictly inside
   the map (and do not meet) can only exist if fragments were concatenated.

Usage:  python check_contour_jumps.py [map ...]
"""
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
DIST_TOL_M = 8.0          # a legit simplified segment stays well under this
BORDER_FRAC = 0.004       # "on the border" = within 0.4 % of the map size


class Grid:
    """The exact grid the contours were traced on, so the test is apples-to-apples."""

    def __init__(self, stem):
        h = np.load(os.path.join(OUT, f"{stem}_heightmap.npz"))["h"]
        doc = json.load(open(os.path.join(OUT, f"{stem}_contours.json"), encoding="utf-8"))
        self.meta = doc["meta"]
        self.doc = doc
        b = self.meta["bounds"]
        self.b = b
        step = max(int(self.meta.get("sampleStep", 1)), 1)
        # replicate make_contours.sample_grid
        self.h = h[::step, ::step].astype(np.float64)
        cs = self.meta["cellSizeM"]
        self.cell = cs * step
        self.xs0 = self.meta["originX"] + step * 0.5 * cs
        self.ys0 = self.meta["originY"] + step * 0.5 * cs
        self.w = b["maxX"] - b["minX"]
        self.hgt = b["maxY"] - b["minY"]
        ny, nx = self.h.shape
        gy, gx = np.gradient(self.h, self.cell, self.cell)
        self.grad = np.hypot(gx, gy)
        self.ny, self.nx = ny, nx

    def index(self, x, y):
        fx = np.clip((x - self.xs0) / self.cell, 0, self.nx - 1.001)
        fy = np.clip((y - self.ys0) / self.cell, 0, self.ny - 1.001)
        return fx, fy

    def sample(self, arr, x, y):
        fx, fy = self.index(x, y)
        x0, y0 = fx.astype(int), fy.astype(int)
        tx, ty = fx - x0, fy - y0
        return (arr[y0, x0] * (1 - tx) * (1 - ty) + arr[y0, x0 + 1] * tx * (1 - ty)
                + arr[y0 + 1, x0] * (1 - tx) * ty + arr[y0 + 1, x0 + 1] * tx * ty)

    def on_border(self, x, y):
        m = BORDER_FRAC * max(self.w, self.hgt)
        return (min(x - self.b["minX"], self.b["maxX"] - x,
                    y - self.b["minY"], self.b["maxY"] - y) <= m)


def analyse(stem, nsample=10, verbose=False):
    g = Grid(stem)
    # a legitimate simplified segment may deviate by about the RDP tolerance, and the
    # contour vertices themselves are quantised to the sampling cell
    tol_m = max(2.5 * float(g.meta.get("simplifyTolM") or 0.0), 1.5 * g.cell)
    jumps = []
    open_inside = []
    for ci, c in enumerate(g.doc["contours"]):
        lv = c["elev"]
        for li, line in enumerate(c["lines"]):
            a = np.asarray(line, float).reshape(-1, 2)
            if len(a) < 2:
                continue
            # test 2: open polyline with both ends strictly inside
            closed = np.hypot(*(a[0] - a[-1])) <= g.cell
            if not closed and not (g.on_border(*a[0]) or g.on_border(*a[-1])):
                open_inside.append((ci, li, lv, a[0].tolist(), a[-1].tolist(),
                                    float(np.hypot(*(a[0] - a[-1])))))
            # test 1: distance of each segment from the isoline
            for k in range(len(a) - 1):
                p0, p1 = a[k], a[k + 1]
                L = float(np.hypot(*(p1 - p0)))
                if L < 4 * g.cell:
                    continue
                t = np.linspace(0.15, 0.85, nsample)
                xs = p0[0] + (p1[0] - p0[0]) * t
                ys = p0[1] + (p1[1] - p0[1]) * t
                err = np.abs(g.sample(g.h, xs, ys) - lv)
                sl = g.sample(g.grad, xs, ys)
                dist = err / np.maximum(sl, 1e-4)
                if dist.max() > tol_m:
                    jumps.append((ci, li, lv, L, float(dist.max()), p0.tolist(), p1.tolist()))
    return g, jumps, open_inside


def main():
    stems = sys.argv[1:]
    if not stems:
        stems = sorted(os.path.basename(p)[:-len("_contours.json")]
                       for p in glob.glob(os.path.join(OUT, "*_contours.json")))
    total_j = total_o = bad_maps = 0
    worst = []
    for s in stems:
        try:
            g, jumps, open_inside = analyse(s)
        except Exception as e:
            print(f"{s}: FAILED {type(e).__name__}: {e}")
            continue
        total_j += len(jumps)
        total_o += len(open_inside)
        nlines = sum(len(c["lines"]) for c in g.doc["contours"])
        if jumps or open_inside:
            bad_maps += 1
        dmax = max((j[4] for j in jumps), default=0.0)
        print(f"{s:30s} polylines {nlines:5d}  jumps {len(jumps):4d} (worst {dmax:8.1f} m)  "
              f"open-inside {len(open_inside):3d}")
        if jumps:
            worst.append((dmax, s, jumps[:3]))
    print(f"\nmaps {len(stems)}, with problems {bad_maps}; "
          f"jump segments {total_j}; open-inside polylines {total_o}")
    worst.sort(reverse=True)
    for d, s, js in worst[:5]:
        print(f"  {s}: worst {d:.0f} m, e.g. {js[:1]}")
    return 1 if (total_j or total_o) else 0


if __name__ == "__main__":
    sys.exit(main())
