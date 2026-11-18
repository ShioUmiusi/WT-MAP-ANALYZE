"""Check a hybrid contour file (whole level from lmap + battle tile from HM2).

Two independent things can go wrong and each gets its own assertion:

1. **Seam** - if the coarse layer was not blanked inside the tile, both layers draw the
   same ground and the map shows doubled lines; if the mask was too big, a ring of
   terrain has no contours at all. So: no coarse vertex inside the tile, and every
   detail vertex inside the tile.
2. **Placement** - contour vertices must sit on the height they claim. Each vertex is
   re-sampled (bilinear) from the same grid the lines came from; a shifted or mirrored
   coordinate frame shows up immediately as a large |height - elev|.

    python check_hybrid_contours.py avg_soviet_range            # one map from out/
    python check_hybrid_contours.py --all                       # every hybrid file in out/
    python check_hybrid_contours.py --dir dataset/contours --all  # assert what ships
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wt_maps import load_map_heightmap  # noqa: E402
from wt_paths import levels_dir  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EDGE_SLACK_M = 1.0        # a vertex may sit on the tile edge, not inside it by more
MAX_MEDIAN_FRAC = 0.5     # median |height - elev| must stay under half the interval
MAX_P99_FRAC = 1.0        # p99 may reach one interval (steep ground, RDP simplification)


def bilinear(heights, info, x, y):
    """Sample the source grid at world (x, y); NaN outside the grid."""
    col = (x - info.ofs_x) / info.cell_size - 0.5
    row = (y - info.ofs_y) / info.cell_size - 0.5
    h, w = heights.shape
    out = np.full(np.shape(x), np.nan, dtype=np.float64)
    ok = (col >= 0) & (col <= w - 1) & (row >= 0) & (row <= h - 1)
    if not ok.any():
        return out
    c0 = np.clip(np.floor(col[ok]).astype(int), 0, w - 2)
    r0 = np.clip(np.floor(row[ok]).astype(int), 0, h - 2)
    fc = np.clip(col[ok] - c0, 0, 1)
    fr = np.clip(row[ok] - r0, 0, 1)
    a = heights[r0, c0] * (1 - fc) + heights[r0, c0 + 1] * fc
    b = heights[r0 + 1, c0] * (1 - fc) + heights[r0 + 1, c0 + 1] * fc
    out[ok] = a * (1 - fr) + b * fr
    return out


def vertices(layers):
    for c in layers or []:
        for line in c.get("lines") or []:
            a = np.asarray(line, dtype=np.float64).reshape(-1, 2)
            if len(a):
                yield float(c.get("elev") or 0.0), a


def fidelity(layers, heights, info):
    """median / p99 of |sampled height - contour elev| over all vertices."""
    diffs = []
    for elev, a in vertices(layers):
        s = bilinear(heights, info, a[:, 0], a[:, 1])
        d = np.abs(s - elev)
        diffs.append(d[np.isfinite(d)])
    d = np.concatenate(diffs) if diffs else np.array([])
    if not d.size:
        return None
    return float(np.median(d)), float(np.percentile(d, 99)), int(d.size)


def check_one(stem, path, levels=None):
    bad = []
    doc = json.load(open(path, encoding="utf-8"))
    meta = doc["meta"]
    detail = doc.get("contoursDetail")
    det = meta.get("detail")
    if detail is None:
        return ["%s: no contoursDetail (not a hybrid file)" % stem]
    b = meta["bounds"]
    db = det["bounds"]
    if db["minX"] < b["minX"] - 1 or db["maxX"] > b["maxX"] + 1 \
            or db["minY"] < b["minY"] - 1 or db["maxY"] > b["maxY"] + 1:
        bad.append("%s: detail bounds %s outside bounds %s" % (stem, db, b))
    if not det["cellSizeM"] < meta["cellSizeM"]:
        bad.append("%s: detail cell %g m not finer than the coarse %g m"
                   % (stem, det["cellSizeM"], meta["cellSizeM"]))
    if len(detail) < 3:
        print("  note: flat tile, only %d detail level(s)" % len(detail))

    inside = 0
    for _elev, a in vertices(doc.get("contours")):
        inside += int(((a[:, 0] > db["minX"] + EDGE_SLACK_M) & (a[:, 0] < db["maxX"] - EDGE_SLACK_M)
                       & (a[:, 1] > db["minY"] + EDGE_SLACK_M) & (a[:, 1] < db["maxY"] - EDGE_SLACK_M)).sum())
    if inside:
        bad.append("%s: %d coarse contour vertices inside the tile (mask did not take)" % (stem, inside))

    outside = 0
    for _elev, a in vertices(detail):
        outside += int(((a[:, 0] < db["minX"] - EDGE_SLACK_M) | (a[:, 0] > db["maxX"] + EDGE_SLACK_M)
                        | (a[:, 1] < db["minY"] - EDGE_SLACK_M) | (a[:, 1] > db["maxY"] + EDGE_SLACK_M)).sum())
    if outside:
        bad.append("%s: %d detail contour vertices outside the tile" % (stem, outside))

    if levels:
        lv = os.path.join(levels, stem + ".bin")
        if os.path.exists(lv):
            hc, ic = load_map_heightmap(lv, max_dim=1024, prefer="landmesh")
            hf, iff = load_map_heightmap(lv, prefer="hm2")
            fc = fidelity(doc.get("contours"), hc, ic)
            fd = fidelity(detail, hf, iff)
            for tag, got, interval in (("coarse", fc, meta["contourIntervalM"]),
                                       ("detail", fd, det["contourIntervalM"])):
                if not got:
                    bad.append("%s: %s fidelity unmeasurable" % (stem, tag))
                    continue
                med, p99, n = got
                flag = "" if (med <= interval * MAX_MEDIAN_FRAC and p99 <= interval * MAX_P99_FRAC) else "  FAIL"
                print("  %-10s n=%-6d median %.2f m  p99 %.2f m  (interval %g m)%s"
                      % (tag, n, med, p99, interval, flag))
                if flag:
                    bad.append("%s: %s vertices sit %.2f m (median) off their own elevation"
                               % (stem, tag, med))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("maps", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dir", default=os.path.join(HERE, "out"))
    ap.add_argument("--no-height-check", action="store_true")
    args = ap.parse_args()
    if args.all:
        paths = sorted(glob.glob(os.path.join(args.dir, "*_contours.json")))
    elif args.maps:
        paths = [os.path.join(args.dir, m + "_contours.json" if not m.endswith(".json") else m)
                 for m in args.maps]
    else:
        ap.error("give map names or --all")
    bad = []
    hybrid = 0
    for p in paths:
        stem = os.path.basename(p)[:-len("_contours.json")]
        try:
            doc = json.load(open(p, encoding="utf-8"))
        except Exception as e:                                    # noqa: BLE001
            bad.append("%s: unreadable (%s)" % (stem, e))
            continue
        if "contoursDetail" not in doc:
            continue
        hybrid += 1
        print("%s" % stem)
        bad += check_one(stem, p, None if args.no_height_check else levels_dir(required=False))
    print("check: %d hybrid file(s), %d problem(s)" % (hybrid, len(bad)))
    for b in bad:
        print("  " + b)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
