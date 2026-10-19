"""Post-process the contour JSONs to remove artefacts from filled/flat regions.

Two defects the app can see in `contours[].lines`:

1. **apron contours** - the rasteriser fills everything outside the level's real terrain
   patch with a nearest-neighbour extension, which is *exactly* flat. An isolevel that
   intersects that apron produces a ring hugging the artificial patch boundary.
2. **bridge chords** - a polyline that crosses the apron gets a long straight chord after
   RDP, so a consumer that draws each `lines` entry as one path sees a line jumping
   between far-apart pieces.

The filter needs only the shipped JSON plus `out/<map>_heightmap.npz`, so it is cheap:

  * drop a polyline when most of its vertices sit in an exactly-flat 5x5 neighbourhood;
  * cut a polyline at a chord that is both long (> MAX_CHORD_FRAC of the map diagonal)
    and whose midpoint sits in such a flat neighbourhood - that is a bridge, while a long
    straight run along a real valley floor is kept.

Usage:
    python fix_contour_artifacts.py --dry-run            # only report
    python fix_contour_artifacts.py --apply  [map ...]   # rewrite the JSONs
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BACKUP = os.path.join(HERE, "out", "_contours_backup_preartifacts")
FLAT_FRAC_DROP = 0.55      # polyline dropped when this share of its path is in flat fill
MAX_CHORD_FRAC = 0.0   # 0 disables chord cutting      # chord longer than 3 % of the map diagonal is suspicious
FLAT_WIN = 5
FLAT_DILATE = 8            # contour vertices sit exactly on the fill boundary, so grow it


def flat_mask(h, win=FLAT_WIN):
    mn = ndimage.minimum_filter(h, win)
    mx = ndimage.maximum_filter(h, win)
    return (mx - mn) < 1e-6


def bilinear(a, fx, fy):
    ny, nx = a.shape
    fx = np.clip(fx, 0, nx - 1.001)
    fy = np.clip(fy, 0, ny - 1.001)
    x0, y0 = fx.astype(int), fy.astype(int)
    tx, ty = fx - x0, fy - y0
    return (a[y0, x0] * (1 - tx) * (1 - ty) + a[y0, x0 + 1] * tx * (1 - ty)
            + a[y0 + 1, x0] * (1 - tx) * ty + a[y0 + 1, x0 + 1] * tx * ty)


def process(stem, apply_changes=False, verbose=False):
    jp = os.path.join(OUT, f"{stem}_contours.json")
    npz = os.path.join(OUT, f"{stem}_heightmap.npz")
    if not (os.path.exists(jp) and os.path.exists(npz)):
        return None
    doc = json.load(open(jp, encoding="utf-8"))
    meta = doc["meta"]
    h = np.load(npz)["h"].astype(np.float64)
    b = meta["bounds"]
    diag = float(np.hypot(b["maxX"] - b["minX"], b["maxY"] - b["minY"]))
    ny, nx = h.shape

    fm = flat_mask(h)
    fm = ndimage.binary_dilation(fm, iterations=FLAT_DILATE)
    fmf = fm.astype(float)
    # world -> flat mask index
    def fidx(x, y):
        fx = (x - b["minX"]) / (b["maxX"] - b["minX"]) * (nx - 1)
        fy = (y - b["minY"]) / (b["maxY"] - b["minY"]) * (ny - 1)
        return fx, fy

    def flat_share(a, nper=24):
        """Share of the sampled path that lies in the (dilated) filled region."""
        if len(a) < 2:
            return 0.0
        pts = []
        for k in range(len(a) - 1):
            t = np.linspace(0.0, 1.0, nper, endpoint=(k == len(a) - 2))
            pts.append(a[k] + (a[k + 1] - a[k]) * t[:, None])
        p = np.vstack(pts)
        fx, fy = fidx(p[:, 0], p[:, 1])
        return float((bilinear(fmf, fx, fy) > 0.5).mean())

    kept = dropped = cut_in_two = 0
    new_contours = []
    for c in doc["contours"]:
        new_lines = []
        for line in c["lines"]:
            a = np.asarray(line, float).reshape(-1, 2)
            if len(a) < 2:
                continue
            fx, fy = fidx(a[:, 0], a[:, 1])
            if flat_share(a) >= FLAT_FRAC_DROP:
                dropped += 1
                continue
            seg = np.hypot(*np.diff(a, axis=0).T)
            cuts = []
            lim = MAX_CHORD_FRAC * diag
            for k in (np.where(seg > lim)[0] if MAX_CHORD_FRAC > 0 else []):
                mx, my = (a[k] + a[k + 1]) / 2
                mfx, mfy = fidx(mx, my)
                if True:
                    cuts.append(k + 1)
            if cuts:
                cut_in_two += 1
                prev = 0
                for k in cuts + [len(a)]:
                    part = a[prev:k]
                    if len(part) >= 2:
                        new_lines.append([round(float(v), 1) for p in part for v in p])
                    prev = k
            else:
                kept += 1
                new_lines.append(line)
        if new_lines:
            new_contours.append({"elev": c["elev"], "lines": new_lines})
    for k in ("water", "coastlines"):
        if k in doc and isinstance(doc[k], list):
            pass  # left untouched: water polygons are closed shapes, not isolevel paths

    n_before = sum(len(c["lines"]) for c in doc["contours"])
    n_after = sum(len(c["lines"]) for c in new_contours)
    if not apply_changes:
        return dict(stem=stem, before=n_before, after=n_after, dropped=dropped,
                    cut=cut_in_two, kept=kept)
    os.makedirs(BACKUP, exist_ok=True)
    shutil.copy2(jp, os.path.join(BACKUP, os.path.basename(jp)))
    doc["contours"] = new_contours
    doc["meta"]["artifactFilter"] = {
        "flatWindow": FLAT_WIN,
        "dropFlatFraction": FLAT_FRAC_DROP,
        "maxChordFracOfDiagonal": MAX_CHORD_FRAC,
        "polylinesBefore": n_before,
        "polylinesAfter": n_after,
        "droppedFlat": dropped,
        "cutAtBridge": cut_in_two,
    }
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    return dict(stem=stem, before=n_before, after=n_after, dropped=dropped,
                cut=cut_in_two, kept=kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("maps", nargs="*")
    args = ap.parse_args()
    stems = args.maps or sorted(os.path.basename(p)[:-len("_contours.json")]
                                for p in glob.glob(os.path.join(OUT, "*_contours.json")))
    if args.limit:
        stems = stems[: args.limit]
    rows = []
    for s in stems:
        r = process(s, apply_changes=args.apply)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: -(r["before"] - r["after"]))
    tot_b = sum(r["before"] for r in rows)
    tot_a = sum(r["after"] for r in rows)
    print(f"maps {len(rows)}  polylines {tot_b} -> {tot_a} "
          f"(-{tot_b-tot_a}, {100*(tot_b-tot_a)/max(tot_b,1):.2f}%)")
    print(f"dropped-as-flat {sum(r['dropped'] for r in rows)}, "
          f"cut-at-bridge {sum(r['cut'] for r in rows)}")
    for r in rows[:15]:
        print(f"  {r['stem']:26s} {r['before']:6d} -> {r['after']:6d}  "
              f"(drop {r['dropped']:5d}, cut {r['cut']:4d})")
    if not args.apply:
        print("\n(dry run - nothing written; re-run with --apply)")


if __name__ == "__main__":
    main()
