"""Drop bathymetric contour levels from the shipped JSONs.

Contours at or below the water level trace the sea floor. That data is clamped and flat
(`air_denmark` is 15 % exactly -50 m), so marching squares produces degenerate polylines
that wander across the sea and connect unrelated coastlines - exactly the "straight lines
jumping between places" a map renderer shows.

`make_contours.py` now skips those levels; this script applies the same rule to the files
that were generated before, without re-decoding the levels (dropping whole levels is
exactly what the generator would have done, so the result is not an approximation).

    python drop_subwater_contours.py --dry-run
    python drop_subwater_contours.py --apply
"""
import argparse
import glob
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BACKUP = os.path.join(HERE, "out", "_contours_backup_presubwater")


def water_level_of(meta):
    wl = meta.get("waterLevel")
    if wl is not None:
        return float(wl)
    lo, hi = meta.get("heightMin"), meta.get("heightMax")
    if lo is not None and hi is not None and lo < 0 < hi:
        return 0.0
    return None


def process(stem, apply_changes):
    jp = os.path.join(OUT, f"{stem}_contours.json")
    if not os.path.exists(jp):
        return None
    doc = json.load(open(jp, encoding="utf-8"))
    meta = doc["meta"]
    wl = water_level_of(meta)
    before_levels = len(doc["contours"])
    before_lines = sum(len(c["lines"]) for c in doc["contours"])
    before_pts = sum(len(l) // 2 for c in doc["contours"] for l in c["lines"])
    if wl is None:
        return dict(stem=stem, wl=None, lv=0, lines=0, pts=0, before=before_lines)
    kept = [c for c in doc["contours"] if c["elev"] > wl]
    dropped_levels = before_levels - len(kept)
    after_lines = sum(len(c["lines"]) for c in kept)
    after_pts = sum(len(l) // 2 for c in kept for l in c["lines"])
    if not dropped_levels:
        return dict(stem=stem, wl=wl, lv=0, lines=0, pts=0, before=before_lines)
    if apply_changes:
        os.makedirs(BACKUP, exist_ok=True)
        shutil.copy2(jp, os.path.join(BACKUP, os.path.basename(jp)))
        doc["contours"] = kept
        f = doc["meta"].setdefault("artifactFilter", {})
        f["droppedSubWaterLevels"] = dropped_levels
        f["waterLevelUsed"] = wl
        f["polylinesAfterSubWater"] = after_lines
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, separators=(",", ":"))
    return dict(stem=stem, wl=wl, lv=dropped_levels, lines=before_lines - after_lines,
                pts=before_pts - after_pts, before=before_lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("maps", nargs="*")
    args = ap.parse_args()
    stems = args.maps or sorted(os.path.basename(p)[:-len("_contours.json")]
                                for p in glob.glob(os.path.join(OUT, "*_contours.json")))
    rows = []
    for s in stems:
        r = process(s, args.apply)
        if r and r["lv"]:
            rows.append(r)
    rows.sort(key=lambda r: -r["lines"])
    tl = sum(r["lines"] for r in rows)
    tp = sum(r["pts"] for r in rows)
    tb = sum(r["before"] for r in rows)
    print(f"maps with sub-water levels {len(rows)}; dropped {sum(r['lv'] for r in rows)} levels, "
          f"{tl} polylines, {tp} points")
    for r in rows[:16]:
        print(f"  {r['stem']:26s} waterLevel {r['wl']:6.1f}  -{r['lv']} levels  "
              f"{r['lines']:5d} polylines removed")
    if not args.apply:
        print("\n(dry run - nothing written; re-run with --apply)")


if __name__ == "__main__":
    main()
