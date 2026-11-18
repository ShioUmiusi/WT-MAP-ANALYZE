"""Regenerate terrain for levels whose HM2 only covers the battle tile.

Ground-battle levels (`avg_*` / `avn_*` / `hvg_*`) ship an HM2 heightmap that covers
just the ~2 km playable tile, while the level itself (and every airfield / air-spawn in
the mission data) spans the whole `lmap` LandMesh frame. Reading the HM2 alone therefore
produced contours that sit inside the battlefield, so a consumer drawing a map frame from
them put the frame in the battle area and left the airfields outside it.

This script finds those levels from the level headers (no decoding needed), rebuilds
their contours from the LandMesh (`make_contours.py --source landmesh`), and then rebuilds
the land mask from the same heights so the two layers stay consistent.

    python regen_wide_levels.py                 # report only
    python regen_wide_levels.py --apply         # contours + masks
    python regen_wide_levels.py --apply --maps avg_finland avg_berlin
"""
import argparse
import json
import os
import struct
import subprocess
import sys
import time

from wt_hmap import read_level_blocks
from wt_paths import levels_dir

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
WIDE_RATIO = 1.5          # lmap span / hm2 span above which the HM2 is only a sub-tile


def spans(path):
    """(landmesh_span_m, hm2_span_m) from the level header, no decoding."""
    d, blocks = read_level_blocks(path)
    if b"lmap" not in blocks:
        return None, None
    off, _sz = blocks[b"lmap"]
    _magic, _ver, _grid, land, sx, sy, _ox, _oy = struct.unpack_from("<4sIffiiii", d, off)
    land_span = max(sx, sy) * land if land > 0 else None
    hm2_span = None
    if b"\x00HM2" in blocks:
        ho, _hs = blocks[b"\x00HM2"]
        cell = struct.unpack_from("<f", d, ho)[0]
        w, hm = struct.unpack_from("<2I", d, ho + 20)
        hm2_span = max(w & 0xFFFFFF, hm & 0x7FFFFFFF) * cell
    return land_span, hm2_span


def wide_levels(names=None):
    rows = []
    L = levels_dir()
    for fn in sorted(os.listdir(L)):
        if not fn.endswith(".bin"):
            continue
        stem = fn[:-4]
        if names and stem not in names:
            continue
        land_span, hm2_span = spans(os.path.join(L, fn))
        if not land_span or not hm2_span or land_span <= hm2_span * WIDE_RATIO:
            continue
        rows.append({"map": stem, "landSpanM": land_span, "hm2SpanM": hm2_span})
    return rows


def contours_bounds(stem):
    p = os.path.join(OUT, stem + "_contours.json")
    if not os.path.exists(p):
        return None
    meta = json.load(open(p, encoding="utf-8"))["meta"]
    b = meta["bounds"]
    return meta, (b["maxX"] - b["minX"]), (b["maxY"] - b["minY"])


def run(script, args):
    r = subprocess.run([sys.executable, os.path.join(HERE, script)] + args,
                       cwd=HERE, capture_output=True, text=True)
    return r.returncode, (r.stdout or "").strip().splitlines(), (r.stderr or "").strip().splitlines()


def check(dataset_dir):
    """Assert the shipped artifacts put the whole level inside the frame.

    Run after out/ has been synced into dataset/. Fails (exit 1) when a wide level still
    carries battle-tile contours, when meta.levelBounds disagrees with them, when a runway
    or air-spawn sits outside the frame, or when the land mask covers a different rectangle
    than the contours.
    """
    cdir = os.path.join(dataset_dir, "contours")
    mdir = os.path.join(dataset_dir, "landmask")
    bad = 0
    n = 0
    for r in wide_levels():
        stem = r["map"]
        n += 1
        doc = json.load(open(os.path.join(cdir, stem + "_contours.json"), encoding="utf-8"))
        meta = doc["meta"]
        b = meta["bounds"]
        span = max(b["maxX"] - b["minX"], b["maxY"] - b["minY"])
        lb = meta.get("levelBounds")
        if span <= r["hm2SpanM"] * WIDE_RATIO:
            print("  %-28s FAIL contours span %.0f m (hm2 tile %.0f m)" % (stem, span, r["hm2SpanM"]))
            bad += 1
            continue
        if not lb:
            print("  %-28s FAIL no meta.levelBounds" % stem)
            bad += 1
            continue
        if lb["minX"] < b["minX"] - 1 or lb["maxX"] > b["maxX"] + 1 \
                or lb["minY"] < b["minY"] - 1 or lb["maxY"] > b["maxY"] + 1:
            print("  %-28s FAIL levelBounds outside the raster" % stem)
            bad += 1
            continue
        for key in ("airfields", "airfieldsOutsideBaseMap"):
            for a in doc.get(key) or []:
                for pt in (a.get("center"),) + tuple(a.get("ends") or ()):
                    if not pt:
                        continue
                    if not (lb["minX"] <= pt[0] <= lb["maxX"] and lb["minY"] <= pt[1] <= lb["maxY"]):
                        print("  %-28s FAIL runway %s at %s outside levelBounds" % (stem, a.get("name"), pt))
                        bad += 1
                        raise SystemExit(1)
        mask_path = os.path.join(mdir, stem + ".json")
        if os.path.exists(mask_path):
            mb = json.load(open(mask_path, encoding="utf-8"))["bounds"]
            if abs(mb["maxX"] - b["maxX"]) > 1 or abs(mb["maxY"] - b["maxY"]) > 1:
                print("  %-28s FAIL mask bounds %s != contours bounds %s" % (stem, mb, b))
                bad += 1
    print("check: %d wide level(s), %d problem(s)" % (n, bad))
    return 0 if bad == 0 else 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="regenerate even when the contours already span the level")
    ap.add_argument("--maps", nargs="*", default=None)
    ap.add_argument("--check", action="store_true",
                    help="verify dataset/ (run after syncing out/); asserts frame == level")
    args = ap.parse_args()

    if args.check:
        return check(os.path.join(HERE, "dataset"))
    rows = wide_levels(args.maps)
    print("%d level(s) whose lmap is wider than the HM2 tile (ratio > %.1f)" % (len(rows), WIDE_RATIO))
    todo = []
    for r in rows:
        got = contours_bounds(r["map"])
        if got and not args.force and got[1] > r["hm2SpanM"] * WIDE_RATIO:
            print("  %-28s contours already span %.0f m, skipped" % (r["map"], got[1]))
            continue
        todo.append(r)
        print("  %-28s lmap %.0f m, hm2 %.0f m" % (r["map"], r["landSpanM"], r["hm2SpanM"]))
    if not args.apply:
        print("\n(report only - re-run with --apply)")
        return 0

    ok = fail = 0
    t_all = time.time()
    for i, r in enumerate(todo, 1):
        stem = r["map"]
        t0 = time.time()
        rc, out, err = run("make_contours.py", [stem, "--source", "landmesh"])
        if rc != 0:
            fail += 1
            print("[%2d/%2d] %-28s FAIL contours rc=%d %s" % (i, len(todo), stem, rc, (err or [""])[-1][:90]), flush=True)
            continue
        rc2, _out2, err2 = run("make_landmask_json.py", [stem])
        got = contours_bounds(stem)
        print("[%2d/%2d] %-28s %5.1fs span %.0f m x %.0f m, mask %s"
              % (i, len(todo), stem, time.time() - t0,
                 got[1] if got else -1, got[2] if got else -1,
                 "ok" if rc2 == 0 else "FAIL " + (err2 or [""])[-1][:60]), flush=True)
        if rc2 == 0:
            ok += 1
        else:
            fail += 1
    print("\ndone: ok=%d fail=%d in %.1f min" % (ok, fail, (time.time() - t_all) / 60))
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
