"""Copy the airfield / runway / respawn records into the contour JSON.

The delivery contract always had an optional `airfields` key, but the records ended up in a
separate `dataset/airfields/<map>.json`. Consumers that only read
`dataset/contours/<map>_contours.json` therefore never saw them. This merges a compact copy
into the contour file so one file per map is enough; the standalone file stays as the
authoritative record (it also carries `missions`, `frame` and the out-of-frame lists).

Compact means: the per-entry `level` and `mission` strings are dropped (they are constant
per map and are already in the standalone file), which keeps the payload small.

    python merge_airfields_into_contours.py            # report only
    python merge_airfields_into_contours.py --apply
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset")
CONTOURS = os.path.join(DATASET, "contours")
AIRFIELDS = os.path.join(DATASET, "airfields")

AIRFIELD_KEYS = ("name", "kind", "unitClass", "center", "headingDeg", "reciprocalDeg",
                 "takeoffHeadingDeg", "spawnEnd", "ends", "lengthM", "widthM", "altM",
                 "source", "confidence")
RESPAWN_KEYS = ("kind", "name", "pos", "altM", "source")


def compact(rows, keys):
    out = []
    for r in rows:
        out.append({k: r[k] for k in keys if k in r})
    return out


def process(stem, apply_changes):
    cp = os.path.join(CONTOURS, f"{stem}_contours.json")
    if not os.path.exists(cp):
        return None
    doc = json.load(open(cp, encoding="utf-8"))
    ap = os.path.join(AIRFIELDS, f"{stem}.json")
    n_af = n_rs = 0
    if os.path.exists(ap):
        src = json.load(open(ap, encoding="utf-8"))
        af = compact(src.get("airfields") or [], AIRFIELD_KEYS)
        rs = compact(src.get("respawns") or [], RESPAWN_KEYS)
        out_af = compact(src.get("airfieldsOutsideBaseMap") or [], AIRFIELD_KEYS)
        out_rs = compact(src.get("respawnsOutsideBaseMap") or [], RESPAWN_KEYS)
        n_af, n_rs = len(af), len(rs)
    else:
        af = rs = out_af = out_rs = []
    doc["airfields"] = af
    doc["respawns"] = rs
    if out_af:
        doc["airfieldsOutsideBaseMap"] = out_af
    if out_rs:
        doc["respawnsOutsideBaseMap"] = out_rs
    doc["meta"]["airfields"] = {
        "source": f"dataset/airfields/{stem}.json" if os.path.exists(ap) else None,
        "count": n_af, "respawnCount": n_rs,
        "outsideBaseMap": len(out_af),
        "note": ("runway centrelines in replay-world-meters; headingDeg is the "
                 "ends[0]->ends[1] bearing, take-off runs takeoffHeadingDeg; altM is "
                 "absolute altitude, not height above ground"),
    }
    size_before = os.path.getsize(cp)
    if apply_changes:
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(doc, f, separators=(",", ":"), ensure_ascii=False)
        size_after = os.path.getsize(cp)
    else:
        size_after = len(json.dumps(doc, separators=(",", ":"), ensure_ascii=False))
    return dict(stem=stem, af=n_af, rs=n_rs, before=size_before, after=size_after)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("maps", nargs="*")
    args = ap.parse_args()
    stems = args.maps or sorted(os.path.basename(p)[:-len("_contours.json")]
                                for p in glob.glob(os.path.join(CONTOURS, "*_contours.json")))
    rows = []
    for s in stems:
        r = process(s, args.apply)
        if r:
            rows.append(r)
    tb = sum(r["before"] for r in rows)
    ta = sum(r["after"] for r in rows)
    n_af = sum(r["af"] for r in rows)
    n_rs = sum(r["rs"] for r in rows)
    print(f"maps {len(rows)}, runways embedded {n_af}, respawns {n_rs}")
    print(f"contours total {tb/1e6:.1f} MB -> {ta/1e6:.1f} MB (+{(ta-tb)/1e6:.1f} MB)")
    for r in sorted(rows, key=lambda r: -(r["after"] - r["before"]))[:6]:
        print(f"  {r['stem']:26s} {r['af']:3d} runways  +{(r['after']-r['before'])/1024:6.1f} KB")
    if not args.apply:
        print("\n(report only - re-run with --apply)")


if __name__ == "__main__":
    main()
