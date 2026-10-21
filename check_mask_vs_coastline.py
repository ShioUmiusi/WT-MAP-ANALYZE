"""Cross-check the mask against the coastline polylines we already ship.

`<map>_contours.json` has `coastlines` in world metres, produced from the same heightmap
and already validated against the game's own map. If the stored mask is in the orientation
it claims, every coastline vertex must sit on a land/water transition of the mask. A
vertically flipped mask puts them in the wrong half and the hit rate collapses.

This is colour-independent, so it also works for maps whose base map has no blue water.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


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


def transition(m):
    """Pixels that touch the opposite class (4-neighbour)."""
    t = np.zeros_like(m)
    t[1:, :] |= m[1:, :] != m[:-1, :]
    t[:-1, :] |= m[:-1, :] != m[1:, :]
    t[:, 1:] |= m[:, 1:] != m[:, :-1]
    t[:, :-1] |= m[:, :-1] != m[:, 1:]
    return t


def score(stem, flip=False):
    p = os.path.join(OUT, "landmask", f"{stem}.json")
    cj = os.path.join(OUT, f"{stem}_contours.json")
    if not (os.path.exists(p) and os.path.exists(cj)):
        return None
    doc = json.load(open(p, encoding="utf-8"))
    m = decode(doc)
    if flip:
        m = m[::-1, :]
    hh, ww = m.shape
    b = doc["bounds"]
    cellx = (b["maxX"] - b["minX"]) / ww
    celly = (b["maxY"] - b["minY"]) / hh
    t = transition(m)
    coast = json.load(open(cj, encoding="utf-8")).get("coastlines") or []
    if not coast:
        return None
    cols, rows = [], []
    npts = 0
    for line in coast:
        a = np.asarray(line, float).reshape(-1, 2)
        c = (a[:, 0] - b["minX"]) / cellx - 0.5
        r = (b["maxY"] - a[:, 1]) / celly - 0.5
        ok = (c >= 0) & (c < ww - 1) & (r >= 0) & (r < hh - 1)
        if ok.any():
            cols.append(c[ok])
            rows.append(r[ok])
            npts += int(ok.sum())
    if not npts:
        return None
    c = np.concatenate(cols)
    r = np.concatenate(rows)
    ci = np.round(c).astype(int)
    ri = np.round(r).astype(int)
    hit1 = float(t[ri, ci].mean())
    # allow one pixel of slack
    hit2 = np.zeros(len(ci), bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rr = np.clip(ri + dr, 0, hh - 1)
            cc = np.clip(ci + dc, 0, ww - 1)
            hit2 |= t[rr, cc]
    return dict(stem=stem, n=npts, hit1=hit1, hit2=float(hit2.mean()))


def main():
    stems = sys.argv[1:] or sorted(
        os.path.basename(p)[:-5] for p in os.listdir(os.path.join(OUT, "landmask"))
        if p.endswith(".json") and p != "index.json")
    rows = []
    for s in stems:
        a = score(s)
        if a is None:
            continue
        bfl = score(s, flip=True)
        rows.append((a, bfl))
    rows.sort(key=lambda r: r[0]["hit1"])
    print(f"{'map':26s} {'coastPts':>9s} {'hit':>7s} {'hit±1px':>8s} {'ifFlipped':>10s}")
    for a, bfl in rows:
        flag = "" if bfl is None or a["hit2"] >= bfl["hit2"] else "   <-- flipped fits better"
        print(f"{a['stem']:26s} {a['n']:9d} {100*a['hit1']:6.1f}% {100*a['hit2']:7.1f}% "
              f"{(100*bfl['hit2']) if bfl else float('nan'):9.1f}%{flag}")
    if rows:
        h = np.array([r[0]["hit2"] for r in rows])
        print(f"\nmedian coastline hit (+-1 px): {100*np.median(h):.1f}%  over {len(rows)} maps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
