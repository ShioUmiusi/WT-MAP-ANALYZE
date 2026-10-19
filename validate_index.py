"""Validate every generated contour JSON and build out/index.json."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

rows = []
errors = []
for fn in sorted(os.listdir(OUT)):
    if not fn.endswith("_contours.json"):
        continue
    path = os.path.join(OUT, fn)
    try:
        doc = json.load(open(path, encoding="utf-8"))
        m = doc["meta"]
        b = m["bounds"]
        assert doc["contours"], "no contours"
        npts = sum(len(l) // 2 for c in doc["contours"] for l in c["lines"])
        for c in doc["contours"]:
            assert isinstance(c["elev"], (int, float))
            for l in c["lines"]:
                assert len(l) >= 4 and len(l) % 2 == 0
                for i in range(0, len(l), 2):
                    x, y = l[i], l[i + 1]
                    assert b["minX"] - 1 <= x <= b["maxX"] + 1, (fn, x)
                    assert b["minY"] - 1 <= y <= b["maxY"] + 1, (fn, y)
        rows.append({
            "map": m["map"],
            "source": m.get("source", "hm2"),
            "px": [m["widthPx"], m["heightPx"]],
            "cellM": m["cellSizeM"],
            "spanM": [round(b["maxX"] - b["minX"], 1), round(b["maxY"] - b["minY"], 1)],
            "bounds": b,
            "heightM": [m["heightMin"], m["heightMax"]],
            "intervalM": m["contourIntervalM"],
            "levels": len(doc["contours"]),
            "points": npts,
            "water": "water" in doc,
            "file": fn,
        })
    except Exception as e:
        errors.append((fn, repr(e)))

rows.sort(key=lambda r: r["map"])
n_hm2 = sum(1 for r in rows if r["source"] == "hm2")
n_lm = len(rows) - n_hm2
idx = {
    "schema": 1,
    "crs": "replay-world-meters",
    "note": "War Thunder level terrain extracted from levels/<map>.bin. "
            "source=hm2 means decoded from the HM2 heightmap block; "
            "source=landmesh means rebuilt from the lmap LandMesh triangle mesh "
            "(validated against HM2: median error 0.14 m over 90 cross-checked levels). "
            "x = world X (east), y = world north/Z; bounds in meters.",
    "count": len(rows),
    "bySource": {"hm2": n_hm2, "landmesh": n_lm},
    "maps": rows,
}
with open(os.path.join(OUT, "index.json"), "w", encoding="utf-8") as f:
    json.dump(idx, f, ensure_ascii=False, indent=1)

print(f"validated {len(rows)} contour files, {len(errors)} errors")
print(f"  by source: hm2={n_hm2} landmesh={n_lm}")
for e in errors[:20]:
    print("  ERR", e)
big = sorted(rows, key=lambda r: -r["points"])[:5]
print("largest:", [(r["map"], r["points"]) for r in big])
print("index ->", os.path.join(OUT, "index.json"))
