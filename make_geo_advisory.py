"""Write out/geo/ADVISORY.json: which georeferences are hints only, never coordinates.

Rule: anything matched only at the global z5 terrain level (`dem-coarse-only`) carries a
+-8 km uncertainty and has no independent label evidence, so it must not be used as a
coordinate. `air_grand_canyon` is called out separately: the level's terrain is not a
faithful copy of the real canyon (local NCC 0.55 vs a 0.69 false global peak).
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GEO = os.path.join(HERE, "out", "geo")

idx = json.load(open(os.path.join(GEO, "index.json"), encoding="utf-8"))
maps = idx["maps"]

hint_only = {}
for stem, v in sorted(maps.items()):
    conf = v.get("confidence")
    method = v.get("method")
    if conf == "unresolved" or not conf:
        continue
    if method == "dem-coarse-only":
        hint_only[stem] = {
            "confidence": conf,
            "method": method,
            "bboxUncertaintyKm": v.get("bboxUncertaintyKm"),
            "reason": "matched only at global z5 DEM (~4.9 km/px), no independent label evidence",
        }

EXTRA = {
    "air_grand_canyon": ("the level's terrain is not a faithful copy of the real Grand Canyon "
                         "(local NCC 0.55 vs 0.69 for a false global peak); bbox points at "
                         "east Greenland and is wrong"),
}
for stem, why in EXTRA.items():
    v = maps.get(stem, {})
    if stem in maps and v.get("confidence") not in (None, "unresolved"):
        hint_only[stem] = {"confidence": v.get("confidence"), "method": v.get("method"),
                           "bboxUncertaintyKm": v.get("bboxUncertaintyKm"), "reason": why}

doc = {
    "schema": "wt-geo-advisory/1",
    "note": ("These levels have a bbox in out/geo/<map>.json but it must NOT be used as a "
             "coordinate. Use it only as a hint for 'which real region is this'. Anything not "
             "listed here and with confidence high/medium can be used as a coordinate, "
             "respecting bboxUncertaintyKm."),
    "hintOnly": hint_only,
}
p = os.path.join(GEO, "ADVISORY.json")
with open(p, "w", encoding="utf-8") as f:
    json.dump(doc, f, ensure_ascii=False, indent=1, sort_keys=True)
print(f"wrote {p}: {len(hint_only)} hint-only georeferences")
for k, v in hint_only.items():
    print(f"  {k:24s} {v['method']:18s} {v['confidence']:7s} ±{v['bboxUncertaintyKm']} km")
