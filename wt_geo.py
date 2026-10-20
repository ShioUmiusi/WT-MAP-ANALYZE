"""wt_geo.py -- real-world geographic reference (WGS84) for War Thunder levels.

What this computes
------------------
For every level in the game install (``<WT_GAME_DIR>/levels/<name>.bin``) we answer: which
piece of the real Earth does its playable world cover?  The output is a
latitude/longitude bounding box plus its four corners.

Method (all offline except the SRTM tile download)
--------------------------------------------------
1. **OCR the map texture.**  Each level's minimap texture
   (``<level>_map`` in the Dagor ``.dxp.bin`` texture packs, see
   ``wt_textures.py``) is a real satellite/terrain mosaic with place-name
   labels printed on it (``KABUL``, ``Charikar``, ...).  RapidOCR reads the
   labels and their pixel positions.
2. **Gazetteer lookup.**  Labels are matched against an offline GeoNames
   ``allCountries`` extract (feature classes P/A/L/S/T/H/U/V) with
   ``rapidfuzz``.  A name can be many places, so every label keeps a list of
   candidates.
3. **RANSAC affine.**  Candidate combinations are fitted with a north-up
   affine ``lon = a*px + b``, ``lat = c*py + d``; the winning combination is
   the one with the most inliers and the smallest residual.  This only gives a
   *rough* location (labels are drawn beside a town, not on it), good to a
   few km -- enough to say "somewhere near Kabul".
4. **DEM alignment.**  The level's own heightmap
   (``out/<map>_heightmap.npz``, ``replay-world-meters``) is correlated with
   SRTM 1-arcsecond elevation (AWS *Terrain Tiles* ``skadi`` bucket).  We fit
   ``lon0``, ``lat0`` (where world origin (0,0) sits on Earth) plus ``k``, the
   real-metre length of one world unit, by maximising the normalised
   cross-correlation of the two elevation fields.  This is the authoritative
   step: it is independent of the labels and typically reaches NCC 0.9+.
5. **Cross-checks.**  NCC peak, label residual at the fitted placement, and
   bbox consistency are all recorded in the per-map JSON.

Usage
-----
    python wt_geo.py ocr                 # OCR every map texture -> out/_geo_cache/ocr_hq.json
    python wt_geo.py gazetteer           # build the candidate gazetteer -> out/_geo_cache/gaz_cand.json
    python wt_geo.py solve --map air_afghan
    python wt_geo.py all                 # solve everything -> out/geo/<map>.json
    python wt_geo.py report              # out/geo/index.json + console stats

Only ``out/geo/`` and ``out/_geo_cache/`` are written; the game directory is read
only.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time
import unicodedata
import urllib.request
import zipfile

import numpy as np

from wt_paths import game_dir

WS = os.path.dirname(os.path.abspath(__file__))
SCRATCH = os.path.join(WS, "out", "_geo_cache")
OUTGEO = os.path.join(WS, "out", "geo")
SRTM_CACHE = os.path.join(SCRATCH, "srtm")
GEOZIP = os.path.join(SCRATCH, "allCountries.zip")
GEONAMES_URL = "https://download.geonames.org/export/dump/allCountries.zip"
SRTM_URL = "https://elevation-tiles-prod.s3.amazonaws.com/skadi/{hem}/{tile}.hgt.gz"

# World units: every level's playable square is 131072 x 131072 world units,
# centred on (0,0) and stored as heightmap row 0 = south, column 0 = west.
WORLD_HALF = 65536.0
M_PER_DEG_LAT = 110540.0
M_PER_DEG_LON_EQ = 111320.0

# sxriptable PDF: (lon = lon0 + X * k / (111320 cos(lat)), lat = lat0 + Y * k / 110540)
# -> one world unit has a real length of k metres (fit per map)

def texture_packs():
    """Dagor texture packs that hold map images: (label, path) in load order."""
    game = game_dir()
    return [
        ("base", os.path.join(game, "content", "base", "res", "locations_maps.dxp.bin")),
        ("hq", os.path.join(game, "content.hq", "hq_tex", "res",
                            "hq_tex_locations_maps.dxp.bin")),
        ("hq2", os.path.join(game, "content", "base", "res", "locations_maps-hq.dxp.bin")),
        ("patch", os.path.join(game, "patch", "content", "base", "res",
                               "locations_maps.dxp.bin")),
    ]

KEEP_FEATURE_CLASSES = set("PALSTHUV")


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def log(*a):
    print(*a, flush=True)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, **kw)


def level_list():
    import glob
    paths = glob.glob(os.path.join(game_dir(), "levels", "*.bin"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def norm_text(s):
    """Fold to a comparable ASCII-ish key: strip diacritics, lowercase, alnum only."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c for c in s.lower() if c.isalnum())


def label_variants(text):
    """Whole normalized string plus individual words worth matching."""
    n = norm_text(text)
    out = {n} if len(n) >= 3 else set()
    for tok in n.split():
        if len(tok) >= 4:
            out.add(tok)
    # norm_text drops spaces, so recover word tokens from the raw string too
    for tok in unicodedata.normalize("NFKD", text).split():
        t = norm_text(tok)
        if len(t) >= 4:
            out.add(t)
    return out


# ---------------------------------------------------------------------------
# texture packs
# ---------------------------------------------------------------------------
class TextureBank:
    """Lazy access to the Dagor texture packs that hold the map images."""

    def __init__(self):
        self._packs = {}

    def pack(self, name):
        if name not in self._packs:
            from wt_textures import TexturePack
            path = dict(texture_packs())[name]
            self._packs[name] = TexturePack(path)
        return self._packs[name]

    def best(self, level):
        """Highest-resolution texture for a level -> dict or None."""
        best = None
        for name, _ in texture_packs():
            pk = self.pack(name)
            for suffix in ("_map", "_map$hq"):
                i = pk.index_of(level + suffix)
                if i < 0:
                    continue
                h = pk.headers[i]
                if h.w < 512:          # skip 256px thumbnails
                    continue
                if best is None or h.w * h.h > best["w"] * best["h"]:
                    best = {"pack": name, "index": i, "w": h.w, "h": h.h}
        return best

    def image(self, level):
        b = self.best(level)
        if b is None:
            return None, None
        return self.pack(b["pack"]).to_pil(b["index"]).convert("RGB"), b


# ---------------------------------------------------------------------------
# step 1: OCR
# ---------------------------------------------------------------------------
def ocr_texture(img):
    """Run RapidOCR on a PIL image -> list of label dicts."""
    from rapidocr_onnxruntime import RapidOCR
    global _OCR
    try:
        _OCR
    except NameError:
        _OCR = RapidOCR()
    r, _ = _OCR(np.asarray(img))
    out = []
    if not r:
        return out
    for box, txt, score in r:
        box = np.asarray(box, float)
        cx, cy = box.mean(0)
        out.append({
            "text": txt,
            "score": round(float(score), 4),
            "px": round(float(cx), 2),
            "py": round(float(cy), 2),
            "box": [[round(float(a), 2), round(float(b), 2)] for a, b in box],
            "w": img.width,
            "h": img.height,
        })
    return out


def cmd_ocr(force=False):
    """OCR every level texture at its best resolution."""
    bank = TextureBank()
    path = os.path.join(SCRATCH, "ocr_hq.json")
    done = {} if force else (load_json(path, {}) or {})
    levels = level_list()
    t0 = time.time()
    for n, lv in enumerate(levels):
        if lv in done:
            continue
        img, meta = bank.image(lv)
        if img is None:
            continue
        labs = ocr_texture(img)
        done[lv] = {"w": img.width, "h": img.height, "pack": meta["pack"], "labels": labs}
        log(f"[{n+1}/{len(levels)}] {lv} {img.width}px: {len(labs)} labels "
            f"({time.time()-t0:.0f}s)")
        if n % 5 == 0:
            save_json(path, done)
    save_json(path, done)
    log("wrote", path, len(done), "levels")


def collect_labels():
    """Merge the 1024px and 2048px OCR runs into one label list per level.

    Labels found by both runs are de-duplicated by normalized text; the hq run
    positions (same normalized coordinates, larger image) are preferred.
    """
    base = load_json(os.path.join(SCRATCH, "ocr_raw.json"), {}) or {}
    hq = load_json(os.path.join(SCRATCH, "ocr_hq.json"), {}) or {}
    out = {}
    for lv in sorted(set(base) | set(hq)):
        cand = hq.get(lv) or base.get(lv)
        if not cand:
            continue
        labs = list(cand.get("labels") or [])
        best = cand
        for src in (base.get(lv), hq.get(lv)):
            if src and len(src.get("labels") or []) > len(labs):
                labs = list(src["labels"])
                best = src
        out[lv] = {"w": best["w"], "h": best["h"], "labels": labs}
    return out


# ---------------------------------------------------------------------------
# step 2: gazetteer
# ---------------------------------------------------------------------------
GAZ_NPZ = os.path.join(SCRATCH, "gaz.npz")
GAZ_PREFIX = 4          # prefix length used to shard the name index
GAZ_STRIDE = 24


def cmd_gazetteer(force=False):
    """Build the filtered GeoNames extract for exactly the labels we have.

    Keeps every populated-place / terrain / hydro feature that could plausibly
    match one of our OCR labels, either by sharing the first 4 characters or by
    sharing the first 2 characters *and* an almost equal length (which is what
    rescues names the OCR mangled in the middle).  Stored as a compact npz
    (name chars, lengths, lat, lon, feature class, country, prefix index).
    """
    labels = collect_labels()
    variants = set()
    for lv, d in labels.items():
        for l in d["labels"]:
            variants |= label_variants(l["text"])
    var4 = {v[:4] for v in variants if len(v) >= 4}
    var2 = {}
    for v in variants:
        if len(v) >= 3:
            var2.setdefault(v[:2], set()).add(len(v))
    log("label variants", len(variants), "prefix4", len(var4), "prefix2", len(var2))

    if not os.path.exists(GEOZIP):
        log("downloading", GEONAMES_URL)
        urllib.request.urlretrieve(GEONAMES_URL, GEOZIP)

    names = []
    lats = []
    lons = []
    cls = []
    ccs = []
    t0 = time.time()
    with zipfile.ZipFile(GEOZIP) as z:
        with z.open("allCountries.txt") as f:
            for k, line in enumerate(io.TextIOWrapper(f, encoding="utf-8")):
                c = line.rstrip("\n").split("\t")
                if len(c) < 9 or c[6] not in KEEP_FEATURE_CLASSES:
                    continue
                lat, lon = float(c[4]), float(c[5])
                cc = c[8]
                for raw in (c[1], c[2]):
                    n = norm_text(raw)
                    ln = len(n)
                    if ln < 3:
                        continue
                    keep = n[:4] in var4
                    if not keep:
                        lens = var2.get(n[:2])
                        if lens and any(abs(ln - L) <= 2 for L in lens):
                            keep = True
                    if not keep:
                        continue
                    names.append(n[:GAZ_STRIDE])
                    lats.append(lat)
                    lons.append(lon)
                    cls.append(c[6])
                    ccs.append(cc)
                if k % 4000000 == 0:
                    log(k, len(names), f"{time.time()-t0:.0f}s")

    order = sorted(range(len(names)), key=lambda i: names[i])
    names = [names[i] for i in order]
    lats = np.array([lats[i] for i in order], np.float32)
    lons = np.array([lons[i] for i in order], np.float32)
    cls = [cls[i] for i in order]
    ccs = [ccs[i] for i in order]
    buflen = max(len(n) for n in names)
    buf = np.zeros((len(names), buflen), np.uint8)
    lens = np.zeros(len(names), np.uint8)
    for i, s in enumerate(names):
        b = s.encode("ascii", "ignore")
        buf[i, :len(b)] = np.frombuffer(b, np.uint8)
        lens[i] = len(b)
    prefixes_out = []
    starts = []
    p = None
    for i, n in enumerate(names):
        pfx = n[:GAZ_PREFIX]
        if pfx != p:
            prefixes_out.append(pfx)
            starts.append(i)
            p = pfx
    starts.append(len(names))
    np.savez_compressed(
        GAZ_NPZ, name=buf, namelen=lens, lat=lats, lon=lons,
        cls=np.array([ord(x) for x in cls], np.uint8),
        cc=np.array([ord(x[0]) * 256 + ord(x[1]) if len(x) > 1 else 0 for x in ccs],
                    np.uint16),
        prefixes=np.array(prefixes_out, dtype=f"<U{GAZ_PREFIX}"),
        prefix_start=np.array(starts, np.int64))
    log("wrote gaz.npz rows=", len(names), "prefixes", len(prefixes_out))


class Gazetteer:
    """Offline place-name -> lat/lon candidate lookup with fuzzy matching."""

    def __init__(self, path=None):
        path = path or GAZ_NPZ
        if not os.path.exists(path):
            raise SystemExit(f"missing {path}; run `python wt_geo.py gazetteer` first")
        d = np.load(path)
        self.buf = d["name"]
        self.lens = d["namelen"]
        self.lat = d["lat"]
        self.lon = d["lon"]
        self.cls = d["cls"]
        self.cc = d["cc"]
        self.prefixes = [str(x) for x in d["prefixes"]]
        self.pstart = d["prefix_start"]
        self.pindex = {p: i for i, p in enumerate(self.prefixes)}
        self.prefix_groups = {}
        for p in self.prefixes:
            for plen in (3, 2):
                if len(p) >= plen:
                    self.prefix_groups.setdefault(p[:plen], []).append(p)
        self._names = {}

    def _bucket_names(self, pfx):
        i = self.pindex.get(pfx)
        if i is None:
            return None, None
        if pfx not in self._names:
            a, b = int(self.pstart[i]), int(self.pstart[i + 1])
            buf = self.buf[a:b]
            ln = self.lens[a:b]
            self._names[pfx] = [bytes(row[:l]).decode("ascii") for row, l in zip(buf, ln)]
            if len(self._names) > 400:
                self._names.pop(next(iter(self._names)))
        a = int(self.pstart[i])
        return self._names[pfx], a

    def candidates(self, text, topn=8, min_score=70.0):
        """Fuzzy-match a label; falls back to shorter prefixes when the 4-char
        shard has no hit (OCR often mangles a character or two)."""
        from rapidfuzz import fuzz, process
        seen = {}

        def scan(v, prefixes):
            for pfx in prefixes:
                rows, base = self._bucket_names(pfx)
                if not rows:
                    continue
                hits = process.extract(v, rows, scorer=fuzz.ratio,
                                       limit=topn * 3,
                                       score_cutoff=min(min_score, 92.0))
                for nm, score, idx in hits:
                    j = base + idx
                    key = (round(float(self.lat[j]), 3), round(float(self.lon[j]), 3))
                    s = float(score)
                    if len(v) < 5:
                        s -= 8.0
                    if key not in seen or seen[key][0] < s:
                        cc = int(self.cc[j])
                        seen[key] = (s, nm, float(self.lat[j]), float(self.lon[j]),
                                     chr(int(self.cls[j])),
                                     (chr(cc >> 8) + chr(cc & 0xFF)) if cc else "", len(v))

        for v in label_variants(text):
            if len(v) < GAZ_PREFIX:
                continue
            rows, _ = self._bucket_names(v[:GAZ_PREFIX])
            if rows:
                scan(v, [v[:GAZ_PREFIX]])
            else:
                for plen in (3, 2):
                    if len(v) < plen:
                        continue
                    cand = self.prefix_groups.get(v[:plen])
                    if cand:
                        scan(v, cand)
        out = sorted(seen.values(), key=lambda t: -t[0])[:topn]
        return [{"score": s, "name": nm, "lat": la, "lon": lo,
                 "cls": cl, "fcode": "", "cc": cc, "matched": raw}
                for s, nm, la, lo, cl, cc, raw in out]


# ---------------------------------------------------------------------------
# step 3: affine fit / RANSAC
# ---------------------------------------------------------------------------
def fit_affine(p0):
    """p0: [(px,py,lat,lon)] -> (a,b,c,d, rms_km, resid_km[])."""
    p = np.asarray(p0, float)
    px, py, lat, lon = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    if len(p) < 2 or np.ptp(px) < 1 or np.ptp(py) < 1:
        return None
    al, bl = np.polyfit(px, lon, 1)
    cc, dd = np.polyfit(py, lat, 1)
    rl = (al * px + bl - lon) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat))
    rt = (cc * py + dd - lat) * M_PER_DEG_LAT
    resid = np.hypot(rl, rt) / 1000.0
    return al, bl, cc, dd, float(np.sqrt((resid ** 2).mean())), resid


def affine_bbox(abcd, w, h):
    a, b, c, d = abcd[:4]
    lons = [a * 0 + b, a * w + b]
    lats = [c * 0 + d, c * h + d]
    return {"minLon": min(lons), "maxLon": max(lons),
            "minLat": min(lats), "maxLat": max(lats)}


def ransac_affine(label_cands, w, h, iters=4000, rng=None):
    """Fit pixel->lon/lat over candidate place assignments.

    label_cands: list of (label dict, [candidate dicts]) for labels on one map.
    Returns dict with affine, inliers, rms or None.
    """
    rng = rng or np.random.default_rng(0)
    usable = [(i, l, c) for i, (l, c) in enumerate(label_cands) if c]
    if len(usable) < 2:
        return None
    best = None
    max_size = 400e3   # a WT map is < 400 km across; anything bigger is nonsense
    for _ in range(iters):
        i1, i2 = rng.choice(len(usable), 2, replace=False)
        l1, c1s = usable[i1][1], usable[i1][2]
        l2, c2s = usable[i2][1], usable[i2][2]
        if abs(l1["px"] - l2["px"]) < 20 or abs(l1["py"] - l2["py"]) < 20:
            continue
        cand1 = c1s[rng.integers(0, min(4, len(c1s)))]
        cand2 = c2s[rng.integers(0, min(4, len(c2s)))]
        abcd = fit_affine([(l1["px"], l1["py"], cand1["lat"], cand1["lon"]),
                           (l2["px"], l2["py"], cand2["lat"], cand2["lon"])])
        if abcd is None:
            continue
        a, b, c, d = abcd[:4]
        ew = abs(a) * w * M_PER_DEG_LON_EQ * math.cos(math.radians(c * h / 2 + d))
        ns = abs(c) * h * M_PER_DEG_LAT
        if not (30e3 < ew < max_size and 30e3 < ns < max_size):
            continue
        if not (0.45 < ew / ns < 2.2):
            continue
        # count inliers
        inl = []
        for j, (lj, cjs) in enumerate(label_cands):
            if not cjs:
                continue
            for cj in cjs[:5]:
                plon = a * lj["px"] + b
                plat = c * lj["py"] + d
                dkm = math.hypot((plon - cj["lon"]) * M_PER_DEG_LON_EQ *
                                 math.cos(math.radians(cj["lat"])),
                                 (plat - cj["lat"]) * M_PER_DEG_LAT) / 1000.0
                if dkm < 6.0:
                    inl.append((j, cj, dkm))
                    break
        score = (len(inl), -sum(x[2] for x in inl))
        if best is None or score > best[0]:
            best = (score, abcd, inl)
    if not best:
        return None
    _, abcd, inl = best
    # refit on inliers
    pts = [(label_cands[j][0]["px"], label_cands[j][0]["py"], cj["lat"], cj["lon"])
           for j, cj, _ in inl]
    if len(pts) >= 2:
        abcd = fit_affine(pts)
    return {"affine": abcd, "inliers": inl, "n_inliers": len(inl)}


# ---------------------------------------------------------------------------
# step 4: SRTM elevation alignment
# ---------------------------------------------------------------------------
_SRTM = {}
SRTM_TILE_BUDGET = 20          # hard cap of 1-degree tiles downloaded per level
_srtm_used = {"n": 0, "tiles": set()}


def srtm_reset_budget():
    _srtm_used["n"] = 0
    _srtm_used["tiles"] = set()


def srtm_tile(lat_deg, lon_deg):
    """1x1 degree SRTM 1" tile as float32 (N,N), row 0 = north edge."""
    key = (int(math.floor(lat_deg)), int(math.floor(lon_deg)))
    if key in _SRTM:
        return _SRTM[key]
    os.makedirs(SRTM_CACHE, exist_ok=True)
    la, lo = key
    ns = "N" if la >= 0 else "S"
    ew = "E" if lo >= 0 else "W"
    name = f"{ns}{abs(la):02d}{ew}{abs(lo):03d}"
    path = os.path.join(SRTM_CACHE, name + ".hgt")
    if not os.path.exists(path):
        if key not in _srtm_used["tiles"]:
            if len(_srtm_used["tiles"]) >= SRTM_TILE_BUDGET:
                raise RuntimeError(
                    f"SRTM download budget exhausted ({SRTM_TILE_BUDGET} tiles) "
                    f"for this level")
            _srtm_used["tiles"].add(key)
        url = SRTM_URL.format(hem=f"{ns}{abs(la):02d}", tile=name)
        req = urllib.request.Request(url, headers={"User-Agent": "wt-geo/1.0"})
        raw = urllib.request.urlopen(req, timeout=180).read()
        data = raw if raw[:2] != b"\x1f\x8b" else __import__("gzip").decompress(raw)
        if len(data) != 3601 * 3601 * 2:
            raise ValueError(f"{name}: unexpected hgt size {len(data)}")
        with open(path, "wb") as f:
            f.write(data)
    a = np.fromfile(path, dtype=">i2").reshape(3601, 3601).astype(np.float32)
    a[a < -1000] = np.nan
    if len(_SRTM) > 40:
        _SRTM.pop(next(iter(_SRTM)))
    _SRTM[key] = a
    return a


def srtm_sample(lon, lat):
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    out = np.full(lon.shape, np.nan, np.float32)
    la0 = np.floor(lat).astype(int)
    lo0 = np.floor(lon).astype(int)
    for kla in np.unique(la0):
        for klo in np.unique(lo0[la0 == kla]):
            m = (la0 == kla) & (lo0 == klo)
            t = srtm_tile(kla, klo)
            y = np.clip((kla + 1 - lat[m]) * 3600, 0, 3599.999)
            x = np.clip((lon[m] - klo) * 3600, 0, 3599.999)
            y0 = y.astype(int)
            x0 = x.astype(int)
            fy = (y - y0).astype(np.float32)
            fx = (x - x0).astype(np.float32)
            out[m] = (t[y0, x0] * (1 - fx) * (1 - fy) + t[y0, x0 + 1] * fx * (1 - fy)
                      + t[y0 + 1, x0] * (1 - fx) * fy + t[y0 + 1, x0 + 1] * fx * fy)
    return out


class LocalDem:
    """SRTM sampled once onto a local equirectangular grid for fast lookup."""

    def __init__(self, lon0, lat0, lon1, lat1, res_m=30.0):
        from scipy.ndimage import map_coordinates  # noqa: F401  (import check)
        self.lon0, self.lat0 = lon0, lat0
        self.dlat = res_m / M_PER_DEG_LAT
        self.dlon = res_m / (M_PER_DEG_LON_EQ * math.cos(math.radians((lat0 + lat1) / 2)))
        self.na = int(math.ceil((lat1 - lat0) / self.dlat)) + 2
        self.nl = int(math.ceil((lon1 - lon0) / self.dlon)) + 2
        LA, LO = np.meshgrid(lat0 + np.arange(self.na) * self.dlat,
                             lon0 + np.arange(self.nl) * self.dlon, indexing="ij")
        z = srtm_sample(LO, LA)
        valid = np.isfinite(z)
        if valid.mean() < 0.2:
            raise ValueError("no SRTM coverage in window")
        z = np.where(valid, z, np.nanmedian(z)).astype(np.float32)
        self.z = z

    def sample(self, lon, lat):
        from scipy.ndimage import map_coordinates
        y = (lat - self.lat0) / self.dlat
        x = (lon - self.lon0) / self.dlon
        return map_coordinates(self.z, [y, x], order=1, mode="nearest")


def load_heightmap(level):
    z = np.load(os.path.join(WS, "out", f"{level}_heightmap.npz"))
    h = z["h"].astype(np.float32)
    meta = json.load(open(os.path.join(WS, "out", f"{level}_contours.json")))["meta"]
    return h, meta


def _block_mean(a, f):
    n0 = a.shape[0] // f * f
    n1 = a.shape[1] // f * f
    return a[:n0, :n1].reshape(n0 // f, f, n1 // f, f).mean(axis=(1, 3))


def game_grid(level, factor=3):
    """Heightmap block-averaged by `factor`; returns grid, world X/Y axes, cell size.

    Row 0 of the heightmap is the SOUTH edge of the world square (verified
    against SRTM: the template has to be flipped to match north-up elevation),
    column 0 is the WEST edge.
    """
    h, meta = load_heightmap(level)
    b = meta["bounds"]
    ncol = h.shape[1]
    cell = (b["maxX"] - b["minX"]) / ncol
    g = _block_mean(h, factor)
    m = factor * cell
    X = b["minX"] + (np.arange(g.shape[1]) + 0.5) * m
    Y = b["minY"] + (np.arange(g.shape[0]) + 0.5) * m
    return g, X, Y, m


def world_half(level):
    """Half width / half height of the level's world square, in world units.

    The world square is NOT always 131072 units: it ranges from 1024 to 262144
    across the 164 levels (most are 4096 or 65568), so it has to be read from
    ``out/<level>_contours.json`` rather than assumed.
    """
    try:
        b = json.load(open(os.path.join(WS, "out", f"{level}_contours.json")))["meta"]["bounds"]
        return (b["maxX"] - b["minX"]) / 2, (b["maxY"] - b["minY"]) / 2
    except Exception:
        return WORLD_HALF, WORLD_HALF


def dem_bbox(lon0, lat0, k, half_x=WORLD_HALF, half_y=None):
    if half_y is None:
        half_y = half_x
    dlon = half_x * k / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)))
    dlat = half_y * k / M_PER_DEG_LAT
    return {"minLon": lon0 - dlon, "maxLon": lon0 + dlon,
            "minLat": lat0 - dlat, "maxLat": lat0 + dlat}


def _window_deg(lonc, latc, half_m):
    """Degrees covering +/-half_m metres around (lonc, latc)."""
    dlat = half_m / M_PER_DEG_LAT
    dlon = half_m / (M_PER_DEG_LON_EQ * math.cos(math.radians(latc)))
    return [lonc - dlon, latc - dlat, lonc + dlon, latc + dlat]


def dem_coarse(level, lonc, latc, half_m=100e3, step=None):
    """cv2 normalised cross-correlation of the heightmap against an SRTM window.

    The window is sized in metres (it has to be bigger than the world square),
    and the comparison happens at roughly 160 m per pixel whatever the heightmap
    resolution is.  Returns the centre (lon0, lat0) of the world square assuming
    one world unit is one metre; the k factor is fitted afterwards.
    """
    import cv2
    h, meta = load_heightmap(level)
    if step is None:
        step = max(1, int(round(160.0 / meta["cellSizeM"])))
    mpx = meta["cellSizeM"] * step
    tpl = h[::step, ::step].astype(np.float32)
    tpl = np.nan_to_num(tpl, nan=float(np.nanmedian(tpl)))
    dlat = mpx / M_PER_DEG_LAT
    dlon = mpx / (M_PER_DEG_LON_EQ * math.cos(math.radians(latc)))
    nlat = max(int(2 * half_m / mpx) + 1, tpl.shape[0] + 2)
    nlon = max(int(2 * half_m / mpx) + 1, tpl.shape[1] + 2)
    glat = latc + (np.arange(nlat) - nlat / 2) * dlat          # ascending (south..north)
    glon = lonc + (np.arange(nlon) - nlon / 2) * dlon
    LA, LO = np.meshgrid(glat[::-1], glon, indexing="ij")      # row 0 = north
    dem = srtm_sample(LO, LA)
    if np.isfinite(dem).mean() < 0.5:
        return None
    dem = np.nan_to_num(dem, nan=float(np.nanmedian(dem))).astype(np.float32)
    t = tpl[::-1]                                              # heightmap -> north-up
    if t.shape[0] > dem.shape[0] or t.shape[1] > dem.shape[1]:
        return None
    r = cv2.matchTemplate(dem, t, cv2.TM_CCOEFF_NORMED)
    _, mx, _, mxl = cv2.minMaxLoc(r)
    c, rr = mxl
    return {"ncc": float(mx),
            "lon0": float(glon[0] + (c + t.shape[1] / 2) * dlon),
            "lat0": float(glat[-1] - (rr + t.shape[0] / 2) * dlat),
            "step_m": mpx}


def dem_refine(level, lon0, lat0, factor=None, k0=1.0, half_m=125e3):
    """Fit (lon0, lat0, k) by maximising NCC of heightmap vs SRTM.

    Both fields are brought to the same effective resolution (~96 m per pixel)
    before comparing, so the correlation is not limited by resampling.  The SRTM
    cache window is sized in metres so it always contains the whole square even
    for k up to 1.5 at high latitudes.
    """
    from scipy.optimize import minimize
    if factor is None:
        cell = load_heightmap(level)[1]["cellSizeM"]
        factor = max(1, int(round(96.0 / cell)))
    g, X, Y, m = game_grid(level, factor)
    win = _window_deg(lon0, lat0, half_m)
    loc = LocalDem(*win, res_m=m)
    rng = np.random.default_rng(0)
    ii = rng.integers(0, g.shape[0], 120000)
    jj = rng.integers(0, g.shape[1], 120000)
    Xs, Ys, Z = X[jj], Y[ii], g[ii, jj].astype(np.float64)
    ok = np.isfinite(Z)
    Xs, Ys, Z = Xs[ok], Ys[ok], Z[ok]
    Z = Z - Z.mean()
    Zn = float(np.sqrt((Z * Z).sum()))

    def ncc(p):
        lo0, la0, k = p
        lat = la0 + k * Ys / M_PER_DEG_LAT
        lon = lo0 + k * Xs / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat)))
        d = loc.sample(lon, lat).astype(np.float64)
        d = d - d.mean()
        return -float((Z * d).sum() / (Zn * np.sqrt((d * d).sum())))

    x0 = [lon0, lat0, k0]
    r = minimize(ncc, x0, method="Nelder-Mead",
                 bounds=[(lon0 - 0.3, lon0 + 0.3), (lat0 - 0.25, lat0 + 0.25), (0.6, 1.5)],
                 options={"initial_simplex": [x0,
                                              [x0[0] + 0.01, x0[1], x0[2]],
                                              [x0[0], x0[1] + 0.01, x0[2]],
                                              [x0[0], x0[1], x0[2] * 1.02]],
                          "xatol": 1e-5, "fatol": 1e-6, "maxiter": 500})
    hx, hy = world_half(level)
    p = [float(r.x[0]), float(r.x[1]), float(r.x[2])]
    out = {"ncc": -float(r.fun), "lon0": p[0], "lat0": p[1], "k": p[2],
           "bbox": dem_bbox(*p, hx, hy)}
    # How sharply is the position pinned down?  Scan translations around the
    # optimum with k frozen and measure the NCC half-width.
    try:
        out["uncertaintyKm"] = ncc_halfwidth(ncc, p)
    except Exception:
        out["uncertaintyKm"] = None
    return out


def ncc_halfwidth(ncc, p, max_m=2000.0, steps=9):
    """NCC half-width (km) of the translation peak: the radius at which the
    correlation falls to half way between peak and the surrounding floor."""
    lon0, lat0, k = p
    dlat_m = 1.0 / (M_PER_DEG_LAT)                 # metres -> degrees
    dlon_scale = 1.0 / (M_PER_DEG_LON_EQ * math.cos(math.radians(lat0)))
    offs = np.linspace(-max_m, max_m, steps)
    prof = []
    peak = -ncc(p)
    for d in offs:
        prof.append((-ncc([lon0 + d * dlon_scale, lat0, k]),
                     -ncc([lon0, lat0 + d * dlat_m, k])))
    prof = np.array(prof)
    floor = float(np.median([prof[0].min(), prof[-1].min()]))
    half = floor + (peak - floor) / 2.0
    # largest |d| still above half level, interpolated
    r = 0.0
    for col in (0, 1):
        good = np.abs(offs)[prof[:, col] >= half]
        if good.size:
            r = max(r, float(good.max()))
        else:
            r = max(r, float(np.abs(offs).min()))
    if r >= max_m * 0.95:            # never falls to half: very flat peak
        return round(max_m / 1000.0, 2)
    return round(max(r, 150.0) / 1000.0, 2)


# ---------------------------------------------------------------------------
# step 4b: global DEM prescreen (AWS terrarium z=5) for maps without labels
# ---------------------------------------------------------------------------
TERRARIUM_DIR = os.path.join(SCRATCH, "terrarium")
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TERRARIUM_Z = 5
TERRARIUM_N = 2 ** TERRARIUM_Z
TERRARIUM_PX = 256
_MOSAIC = {}


def terrarium_tile(z, x, y):
    from PIL import Image
    os.makedirs(TERRARIUM_DIR, exist_ok=True)
    p = os.path.join(TERRARIUM_DIR, f"{z}_{x}_{y}.png")
    if not os.path.exists(p):
        req = urllib.request.Request(TERRARIUM_URL.format(z=z, x=x, y=y),
                                     headers={"User-Agent": "wt-geo/1.0"})
        data = urllib.request.urlopen(req, timeout=60).read()
        with open(p, "wb") as f:
            f.write(data)
    a = np.asarray(Image.open(p).convert("RGB"), np.float32)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768


def terrarium_mosaic(force=False):
    """The whole world at z=5 as one float32 array (8192 x 8192, ~4.9 km/px)."""
    if "world" in _MOSAIC and not force:
        return _MOSAIC["world"]
    npy = os.path.join(TERRARIUM_DIR, f"world_z{TERRARIUM_Z}.npy")
    if os.path.exists(npy) and not force:
        _MOSAIC["world"] = np.load(npy)
        return _MOSAIC["world"]
    n = TERRARIUM_N * TERRARIUM_PX
    W = np.zeros((n, n), np.float32)
    for x in range(TERRARIUM_N):
        for y in range(TERRARIUM_N):
            W[y * TERRARIUM_PX:(y + 1) * TERRARIUM_PX, x * TERRARIUM_PX:(x + 1) * TERRARIUM_PX] = \
                terrarium_tile(TERRARIUM_Z, x, y)
        log(f"  terrarium column {x+1}/{TERRARIUM_N}")
    np.save(npy, W)
    _MOSAIC["world"] = W
    return W


def merc_to_px(lon, lat):
    n = TERRARIUM_N * TERRARIUM_PX
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(max(min(lat, 85.0), -85.0))
    y = (1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2 * n
    return x, y


def px_to_merc(x, y):
    n = TERRARIUM_N * TERRARIUM_PX
    lon = x / n * 360.0 - 180.0
    lat_r = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    return lon, math.degrees(lat_r)


def prescreen_level(level, topn=5, lat_step=1.5, mosaic=None):
    """Global terrain NCC prescreen: returns candidate world-square centres.

    The level heightmap (row 0 = south) is flipped to north-up, resized to the
    pixel footprint of a square of `size_m` metres at each latitude band, and
    slid over the z=5 terrarium mosaic with normalised cross-correlation.  Only
    levels whose world square is large enough to survive ~5 km pixels produce
    useful candidates.
    """
    import cv2
    if mosaic is None:
        mosaic = terrarium_mosaic()
    h, meta = load_heightmap(level)
    hx, hy = world_half(level)
    size_m = 2 * max(hx, hy)
    tpl0 = cv2.resize(np.nan_to_num(h, nan=float(np.nanmedian(h))), (48, 48),
                      interpolation=cv2.INTER_AREA)
    tpl0 = tpl0[::-1]                       # row 0 = south -> north-up
    n = mosaic.shape[0]
    cands = []
    for latc in np.arange(-68, 69, lat_step):
        mpp = (2 * math.pi * 6378137 / n) * math.cos(math.radians(latc))
        side = int(round(size_m / mpp))
        if side < 8 or side > 500:
            continue
        tpl = cv2.resize(tpl0, (side, side), interpolation=cv2.INTER_AREA)
        _, y0 = merc_to_px(0.0, min(latc + 3, 84))
        _, y1 = merc_to_px(0.0, max(latc - 3, -84))
        ya = max(int(min(y0, y1)) - side, 0)
        yb = min(int(max(y0, y1)) + side, n - 1)
        if yb - ya < side + 2:
            continue
        r = cv2.matchTemplate(mosaic[ya:yb, :], tpl, cv2.TM_CCOEFF_NORMED)
        flat = r.ravel()
        idx = np.argpartition(flat, -4)[-4:]
        for i in idx:
            rr, cc = divmod(int(i), r.shape[1])
            lon, lat = px_to_merc(cc + side / 2.0, ya + rr + side / 2.0)
            cands.append((float(flat[i]), lon, lat, side))
        del r, flat
    cands.sort(key=lambda t: -t[0])
    out = []
    for c in cands:
        if all(abs(c[1] - o[1]) > 1.5 or abs(c[2] - o[2]) > 1.5 for o in out):
            # keep only candidates that sit on land: the terrarium mosaic
            # includes bathymetry, so open ocean produces spurious peaks that
            # SRTM (land only) can never confirm
            x, y = merc_to_px(c[1], c[2])
            s = c[3]
            x, y = int(x), int(y)
            win = mosaic[max(y - s // 2, 0):y + s // 2,
                         max(x - s // 2, 0):x + s // 2]
            if win.size == 0 or float((win > -100).mean()) < 0.25:
                continue
            out.append(c)
        if len(out) >= topn:
            break
    return [{"ncc": c[0], "lon": c[1], "lat": c[2], "sidePx": c[3]} for c in out]


def terr_window(lonc, latc, half_m, z=10):
    """Stitch the terrarium tiles covering a window; return a mercator image.

    Tiles cache under out/_geo_cache/terrarium/<z>_<x>_<y>.png.  Returns
    (array, x0, y0, mpp) where x0/y0 are the mercator pixel coordinates of the
    array's top-left corner at this zoom and mpp is metres per pixel at the
    centre latitude.
    """
    from PIL import Image
    n = 2 ** z
    x0f, y0f = merc_to_px_z(lonc - half_m / (M_PER_DEG_LON_EQ * math.cos(
        math.radians(latc))), latc + half_m / M_PER_DEG_LAT, z)
    x1f, y1f = merc_to_px_z(lonc + half_m / (M_PER_DEG_LON_EQ * math.cos(
        math.radians(latc))), latc - half_m / M_PER_DEG_LAT, z)
    tx0, tx1 = int(math.floor(x0f / 256)), int(math.floor(x1f / 256))
    ty0, ty1 = int(math.floor(y0f / 256)), int(math.floor(y1f / 256))
    tx0, ty0 = max(tx0, 0), max(ty0, 0)
    tx1, ty1 = min(tx1, n - 1), min(ty1, n - 1)
    if tx1 < tx0 or ty1 < ty0:
        raise ValueError("window outside the terrarium pyramid")
    H = (ty1 - ty0 + 1) * 256
    W = (tx1 - tx0 + 1) * 256
    arr = np.zeros((H, W), np.float32)
    for tx in range(tx0, tx1 + 1):
        for ty in range(ty0, ty1 + 1):
            arr[(ty - ty0) * 256:(ty - ty0 + 1) * 256,
                (tx - tx0) * 256:(tx - tx0 + 1) * 256] = terrarium_tile(z, tx, ty)
    mpp = (2 * math.pi * 6378137 / (256 * n)) * math.cos(math.radians(latc))
    return arr, tx0 * 256, ty0 * 256, mpp


def merc_to_px_z(lon, lat, z):
    """lon/lat (scalars or arrays) -> pixel coordinates in a z-level mercator grid."""
    n = 256 * (2 ** z)
    lon = np.asarray(lon, np.float64)
    lat = np.asarray(lat, np.float64)
    x = (lon + 180.0) / 360.0 * n
    lat_c = np.clip(lat, -85.0, 85.0)
    lat_r = np.radians(lat_c)
    y = (1.0 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / np.pi) / 2 * n
    return x, y


class LocalTerr:
    """Fast bilinear sampling of a stitched terrarium tile window."""

    def __init__(self, lonc, latc, half_m, z=10):
        self.z = z
        self.arr, self.x0, self.y0, self.mpp = terr_window(lonc, latc, half_m, z)
        self.arr = np.where(np.isfinite(self.arr), self.arr,
                            float(np.nanmedian(self.arr[np.isfinite(self.arr)]))
                            if np.isfinite(self.arr).any() else 0.0)

    def sample(self, lon, lat):
        from scipy.ndimage import map_coordinates
        x, y = merc_to_px_z(lon, lat, self.z)
        return map_coordinates(self.arr, [y - self.y0, x - self.x0], order=1,
                               mode="nearest")


def terr_refine(level, lon0, lat0, z=10, k0=1.0, half_m=125e3, factor=None):
    """Same fit as dem_refine but against the terrarium pyramid instead of SRTM.

    No download budget and no coverage holes over land, which is why the
    terrain-only route uses it: z=10 is ~150 m per pixel, close enough to the
    96 m game grid.
    """
    from scipy.optimize import minimize
    if factor is None:
        cell = load_heightmap(level)[1]["cellSizeM"]
        factor = max(1, int(round(150.0 / cell)))
    g, X, Y, m = game_grid(level, factor)
    loc = LocalTerr(lon0, lat0, half_m, z)
    rng = np.random.default_rng(0)
    ii = rng.integers(0, g.shape[0], 120000)
    jj = rng.integers(0, g.shape[1], 120000)
    Xs, Ys, Z = X[jj], Y[ii], g[ii, jj].astype(np.float64)
    ok = np.isfinite(Z)
    Xs, Ys, Z = Xs[ok], Ys[ok], Z[ok]
    Z = Z - Z.mean()
    Zn = float(np.sqrt((Z * Z).sum()))

    def ncc(p):
        lo0, la0, k = p
        lat = la0 + k * Ys / M_PER_DEG_LAT
        lon = lo0 + k * Xs / (M_PER_DEG_LON_EQ * np.cos(np.radians(lat)))
        d = loc.sample(lon, lat).astype(np.float64)
        d = d - d.mean()
        return -float((Z * d).sum() / (Zn * np.sqrt((d * d).sum())))

    x0 = [lon0, lat0, k0]
    r = minimize(ncc, x0, method="Nelder-Mead",
                 bounds=[(lon0 - 0.35, lon0 + 0.35), (lat0 - 0.3, lat0 + 0.3),
                         (0.75, 1.4)],
                 options={"initial_simplex": [x0,
                                              [x0[0] + 0.01, x0[1], x0[2]],
                                              [x0[0], x0[1] + 0.01, x0[2]],
                                              [x0[0], x0[1], x0[2] * 1.02]],
                          "xatol": 1e-5, "fatol": 1e-6, "maxiter": 500})
    hx, hy = world_half(level)
    p = [float(r.x[0]), float(r.x[1]), float(r.x[2])]
    out = {"ncc": -float(r.fun), "lon0": p[0], "lat0": p[1], "k": p[2],
           "bbox": dem_bbox(*p, hx, hy), "z": z, "mpp": round(loc.mpp, 1)}
    try:
        out["uncertaintyKm"] = ncc_halfwidth(ncc, p)
    except Exception:
        out["uncertaintyKm"] = None
    return out


def solve_dem_only(level, std, verbose=True, topn=2):
    """Locate a level purely from its terrain, no OCR labels involved."""
    res = {"map": level, "crs": "WGS84", "bounds": None, "corners": None,
           "method": None, "residualKm": None, "anchorResidualKm": None,
           "bboxUncertaintyKm": None, "anchors": [], "crosscheck": {},
           "confidence": "unresolved", "note": ""}
    hx, hy = world_half(level)
    half_m = max(60e3, 0.62 * 2 * max(hx, hy))
    try:
        cands = prescreen_level(level, topn=max(topn + 2, 5))
    except Exception as e:
        res["note"] = f"global terrain prescreen failed ({type(e).__name__}: {e})"
        return res
    if not cands:
        res["note"] = ("no dem peak: world square too small for the ~5 km global "
                       "DEM prescreen")
        res["crosscheck"]["heightStdM"] = round(std, 1)
        return res
    srtm_reset_budget()
    accepted = None
    refine_err = None
    tried = []
    for c in cands[:topn]:
        try:
            r = terr_refine(level, c["lon"], c["lat"], z=9, half_m=half_m)
        except Exception as e:
            refine_err = f"{type(e).__name__}: {e}"
            if verbose:
                log(f"  {level}: terrarium refine at candidate failed ({refine_err})")
            continue
        drift = math.hypot((r["lon0"] - c["lon"]) * math.cos(math.radians(c["lat"])),
                           r["lat0"] - c["lat"])
        tried.append({"seedNcc": round(c["ncc"], 3), "refinedNcc": round(r["ncc"], 3),
                      "lon": round(r["lon0"], 3), "lat": round(r["lat0"], 3),
                      "driftDeg": round(drift, 3)})
        if r["ncc"] >= 0.78 and drift <= 0.4:
            accepted = dict(r, seed=(c["lon"], c["lat"]), prescreenNcc=c["ncc"])
            break
    margin = cands[0]["ncc"] - (cands[1]["ncc"] if len(cands) > 1 else 0)
    if accepted is None:
        c = cands[0]
        hx, hy = world_half(level)
        bb = dem_bbox(c["lon"], c["lat"], 1.0, hx, hy)
        res["bounds"] = {k: round(float(v), 6) for k, v in bb.items()}
        res["corners"] = {
            "nw": [round(bb["minLon"], 6), round(bb["maxLat"], 6)],
            "ne": [round(bb["maxLon"], 6), round(bb["maxLat"], 6)],
            "sw": [round(bb["minLon"], 6), round(bb["minLat"], 6)],
            "se": [round(bb["maxLon"], 6), round(bb["minLat"], 6)],
        }
        res["method"] = "dem-coarse-only"
        res["confidence"] = "low"
        res["bboxUncertaintyKm"] = 8.0
        res["demNcc"] = round(cands[0]["ncc"], 4)
        res["crosscheck"] = {
            "demNcc": round(cands[0]["ncc"], 4),
            "bboxUncertaintyKm": 8.0,
            "heightStdM": round(std, 1),
            "prescreenMargin": round(margin, 4),
            "refineAttempts": tried,
            "refine": "failed" + (f" ({refine_err})" if refine_err else
                                  " (weak correlation)"),
            "prescreenCandidates": [{"ncc": round(x["ncc"], 3),
                                     "lon": round(x["lon"], 3),
                                     "lat": round(x["lat"], 3)} for x in cands[:5]],
        }
        res["note"] = (f"coarse-only: z5 global prescreen peak {cands[0]['ncc']:.2f} at "
                       f"{cands[0]['lon']:.2f},{cands[0]['lat']:.2f}; z10 refine failed "
                       f"({refine_err or 'weak correlation'}); position good to ~5-10 km; "
                       f"terrain only, no OCR labels used")
        return res
    best = accepted
    bb = best["bbox"]
    res["bounds"] = {k: round(float(v), 6) for k, v in bb.items()}
    res["corners"] = {
        "nw": [round(bb["minLon"], 6), round(bb["maxLat"], 6)],
        "ne": [round(bb["maxLon"], 6), round(bb["maxLat"], 6)],
        "sw": [round(bb["minLon"], 6), round(bb["minLat"], 6)],
        "se": [round(bb["maxLon"], 6), round(bb["minLat"], 6)],
    }
    res["method"] = "dem-only"
    res["bboxUncertaintyKm"] = best.get("uncertaintyKm")
    res["demNcc"] = round(best["ncc"], 4)
    res["crosscheck"] = {
        "demNcc": round(best["ncc"], 4),
        "prescreenTopNcc": round(cands[0]["ncc"], 4),
        "prescreenMargin": round(margin, 4),
        "worldUnitM": round(best["k"], 4),
        "heightStdM": round(std, 1),
        "bboxUncertaintyKm": best.get("uncertaintyKm"),
        "demSource": "terrarium z10",
        "prescreenCandidates": [{"ncc": round(c["ncc"], 3),
                                 "lon": round(c["lon"], 3),
                                 "lat": round(c["lat"], 3)} for c in cands[:5]],
    }
    ncc = best["ncc"]
    if ncc >= 0.93 and margin >= 0.04:
        res["confidence"] = "high"
    elif ncc >= 0.90:
        res["confidence"] = "medium"
    else:
        # a terrain-only match below 0.90 is a hint, not a fix
        res["confidence"] = "low"
    res["note"] = (f"terrain-only match (no OCR labels); z5 prescreen peak "
                   f"{cands[0]['ncc']:.2f} at {cands[0]['lon']:.2f},"
                   f"{cands[0]['lat']:.2f}, z10 refined NCC {ncc:.3f}, margin "
                   f"{margin:.3f}")
    return res
def bbox_size_m(bounds):
    """Real-world size of a lon/lat bbox in metres (local equirectangular)."""
    mid = (bounds["minLat"] + bounds["maxLat"]) / 2
    ew = (bounds["maxLon"] - bounds["minLon"]) * M_PER_DEG_LON_EQ * math.cos(
        math.radians(mid))
    ns = (bounds["maxLat"] - bounds["minLat"]) * M_PER_DEG_LAT
    return ew, ns


def selfcheck(level, bounds, tol_pct=10.0):
    """Sanity-check a georeference against the level's own world bounds.

    Checks that the bbox is non-degenerate, that its real size matches the
    level's 131072 x 131072 world square within `tol_pct` (one world unit is
    `k` real metres, so a k far from 1 fails this), and that the
    square-in-metres assumption holds (EW == NS).
    """
    out = {"ok": False, "sizeErrPct": None, "squareErrPct": None, "bboxKm": None}
    if not bounds or bounds["minLon"] >= bounds["maxLon"] or bounds["minLat"] >= bounds["maxLat"]:
        return out
    try:
        meta = json.load(open(os.path.join(WS, "out", f"{level}_contours.json")))["meta"]
        W = meta["bounds"]["maxX"] - meta["bounds"]["minX"]
        H = meta["bounds"]["maxY"] - meta["bounds"]["minY"]
    except Exception:
        W = H = 2 * WORLD_HALF
    ew, ns = bbox_size_m(bounds)
    out["bboxKm"] = [round(float(ew) / 1000, 1), round(float(ns) / 1000, 1)]
    out["sizeErrPct"] = round(float(max(abs(ew - W) / W, abs(ns - H) / H) * 100), 2)
    out["squareErrPct"] = round(float(abs(ew - ns) / max(ew, ns) * 100), 2)
    out["ok"] = bool(out["sizeErrPct"] < tol_pct)
    return out


def land_frac_map(level):
    """Fraction of the level heightmap above sea level (h > 0)."""
    try:
        h, _ = load_heightmap(level)
        v = h[np.isfinite(h)]
        return float((v > 0).mean()) if v.size else None
    except Exception:
        return None


def dem_grid(bounds, n=64, z=8):
    """Sample the terrarium pyramid over a lon/lat bbox -> (dem, land mask)."""
    lonc = (bounds["minLon"] + bounds["maxLon"]) / 2
    latc = (bounds["minLat"] + bounds["maxLat"]) / 2
    ew, ns = bbox_size_m(bounds)
    half = max(ew, ns) / 2 * 1.15
    loc = LocalTerr(lonc, latc, half, z)
    lon = np.linspace(bounds["minLon"], bounds["maxLon"], n)
    lat = np.linspace(bounds["maxLat"], bounds["minLat"], n)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    z_ = loc.sample(LO, LA)
    return z_, (z_ > 0)


def texture_sea_mask(level, n=64):
    """Sea mask from the level's exported map texture (None if unavailable)."""
    from PIL import Image
    cand = [os.path.join(WS, "out", "textures", level + "_map_hq.png"),
            os.path.join(WS, "out", "textures", level + "_map.png")]
    path = next((c for c in cand if os.path.exists(c)), None)
    a = None
    if path:
        a = np.asarray(Image.open(path).convert("RGB").resize((n, n), Image.BILINEAR),
                       np.float32)
    else:
        try:
            img, _ = TextureBank().image(level)
            if img is not None:
                a = np.asarray(img.resize((n, n), Image.BILINEAR), np.float32)
        except Exception:
            a = None
    if a is None:
        return None
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    # WT map textures paint the sea a flat blue; land is never that blue
    sea = (b > 55) & (b > r + 25) & (b > g + 15)
    return sea


def _valid_geo(level, res, verbose=False):
    """Independent veto for terrain-only answers.

    Two cheap checks that catch a wrong-but-confident DEM peak:
      * land/sea fraction of the level heightmap vs the matched DEM window;
      * IoU between the texture's sea mask and the DEM land/sea mask.
    """
    b = res.get("bounds")
    if not b:
        return
    method = res.get("method") or ""
    cc = res.setdefault("crosscheck", {})
    lfm = land_frac_map(level)
    if lfm is not None:
        cc["landFracMap"] = round(lfm, 4)
    if method not in ("dem-only", "dem-coarse-only", "ocr+gazetteer+dem-match",
                      "ocr+gazetteer+lsq"):
        return
    try:
        dem, land = dem_grid(b)
    except Exception as e:
        cc["landcheck"] = f"skipped ({type(e).__name__})"
        return
    lfd = float(land.mean())
    cc["landFracDem"] = round(lfd, 4)
    reasons = []
    nccv = res.get("demNcc", cc.get("demNcc", cc.get("nccPeak"))) or 0.0
    labelled_strong = ((res.get("method") == "ocr+gazetteer+dem-match") and nccv >= 0.90)
    if lfm is not None and lfm > 0.01 and lfd > 0.01:
        ratio = max(lfm / lfd, lfd / lfm)
        cc["landFracRatio"] = round(ratio, 3)
        ratio_limit = 5.0 if labelled_strong else 3.0
        if ratio > ratio_limit:
            reasons.append(f"land/sea mismatch (map {lfm:.2f} vs dem {lfd:.2f}, "
                           f"ratio {ratio:.1f}x)")
    # a flat land/water swap is wrong no matter how strong the labels are
    if lfm is not None and ((lfm < 0.05 and lfd > 0.5) or (lfd < 0.05 and lfm > 0.5)):
        reasons.append(f"land/water swapped (map {lfm:.2f} vs dem {lfd:.2f})")
    try:
        tm = texture_sea_mask(level)
    except Exception:
        tm = None
    if tm is not None:
        sea_dem = ~land
        inter = float((tm & sea_dem).sum())
        union = float((tm | sea_dem).sum())
        cc["textureSeaFrac"] = round(float(tm.mean()), 3)
        if tm.mean() > 0.15 and sea_dem.mean() < 0.02:
            reasons.append(f"texture shows water ({tm.mean():.0%}) but the DEM "
                           f"window is dry land")
        if tm.mean() > 0.15 and sea_dem.mean() > 0.15 and union > 0:
            iou = inter / union
            cc["textureWaterIou"] = round(iou, 3)
            if iou < 0.35:
                reasons.append(f"texture water mask IoU {iou:.2f} < 0.35")
            elif iou < 0.60 and res.get("confidence") == "high":
                res["confidence"] = "medium"
                res["note"] = ((res.get("note") or "") +
                               f" [low texture water IoU {iou:.2f}]").strip()
        else:
            cc["textureLandIou"] = None
    if reasons:
        res["bounds"] = None
        res["corners"] = None
        res["confidence"] = "unresolved"
        res["rejected"] = "; ".join(reasons)
        res["note"] = ((res.get("note") or "") + " REJECTED: " +
                       "; ".join(reasons)).strip()
    else:
        res.pop("rejected", None)


def _apply_selfcheck(level, res):
    pre_bounds = bool(res.get("bounds"))
    pre_conf = res.get("confidence")
    chk = selfcheck(level, res.get("bounds") or {})
    cc = res.setdefault("crosscheck", {})
    cc["sizeErrPct"] = chk["sizeErrPct"]
    cc["squareErrPct"] = chk["squareErrPct"]
    cc["sizeCheckOk"] = chk["ok"]
    if chk["bboxKm"]:
        cc["worldSquareKm"] = chk["bboxKm"]
    if res.get("bounds") and not chk["ok"]:
        # the bbox does not agree with the level's own world square: demote it
        if res.get("confidence") in ("high", "medium"):
            res["confidence"] = "low"
            res["note"] = ((res.get("note") or "") +
                           f" [demoted: bbox size error {chk['sizeErrPct']}% vs "
                           f"world square]").strip()
    _valid_geo(level, res)
    # confidence discipline: a terrain-only match below NCC 0.90 is a hint, not
    # a fix; coarse-only is always low
    if res.get("bounds"):
        m = res.get("method") or ""
        ncc = res.get("demNcc", cc.get("demNcc", cc.get("nccPeak"))) or 0.0
        if m == "dem-coarse-only":
            res["confidence"] = "low"
        elif m == "dem-only" and ncc < 0.90:
            res["confidence"] = "low"
    cc["solvedBeforeVeto"] = pre_bounds
    cc["confidenceBeforeVeto"] = pre_conf
    if res.get("method") and res.get("bounds"):
        # one canonical name for the correlation used to place the bbox
        res.setdefault("demNcc", cc.get("demNcc", cc.get("nccPeak")))
        res.setdefault("anchorResidualKm", cc.get("anchorResidualKm"))
        res.setdefault("bboxUncertaintyKm", cc.get("bboxUncertaintyKm"))
        if cc.get("nccPeak") is None and cc.get("demNcc") is not None:
            cc["nccPeak"] = cc["demNcc"]
    return chk


def cmd_verify(rewrite=True):
    """Re-run the bbox sanity check on every written result."""
    import collections
    stats = collections.Counter()
    bad = []
    for lv in level_list():
        p = os.path.join(OUTGEO, lv + ".json")
        r = load_json(p)
        if not r:
            continue
        if r.get("bounds"):
            chk = _apply_selfcheck(lv, r)
            stats["checked"] += 1
            stats["ok" if chk["ok"] else "sizeErrPct>10"] += 1
            if not chk["ok"]:
                bad.append((lv, chk["sizeErrPct"], chk["bboxKm"]))
        else:
            stats["no bounds (unresolved)"] += 1
        if rewrite:
            save_json(p, r)
    print(dict(stats))
    for b in bad:
        print("  size error: %s %.2f%% bbox=%s" % b)


def _anchor_only_result(res, raw, label_cands, w, hgt, dem_err):
    """Fallback result built from the label affine alone (km-scale accuracy)."""
    ab = affine_bbox(raw["affine"], w, hgt)
    if ab["minLon"] >= ab["maxLon"] or ab["minLat"] >= ab["maxLat"]:
        res["note"] = "degenerate affine fit"
        return res
    ew = (ab["maxLon"] - ab["minLon"]) * M_PER_DEG_LON_EQ * math.cos(
        math.radians((ab["minLat"] + ab["maxLat"]) / 2))
    ns = (ab["maxLat"] - ab["minLat"]) * M_PER_DEG_LAT
    if not (30e3 < ew < 400e3 and 30e3 < ns < 400e3):
        res["note"] = f"affine-implied map size implausible ({ew/1000:.0f}x{ns/1000:.0f} km)"
        return res
    res["bounds"] = {k: round(v, 6) for k, v in ab.items()}
    res["corners"] = {
        "nw": [round(ab["minLon"], 6), round(ab["maxLat"], 6)],
        "ne": [round(ab["maxLon"], 6), round(ab["maxLat"], 6)],
        "sw": [round(ab["minLon"], 6), round(ab["minLat"], 6)],
        "se": [round(ab["maxLon"], 6), round(ab["minLat"], 6)],
    }
    res["method"] = "ocr+gazetteer+lsq"
    res["residualKm"] = round(float(raw["affine"][4]), 2)
    res["anchors"] = [{"text": label_cands[j][0]["text"],
                       "lat": round(cj["lat"], 5), "lon": round(cj["lon"], 5),
                       "px": label_cands[j][0]["px"], "py": label_cands[j][0]["py"],
                       "score": round(cj["score"], 1),
                       "gazetteer": cj["name"], "offsetKm": round(dkm, 2)}
                      for j, cj, dkm in raw["inliers"]]
    res["crosscheck"] = {
        "worldSquareKm": [round(ew / 1000, 1), round(ns / 1000, 1)],
        "anchorsUsed": raw["n_inliers"],
        "aspectErrPct": round(abs(ew - ns) / max(ew, ns) * 100, 2),
    }
    res["confidence"] = "medium" if raw["n_inliers"] >= 4 else "low"
    res["note"] = ("no DEM refinement"
                   + (f" ({dem_err})" if dem_err else "")
                   + f"; bbox from {raw['n_inliers']} label anchors, km-scale accuracy")
    return res


def _solve_map(level, gaz=None, verbose=True):
    gaz = gaz or Gazetteer()
    labels_all = collect_labels().get(level)
    res = {"map": level, "crs": "WGS84", "bounds": None, "corners": None,
           "method": None, "residualKm": None, "anchorResidualKm": None,
           "bboxUncertaintyKm": None, "anchors": [],
           "crosscheck": {}, "confidence": "unresolved", "note": ""}
    try:
        h, meta = load_heightmap(level)
    except Exception as e:
        res["note"] = f"no heightmap: {e}"
        return res
    flat = h[np.isfinite(h)]
    std = float(flat.std()) if flat.size else 0.0
    res["crosscheck"]["heightStdM"] = round(std, 1)
    if std < 20.0:
        res["note"] = (f"flat terrain: heightmap std only {std:.1f} m, no DEM "
                       f"signal to match against")
        return res
    if not labels_all or not labels_all["labels"]:
        if verbose:
            log(f"  {level}: no labels, std {std:.0f} m -> terrain-only search")
        return solve_dem_only(level, std, verbose)

    w, hgt = labels_all["w"], labels_all["h"]
    label_cands = []
    for l in labels_all["labels"]:
        cs = gaz.candidates(l["text"], topn=8)
        cs = [c for c in cs if c["score"] >= 70]
        label_cands.append((l, cs))
    n_with = sum(1 for _, c in label_cands if c)
    if n_with == 0:
        if verbose:
            log(f"  {level}: {len(label_cands)} labels, none in gazetteer "
                f"-> terrain-only search")
        alt = solve_dem_only(level, std, verbose)
        alt["note"] = (f"{len(label_cands)} OCR labels, none resolved in the "
                       f"gazetteer; " + (alt.get("note") or ""))
        return alt

    raw = ransac_affine(label_cands, w, hgt)
    centers = []
    if raw and raw["n_inliers"] >= 2:
        bb = affine_bbox(raw["affine"], w, hgt)
        centers.append(((bb["minLon"] + bb["maxLon"]) / 2,
                        (bb["minLat"] + bb["maxLat"]) / 2, "affine"))
        res["_affine"] = raw["affine"]
        res["_affineBbox"] = bb
    # single-label hypotheses: assume the texture spans the world square
    scale_lon = 2 * WORLD_HALF / (w * M_PER_DEG_LON_EQ)
    scale_lat = 2 * WORLD_HALF / (hgt * M_PER_DEG_LAT)
    for l, cs in label_cands:
        for c in cs[:3]:
            lon0 = c["lon"] + (w / 2 - l["px"]) * scale_lon / math.cos(
                math.radians(c["lat"]))
            lat0 = c["lat"] + (hgt / 2 - l["py"]) * scale_lat
            centers.append((lon0, lat0, "label:" + c["name"]))
    # keep at most 4 distinct hypotheses
    uniq = []
    for lon, lat, tag in centers:
        if all(abs(lon - u[0]) > 0.05 or abs(lat - u[1]) > 0.05 for u in uniq):
            uniq.append((lon, lat, tag))
    centers = uniq[:4]

    srtm_reset_budget()
    best = None
    best_coarse = None
    dem_err = None
    for lon, lat, tag in centers:
        try:
            c = dem_coarse(level, lon, lat, half_m=95e3)
        except Exception as e:
            dem_err = f"{type(e).__name__}: {e}"
            if verbose:
                log(f"  {level}: SRTM coarse step failed for {tag}: {dem_err}")
            break
        if c is None:
            continue
        if best_coarse is None or c["ncc"] > best_coarse[0]:
            best_coarse = (c["ncc"], tag, c["lon0"], c["lat0"])
        if c["ncc"] < 0.45:
            continue        # not worth a full refinement
        try:
            r = dem_refine(level, c["lon0"], c["lat0"], half_m=95e3)
        except Exception as e:
            dem_err = f"{type(e).__name__}: {e}"
            if verbose:
                log(f"  {level}: SRTM refine step failed for {tag}: {dem_err}")
            break
        if best is None or r["ncc"] > best["ncc"]:
            best = dict(r, hypothesis=tag, seed=(lon, lat), coarseNcc=c["ncc"])
        if best["ncc"] >= 0.85:
            break        # good enough, do not spend more tiles

    if best is None:
        # no usable DEM match: fall back to the label affine (km-scale accuracy)
        if raw and raw["n_inliers"] >= 3:
            r = _anchor_only_result(res, raw, label_cands, w, hgt, dem_err)
            if best_coarse:
                r["crosscheck"]["bestCoarseNcc"] = round(best_coarse[0], 4)
            return r
        res["note"] = ("labels could not be aligned"
                       + (f" ({dem_err})" if dem_err else "")
                       + "; too few consistent label anchors for an affine fit; ")
        if best_coarse:
            res["crosscheck"]["bestCoarseNcc"] = round(best_coarse[0], 4)
        alt = solve_dem_only(level, std, verbose)
        alt["note"] = res["note"] + (alt.get("note") or "")
        if alt.get("bounds"):
            alt["crosscheck"]["bestCoarseNcc"] = res["crosscheck"].get("bestCoarseNcc")
        return alt
    if best["ncc"] < 0.55 and raw and raw["n_inliers"] >= 3:
        alt = _anchor_only_result(res, raw, label_cands, w, hgt, dem_err)
        if alt["bounds"] is not None:
            return alt

    bb = best["bbox"]
    # For every label pick the gazetteer candidate that agrees best with the
    # Anchor bookkeeping.  The label affine gives each label's position *in the
    # texture*; the DEM placement gives it in the world square.  They agree only
    # when the texture covers the whole square (it does not always), so both are
    # reported: `residualKm` is the rms residual of the label affine itself (a
    # measure of how self-consistent the label placement is), and
    # `labelVsDemKm` is how far the texture mapping sits from the DEM mapping.
    a_res = []
    if raw and raw["n_inliers"] >= 3:
        for j, cj, dkm in raw["inliers"]:
            l = label_cands[j][0]
            a_res.append(dkm)
            res["anchors"].append({"text": l["text"], "lat": round(cj["lat"], 5),
                                   "lon": round(cj["lon"], 5),
                                   "px": l["px"], "py": l["py"],
                                   "score": round(cj["score"], 1),
                                   "gazetteer": cj["name"],
                                   "offsetKm": round(dkm, 2)})
    ew = (bb["maxLon"] - bb["minLon"]) * M_PER_DEG_LON_EQ * math.cos(
        math.radians(best["lat0"]))
    ns = (bb["maxLat"] - bb["minLat"]) * M_PER_DEG_LAT
    cc = {
        "nccPeak": round(best["ncc"], 4),
        "coarseNcc": round(best.get("coarseNcc", 0.0), 4),
        "worldUnitM": round(best["k"], 4),
        "worldSquareKm": [round(ew / 1000, 1), round(ns / 1000, 1)],
        "labelResidKm": [round(x, 1) for x in sorted(a_res)],
        "anchorsUsed": len(a_res),
    }
    # How far does the label-only mapping sit from the DEM mapping?  A large
    # value means the texture covers a different area than the playable square.
    if res.get("_affineBbox") and ew > 0:
        ab = res["_affineBbox"]
        ew2 = (ab["maxLon"] - ab["minLon"]) * M_PER_DEG_LON_EQ * math.cos(
            math.radians((ab["minLat"] + ab["maxLat"]) / 2))
        ns2 = (ab["maxLat"] - ab["minLat"]) * M_PER_DEG_LAT
        cc["textureExtentKm"] = [round(ew2 / 1000, 1), round(ns2 / 1000, 1)]
        cc["textureVsWorldScale"] = round(max(ew2 / ew, ns2 / ns), 3)
        cc["textureCenterOffsetKm"] = round(math.hypot(
            ((ab["minLon"] + ab["maxLon"]) - (bb["minLon"] + bb["maxLon"])) / 2 *
            M_PER_DEG_LON_EQ * math.cos(math.radians(best["lat0"])),
            ((ab["minLat"] + ab["maxLat"]) - (bb["minLat"] + bb["maxLat"])) / 2 *
            M_PER_DEG_LAT) / 1000, 2)
        res["textureBounds"] = {k: round(v, 6) for k, v in ab.items()}
    res["bounds"] = {k: round(v, 6) for k, v in bb.items()}
    res["corners"] = {
        "nw": [round(bb["minLon"], 6), round(bb["maxLat"], 6)],
        "ne": [round(bb["maxLon"], 6), round(bb["maxLat"], 6)],
        "sw": [round(bb["minLon"], 6), round(bb["minLat"], 6)],
        "se": [round(bb["maxLon"], 6), round(bb["minLat"], 6)],
    }
    res["method"] = "ocr+gazetteer+dem-match"
    # Two different numbers, deliberately kept apart:
    #   anchorResidualKm  - scatter of the OCR labels around the fitted texture
    #                       affine (label text is drawn beside a town, so this is
    #                       a few km even when everything is right);
    #   bboxUncertaintyKm - how tightly the DEM correlation pins the bbox down
    #                       (NCC half-width); this is the number to trust.
    res["anchorResidualKm"] = round(float(raw["affine"][4]), 2) if raw else None
    res["residualKm"] = res["anchorResidualKm"]
    res["bboxUncertaintyKm"] = best.get("uncertaintyKm")
    cc["bboxUncertaintyKm"] = best.get("uncertaintyKm")
    res["crosscheck"] = cc
    ncc = best["ncc"]
    if ncc >= 0.90:
        res["confidence"] = "high"
    elif ncc >= 0.75:
        res["confidence"] = "medium"
    elif ncc >= 0.55:
        res["confidence"] = "low"
    elif raw and raw["n_inliers"] >= 3:
        # weak terrain correlation, but the place-name labels still agree on a
        # consistent affine: keep the km-scale answer, clearly marked
        alt = _anchor_only_result(res, raw, label_cands, w, hgt,
                                 f"DEM correlation only {ncc:.2f}")
        return alt
    else:
        res["confidence"] = "unresolved"
        res["bounds"] = None
        res["corners"] = None
        res["note"] = f"DEM correlation too weak (NCC {ncc:.2f})"
    res["note"] = (res["note"] + " " if res["note"] else "") + (
        f"label hypothesis {best['hypothesis']}; fitted world unit = "
        f"{best['k']:.3f} real m; {n_with}/{len(label_cands)} labels resolved")
    return res


def solve_map(level, gaz=None, verbose=True):
    """Georeference one level; the result is always sanity-checked first."""
    res = _solve_map(level, gaz, verbose)
    _apply_selfcheck(level, res)
    return res


def cmd_solve(mapname):
    r = solve_map(mapname)
    r.pop("_affine", None)
    r.pop("_affineBbox", None)
    print(json.dumps(r, ensure_ascii=False, indent=1))
    return r


def cmd_all(only=None, skip_existing=False):
    gaz = Gazetteer()
    levels = level_list()
    os.makedirs(OUTGEO, exist_ok=True)
    stats = {"ok": 0, "fail": 0}
    for n, lv in enumerate(levels):
        dst = os.path.join(OUTGEO, lv + ".json")
        if only and lv not in only:
            continue
        if skip_existing and os.path.exists(dst):
            continue
        t0 = time.time()
        try:
            r = solve_map(lv, gaz)
        except Exception as e:
            r = {"map": lv, "crs": "WGS84", "bounds": None, "corners": None,
                 "method": None, "residualKm": None, "anchors": [],
                 "crosscheck": {}, "confidence": "unresolved",
                 "note": f"solver error: {type(e).__name__}: {e}"}
        r.pop("_affine", None)
        r.pop("_affineBbox", None)
        save_json(dst, r)
        stats["ok" if r["confidence"] != "unresolved" else "fail"] += 1
        log(f"[{n+1}/{len(levels)}] {lv:34s} {r['confidence']:10s} "
            f"{r.get('crosscheck', {}).get('nccPeak')} ({time.time()-t0:.0f}s)")
    cmd_verify()
    cmd_report()


def cmd_report():
    import collections
    levels = level_list()
    rows = []
    for lv in levels:
        r = load_json(os.path.join(OUTGEO, lv + ".json"))
        if r:
            rows.append(r)
    by_conf = collections.Counter(r["confidence"] for r in rows)
    by_method = collections.Counter(r.get("method") or "none" for r in rows)
    ares = [r.get("anchorResidualKm") for r in rows
            if r.get("anchorResidualKm") is not None]
    unc = [r.get("bboxUncertaintyKm") for r in rows
           if r.get("bboxUncertaintyKm") is not None]
    nccs = [r.get("demNcc", r.get("crosscheck", {}).get("nccPeak")) for r in rows]
    nccs = [x for x in nccs if x is not None]
    solved = [r for r in rows if r.get("bounds")]
    unresolved = [{"map": r["map"],
                   "reason": ("rejected: " + r["rejected"]) if r.get("rejected")
                   else (r.get("note") or "").strip()}
                  for r in rows if not r.get("bounds")]
    # relief buckets: how many maps are even *findable* by terrain correlation
    std = {}
    stdp = os.path.join(SCRATCH, "std.json")
    if os.path.exists(stdp):
        std = {k: v["stdM"] for k, v in load_json(stdp).items()}
    buckets = [("std>200", 200, 1e9), ("std 50-200", 50, 200),
               ("std 20-50", 20, 50), ("std<20", -1e9, 20)]
    by_bucket = {}
    for name, lo, hi in buckets:
        keys = [lv for lv in levels if lo < std.get(lv, 0) <= hi]
        ok = sum(1 for lv in keys
                 if any(r["map"] == lv and r.get("bounds") for r in rows))
        by_bucket[name] = {"levels": len(keys), "solved": ok,
                           "rate": round(100.0 * ok / len(keys), 1) if keys else None}
    pre_bounds = [r for r in rows if r.get("crosscheck", {}).get("solvedBeforeVeto")]
    pre_conf = collections.Counter(
        r.get("crosscheck", {}).get("confidenceBeforeVeto") or "unresolved"
        for r in rows if r.get("crosscheck", {}).get("solvedBeforeVeto"))
    idx = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "beforeVeto": {
            "solved": len(pre_bounds),
            "byConfidence": dict(pre_conf),
            "note": "results as produced by the matcher, before the independent "
                    "land/sea + texture-water veto",
        },
        "crs": "WGS84",
        "levels": len(levels),
        "solved": len(solved),
        "unresolved": len(levels) - len(solved),
        "byConfidence": dict(by_conf),
        "byMethod": dict(by_method),
        "byReliefBucket": by_bucket,
        "anchorResidualKm": {
            "median": round(float(np.median(ares)), 2) if ares else None,
            "p90": round(float(np.percentile(ares, 90)), 2) if ares else None,
        },
        "bboxUncertaintyKm": {
            "median": round(float(np.median(unc)), 2) if unc else None,
            "p90": round(float(np.percentile(unc, 90)), 2) if unc else None,
            "max": round(float(max(unc)), 2) if unc else None,
        },
        "demNccMedian": round(float(np.median(nccs)), 3) if nccs else None,
        "unresolvedReasons": _group_reasons(unresolved),
        "rejected": {r["map"]: {"reason": r["rejected"],
                                "landFracMap": r.get("crosscheck", {}).get("landFracMap"),
                                "landFracDem": r.get("crosscheck", {}).get("landFracDem"),
                                "textureWaterIou": r.get("crosscheck", {}).get("textureWaterIou"),
                                "demNcc": r.get("demNcc")}
                     for r in rows if r.get("rejected")},
        "maps": {r["map"]: {"confidence": r["confidence"],
                            "bounds": r.get("bounds"),
                            "method": r.get("method"),
                            "demNcc": r.get("demNcc"),
                            "anchorResidualKm": r.get("anchorResidualKm"),
                            "bboxUncertaintyKm": r.get("bboxUncertaintyKm"),
                            "heightStdM": std.get(r["map"]),
                            "crosscheck": r.get("crosscheck", {})} for r in rows},
    }
    save_json(os.path.join(OUTGEO, "index.json"), idx)
    print(json.dumps({k: v for k, v in idx.items() if k != "maps"},
                     ensure_ascii=False, indent=1))


def _group_reasons(unresolved):
    """Group unresolved maps by reason category (the raw notes carry numbers)."""
    import collections
    groups = collections.defaultdict(list)
    pats = [
        ("flat terrain", "flat terrain (heightmap std < 20 m, no DEM signal)"),
        ("world square too small", "world square too small for the ~5 km global DEM"),
        ("no dem peak", "no DEM peak found (global prescreen + refine)"),
        ("labels not in gazetteer", "labels found but not in the gazetteer"),
        ("labels could not be aligned", "labels found but inconsistent"),
        ("no readable place-name labels", "texture has no labels"),
        ("no heightmap", "no heightmap"),
        ("no SRTM coverage", "elevation source had no coverage"),
        ("rejected:", "rejected by an independent land/sea or texture check"),
    ]
    for u in unresolved:
        reason = (u["reason"] or "unknown").strip()
        for pat, key in pats:
            if pat in reason:
                groups[key].append(u["map"])
                break
        else:
            groups["other: " + reason[:60]].append(u["map"])
    return dict(sorted(groups.items(), key=lambda kv: -len(kv[1])))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ocr")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("gazetteer")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("solve")
    p.add_argument("--map", required=True)
    p = sub.add_parser("all")
    p.add_argument("--only", nargs="*")
    p.add_argument("--skip-existing", action="store_true")
    sub.add_parser("report")
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "ocr":
        cmd_ocr(a.force)
    elif a.cmd == "gazetteer":
        cmd_gazetteer(a.force)
    elif a.cmd == "solve":
        cmd_solve(a.map)
    elif a.cmd == "all":
        cmd_all(a.only, a.skip_existing)
    elif a.cmd == "report":
        cmd_report()
    elif a.cmd == "verify":
        cmd_verify()


if __name__ == "__main__":
    main()
