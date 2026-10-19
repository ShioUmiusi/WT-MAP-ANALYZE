"""Offline airfield / runway / respawn extraction for War Thunder levels.

War Thunder does not store airfield geometry in the level binary (`levels/<map>.bin`):
the level carries terrain (HM2/lmap), splines (`hspl`, only roads/rivers/decals),
rendinst placement (`RIGz`, airfield *buildings* only) and lights (`Obj`).
Airfields are **mission data**: the mission spawns a composit objectgroup
(`dynaf_*`, e.g. `dynaf_pg_1line_3000_universal`) whose definition carries the
runway rectangle (`airfield{ start/end/width/spawn[] }`) in local space, and the
mission instance supplies the TMatrix.  Both live in Dagor vromfs packs:

  * geometry: ``aces.vromfs.bin``  -> ``gamedata/objectgroups/dynaf_*.blk``
  * placement: ``mis.vromfs.bin``  -> ``gamedata/missions/**/*.blk``

This module unpacks those vromfs packs (VRFx + XOR-obfuscated zstd, shared BBF
namemap, optional zstd dictionary), parses the binary DataBlock dumps and turns
the runway rectangles into `replay-world-meters` world coordinates, i.e. the
same (x = east, y = north/Z) frame used by ``out/<map>_contours.json``.

Verification (see docs/METHOD-VALIDATION.md):
  * 100% of the 54 ``dynaf_*`` unit classes used by missions have a geometry
    definition (73 definitions found).
  * heading cross-checked against replay-derived runways for air_skyscraper_city:
    4/4 headings agree to 0.1 deg, and the derived centre sits L/2 further along
    the axis because the replay code assumed the marker was the runway centre
    while it is actually the composit origin (runway is offset by 100/-208.5 m).
  * runway altitude vs terrain: median |dz| of a few metres on every map with a
    heightmap (see `--verify`).

Usage:
    python wt_airfields.py --all                 # write out/airfields/*.json + index.json
    python wt_airfields.py --map air_afghan --print
    python wt_airfields.py --verify air_afghan   # altitude residual against heightmap
    python wt_airfields.py --catalog             # dump the dynaf geometry catalog
"""
import argparse
import glob
import json
import math
import os
import struct
import sys

import numpy as np
import zstandard

HERE = os.path.dirname(os.path.abspath(__file__))
GAME = ""        # game install root; resolved from WT_GAME_DIR on first use
LEVELS = ""
OUT_DIR = os.path.join(HERE, "out", "airfields")
MAGIC = b"\x28\xb5\x2f\xfd"
XOR_KEYS = (0xAA55AA55, 0xF00FF00F, 0xAA55AA55, 0x12481248)
SHARED_NM_NAME = "\xff?nm"


def _require_game():
    """Fill GAME/LEVELS from WT_GAME_DIR (see wt_paths.py)."""
    global GAME, LEVELS
    if not GAME:
        from wt_paths import game_dir
        GAME = game_dir()
        LEVELS = os.path.join(GAME, "levels")
    return GAME

# ---------------------------------------------------------------- vromfs / BBF


def _deobf(b):
    """Undo Dagor's zstd obfuscation (first and last 16 bytes XOR)."""
    b = bytearray(b)
    n = len(b)
    if n < 16:
        return bytes(b)
    w = struct.unpack_from("<4I", b, 0)
    struct.pack_into("<4I", b, 0, *[w[i] ^ XOR_KEYS[i] for i in range(4)])
    if n >= 32:
        o = n - 16
        w = struct.unpack_from("<4I", b, o)
        struct.pack_into("<4I", b, o, *[w[i] ^ XOR_KEYS[3 - i] for i in range(4)])
    return bytes(b)


def _cstr(b, o):
    return b[o:b.index(b"\x00", o)].decode("latin-1")


class Vromfs:
    """Unpacked Dagor .vromfs.bin (VRFx): name list + data spans + shared namemap."""

    def __init__(self, path):
        d = open(path, "rb").read()
        label, _target, full, hw = struct.unpack_from("<4sIII", d, 0)
        if label != b"VRFx":
            raise ValueError("not a VRFx vromfs: %s" % path)
        packed = hw & 0x3FFFFFFF
        body = zstandard.ZstdDecompressor().decompress(
            _deobf(d[24:24 + packed]), max_output_size=full)
        self.body = body
        u64 = struct.unpack_from("<Q", body, 0)[0]
        map_ofs, map_cnt = u64 & 0xFFFFFFFF, u64 >> 32
        offs = [struct.unpack_from("<Q", body, map_ofs + i * 8)[0]
                for i in range(map_cnt)]
        names = []
        for o in offs:
            try:
                names.append(_cstr(body, o))
            except Exception:
                names.append(None)
        self.names = names
        # file records follow the name blob (8-byte aligned):
        #   {u64 flags; u32 ofs; u32 size}
        end = 0
        for o in offs:
            if 0 <= o < len(body):
                e = body.find(b"\x00", o)
                if e > 0:
                    end = max(end, e + 1)
        tab = (end + 7) // 8 * 8
        entries = []
        for i in range(map_cnt):
            try:
                ofs, sz = struct.unpack_from("<II", body, tab + i * 16 + 8)
            except struct.error:
                break
            entries.append((ofs, sz))
        self.entries = entries
        self._index = {}
        for i, n in enumerate(names):
            if n is not None:
                self._index.setdefault(n, i)
        self._dict = None
        self._shared_nm = None

    def raw(self, name):
        i = self._index[name]
        ofs, sz = self.entries[i]
        return self.body[ofs:ofs + sz]

    def dict_data(self):
        if self._dict is None:
            self._dict = False
            for n in self.names:
                if n and n.endswith(".dict"):
                    self._dict = zstandard.ZstdCompressionDict(self.raw(n))
                    break
        return self._dict or None

    def data(self, name):
        """Decompressed payload of one file (BBF label byte + zstd[+dict])."""
        raw = self.raw(name)
        if not raw:
            return b""
        if raw[:4] == MAGIC:
            return zstandard.ZstdDecompressor().decompress(
                raw, max_output_size=1 << 28)
        if len(raw) > 5 and raw[1:5] == MAGIC:
            for dd in ([self.dict_data()] if self.dict_data() else []) + [None]:
                try:
                    return zstandard.ZstdDecompressor(dict_data=dd).decompress(
                        raw[1:], max_output_size=1 << 28)
                except Exception:
                    pass
        return raw

    def shared_namemap(self):
        """The vromfs-wide BBF namemap (list of names) or []."""
        if self._shared_nm is None:
            self._shared_nm = []
            if self.names and self.names[-1] == SHARED_NM_NAME:
                raw = self.raw(SHARED_NM_NAME)
                if len(raw) > 40:
                    pay = zstandard.ZstdDecompressor().decompress(
                        raw[40:], max_output_size=1 << 27)
                    self._shared_nm = _read_names(pay)
        return self._shared_nm


def _read_varint(buf, p):
    v = 0
    for i in range(10):
        b = buf[p]
        p += 1
        v |= (b & 0x7F) << (7 * i)
        if not (b & 0x80):
            break
    return v, p


def _read_names(buf):
    """dblk::read_names_base: varint count, varint byte size, NUL-separated names."""
    try:
        cnt, p = _read_varint(buf, 0)
        if not cnt:
            return []
        size, p = _read_varint(buf, p)
        blob = buf[p:p + size]
        names = blob.split(b"\x00")
        if len(names) > cnt:
            names = names[:cnt]
        return [s.decode("latin-1") for s in names]
    except Exception:
        return []


TYPE_SIZE = (0, 8, 4, 4, 8, 12, 16, 8, 12, 1, 4, 48, 8, 16, 0)


class Blk:
    """One node of a binary DataBlock dump."""

    __slots__ = ("name", "params", "blocks")

    def __init__(self, name=""):
        self.name = name
        self.params = []       # [(name, type, value)]
        self.blocks = []

    def get(self, name, default=None):
        for k, _t, v in self.params:
            if k == name:
                return v
        return default

    def all(self, name):
        return [v for k, _t, v in self.params if k == name]

    def walk(self):
        yield self
        for b in self.blocks:
            yield from b.walk()

    def child(self, name):
        for b in self.blocks:
            if b.name == name:
                return b
        return None


def _decode_param(ptype, v, complex_data, names):
    if ptype == 1:                                  # string
        if v & 0x80000000:
            idx = v & 0x7FFFFFFF
            return names[idx] if idx < len(names) else None
        return _cstr(complex_data, v) if v < len(complex_data) else None
    if ptype == 2:                                  # int
        return struct.unpack("<i", struct.pack("<I", v))[0]
    if ptype == 3:                                  # real
        return struct.unpack("<f", struct.pack("<I", v))[0]
    if ptype == 9:                                  # bool
        return bool(v)
    if ptype == 10:                                 # e3dcolor
        return ((v >> 24) & 0xFF, (v >> 16) & 0xFF, (v >> 8) & 0xFF, v & 0xFF)
    if ptype == 12:                                 # int64
        return struct.unpack("<q", complex_data[v:v + 8])[0] \
            if v + 8 <= len(complex_data) else None
    sz = TYPE_SIZE[ptype] if ptype < len(TYPE_SIZE) else 0
    if sz and v + sz <= len(complex_data):
        raw = complex_data[v:v + sz]
        if ptype in (4, 5, 6, 11):
            return struct.unpack("<%df" % (sz // 4), raw)
        if ptype in (7, 8, 13):
            return struct.unpack("<%di" % (sz // 4), raw)
        return raw
    return None


def load_blk(payload, shared_names):
    """Parse an uncompressed BBF binary DataBlock dump (label byte already stripped)."""
    if not payload:
        raise ValueError("empty blk payload")
    nr = _read_names(payload)
    p = 0
    try:
        cnt, p = _read_varint(payload, 0)
        if cnt:
            _size, p = _read_varint(payload, p)
            p += _size
        names = list(shared_names) + nr
        nblocks, p = _read_varint(payload, p)
        nparams, p = _read_varint(payload, p)
        complex_sz, p = _read_varint(payload, p)
    except Exception as e:
        raise ValueError("bad blk header: %r" % (e,))
    complex_data = payload[p:p + complex_sz]
    p += complex_sz
    # NB: on disk there is no padding between complex data and params
    params_raw = payload[p:p + nparams * 8]
    p += nparams * 8
    descs = []
    for _ in range(nblocks):
        name_id, p = _read_varint(payload, p)
        pcnt, p = _read_varint(payload, p)
        bcnt, p = _read_varint(payload, p)
        fblock, p = (_read_varint(payload, p) if bcnt else (0, p))
        descs.append((name_id, pcnt, bcnt, fblock))
    nodes = [Blk(names[nid - 1] if 0 < nid <= len(names) else "")
             for nid, _pc, _bc, _fb in descs]
    for i, (_nid, _pc, bcnt, fblock) in enumerate(descs):
        for j in range(fblock, fblock + bcnt):
            if 0 <= j < nblocks:
                nodes[i].blocks.append(nodes[j])
    po = 0
    for i, (_nid, pcnt, _bc, _fb) in enumerate(descs):
        node = nodes[i]
        for _ in range(pcnt):
            w, v = struct.unpack_from("<II", params_raw, po)
            po += 8
            name_id = w & 0xFFFFFF
            ptype = (w >> 24) & 0xFF
            pname = names[name_id] if name_id < len(names) else ""
            node.params.append((pname, ptype,
                               _decode_param(ptype, v, complex_data, names)))
    return nodes[0]


# ------------------------------------------------------------- geometry catalog


class AirfieldGeom:
    """Runway rectangle of one ``dynaf_*`` composit, in composit-local metres."""

    __slots__ = ("cls", "file", "start", "end", "width", "spawns", "place_above")

    def __repr__(self):
        return (f"<{self.cls} L={self.length():.0f} W={self.width} "
                f"spawns={len(self.spawns)}>" if self.along() else f"<{self.cls} no runway>")

    def along(self):
        return self.start is not None and self.end is not None

    def length(self):
        if not self.along():
            return 0.0
        return math.dist(self.start, self.end)


def load_geometry_catalog(aces):
    """``gamedata/objectgroups/dynaf_*.blk`` -> {lower class name: AirfieldGeom}."""
    cat = {}
    for fn in aces.names:
        if not fn or not fn.endswith(".blk"):
            continue
        if "/objectgroups/dynaf" not in fn.lower():
            continue
        base = os.path.basename(fn)[:-4].lower()
        try:
            root = load_blk(aces.data(fn), aces.shared_namemap())
        except Exception:
            continue
        af = root.child("airfield")
        g = AirfieldGeom()
        g.cls = base
        g.file = fn
        g.start = af.get("start") if af is not None else None
        g.end = af.get("end") if af is not None else None
        g.width = af.get("width") if af is not None else None
        g.spawns = af.all("spawn") if af is not None else []
        g.place_above = bool(af.get("placeAboveTerrain")) if af is not None else False
        cat[base] = g
    return cat


# ---------------------------------------------------------------- mission scan


AIRFIELD_CLASS_PREFIX = "dynaf"

# Template "part" files (airfields/, templates/, static_units/, ...) are saved with
# whatever level happened to be open in the editor, so their embedded level is not
# trustworthy.  Their contents are attributed to the mission that *imports* them.
_PART_DIRS = ("/templates/", "/airfields/", "/bomb_bases/", "/static_units/",
              "/frontline/", "/objectgroups/")


def _is_part_path(path):
    p = "/" + path.lower()
    return any(d in p for d in _PART_DIRS)


def _mission_level(root):
    """Only the root mission_settings.mission.level."""
    ms = root.child("mission_settings")
    if ms is None:
        return None
    m = ms.child("mission")
    if m is None:
        return None
    lv = m.get("level")
    return lv if isinstance(lv, str) else None


def _make_loader(mis, aces):
    index = {}
    for v in (mis, aces):
        for n in v.names:
            if n:
                index.setdefault(n.lower(), (v, n))

    def load(path):
        ent = index.get(path.lower())
        if ent is None:
            return None
        v, n = ent
        try:
            return load_blk(v.data(n), v.shared_namemap())
        except Exception:
            return None

    return load


def _collect_units(root, out_inst, out_spawn):
    """Extract dynaf airfield instances and *air_spawn* areas from one blk tree."""
    for b in root.walk():
        if b.name == "objectGroups":
            cls = (b.get("unit_class") or "").lower()
            tm = b.get("tm")
            if not tm or len(tm) < 12 or not cls.startswith(AIRFIELD_CLASS_PREFIX):
                continue
            out_inst.append({"name": b.get("name") or cls, "cls": cls,
                             "tm": list(tm)})
    areas = root.child("areas")
    if areas is not None:
        for a in areas.blocks:
            nm = (a.name or "").lower()
            if "air_spawn" not in nm:
                continue
            # `briefing_*_air_spawn` areas are briefing/cinematic markers, not spawns
            if "briefing" in nm:
                continue
            tm = a.get("tm")
            if not tm or len(tm) < 12:
                continue
            out_spawn.append({"name": a.name, "type": a.get("type") or "",
                              "tm": list(tm)})


def _summarize(root):
    imports = []
    imp = root.child("imports")
    if imp is not None:
        for rec in imp.blocks:
            f = rec.get("file")
            if isinstance(f, str) and f:
                imports.append(f)
    inst, spawn = [], []
    _collect_units(root, inst, spawn)
    return {"level": _mission_level(root), "imports": imports,
            "inst": inst, "spawn": spawn}


def scan_missions(mis, aces, known_levels):
    """Attribute every airfield/airspawn to the level its *mission* declares.

    Rules (see docs/METHOD-VALIDATION.md):
      * the level of a mission is the exact ``mission_settings.mission.level``
        string (``levels/<stem>.bin``); no name/prefix matching anywhere;
      * a blk that is imported by another blk is a *part*, not a mission.  Its own
        ``level:`` is editor residue (often ``levels/water.bin`` or some other
        map) and is ignored; its units are attributed to the importing mission;
      * files under template-ish directories and files whose basename starts with
        ``template_`` are never treated as top-level missions;
      * ``levels/water.bin`` (dev placeholder) is dropped entirely.

    Returns (instances, airspawns) where each record carries the exact level, the
    defining file and the root mission file.
    """
    load = _make_loader(mis, aces)
    files = [fn for fn in mis.names
             if fn and fn.endswith(".blk") and fn.startswith("gamedata/missions/")]
    summaries = {}
    imported = set()
    for fn in files:
        root = load(fn)
        if root is None:
            continue
        s = _summarize(root)
        summaries[fn] = s
        for inc in s["imports"]:
            imported.add(inc.lower())
    by_lower = {fn.lower(): fn for fn in summaries}

    instances, airspawns = [], []
    unattributed = []
    n_roots = 0
    for fn in files:
        if fn not in summaries:
            continue
        if fn.lower() in imported or _is_part_path(fn):
            continue
        if os.path.basename(fn).lower().startswith("template_"):
            continue
        level = summaries[fn]["level"]
        if not level or level not in known_levels or level.endswith("/water.bin"):
            unattributed.append({
                "mission": fn, "level": level,
                "reason": ("no level field" if not level else
                           "level not shipped" if level not in known_levels else
                           "levels/water.bin dev placeholder")})
            continue
        n_roots += 1
        visited = set()

        def visit(path):
            key = path.lower()
            if key in visited:
                return
            visited.add(key)
            real = by_lower.get(key)
            if real is None:
                return                      # e.g. objectgroups blk from aces
            s = summaries[real]
            for it in s["inst"]:
                instances.append(dict(it, level=level, file=real, mission=fn))
            for sp in s["spawn"]:
                airspawns.append(dict(sp, level=level, file=real, mission=fn))
            for inc in s["imports"]:
                visit(inc)

        visit(fn)
    if os.environ.get("WT_AIRFIELDS_DEBUG"):
        print("  root missions used: %d, unattributable root missions: %d"
              % (n_roots, len(unattributed)))
        print("  part files (imported by someone): %d" % len(imported))
    return instances, airspawns, unattributed


# ------------------------------------------------------------ world conversion


def world_point(tm, p):
    """Dagor TMatrix (3x3 row-major then translation, 12 floats) * local point.

    Dagor applies a point as p.x*row0 + p.y*row1 + p.z*row2 + t, with the three
    rows flattened first (tm[0..2], tm[3..5], tm[6..8]) and t = tm[9..11].
    """
    x = p[0] * tm[0] + p[1] * tm[3] + p[2] * tm[6] + tm[9]
    y = p[0] * tm[1] + p[1] * tm[4] + p[2] * tm[7] + tm[10]
    z = p[0] * tm[2] + p[1] * tm[5] + p[2] * tm[8] + tm[11]
    return (x, y, z)


def heading_deg(dx, dz):
    """Compass heading clockwise from +Z (north), degrees in [0, 360)."""
    return math.degrees(math.atan2(dx, dz)) % 360.0


def _tm_axis_row(tm, row):
    """Length of basis row `row` (0..2) of the 3x3 part, i.e. the scale of that axis."""
    i = row * 3
    return math.dist((0.0, 0.0, 0.0), (tm[i], tm[i + 1], tm[i + 2]))


def build_airfield(inst, geom):
    """Turn one mission instance + geometry into the public airfield record."""
    tm = inst["tm"]
    start_w = world_point(tm, geom.start)     # local +X threshold (aircraft spawn side)
    end_w = world_point(tm, geom.end)         # local -X far threshold
    cx = (start_w[0] + end_w[0]) * 0.5
    cz = (start_w[2] + end_w[2]) * 0.5
    cy = (start_w[1] + end_w[1]) * 0.5
    dx, dz = start_w[0] - end_w[0], start_w[2] - end_w[2]
    length = math.hypot(dx, dz)
    rect = dict(
        name=inst["name"],
        kind="runway",
        unitClass=inst["cls"],
        level=inst["level"],
        mission=inst["mission"],
        center=[round(cx, 1), round(cz, 1)],
        headingDeg=round(heading_deg(dx, dz), 1),
        # axis bearing: headingDeg runs ends[0] -> ends[1] (0 = north, clockwise).
        # reciprocalDeg = headingDeg + 180, and is the take-off direction because
        # the dynaf spawn points sit at the ends[1] (local +X) threshold.
        reciprocalDeg=round((heading_deg(dx, dz) + 180.0) % 360.0, 1),
        takeoffHeadingDeg=round((heading_deg(dx, dz) + 180.0) % 360.0, 1),
        spawnEnd=1,
        ends=[[round(end_w[0], 1), round(end_w[2], 1)],
              [round(start_w[0], 1), round(start_w[2], 1)]],
        lengthM=round(length, 1),
        widthM=round(float(geom.width), 1) if geom.width is not None else None,
        altM=round(cy, 1),
        source="mission",
        confidence="high",
        geometryFile=geom.file,
        missionFile=inst["file"],
        note=("ends[0] = local -X threshold, ends[1] = local +X threshold where the "
              "dynaf spawn points sit. headingDeg is the ends[0]->ends[1] axis bearing "
              "(true north 0, clockwise); take-off runs ends[1]->ends[0], i.e. "
              "takeoffHeadingDeg = reciprocalDeg = headingDeg+180. altM is absolute "
              "world altitude in metres (not AGL). Geometry: airfield start/end/width "
              "from the dynaf composit definition."),
    )
    spawns = []
    for sp in geom.spawns:
        w = world_point(tm, sp)
        spawns.append({
            "kind": "airfield",
            "level": inst["level"],
            "mission": inst["mission"],
            "pos": [round(w[0], 1), round(w[2], 1)],
            "altM": round(w[1], 1),
            "source": "mission",
            "name": inst["name"],
            "unitClass": inst["cls"],
        })
    return rect, spawns


def build_airspawn(rec):
    tm = rec["tm"]
    w = (tm[9], tm[10], tm[11])
    axes = [round(_tm_axis_row(tm, i), 1) for i in range(3)]
    return {
        "kind": "airspawn",
        "level": rec["level"],
        "mission": rec["mission"],
        "pos": [round(w[0], 1), round(w[2], 1)],
        "altM": round(w[1], 1),
        "source": "mission",
        "name": rec["name"],
        "areaType": rec.get("type") or "",
        # lengths of the three transformed local axes of the area's TMatrix
        # (for a Box area this is the box size along each local axis; a Sphere
        # uses axes 0 and 1 as its radii, axis 2 is then unused/1)
        "matrixAxesM": axes,
        "missionFile": rec["file"],
    }


# ------------------------------------------------------------------ level frame


def level_world_bounds(mapname):
    """World (x,z) box of the level's land mesh, i.e. the frame replay coordinates use.

    Parsed from the ``lmap`` block header (LandMeshManager::loadDump):
    ``gridCell, landCell, mapSizeX, mapSizeY, originX, originY``.
    For old levels that only carry a *detail* HM2 (2 m cells over a 4 km patch)
    this box is far larger than ``out/<map>_contours.json`` meta.bounds; the
    contours artifact covers just that detail patch, while airfields live in the
    full land-mesh frame reported here.
    """
    path = os.path.join(LEVELS, mapname + ".bin")
    try:
        from wt_hmap import read_level_blocks
        d, blocks = read_level_blocks(path)
        if b"lmap" not in blocks:
            return None
        o, _n = blocks[b"lmap"]
        _magic, _ver, _grid, land, sx, sy, ox, oy = struct.unpack_from(
            "<4sIffiiii", d, o)
        if not (land > 0 and 0 < sx <= 4096 and 0 < sy <= 4096):
            return None
        return {"minX": float(ox * land), "maxX": float((ox + sx) * land),
                "minY": float(oy * land), "maxY": float((oy + sy) * land),
                "landCellSizeM": float(land),
                "cells": [int(sx), int(sy)]}
    except Exception:
        return None


# ------------------------------------------------------------------- per level


TEXTURE_INDEX = os.path.join(HERE, "out", "textures", "index.json")


def base_map_bounds(mapname):
    """World bounds of the exported base map (out/textures/index.json worldBounds).

    This is the frame the application-side map view is built on.  For air-battle
    maps it equals the level's land-mesh frame; for the legacy tank maps the
    exported base texture only covers the ~4 km battlefield inside a much larger
    level, so airfields can legitimately lie outside it (they are reported
    separately as `airfieldsOutsideBaseMap`).
    """
    try:
        d = json.load(open(TEXTURE_INDEX))
        v = d.get("maps", {}).get(mapname)
        if v and v.get("worldBounds"):
            return v["worldBounds"]
    except Exception:
        pass
    m = load_contours_meta(mapname)
    return m["bounds"] if m else None


def _in_bounds(b, x, z, tol=1.0):
    return (b["minX"] - tol <= x <= b["maxX"] + tol and
            b["minY"] - tol <= z <= b["maxY"] + tol)


def run_extract(maps=None, verbose=True):
    game = _require_game()
    aces = Vromfs(os.path.join(game, "aces.vromfs.bin"))
    mis = Vromfs(os.path.join(game, "mis.vromfs.bin"))
    if verbose:
        print("loading dynaf geometry catalog from aces.vromfs.bin ...")
    geoms = load_geometry_catalog(aces)
    if verbose:
        print("  %d dynaf geometry definitions" % len(geoms))
    known = {"levels/%s.bin" % os.path.splitext(os.path.basename(p))[0]
             for p in glob.glob(os.path.join(LEVELS, "*.bin"))}
    if verbose:
        print("scanning %d mission blks in mis.vromfs.bin ..." % len(mis.names))
    insts, airspawns, unattributed = scan_missions(mis, aces, known)
    if verbose:
        print("  %d airfield instances, %d airspawn areas" % (len(insts), len(airspawns)))

    by_level = {}
    for it in insts:
        by_level.setdefault(it["level"], []).append(it)
    as_by_level = {}
    for a in airspawns:
        as_by_level.setdefault(a["level"], []).append(a)

    results = {}
    all_levels = {os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(LEVELS, "*.bin"))}
    for mapname in sorted({os.path.splitext(os.path.basename(lv))[0] for lv in by_level}):
        if maps and mapname not in maps:
            continue
        level = "levels/%s.bin" % mapname
        items = by_level.get(level, [])
        lb = level_world_bounds(mapname)
        bm = base_map_bounds(mapname)
        frame = lb or bm                    # level frame is authoritative
        seen = set()
        kept, kept_out, respawns, respawns_out, missions = [], [], [], [], set()
        n_out = 0
        for it in items:
            g = geoms.get(it["cls"])
            if g is None or not g.along():
                continue
            rect, spawns = build_airfield(it, g)
            key = (it["cls"], round(rect["center"][0], 0), round(rect["center"][1], 0))
            missions.add(it["mission"])
            if key in seen:
                continue
            seen.add(key)
            if frame and not _in_bounds(frame, rect["center"][0], rect["center"][1]):
                rect["outsideLevelBounds"] = True
                kept_out.append(rect)
                n_out += 1
                continue
            if bm and not _in_bounds(bm, rect["center"][0], rect["center"][1]):
                rect["outsideBaseMap"] = True
                kept_out.append(rect)
                n_out += 1
                continue
            kept.append(rect)
            respawns.extend(spawns)
        # air spawn areas
        aseen = set()
        for a in as_by_level.get(level, []):
            sp = build_airspawn(a)
            key = (sp["name"], round(sp["pos"][0], 0), round(sp["pos"][1], 0))
            missions.add(a["mission"])
            if key in aseen:
                continue
            aseen.add(key)
            if frame and not _in_bounds(frame, sp["pos"][0], sp["pos"][1]):
                sp["outsideLevelBounds"] = True
                respawns_out.append(sp)
                continue
            if bm and not _in_bounds(bm, sp["pos"][0], sp["pos"][1]):
                sp["outsideBaseMap"] = True
                respawns_out.append(sp)
                continue
            respawns.append(sp)
        kept.sort(key=lambda r: (r["name"], r["center"]))
        kept_out.sort(key=lambda r: (r["name"], r["center"]))
        respawns.sort(key=lambda r: (r["kind"], r["name"], r["pos"]))
        respawns_out.sort(key=lambda r: (r["kind"], r["name"], r["pos"]))
        note = (
            "Extracted offline from aces.vromfs.bin (dynaf_* composit geometry: "
            "airfield start/end/width + spawn points) and mis.vromfs.bin (mission "
            "placement TMatrix). x = world east, y = world north/Z, metres. "
            "altM is absolute world altitude (m), not AGL. Every record carries the "
            "exact `level` it belongs to; no name matching was used. "
            "No airfield geometry exists in levels/%s.bin itself."
            % mapname)
        if kept_out:
            note += (" %d airfield(s) and %d respawn(s) defined by missions for this level "
                     "lie outside the exported base-map bounds (%s) - they are kept in "
                     "airfieldsOutsideBaseMap/respawnsOutsideBaseMap with the reason, "
                     "because the mission data is in the level's land-mesh frame "
                     "(%s) which is larger than the exported texture for this level."
                     % (len(kept_out), len(respawns_out), bm, lb))
        results[mapname] = {
            "map": mapname,
            "crs": "replay-world-meters",
            "levelBounds": lb,
            "baseMapBounds": bm,
            "frame": "level-lmap" if lb else ("base-map" if bm else None),
            "airfields": kept,
            "airfieldsOutsideBaseMap": kept_out,
            "respawns": respawns,
            "respawnsOutsideBaseMap": respawns_out,
            "missions": sorted(missions),
            "note": note,
        }
    return results, geoms, unattributed


# --------------------------------------------------------------- verification


def load_contours_meta(mapname):
    p = os.path.join(HERE, "out", "%s_contours.json" % mapname)
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["meta"]
    except Exception:
        return None


def load_heightmap(mapname):
    p = os.path.join(HERE, "out", "%s_heightmap.npz" % mapname)
    if not os.path.exists(p):
        return None
    try:
        return np.load(p)["h"]
    except Exception:
        return None


def sample_height(h, meta, x, z):
    b = meta["bounds"]
    w = b["maxX"] - b["minX"]
    hh = b["maxY"] - b["minY"]
    c = int((x - b["minX"]) / (w / h.shape[1]))
    r = int((z - b["minY"]) / (hh / h.shape[0]))
    if 0 <= r < h.shape[0] and 0 <= c < h.shape[1]:
        return float(h[r, c])
    return None


def verify(results):
    """Bounds audit + altitude residuals.

    Audit is *edge-exact*: every airfield end and centre and every respawn pos of
    the exported lists must lie inside the level's base-map bounds
    (out/textures/index.json worldBounds) with 1 m tolerance.  Records knowingly
    outside are in the `...OutsideBaseMap` lists and are reported separately.

    Returns [(map, nAf, afViolations, nResp, respViolations, medianAltErrM)].
    """
    rows = []
    viol = []
    for mapname, doc in sorted(results.items()):
        meta = load_contours_meta(mapname)
        h = load_heightmap(mapname)
        bm = doc.get("baseMapBounds") or base_map_bounds(mapname)
        af_v = resp_v = 0
        errs = []
        for af in doc["airfields"]:
            pts = af["ends"] + [af["center"]]
            if bm and not all(_in_bounds(bm, x, z) for x, z in pts):
                af_v += 1
                viol.append((mapname, "airfield", af["name"], af["center"]))
            hs = []
            if meta is not None and h is not None:
                for ex, ez in pts:
                    v = sample_height(h, meta, ex, ez)
                    if v is not None:
                        hs.append(v)
            if hs:
                errs.append(abs(float(np.median(hs)) - af["altM"]))
        for r in doc["respawns"]:
            if bm and not _in_bounds(bm, r["pos"][0], r["pos"][1]):
                resp_v += 1
                viol.append((mapname, r["kind"], r["name"], r["pos"]))
        med = float(np.median(errs)) if errs else None
        rows.append((mapname, len(doc["airfields"]), af_v,
                     len(doc["respawns"]), resp_v, med))
    return rows, viol


# ------------------------------------------------------------------- rendering


def render_map(mapname, doc, out_png):
    """Hillshade + runway rectangles + airfield labels, for eyeball validation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly

    meta = load_contours_meta(mapname)
    h = load_heightmap(mapname)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
    if meta is not None and h is not None:
        b = meta["bounds"]
        gy, gx = np.gradient(h)
        az, alt = np.deg2rad(315.0), np.deg2rad(45.0)
        slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
        aspect = np.arctan2(-gx, gy)
        shade = np.clip(np.sin(alt) * np.sin(slope) +
                        np.cos(alt) * np.cos(slope) * np.cos(az - aspect), 0, 1)
        ax.imshow(shade, cmap="gray", origin="lower",
                  extent=[b["minX"], b["maxX"], b["minY"], b["maxY"]],
                  vmin=0, vmax=1)
    for af in doc["airfields"]:
        (x0, y0), (x1, y1) = af["ends"]
        dx, dy = x1 - x0, y1 - y0
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L * af["widthM"] / 2, dx / L * af["widthM"] / 2
        poly = [(x0 + nx, y0 + ny), (x1 + nx, y1 + ny),
                (x1 - nx, y1 - ny), (x0 - nx, y0 - ny)]
        ax.add_patch(MplPoly(poly, closed=True, facecolor="none",
                             edgecolor="#ff3b30", lw=1.4))
        ax.plot([x0, x1], [y0, y1], color="#ff3b30", lw=0.8, alpha=0.9)
        ax.annotate(af["name"], af["center"], color="#ffd23b",
                    fontsize=6, ha="center", va="bottom")
    for r in doc["respawns"]:
        if r["kind"] == "airspawn":
            ax.plot(r["pos"][0], r["pos"][1], "o", ms=2.5, color="#28c8ff")
    ax.set_aspect("equal")
    ax.set_title("%s  %d airfields  %d respawns" %
                 (mapname, len(doc["airfields"]), len(doc["respawns"])))
    ax.set_xlabel("world X (m)")
    ax.set_ylabel("world Y / Z (m)")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# ------------------------------------------------------------------------ CLI


def cmd_all(maps=None, viz=False):
    results, _geoms, unattributed = run_extract(maps)
    os.makedirs(OUT_DIR, exist_ok=True)
    for mapname, doc in results.items():
        p = os.path.join(OUT_DIR, mapname + ".json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, separators=(",", ":"))
    # all known levels, including those without airfield data
    all_levels = sorted(os.path.splitext(os.path.basename(p))[0]
                        for p in glob.glob(os.path.join(LEVELS, "*.bin")))
    n_af = sum(len(d["airfields"]) for d in results.values())
    n_af_out = sum(len(d["airfieldsOutsideBaseMap"]) for d in results.values())
    n_rs = sum(len(d["respawns"]) for d in results.values())
    n_rs_out = sum(len(d["respawnsOutsideBaseMap"]) for d in results.values())
    index = {
        "schema": "wt-airfields/2",
        "crs": "replay-world-meters",
        "generatedFrom": ["aces.vromfs.bin:gamedata/objectgroups/dynaf_*.blk",
                          "mis.vromfs.bin:gamedata/missions/**/*.blk"],
        "levels": len(all_levels),
        "levelsWithAirfields": len(results),
        "airfields": n_af,
        "airfieldsOutsideBaseMap": n_af_out,
        "respawns": n_rs,
        "respawnsOutsideBaseMap": n_rs_out,
        "runwaysTotal": n_af,
        "maps": {m: {"level": "levels/%s.bin" % m,
                     "airfields": len(d["airfields"]),
                     "airfieldsOutsideBaseMap": len(d["airfieldsOutsideBaseMap"]),
                     "respawns": len(d["respawns"]),
                     "respawnsOutsideBaseMap": len(d["respawnsOutsideBaseMap"]),
                     "levelBounds": d["levelBounds"],
                     "baseMapBounds": d["baseMapBounds"],
                     "runwayLengthsM": sorted({a["lengthM"] for a in d["airfields"]}),
                     "runwayWidthsM": sorted({a["widthM"] for a in d["airfields"]})}
                 for m, d in results.items()},
        "noAirfieldData": [m for m in all_levels if m not in results],
        "unattributableMissions": {
            "count": len(unattributed),
            "note": ("root missions whose own level: field is missing, points to a level "
                     "that is not shipped, or is the levels/water.bin dev placeholder; "
                     "they were not guessed into any map"),
            "list": unattributed,
        },
        "note": ("Airfields inside the exported base-map frame are in `airfields`; "
                 "mission-defined airfields that lie outside that frame (legacy tank "
                 "maps whose exported texture only covers the ~4 km battlefield) are "
                 "kept in `airfieldsOutsideBaseMap` with reason flags.  altM is absolute "
                 "world altitude, matrixAxesM are transformed local axis lengths (Box "
                 "area size / Sphere radii along local axes)."),
    }
    with open(os.path.join(OUT_DIR, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))
    # prune stale per-map files from earlier runs (e.g. the old levels/water.bin dump)
    for p in glob.glob(os.path.join(OUT_DIR, "*.json")):
        m = os.path.splitext(os.path.basename(p))[0]
        if m != "index" and m not in results:
            os.remove(p)
    print("wrote %d map files + index.json -> %s" % (len(results), OUT_DIR))
    print("  runways in base map: %d (+%d outside), respawns: %d (+%d outside)"
          % (n_af, n_af_out, n_rs, n_rs_out))
    if viz:
        viz_dir = os.path.join(OUT_DIR, "_viz")
        os.makedirs(viz_dir, exist_ok=True)
        for mapname, doc in sorted(results.items()):
            if not doc["airfields"]:
                continue
            try:
                render_map(mapname, doc, os.path.join(viz_dir, mapname + ".png"))
            except Exception as e:
                print("  viz failed for", mapname, e)
        print("rendered visualisations ->", viz_dir)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Offline War Thunder airfield extractor")
    ap.add_argument("--all", action="store_true", help="export every level to out/airfields")
    ap.add_argument("--map", help="comma separated map names (default: all)")
    ap.add_argument("--print", dest="do_print", action="store_true")
    ap.add_argument("--verify", metavar="MAPS", nargs="?", const="", default=None)
    ap.add_argument("--viz", action="store_true", help="render validation PNGs")
    ap.add_argument("--catalog", action="store_true", help="dump dynaf geometry catalog")
    args = ap.parse_args(argv)

    maps = None
    if args.map:
        maps = [m.strip() for m in args.map.split(",") if m.strip()]

    if args.catalog:
        game = _require_game()
        aces = Vromfs(os.path.join(game, "aces.vromfs.bin"))
        geoms = load_geometry_catalog(aces)
        out = {k: {"file": g.file,
                   "start": g.start, "end": g.end, "width": g.width,
                   "lengthM": round(g.length(), 1), "spawnCount": len(g.spawns)}
               for k, g in sorted(geoms.items())}
        p = os.path.join(OUT_DIR, "_dynaf_catalog.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print("wrote", p)
        for k, v in list(out.items())[:12]:
            print(" ", k, v)
        return 0

    if args.all or args.map or args.viz:
        results = cmd_all(maps, viz=args.viz)
        if args.do_print:
            for mapname, doc in results.items():
                print("=" * 12, mapname)
                for af in doc["airfields"]:
                    print("  %-26s %-34s center=%s heading=%s L=%s W=%s alt=%s" %
                          (af["name"], af["unitClass"], af["center"],
                           af["headingDeg"], af["lengthM"], af["widthM"], af["altM"]))
        return 0

    if args.verify is not None:
        rmaps = [x for x in (args.verify or "").split(",") if x]
        results, _, unattributed = run_extract(rmaps or None, verbose=False)
        rows, viol = verify(results)
        print("%-30s %5s %5s %6s %5s %8s" %
              ("map", "nAf", "afOut", "nResp", "rsOut", "altErrM"))
        n_af = n_afv = n_rs = n_rsv = 0
        maps_with_data = 0
        for mapname, naf, afv, nrs, rsv, med in rows:
            if naf == 0 and nrs == 0 and rmaps:
                continue
            if naf or nrs:
                maps_with_data += 1
            n_af += naf
            n_afv += afv
            n_rs += nrs
            n_rsv += rsv
            print("%-30s %5d %5d %6d %5d %8s" %
                  (mapname, naf, afv, nrs, rsv,
                   "-" if med is None else "%.1f" % med))
        print("AUDIT: levels=%d levels_with_airfield_data=%d airfields=%d "
              "airfield_bounds_violations=%d respawns=%d respawn_violations=%d"
              % (len(rows), maps_with_data, n_af, n_afv, n_rs, n_rsv))
        if viol:
            print("sample violations:", viol[:10])
        outside = [(m, len(d["airfieldsOutsideBaseMap"]),
                    len(d["respawnsOutsideBaseMap"]))
                   for m, d in sorted(results.items())
                   if d["airfieldsOutsideBaseMap"] or d["respawnsOutsideBaseMap"]]
        print("levels with data kept outside base-map frame:", len(outside),
              "| total outside airfields=%d respawns=%d"
              % (sum(x[1] for x in outside), sum(x[2] for x in outside)))
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
