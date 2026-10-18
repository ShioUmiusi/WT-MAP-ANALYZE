"""Reference LandMesh -> heightmap decoder, validated against HM2 on 60+ levels.

Decoded layout (matches DagorEngine LandMeshManager::loadDump / loadMeshData):

  lmap header, 52 bytes:
    u32 'lndm', u32 version=4, f32 gridCellSize, f32 landCellSize,
    i32 mapSizeX, i32 mapSizeY, i32 originX, i32 originY, u32 useTile,
    u32 meshOfs, u32 detailOfs, u32 tileOfs, u32 rtOfs      all offsets from +36

  at meshAbs = 36 + meshOfs:
    u32 texCnt, u32 matCnt, u32 vdataCnt, u32 mvdHdrSz
    u32 texStrSz; texStrSz bytes of texture names; NOT padded to 4
    u32 blockLen (low 30 bits; top 2 bits = compression tag)
    u32 rawSize;  oodle payload                        <- compressed MatVData

  MatVData (decompressed):
    +0  PatchableTab<mat>    : u32 ofs, u32 cnt
    +16 PatchableTab<vdata>  : u32 ofs, u32 cnt
    VdataHdr, 32 bytes each: u32 vertNum, (stride:8|pkdIdxLo:24), (idxSize:28|pkdIdxHi:4),
                             u32 flags, u32 vDeclOfs, u32 vDeclCnt, u32 pad, u32 pad
    vertex blobs start at mvdHdrSz, back to back, each vertNum*stride + idxSize bytes

  after the MatVData block: a Dagor block stream (u32 len|flags<<30, payload):
    beginBlock                     # cells
      for each cell, row-major over (z, x):
        beginBlock                 # per cell
          beginBlock               # land, LOD_COUNT sub-blocks
            beginBlock; ShaderMesh dump; endBlock   x nLod
          endBlock
          beginBlock; ShaderMesh decal   ; endBlock
          beginBlock; ShaderMesh combined; endBlock
          beginBlock; ShaderMesh patches ; endBlock   (optional)
        endBlock
    endBlock
    beginBlock; u32 size=cellCnt*28; BBox3f bbox[cellCnt]; f32 radius[cellCnt]; endBlock

  ShaderMesh dump (size sz):
    +0  PatchableTab<RElem>: u32 ofs, u32 cnt
    +8  u16 stageEndElemIdx[8]
    +24 i32 _deprecatedMaxMatPass, i32 _resv
    RElem at (dump_start + ofs), 48 bytes:
      +0  PatchablePtr e
      +8  PatchablePtr mat
      +16 PatchablePtr vertexData       -> vdata index
      +24 i32 vdOrderIndex
      +28 i32 sv       start vertex in that vdata
      +32 i32 numv
      +36 i32 si, +40 i32 numf, +44 i32 baseVertex

  Coordinates (LandMeshManager, line 556-564 and 773-777):
    landBbox = union of cellBoundings (+ heightmap world box when an HM2 is present)
    yofs = 0.5*(landBbox[1].y + landBbox[0].y);  yscale = 0.5*(landBbox[1].y - landBbox[0].y)
    vertex is SHORT4 (4 x int16), components 0,1,2 = x, y, z, normalised by 1/32767
    world.x = landCellSize*(cx + originX + 0.5) + q0/32767 * landCellSize*0.5
    world.z = landCellSize*(cz + originY + 0.5) + q2/32767 * landCellSize*0.5
    world.y =                          q1/32767 * yscale + yofs
"""
import os
import struct

import numpy as np

from wt_hmap import read_level_blocks, oodle_decompress

LMAP_HEADER_SIZE = 36
RELEM_SIZE = 48


class Blocks:
    """Minimal IGenLoad block reader: u32 len (low 30 bits), payload follows."""

    def __init__(self, buf, pos):
        self.buf = buf
        self.pos = pos
        self.stack = []

    def begin(self):
        v = struct.unpack_from('<I', self.buf, self.pos)[0]
        self.pos += 4
        n = v & 0x3FFFFFFF
        self.stack.append((self.pos, n))
        return n

    def end(self):
        o, n = self.stack.pop()
        self.pos = o + n

    def rest(self):
        o, n = self.stack[-1]
        return o + n - self.pos


def _parse_shader_mesh(buf, ofs):
    """Return [(vdata_index, sv, numv, si, numf)] for one ShaderMesh dump at `ofs`."""
    ptr, cnt = struct.unpack_from('<2I', buf, ofs)
    if cnt == 0 or cnt > 4096 or ofs + ptr < 0 or ofs + ptr + cnt * RELEM_SIZE > len(buf):
        return []
    base = ofs + ptr
    out = []
    for i in range(cnt):
        o = base + i * RELEM_SIZE
        vd = struct.unpack_from('<I', buf, o + 16)[0]
        sv, numv, si, numf = struct.unpack_from('<4i', buf, o + 28)
        if vd == 0 and sv == 0 and numv == 0:
            continue
        out.append((vd, sv, numv, si, numf))
    return out


def load_land_mesh(path):
    """Parse a level's lmap block. Returns a dict; raises on malformed input."""
    d, blocks = read_level_blocks(path)
    if b'lmap' not in blocks:
        raise ValueError('no lmap block')
    lp, ls = blocks[b'lmap']
    lm = d[lp:lp + ls]

    magic, version = struct.unpack_from('<4sI', lm, 0)
    if magic != b'lndm' or version != 4:
        raise ValueError('bad lmap header %r v%d' % (magic, version))
    grid_cell, land_cell = struct.unpack_from('<2f', lm, 8)
    msx, msy, ox, oy = struct.unpack_from('<4i', lm, 16)
    use_tile = struct.unpack_from('<I', lm, 32)[0]
    mesh_ofs, detail_ofs, tile_ofs, rt_ofs = struct.unpack_from('<4I', lm, 36)
    ncell = msx * msy

    mesh_abs = LMAP_HEADER_SIZE + mesh_ofs
    tex_cnt, mat_cnt, vdata_cnt, hdr_raw = struct.unpack_from('<4I', lm, mesh_abs)
    p = mesh_abs + 16
    tex_str_sz = struct.unpack_from('<I', lm, p)[0]
    p += 4 + tex_str_sz                     # texture-name blob, not padded
    blk = struct.unpack_from('<I', lm, p)[0]
    p += 4
    payload = blk & 0x3FFFFFFF
    raw_sz = struct.unpack_from('<I', lm, p)[0]
    mvd = oodle_decompress(lm[p + 4:p + payload], raw_sz)
    p += payload

    vd_tab, nvd = struct.unpack_from('<2I', mvd, 16)
    hdr_sz = hdr_raw
    if (hdr_sz & 0xE0000000) == 0xC0000000:
        hdr_sz &= ~0xC0000000
    elif hdr_sz & 0x40000000:
        hdr_sz &= ~0x40000000

    vdata = []
    off = hdr_sz
    for i in range(nvd):
        o = vd_tab + i * 32
        vert_num, w1, w2, flags = struct.unpack_from('<4I', mvd, o)
        stride = w1 & 0xFF
        idx_size = w2 & 0x0FFFFFFF
        vdata.append(dict(vertNum=vert_num, stride=stride, idxSize=idx_size,
                          flags=flags, dataOfs=off))
        off += vert_num * stride + idx_size

    # ---- per-cell mesh dumps
    cb = Blocks(lm, p)
    cb.begin()
    cells = []
    for _ in range(ncell):
        cb.begin()
        cb.begin()
        lods = []
        while cb.rest() > 0:
            cb.begin()
            lods.append(_parse_shader_mesh(lm, cb.pos))
            cb.end()
        cb.end()
        cb.begin()
        decal = _parse_shader_mesh(lm, cb.pos)
        cb.end()
        cb.begin()
        combined = _parse_shader_mesh(lm, cb.pos)
        cb.end()
        patches = []
        if cb.rest() > 0:
            cb.begin()
            patches = _parse_shader_mesh(lm, cb.pos)
            cb.end()
        cb.end()
        cells.append(dict(land=lods, decal=decal, combined=combined, patches=patches))

    # ---- cellBoundings: the LAST child block inside `cells`. Its block length *is* the
    # payload size (cellCnt*28 bytes of BBox3f), and it starts exactly at
    # 36+detailOfs - (cellCnt*28+4); the `cells` block then ends at 36+detailOfs.
    bb_len = cb.begin()
    if bb_len != ncell * 28:
        raise ValueError('cellBoundings block len %d != cellCnt*28 %d' % (bb_len, ncell * 28))
    bb = np.frombuffer(lm, dtype='<f4', count=ncell * 6, offset=cb.pos).reshape(ncell, 6)
    cb.end()
    cb.end()

    # ---- vertex data as int16
    raw = []
    ibs = []
    for v in vdata:
        n, st = v['vertNum'], v['stride']
        seg = np.frombuffer(mvd, dtype='<i2', count=n * (st // 2),
                            offset=v['dataOfs']).reshape(n, st // 2)
        raw.append(seg)
        io = v['dataOfs'] + n * st
        ibs.append(mvd[io:io + v['idxSize']])

    return dict(grid=(msx, msy), origin=(ox, oy), cell=land_cell, gridCell=grid_cell,
                useTile=use_tile, texCnt=tex_cnt, matCnt=mat_cnt, vdataCnt=vdata_cnt,
                vdata=vdata, cells=cells, bbox=bb, verts=raw, ib=ibs,
                bboxMin=bb[:, 0:3].copy(), bboxMax=bb[:, 3:6].copy())


def decode_points(lm, lod=0, y_bounds=None):
    """Return (x, y, z) float64 arrays in world space for all LOD-`lod` land vertices.

    y_bounds: optional (ymin, ymax) overriding the landBbox union. When the level has an
    HM2, the engine's landBbox is union(cellBoundings, hmap world box); pass that to match
    it exactly. Default is the cellBoundings union, which differs by < 1 m in practice.
    """
    msx, msy = lm['grid']
    ox, oy = lm['origin']
    lcs = lm['cell']
    bmin, bmax = lm['bboxMin'], lm['bboxMax']
    valid = (bmin < bmax).all(axis=1)
    if y_bounds is None:
        y_bounds = (float(bmin[valid, 1].min()), float(bmax[valid, 1].max()))
    y0, y1 = y_bounds
    yofs = 0.5 * (y1 + y0)
    ysc = 0.5 * (y1 - y0) / 32767.0
    xsc = lcs * 0.5 / 32767.0

    xs, ys, zs = [], [], []
    for ci, c in enumerate(lm['cells']):
        if lod >= len(c['land']) or not c['land'][lod]:
            continue
        cx, cz = ci % msx, ci // msx
        wx0 = lcs * (cx + ox + 0.5)
        wz0 = lcs * (cz + oy + 0.5)
        for (vd, sv, numv, si, numf) in c['land'][lod]:
            q = lm['verts'][vd][sv:sv + numv].astype(np.float64)
            if len(q) == 0:
                continue
            xs.append(wx0 + q[:, 0] * xsc)
            zs.append(wz0 + q[:, 2] * xsc)
            ys.append(q[:, 1] * ysc + yofs)
    if not xs:
        raise ValueError('no land vertices for lod %d' % lod)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)


def decode_triangles(lm, lod=0, y_bounds=None):
    """Return (verts Nx3 world, tris Mx3 local-index) rebuilt from the index buffers.

    Vertices are per-cell (local index base = sv), triangles reference `verts` in
    per-elem local indices, so the two arrays together form a proper TIN.
    """
    msx, _ = lm['grid']
    ox, oy = lm['origin']
    lcs = lm['cell']
    bmin, bmax = lm['bboxMin'], lm['bboxMax']
    valid = (bmin < bmax).all(axis=1)
    if y_bounds is None:
        y_bounds = (float(bmin[valid, 1].min()), float(bmax[valid, 1].max()))
    y0, y1 = y_bounds
    yofs = 0.5 * (y1 + y0)
    ysc = 0.5 * (y1 - y0) / 32767.0
    xsc = lcs * 0.5 / 32767.0

    P = []
    T = []
    base = 0
    for ci, c in enumerate(lm['cells']):
        if lod >= len(c['land']) or not c['land'][lod]:
            continue
        cx, cz = ci % msx, ci // msx
        wx0 = lcs * (cx + ox + 0.5)
        wz0 = lcs * (cz + oy + 0.5)
        for (vd, sv, numv, si, numf) in c['land'][lod]:
            if numv == 0:
                continue
            q = lm['verts'][vd][sv:sv + numv].astype(np.float64)
            P.append(np.stack([wx0 + q[:, 0] * xsc, q[:, 1] * ysc + yofs, wz0 + q[:, 2] * xsc], 1))
            v = lm['vdata'][vd]
            ib16 = not (v['flags'] & 0x40)
            ib = lm['ib'][vd]
            if ib is None or numf == 0:
                base += numv
                continue
            # index values are ABSOLUTE vertex indices into the vdata, unsigned
            # (validated: every elem's indices stay inside [sv, sv+numv) with zero exceptions
            #  on avg_training_ground, air_grand_canyon and air_skyscraper_city)
            a = np.frombuffer(ib, dtype='<u2' if ib16 else '<u4',
                              count=numf * 3, offset=si * (2 if ib16 else 4))
            T.append(a.reshape(-1, 3).astype(np.int64) + base - sv)
            base += numv
    return np.concatenate(P), np.concatenate(T)


def grid_from_points(x, y, z, res=1024, bbox=None):
    """Rasterise the vertex cloud to a (res, res) height grid in world XZ."""
    if bbox is None:
        bbox = (x.min(), z.min(), x.max(), z.max())
    x0, z0, x1, z1 = bbox
    xs = np.clip(((x - x0) / (x1 - x0) * (res - 1)).round().astype(np.int64), 0, res - 1)
    zs = np.clip(((z - z0) / (z1 - z0) * (res - 1)).round().astype(np.int64), 0, res - 1)
    acc = np.zeros((res, res))
    cnt = np.zeros((res, res))
    np.add.at(acc, (zs, xs), y)
    np.add.at(cnt, (zs, xs), 1.0)
    g = np.where(cnt > 0, acc / np.maximum(cnt, 1.0), np.nan)
    nan = ~np.isfinite(g)
    if nan.any():
        from scipy import ndimage
        idx = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
        g = g[tuple(idx)]
    return g.astype(np.float32), bbox


def coverage(lm, lod=0):
    n = len(lm['cells'])
    have = sum(1 for c in lm['cells'] if lod < len(c['land']) and c['land'][lod])
    return have, n
