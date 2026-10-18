"""Legacy (no-HM2) War Thunder levels: rebuild terrain from the lmap LandMesh.

Production wrapper around the validated reference decoder (datalm.py, in this directory;
independently re-verified: median |error| 0.075-0.62 m vs HM2 ground truth on 90 levels,
corr up to 1.00000).

Key layout facts (Dagor `LandMeshManager::loadDump` / `loadMeshData`):

  lmap header, 36 bytes:
    u32 'lndm', u32 version=4, f32 gridCellSize, f32 landCellSize,
    i32 mapSizeX, i32 mapSizeY, i32 originX, i32 originY, u32 useTile,
    u32 meshOfs, u32 detailOfs, u32 tileOfs, u32 rtOfs   (offsets relative to +36)

  at meshAbs = 36 + meshOfs:
    u32 texCnt, matCnt, vdataCnt, mvdHdrSz
    u32 texStrSz; texStrSz bytes of texture names (already 4-aligned in practice)
    u32 blockLen (low 30 bits; top 2 bits are the compression tag), u32 rawSize, oodle data

  after that, a Dagor block stream:
    cells block
      per cell (row-major, z outer): land/lod, decal, combined, [patches] ShaderMesh dumps
      last child: cellBoundings -> u32 size == cellCnt*28, BBox3f[cellCnt]
    (the cells block ends exactly at 36 + detailOfs)

  Each ShaderMesh dump starts with PatchableTab<RElem> (u32 ofs, u32 cnt).
  RElem is 48 bytes: 3 patchable ptrs, then vdOrderIndex, sv, numv, si, numf, baseVertex.
  That elem table IS the vertex ownership map: it says which vdata and which vertex range
  each cell uses. Never guess or evenly split the vertices.

  Coordinates (verified against HM2):
    world.x = landCellSize*(cx + originX + 0.5) + q0/32767 * landCellSize*0.5
    world.z = landCellSize*(cz + originY + 0.5) + q2/32767 * landCellSize*0.5
    world.y =                                  q1/32767 * yscale + yofs
    yofs/yscale from the union of cellBoundings (X/Z are per-cell, Y is global)
    cell index: cx = i % mapSizeX, cz = i // mapSizeX (row-major, z outer)

  Vertices are an irregular triangle mesh (TIN), NOT a regular grid -- a flat cell may
  hold only ~19 vertices while a mountainous one holds ~355. Reshaping to a grid is why
  earlier attempts produced fake seams.
"""
import os

import numpy as np

import datalm  # same directory: the validated LandMesh reference decoder


class LandMeshUnavailable(Exception):
    def __init__(self, reason, detail, path=None):
        self.reason = reason
        self.detail = detail
        self.path = path
        super().__init__(detail)


class LandMeshInfo:
    def __init__(self, lm):
        self.grid_w, self.grid_h = lm["grid"]
        self.cell_size = lm["gridCell"]
        self.land_cell_size = lm["cell"]
        self.origin = lm["origin"]
        self.cell_cnt = self.grid_w * self.grid_h
        self.vdata_cnt = lm["vdataCnt"]
        self.vert_cnt = 0
        self.tri_cnt = 0
        self.source = "landmesh"

    def __repr__(self):
        return (f"<LandMesh {self.grid_w}x{self.grid_h} cells land={self.land_cell_size:.0f}m "
                f"verts={self.vert_cnt} tris={self.tri_cnt} vdata={self.vdata_cnt}>")


def load_land_mesh(path):
    """Return (verts Nx3 world float64, tris Mx3 int32, LandMeshInfo)."""
    try:
        lm = datalm.load_land_mesh(path)
    except Exception as e:
        raise LandMeshUnavailable("parse_failed", f"{type(e).__name__}: {e}", path)
    try:
        V, T = datalm.decode_triangles(lm, lod=0)
    except ValueError as e:
        # e.g. an all-water level: cells exist but carry no land geometry at all
        raise LandMeshUnavailable("no_geometry", f"no land geometry in this level: {e}", path)
    except Exception as e:
        raise LandMeshUnavailable("decode_failed", f"{type(e).__name__}: {e}", path)
    if len(V) == 0 or len(T) == 0:
        raise LandMeshUnavailable("no_geometry", "land mesh has no triangles", path)
    info = LandMeshInfo(lm)
    info.vert_cnt = len(V)
    info.tri_cnt = len(T)
    return V, T.astype(np.int32), info


class LegacyHmapInfo:
    """Attribute-compatible stand-in for wt_hmap.HmapInfo."""

    def __init__(self, width, height, cell_size, ofs_x, ofs_y, h_min, h_scale, landmesh=None):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.ofs_x = ofs_x
        self.ofs_y = ofs_y
        self.h_min = h_min
        self.h_scale = h_scale
        self.source = "landmesh"
        self.landmesh = landmesh
        self.version = None
        self.block_shift = None
        self.hrb_sub_sz = None
        self.chunk_cnt = 0
        self.bw = width
        self.bh = height
        self.mirror = 0

    def __repr__(self):
        return (f"<Hmap(landmesh) {self.width}x{self.height} cell={self.cell_size:.2f}m "
                f"hMin={self.h_min:.2f} hScale={self.h_scale:.2f} "
                f"ofs=({self.ofs_x:.0f},{self.ofs_y:.0f}) {self.landmesh}>")


def rasterise_tin(V, T, res=1024):
    """Rasterise a TIN to a regular height grid by triangle-plane sampling."""
    ok = np.isfinite(V).all(axis=1)
    if not ok.all():
        V = V[ok]
        T = T[ok[T].all(axis=1)]
    x0, x1 = V[:, 0].min(), V[:, 0].max()
    z0, z1 = V[:, 2].min(), V[:, 2].max()
    span = max(x1 - x0, z1 - z0, 1e-9)
    res = int(min(res, max(256, round(span / 8.0))))
    sx = (res - 1) / span
    sz = (res - 1) / span

    acc = np.zeros((res, res))
    cnt = np.zeros((res, res))
    for t in T:
        p0, p1, p2 = V[t[0]], V[t[1]], V[t[2]]
        PX = (np.array([p0[0], p1[0], p2[0]]) - x0) * sx
        PZ = (np.array([p0[2], p1[2], p2[2]]) - z0) * sz
        PY = np.array([p0[1], p1[1], p2[1]])
        den = (PZ[1] - PZ[2]) * (PX[0] - PX[2]) + (PX[2] - PX[1]) * (PZ[0] - PZ[2])
        if abs(den) < 1e-12:
            continue
        mnx = max(int(np.floor(PX.min())), 0)
        mxx = min(int(np.ceil(PX.max())), res - 1)
        mnz = max(int(np.floor(PZ.min())), 0)
        mxz = min(int(np.ceil(PZ.max())), res - 1)
        if mxx < mnx or mxz < mnz:
            continue
        gx_, gz_ = np.meshgrid(np.arange(mnx, mxx + 1), np.arange(mnz, mxz + 1))
        l0 = ((PZ[1] - PZ[2]) * (gx_ - PX[2]) + (PX[2] - PX[1]) * (gz_ - PZ[2])) / den
        l1 = ((PZ[2] - PZ[0]) * (gx_ - PX[2]) + (PX[0] - PX[2]) * (gz_ - PZ[2])) / den
        l2 = 1 - l0 - l1
        ins = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not ins.any():
            continue
        hh = l0 * PY[0] + l1 * PY[1] + l2 * PY[2]
        acc[gz_[ins], gx_[ins]] += hh[ins]
        cnt[gz_[ins], gx_[ins]] += 1

    grid = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
    if not np.isfinite(grid).any():
        raise LandMeshUnavailable("no_tri_coverage", "triangles covered no pixels", None)
    nan = ~np.isfinite(grid)
    if nan.any():
        from scipy import ndimage
        ind = ndimage.distance_transform_edt(nan, return_distances=False, return_indices=True)
        grid = grid[tuple(ind)]
    return grid.astype(np.float32), res, (x0, z0, x0 + span, z0 + span)


def load_heightmap_from_landmesh(path, max_dim=1024):
    """Build a regular height grid for a legacy level. Returns (heights, info)."""
    V, T, lm_info = load_land_mesh(path)
    grid, res, (x0, z0, x1, z1) = rasterise_tin(V, T, res=max_dim)
    info = LegacyHmapInfo(
        width=res, height=res,
        cell_size=(x1 - x0) / max(res - 1, 1),
        ofs_x=x0, ofs_y=z0,
        h_min=float(grid.min()), h_scale=float(max(grid.max() - grid.min(), 1e-6)),
        landmesh=lm_info,
    )
    return grid, info


if __name__ == "__main__":
    import sys
    from wt_paths import levels_dir
    L = levels_dir()
    for name in sys.argv[1:] or ["air_skyscraper_city.bin"]:
        V, T, lm = load_land_mesh(os.path.join(L, name))
        print(f"{name}: {lm}  tris={len(T)}")
        print(f"  x {V[:,0].min():.0f}..{V[:,0].max():.0f}  "
              f"y {V[:,1].min():.1f}..{V[:,1].max():.1f}  "
              f"z {V[:,2].min():.0f}..{V[:,2].max():.0f}")
        h, info = load_heightmap_from_landmesh(os.path.join(L, name))
        print(f"  grid {h.shape} h={h.min():.1f}..{h.max():.1f}")
