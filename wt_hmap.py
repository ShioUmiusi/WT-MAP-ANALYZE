"""War Thunder (Dagor DBLD) heightmap extractor.

Format reverse-engineered from DagorEngine sources:
  prog/gameLibs/heightmap/heightmapPhysHandler.cpp  (HeightmapPhysHandler::loadDump)
  prog/engine/heightMapLand/compressedHeightmap.cpp (CompressedHeightmap::loadData / unpack_chunk)
  prog/dagorInclude/heightMapLand/dag_compressedHeightmap.h
"""
import ctypes
import os
import struct

import numpy as np

HMAP_WIDTH_BITS = 24
HMAP_WIDTH_MASK = (1 << 24) - 1
HMAP_CBLOCK_DELTAC_VER = 2
HIER_GRID_OFFSETS = [0, 1, 5, 21, 85, 341, 1365, 5461, 21845, 87381, 349525,
                     1398101, 5592405, 22369621, 89478485]

TAG = {
    "RqRL": 0x4C527152, "DxP2": 0x32507844, "TEX": 0x58455400, "TEX.": 0x2E584554,
    "lmap": 0x70616D6C, "HM2": 0x324D4800, "LMp2": 0x32704D4C, "hspl": 0x6C707368,
    "spgz": 0x7A677073, "stbl": 0x6C627473, "WBBX": 0x58424257, "Lnav": 0x76616E4C,
    "OCCL": 0x4C43434F, "SCN": 0x4E435300, "Obj": 0x6A624F00, "RIGz": 0x7A474952,
    "FRT": 0x54524600, "END": 0x444E4500,
}

_DLL = None
_oodle = None


def _load_oodle(dll_path=None):
    global _DLL, _oodle
    if _oodle is not None:
        return _oodle
    if dll_path is None:
        dll_path = os.environ.get("WT_OODLE_DLL") or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "vendor", "daKernel-dev.dll")
    if not os.path.isfile(dll_path):
        raise RuntimeError(
            "Oodle kernel DLL not found at %s. Set WT_OODLE_DLL to daKernel-dev.dll "
            "(shipped with Dagor Asset Explorer builds; it is not redistributed here)."
            % dll_path)
    _DLL = ctypes.CDLL(dll_path)
    fn = _DLL[574]  # oodle_decompress(dst, max_original_size, src, compressed_size)
    fn.argtypes = (ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64)
    fn.restype = ctypes.c_int64
    _oodle = fn
    return fn


def oodle_decompress(src, out_size):
    fn = _load_oodle()
    buf = ctypes.create_string_buffer(out_size)
    r = fn(buf, out_size, src, len(src))
    if not r or r <= 0:
        raise RuntimeError(f"oodle decompress failed (want {out_size}, got {r})")
    return buf.raw[:r]


def read_level_blocks(path):
    """Return {tag: (payload_offset, payload_size)} for a levels/*.bin file."""
    d = open(path, "rb").read()
    if d[:8] != b"DBLD3x64":
        raise ValueError("not a DBLD level: " + path)
    meta_sz = struct.unpack_from("<I", d, 12)[0]
    off = meta_sz + 0x10
    blocks = {}
    while off + 8 <= len(d):
        sz = struct.unpack_from("<I", d, off)[0]
        tag = d[off + 4:off + 8]
        blocks[tag] = (off + 8, sz - 4)
        if struct.unpack_from("<I", tag)[0] == TAG["END"]:
            break
        off = off + 8 + sz - 4
    return d, blocks


class HmapInfo:
    def __repr__(self):
        return (f"<Hmap {self.width}x{self.height} cell={self.cell_size} "
                f"hMin={self.h_min} hScale={self.h_scale} ofs=({self.ofs_x},{self.ofs_y}) "
                f"blockShift={self.block_shift} hrbSubSz={self.hrb_sub_sz} chunks={self.chunk_cnt + 1} "
                f"blocks={self.bw}x{self.bh} mirror={self.mirror}>")


def _parse_header(payload):
    info = HmapInfo()
    (info.cell_size, info.h_min, info.h_scale, info.ofs_x, info.ofs_y) = struct.unpack_from("<5f", payload, 0)
    width_version, height_mirrored = struct.unpack_from("<2I", payload, 20)
    info.version = width_version >> HMAP_WIDTH_BITS
    info.width = width_version & HMAP_WIDTH_MASK
    info.mirror = height_mirrored >> 31
    info.height = height_mirrored & 0x7FFFFFFF
    info.exclude_bounding = struct.unpack_from("<4i", payload, 28)
    info.chunk_sz = struct.unpack_from("<I", payload, 44)[0]
    info.block_shift = info.chunk_sz & 0xFF
    info.hrb_sub_sz = (1 << ((info.chunk_sz >> 8) & 0xF)) if (info.chunk_sz & 0xF00) else 0
    info.chunk_bytes = info.chunk_sz & ~0xFFF
    info.block_size_shift = 2 * info.block_shift
    info.bw = info.width >> info.block_shift
    info.bh = info.height >> info.block_shift
    info.nblocks = info.bw * info.bh
    level = (info.width // info.hrb_sub_sz).bit_length() - 1 if info.hrb_sub_sz else 0
    info.hrb_levels = level
    info.hrb_count = HIER_GRID_OFFSETS[level] if level < len(HIER_GRID_OFFSETS) else 0
    info.blocks_per_chunk = info.chunk_bytes >> info.block_size_shift
    info.chunk_cnt = ((info.nblocks + info.blocks_per_chunk - 1) // info.blocks_per_chunk) if info.chunk_bytes else 0
    return info


def _decompress_chunk(data, fmt, out_size):
    if fmt == 0x80:      # OODLE
        return oodle_decompress(data, out_size)
    if fmt == 0x40:      # ZSTD
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=out_size)
    if fmt == 0x60:      # ZLIB
        import zlib
        return zlib.decompress(data)
    if fmt == 0x20:      # LZMA
        import lzma
        return lzma.decompress(data)
    if fmt == 0x00:      # stored
        return data[:out_size]
    raise ValueError(f"unknown compression fmt {fmt:#x}")


class HeightmapUnavailable(Exception):
    """Raised when a level has no usable heightmap source.

    `reason` is a short machine-readable code, `detail` is for humans.
    Callers are expected to catch this instead of letting it escape as a crash.
    """

    def __init__(self, reason, detail, path=None):
        self.reason = reason
        self.detail = detail
        self.path = path
        super().__init__(detail)


def level_tags(path):
    """Return the decoded tag names of a level file, e.g. ['lmap', 'HM2', ...]."""
    _, blocks = read_level_blocks(path)
    return ["".join(chr(c) if 32 <= c < 127 else "" for c in t) for t in blocks]


def has_heightmap(path):
    """True when the level carries an HM2 heightmap block."""
    _, blocks = read_level_blocks(path)
    return b"\x00HM2" in blocks


def load_heightmap(path, verbose=True):
    """Decode the HM2 block of a level file. Returns (heights_float32, HmapInfo).

    Raises HeightmapUnavailable when the level has no HM2 block (legacy levels).
    """
    d, blocks = read_level_blocks(path)
    if b"\x00HM2" not in blocks:
        tags = ["".join(chr(c) if 32 <= c < 127 else "" for c in t) for t in blocks]
        raise HeightmapUnavailable(
            "no_hm2",
            "level has no HM2 heightmap block (legacy format); tags=%s" % ",".join(tags),
            path,
        )
    p, n = blocks[b"\x00HM2"]
    payload = d[p:p + n]
    info = _parse_header(payload)
    if info.version != HMAP_CBLOCK_DELTAC_VER:
        raise NotImplementedError(f"legacy hmap version {info.version} not supported")

    bs = info.block_size_shift
    block_bytes = 1 << bs
    header_bytes = info.nblocks * 4
    variance_bytes = info.nblocks * block_bytes
    hrb_bytes = info.hrb_count * 16

    block_info = None
    variance = None
    hrb = b""
    off = 48
    for chunk_idx in range(info.chunk_cnt + 1):
        csz = int.from_bytes(payload[off:off + 3], "little")
        fmt = payload[off + 3]
        raw = payload[off + 4:off + 4 + csz]
        if chunk_idx == 0:
            if info.chunk_cnt == 0:
                out_size = header_bytes + variance_bytes + hrb_bytes
            else:
                out_size = header_bytes + hrb_bytes
            dec = _decompress_chunk(raw, fmt, out_size)
            block_info = np.frombuffer(dec[:header_bytes], dtype="<u2").reshape(-1, 2)
            if info.chunk_cnt == 0:
                variance = np.frombuffer(dec[header_bytes:header_bytes + variance_bytes], dtype=np.uint8).copy()
        else:
            b0 = (chunk_idx - 1) * info.blocks_per_chunk
            b1 = min(info.nblocks, chunk_idx * info.blocks_per_chunk)
            out_size = (b1 - b0) << bs
            dec = _decompress_chunk(raw, fmt, out_size)
            if variance is None:
                variance = np.zeros(variance_bytes, np.uint8)
            variance[b0 << bs:(b0 << bs) + out_size] = np.frombuffer(dec, dtype=np.uint8)
        if verbose:
            print(f"  chunk {chunk_idx}: csz={csz} fmt={fmt:#x} -> {len(dec)} bytes")
        off += 4 + csz
    if verbose:
        print(f"  trailing bytes: {n - off}")

    # delta decode variance: prefix sum inside each block of block_bytes
    v = variance.reshape(info.nblocks, block_bytes).astype(np.int64)
    v = np.cumsum(v, axis=1).astype(np.uint8)
    variance = v.reshape(-1)

    mn = block_info[:, 0].astype(np.uint32)
    delta = block_info[:, 1].astype(np.uint32)
    var = variance.reshape(info.nblocks, block_bytes).astype(np.uint32)
    vals = np.where(delta[:, None] == 0, mn[:, None],
                    mn[:, None] + (var * delta[:, None] + 127) // 255).astype(np.uint16)

    bwidth = 1 << info.block_shift
    grid = vals.reshape(info.bh, info.bw, bwidth, bwidth)          # [by][bx][bj][bi]
    img = grid.transpose(0, 2, 1, 3).reshape(info.bh * bwidth, info.bw * bwidth)
    heights = info.h_min + img.astype(np.float32) * (info.h_scale / 65535.0)
    return heights, info


if __name__ == "__main__":
    import sys
    from wt_paths import levels_dir
    LEVELS = levels_dir()
    name = sys.argv[1] if len(sys.argv) > 1 else "avg_berlin.bin"
    h, info = load_heightmap(os.path.join(LEVELS, name))
    print(info)
    print("heights: min=%.2f max=%.2f mean=%.2f shape=%s" % (h.min(), h.max(), h.mean(), h.shape))
