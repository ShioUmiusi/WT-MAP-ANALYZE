"""War Thunder texture pack (.dxp.bin / DxP2) reader.

Format reverse-engineered from DagorEngine sources:
  prog/tools/sceneTools/dumpGrp/dumpGrp.cpp        (load_tex_pack / DxpBinData)
  prog/engine/sharedInclude/3d/ddsxTex.h           (ddsx::Header)
  prog/dagorInclude/util/dag_roNameMap.h           (RoNameMap::patchData)
  prog/dagorInclude/gameRes/dag_grpMgr.h, prog/engine/gameRes/grpData.h  (GRP2 packs)

Container layout (little-endian, 64-bit ABI dump):

    u32 label        'DxP2'
    u32 version      2 or 3
    u32 texCount
    u32 dumpSize     size of the descriptor dump that follows

    --- dump starts at 0x10, all pointers are offsets relative to 0x10 ---
    +0x00  PatchableTab<u64>              texNames.map   -> array of u64 name offsets
    +0x10  PatchableTab<ddsx::Header>     texHdr         -> 32-byte headers
    +0x20  PatchableTab<Rec>              texRec

`PatchableTab` here is 16 bytes: u64 ptr, u32 count, u32 reserved.

    Rec v2 = 24 bytes: <q texId, i32 ? , i32 ofs, i32 packedSize, i32 ?>
    Rec v3 = 12 bytes: <i32 ?, i32 ofs, i32 packedSize>

`Rec.ofs` is an ABSOLUTE file offset; the payload occupies `packedSize` bytes there.
Payloads stay contiguous and in increasing order, so `ofs` is enough to slice.

Texture payload is a mip chain of DXT surfaces, compressed per record according to
`ddsx::Header.flags & 0xE0000000` (OODLE / ZSTD / ZLIB / 7ZIP, 0 = stored).
`packedSz` / `memSz` in the header tell the compressed and decoded sizes.

The texture name carries Dagor texture metadata after a '*', e.g.
`air_afghan_map*?acq0-0-0`; the part before '*' is the asset name.
"""
import os
import struct

import numpy as np

from wt_hmap import oodle_decompress

DXP_LABEL = b"DxP2"

FMT_DXT1 = struct.unpack("<I", b"DXT1")[0]
FMT_DXT3 = struct.unpack("<I", b"DXT3")[0]
FMT_DXT5 = struct.unpack("<I", b"DXT5")[0]

FLG_CONTIGUOUS_MIP = 0x0100
FLG_NONPACKED = 0x0200
FLG_REV_MIP_ORDER = 0x00040000
FLG_COMPR_MASK = 0xE0000000
FLG_7ZIP = 0x40000000
FLG_ZLIB = 0x80000000
FLG_ZSTD = 0x20000000
FLG_OODLE = 0x60000000

# bytes per 4x4 block for the DXT families we expect
_BLOCK_BYTES = {FMT_DXT1: 8, FMT_DXT3: 16, FMT_DXT5: 16}
_FOURCC = {FMT_DXT1: b"DXT1", FMT_DXT3: b"DXT3", FMT_DXT5: b"DXT5"}


class DdsxHeader:
    __slots__ = ("label", "d3d_format", "flags", "w", "h", "levels", "hq_part_levels",
                 "depth", "bits_per_pixel", "l_qmip", "m_qmip", "dxt_shift", "u_qmip",
                 "mem_sz", "packed_sz")

    def __init__(self, raw, off):
        (self.label, self.d3d_format, self.flags, self.w, self.h, self.levels,
         self.hq_part_levels, self.depth, self.bits_per_pixel, qmip, uqmip,
         self.mem_sz, self.packed_sz) = \
            struct.unpack_from("<4sIIHHBBHHBBII", raw, off)
        self.l_qmip = qmip & 0xF
        self.m_qmip = qmip >> 4
        self.dxt_shift = uqmip & 0xF
        self.u_qmip = uqmip >> 4

    @property
    def fmt_name(self):
        return _FOURCC.get(self.d3d_format, struct.pack("<I", self.d3d_format))

    @property
    def compression(self):
        c = self.flags & FLG_COMPR_MASK
        return {FLG_OODLE: "oodle", FLG_ZSTD: "zstd", FLG_ZLIB: "zlib",
                FLG_7ZIP: "7zip", 0: "none"}.get(c, "unknown:%#x" % c)

    def level0_size(self):
        """Bytes of the top mip level (no mip chain)."""
        bb = _BLOCK_BYTES.get(self.d3d_format)
        if bb:
            return max(1, (self.w + 3) // 4) * max(1, (self.h + 3) // 4) * bb
        return self.w * self.h * max(1, self.bits_per_pixel) // 8

    def __repr__(self):
        return (f"<ddsx {self.fmt_name.decode('latin1')} {self.w}x{self.h} "
                f"lv={self.levels} {self.compression} mem={self.mem_sz} packed={self.packed_sz}>")


class TexturePack:
    """One .dxp.bin file: a list of named ddsx textures plus their payloads."""

    def __init__(self, path):
        self.path = path
        with open(path, "rb") as f:
            self.data = f.read()
        d = self.data
        label, self.version, self.tex_count, self.dump_size = struct.unpack_from("<4I", d, 0)
        if self.version not in (2, 3):
            raise ValueError(f"{path}: unsupported DxP2 version {self.version}")

        dump = 0x10
        nptr, ncnt = struct.unpack_from("<II", d, dump + 0)
        hptr, hcnt = struct.unpack_from("<II", d, dump + 16)
        rptr, rcnt = struct.unpack_from("<II", d, dump + 32)

        self.names = []
        for i in range(ncnt):
            p, = struct.unpack_from("<Q", d, dump + nptr + i * 8)
            q = dump + p
            e = d.find(b"\0", q, q + 260)
            if e < 0:
                raise ValueError(f"{path}: unterminated name at {q:#x}")
            self.names.append(d[q:e].decode("latin1"))

        self.headers = [DdsxHeader(d, dump + hptr + i * 32) for i in range(hcnt)]

        stride = 24 if self.version == 2 else 12
        self.records = []
        for i in range(rcnt):
            ints = struct.unpack_from("<%dI" % (stride // 4), d, dump + rptr + i * stride)
            if self.version == 2:
                self.records.append((ints[3], ints[4]))
            else:
                self.records.append((ints[1], ints[2]))

        if not (len(self.names) == len(self.headers) == len(self.records)):
            raise ValueError(f"{path}: table size mismatch "
                             f"{len(self.names)}/{len(self.headers)}/{len(self.records)}")

    def __len__(self):
        return len(self.names)

    def asset_name(self, i):
        """Name with the Dagor texture metadata suffix stripped."""
        return self.names[i].split("*", 1)[0]

    def index_of(self, asset_name):
        for i in range(len(self.names)):
            if self.asset_name(i) == asset_name:
                return i
        return -1

    def payload(self, i):
        """Raw (still compressed) payload bytes for texture i."""
        ofs, size = self.records[i]
        return self.data[ofs:ofs + size]

    def raw_mips(self, i):
        """Decoded mip chain bytes for texture i."""
        h = self.headers[i]
        src = self.payload(i)
        if len(src) != h.packed_sz:
            raise ValueError(f"record {i}: payload {len(src)} != header packedSz {h.packed_sz}")
        comp = h.compression
        if comp == "none":
            return src
        if comp == "oodle":
            return oodle_decompress(src, h.mem_sz)
        if comp == "zstd":
            import zstandard
            return zstandard.ZstdDecompressor().decompress(src, max_output_size=h.mem_sz)
        if comp == "zlib":
            import zlib
            return zlib.decompress(src)
        if comp == "7zip":
            import lzma
            return lzma.decompress(src)
        raise ValueError(f"record {i}: unsupported compression {comp}")

    def level0(self, i):
        """Decoded bytes of the top mip level only."""
        h = self.headers[i]
        raw = self.raw_mips(i)
        n = h.level0_size()
        if len(raw) < n:
            raise ValueError(f"record {i}: decoded {len(raw)} < level0 {n}")
        return raw[:n]

    def to_dds(self, i, level0_only=True):
        """Build a standalone .dds file image for texture i."""
        h = self.headers[i]
        fourcc = _FOURCC.get(h.d3d_format)
        if fourcc is None:
            raise ValueError(f"record {i}: unsupported ddsx format {h.fmt_name!r}")
        if level0_only:
            surfs = self.level0(i)
            levels = 1
        else:
            surfs = self.raw_mips(i)
            levels = max(1, h.levels)

        hdr = bytearray(128)
        hdr[0:4] = b"DDS "
        struct.pack_into("<I", hdr, 4, 124)                                   # dwSize
        flags = 0x1 | 0x2 | 0x4 | 0x1000                                      # CAPS|H|W|PF
        if levels > 1:
            flags |= 0x20000 | 0x400000                                       # MIPMAPCOUNT|MIPMAPS
        struct.pack_into("<I", hdr, 8, flags)
        struct.pack_into("<I", hdr, 12, h.h)
        struct.pack_into("<I", hdr, 16, h.w)
        struct.pack_into("<I", hdr, 20, len(surfs))                           # linear size
        struct.pack_into("<I", hdr, 24, 0)                                    # depth
        struct.pack_into("<I", hdr, 28, levels)                               # mip count
        # pixel format at 76
        struct.pack_into("<II4s", hdr, 76, 32, 0x4, fourcc)
        struct.pack_into("<I", hdr, 108, 0x1000)                              # DDSCAPS_TEXTURE
        return bytes(hdr) + surfs

    def to_pil(self, i):
        from PIL import Image
        import io
        return Image.open(io.BytesIO(self.to_dds(i)))

    def summary(self):
        return (f"<TexturePack {os.path.basename(self.path)} v{self.version} "
                f"{len(self)} textures, {self.dump_size} B descriptors, "
                f"{len(self.data)} B file>")


def ddsx_to_dds_bytes(pack, i, level0_only=True):
    return pack.to_dds(i, level0_only)


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    pk = TexturePack(p)
    print(pk.summary())
    for i in range(min(20, len(pk))):
        print(f"  [{i:3d}] {pk.asset_name(i):40s} {pk.headers[i]}")
