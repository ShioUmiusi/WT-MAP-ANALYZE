# 文件格式笔记

本仓库从 Dagor Engine 的关卡文件里解出地形。下面的结论来自对 `levels/*.bin`、`*.dxp.bin`、`*.vromfs.bin` 的实际解析与交叉验证；参考的引擎源码路径附在文末。

## 1. 关卡容器：`levels/<map>.bin`

文件头是 `DBLD3x64`（Dagor Binary Level Data）。`+12` 处是元数据长度，块表从 `0x10 + metaSize` 开始，之后是一串带标签的块：

```
u32 blockSize;  char tag[4];  byte payload[blockSize - 4];
```

解析入口：`wt_hmap.read_level_blocks()`。

本仓库用到的块：

| tag | 内容 |
|---|---|
| `HM2` | 压缩高度图，直接解码 |
| `lmap` | LandMesh 地形渲染网格，旧关卡用它重建地形 |
| `RIGz` | 植被 / 道具实例（含机场建筑，本仓库未使用） |
| `hspl` | 样条：道路、河流、贴花 |
| `Obj` | 灯光对象 |
| `SCN` | 场景模型（未使用） |
| `FRT` | 碰撞 / 射线追踪（未使用） |
| `END` | 块表结束 |

索引 `out/index.json` 与 `dataset/contours/index.json` 里的 `source` 字段标明每张图走的是哪条路：`hm2` 或 `landmesh`。

## 2. `HM2` 块（高度图）

头部字段（小端）：

```
f32 cellSize        每个高度像素对应的米数（2.0 / 32.0 等，随图变化）
f32 hMin            最低高度（米）
f32 hScale          高度范围（米）
f32 worldOfsX       地形左下角世界 X
f32 worldOfsY       地形左下角世界 Y（北 / Z）
u32 widthVersion    高 8 位 = 版本（2 = 分块 delta 压缩），低 24 位 = 宽度（像素）
u32 heightMirrored  bit31 = mirror 标志，低 31 位 = 高度（像素）
i32 excludeBBox[4]  挖洞矩形（像素坐标，本仓库未使用）
u32 chunkSz         位域：bit0-7 = blockShift，bit8-11 = log2(hrbSubSz)，bit12+ = 每块未压缩字节数
```

头部之后是 `chunkCnt + 1` 个压缩块，每块：

```
u24 压缩长度;  u8 压缩方式;  byte 数据
```

压缩方式：`0x00` 原样、`0x20` LZMA、`0x40` ZSTD、`0x60` ZLIB、`0x80` Oodle。战雷的高度图实际用 Oodle。

- `chunkSz & 0xFFF == 0`：单块，解压结果就是完整内容；
- 否则第 0 块只放 BlockInfo 头和高度范围层级表，后续每块是一段方差数据（大图会分块，例如大峡谷 5 块）。

`heightMirrored` 的 bit31 是 Dagor 纹理的 Mirror / Clamp 寻址标志，只影响采样方式，不影响高度数据的方向；本仓库不据此翻转高度图。

它等价于 BC5 / ATI1N 风格的块压缩：每块两个端点（`min`、`delta`）加每像素 1 字节方差。

## 3. 压缩高度图（Dagor `CompressedHeightmap`）

布局：

```
BlockInfo[blocks]              // 每块 4 字节：u16 min、u16 delta
u8 variance[blocks * bs^2]     // 每像素 1 字节，块内 delta 编码（前缀和）
HeightRangeBlock[hrbCount]     // 每层 2x2 高度范围表，本仓库不使用
```

- 像素块大小 `bs = 1 << blockShift`（战雷全部是 8×8）；
- 每个像素：`value = delta == 0 ? min : min + (variance * delta + 127) / 255`；
- 高度：`h = hMin + value * hScale / 65535`；
- **方差字节必须先按 8 字节一组做前缀和**（`np.cumsum`），否则解出来是噪点。这是整套解码里最容易踩的一步。

实现：`wt_hmap.load_heightmap()`。

## 4. `lmap` 块（LandMesh，旧关卡）

旧格式关卡没有 `HM2`，地形只在 LandMesh 三角网里。`lmap` 头部 36 字节：

```
u32 'lndm', u32 version=4, f32 gridCellSize, f32 landCellSize,
i32 mapSizeX, i32 mapSizeY, i32 originX, i32 originY, u32 useTile,
u32 meshOfs, u32 detailOfs, u32 tileOfs, u32 rtOfs      // 相对 +36 的偏移
```

在 `meshAbs = 36 + meshOfs`：

```
u32 texCnt, matCnt, vdataCnt, mvdHdrSz
u32 texStrSz;  texStrSz 字节的纹理名
u32 blockLen（低 30 位；高 2 位是压缩标记）, u32 rawSize, oodle 数据
```

之后是 Dagor 块流，cell 按行主序（z 在外层）排列，每个 cell 依次是 land/lod、decal、combined、可选 patches 的 ShaderMesh dump；块流以 `cellBoundings`（`u32 size == cellCnt*28`，`BBox3f[cellCnt]`）结束。

每个 ShaderMesh dump 以 `PatchableTab<RElem>`（`u32 ofs, u32 cnt`）开头。`RElem` 48 字节，其中 elem 表就是顶点归属表：它说明每个 cell 用哪段 vdata、哪段顶点。**不能按 cell 均匀切分顶点**。

坐标（已与 `HM2` 交叉验证）：

```
world.x = landCellSize*(cx + originX + 0.5) + q0/32767 * landCellSize*0.5
world.z = landCellSize*(cz + originY + 0.5) + q2/32767 * landCellSize*0.5
world.y =                                  q1/32767 * yscale + yofs
```

`yofs` / `yscale` 由所有 `cellBoundings` 的并集确定（X/Z 逐 cell，Y 是全局）。cell 索引 `cx = i % mapSizeX`，`cz = i // mapSizeX`。

顶点是不规则三角网（TIN），不是规则栅格：平地 cell 可能只有约 19 个顶点，山地 cell 可到约 355 个。把 TIN 强行重采样成栅格会在 cell 边界产生假接缝，这是旧实现的主要问题。

实现：`datalm.py`（参考解码器）、`wt_landmesh.py`（生产封装，含重采样）。

## 5. 贴图包：`*.dxp.bin`（DxP2 / GRP2）

小端，64 位 ABI dump：

```
u32 label        'DxP2'
u32 version      2 或 3
u32 texCount
u32 dumpSize     之后描述符 dump 的大小

--- dump 从 0x10 开始，所有指针都是相对 0x10 的偏移 ---
+0x00  PatchableTab<u64>           texNames.map  -> u64 名字偏移数组
+0x10  PatchableTab<ddsx::Header>  texHdr        -> 32 字节头
+0x20  PatchableTab<Rec>           texRec
```

这里的 `PatchableTab` 是 16 字节：`u64 ptr, u32 count, u32 reserved`。`Rec` v2 为 24 字节（`q texId, i32 ?, i32 ofs, i32 packedSize, i32 ?`），v3 为 12 字节（`i32 ?, i32 ofs, i32 packedSize`）。`Rec.ofs` 是**文件绝对偏移**，payload 占 `packedSize` 字节。

纹理 payload 是按 `ddsx::Header.flags & 0xE0000000` 压缩的 DXT mip 链（OODLE / ZSTD / ZLIB / 7ZIP，0 表示未压缩）。名字里 `*` 之后是 Dagor 的纹理元数据（如 `air_afghan_map*?acq0-0-0`），`*` 之前才是资产名。

地图影像分散在 6 个包里，后加载的包覆盖先加载的：`locations_maps`、`locations_maps-hq`、`hq_tex_locations_maps`、`pkg_infantry_level_maps`、`pkg_infantry_hq_level_maps`、`locations_maps_patch`。资产名后缀决定用途，按优先级取：`_map_thumb`、`_tankmap`、`_planemap`、`_airmap`、`_minimap`、`_colormap`、`_map`；带 `$hq` 的是高分辨率变体。

实现：`wt_textures.py`（容器与 DXT 解码）、`export_textures.py`（导出 PNG）、`validate_textures.py`（结构与像素对齐校验）。

## 6. vromfs 与任务数据（机场 / 跑道）

机场不在关卡二进制里，而是任务注入的 composit 对象：

- 几何：`aces.vromfs.bin` → `gamedata/objectgroups/dynaf_*.blk`（`airfield{ start/end/width/spawn[] }`）
- 摆放：`mis.vromfs.bin` → `gamedata/missions/**/*.blk`（TMatrix）

vromfs 容器以 `VRFx` 开头，名字表是共享 BBF。payload 的前 16 与后 16 字节各做一次固定 XOR，解 XOR 后是 zstd 流，个别包还需要 zstd 字典。解出后是 Dagor DataBlock 二进制 dump，字段按标签与声明顺序读取。

`airfield` 字段语义（容易理解错）：

| 字段 | 含义 |
|---|---|
| `start` / `end` | 跑道两端，局部坐标 |
| `width` | 跑道宽度 |
| `spawn[]` | 重生点，局部坐标 |
| 放置矩阵 | 把局部坐标变换到世界坐标（`replay-world-meters`） |

导出结果里：

| 字段 | 含义 |
|---|---|
| `ends[0]` / `ends[1]` | 局部 −X / +X 端；**`ends[1]` 是起飞端** |
| `headingDeg` | `ends[0]→ends[1]` 的轴线方位（正北 0，顺时针） |
| `takeoffHeadingDeg` / `reciprocalDeg` | `headingDeg + 180`，起飞方向用这个 |
| `altM` | 绝对世界海拔 |
| `matrixAxesM` | 区域 TMatrix 三条局部轴的长度（Box 是边长，Sphere 前两轴是半径，Point 是 `[1,1,1]`），不是到原点的距离 |

实现：`wt_airfields.py`。

## 7. 真实地理参考

`wt_geo.py` 对每张图做两件独立的事，再交叉验证：

1. 贴图上的地名标注 → OCR（RapidOCR）→ GeoNames 模糊匹配 → RANSAC 拟合像素到经纬度的北向上仿射；
2. 关卡自己的高度图与 SRTM / terrarium DEM 做归一化互相关，最大化 NCC，反解世界原点的经纬度与比例。

仿射模型（代码注释里的推导）：

```
lon = lon0 + X * k / (111320 * cos(lat))
lat = lat0 + Y * k / 110540
```

其中 `X/Y` 是世界米坐标，`k` 是「一个世界单位对应的真实米数」，按图拟合。两种路线的结果写进 `dataset/geo/<map>.json` 并标置信度。

## 8. 坐标与像素约定

- 输出统一为 `replay-world-meters`：`x` 向东、`y` 向北（Dagor 的 Z），单位米；
- 等高线 JSON 里 `meta.bounds` 用世界米；`contours[*].lines` 是扁平数组 `[x0,y0,x1,y1,...]`，保留 1 位小数；
- 高度图行 0 对应 `bounds.minY`（南端）；`meta.yFlipped=false`；
- 贴图 PNG 左上原点，**行 0 = `maxY`（北）**，与高度图相反；
- 贴图的 `worldBounds` 对陆战图只是 2–4 km 的战斗区域，不是整关 `lmap` 范围。

## 9. 参考源码

格式结论用到的 Dagor Engine 源码（公开仓库）：

- `prog/gameLibs/heightmap/heightmapPhysHandler.cpp`
- `prog/engine/heightMapLand/compressedHeightmap.cpp`
- `prog/dagorInclude/heightMapLand/dag_compressedHeightmap.h`
- `prog/tools/sceneTools/dumpGrp/dumpGrp.cpp`
- `prog/engine/sharedInclude/3d/ddsxTex.h`
- `prog/dagorInclude/gameRes/dag_grpMgr.h`、`prog/engine/gameRes/grpData.h`
- `prog/engine/scene/landMesh/…`（LandMesh 块流）

只读源码做格式对齐，没有复制引擎代码。
