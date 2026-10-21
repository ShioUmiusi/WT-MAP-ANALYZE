# 数据集

仓库里的 `dataset/` 是已经算好的离线数据，可以直接被地图工具、回放播放器或仪表页读取；重新生成只需要本机安装的战雷客户端，步骤见第 10 节。

## 1. 目录

```
dataset/
  contours/                 163 张地图的等高线 + index.json        ~93 MB
  landmask/                 163 张地图的陆水掩膜 + index.json      ~8 MB
  geo/                      164 张地图的经纬度参考 + index.json    ~0.5 MB
  airfields/                164 张地图的机场/跑道/重生点 + index.json ~12 MB
```

`contours/` 是给前端用的静态资源：文件名固定为 `<map>_contours.json`，`<map>` 与关卡文件名（去掉 `.bin`）一致。按地图名单独取一张即可，单图中位约 300 KB，不需要一次加载全部。

## 2. 数据覆盖

| 项 | 数量 |
|---|---|
| 关卡总数 | 164 |
| 有地形的关卡 | 163（`water.bin` 是纯水面占位关卡，跳过） |
| 地形来源 `hm2` | 90（直接解码压缩高度图，精确） |
| 地形来源 `landmesh` | 73（旧格式关卡，由 lmap 三角网重建） |
| 等高线点总数 | 5,420,777 |
| 带水面边界的图 | 138 |
| 机场数据 | 147 / 164 关有数据 |
| 跑道 | 底图内 1,964 条 + 底图外 393 条 |
| 重生点 | 37,255 + 604 |
| 经纬度参考 | 59 关解出（high 6 / medium 14 / low 39），105 关标记 `unresolved` |
| 陆水掩膜 | 163 张（25 张整图陆地，`waterLevel = null`） |
| 掩膜网格 | 高度图分辨率，按整数倍降采样到最大 2048×2048 |

## 3. 等高线索引 `contours/index.json`

```jsonc
{
  "schema": 1,
  "crs": "replay-world-meters",
  "count": 163,
  "bySource": { "hm2": 90, "landmesh": 73 },
  "maps": [
    {
      "map": "air_afghan",
      "source": "hm2",
      "px": [4096, 4096],            // 高度图分辨率
      "cellM": 32.0,                 // 每像素米数
      "spanM": [131072.0, 131072.0], // 覆盖的世界米范围
      "bounds": { "minX": -65536.0, "maxX": 65536.0,
                  "minY": -65536.0, "maxY": 65536.0 },
      "heightM": [1378.19, 5364.0],  // 最低/最高高程
      "intervalM": 200.0,            // 等高距
      "levels": 19,                  // 等高线条数
      "points": 130478,              // 折线点总数
      "water": false,                // 是否带水面边界
      "file": "air_afghan_contours.json"
    }
  ]
}
```

## 4. 单图格式 `<map>_contours.json`

```jsonc
{
  "meta": {
    "crs": "replay-world-meters",
    "map": "air_skyscraper_city",
    "source": "landmesh",            // "hm2" | "landmesh"
    "bounds": { "minX": -32768.0, "maxX": 32800.03,
                "minY": -32768.0, "maxY": 32800.03 },
    "widthPx": 1024, "heightPx": 1024,
    "cellSizeM": 64.03,
    "heightMin": -59.82, "heightMax": 5145.25,
    "contourIntervalM": 200.0,
    "sampleStep": 1,                 // 出线时对高度图的降采样步长
    "simplifyTolM": 22.41,           // Douglas-Peucker 容差（米）
    "originX": -32768.0, "originY": -32768.0,
    "yFlipped": false,
    "waterLevel": 0.0,               // null = 该图没有低于 0 的高度，整图按陆地处
    "artifactFilter": {              // 栅格空边过滤参数与实际丢弃数
      "flatWindow": 5, "dropFlatFraction": 0.55,
      "polylinesBefore": 459, "polylinesAfter": 459,
      "droppedFlat": 0, "cutAtBridge": 0,
      "droppedSubWaterLevels": 1, "waterLevelUsed": 0.0,
      "polylinesAfterSubWater": 453
    },
    "note": "..."                    // 人读的来源与误差说明
  },
  "contours": [
    { "elev": 200.0, "lines": [[x0,y0, x1,y1, ...], ...] }
  ],
  "water":      [[x0,y0, ...]],      // 可选：水面多边形
  "coastlines": [[x0,y0, ...]]       // 可选：与 water 相同，便于按名读取
}
```

约定：

- 坐标单位米，`x` 向东、`y` 向北，`crs = replay-world-meters`；
- 折线是扁平数组，保留 1 位小数；
- `water` / `coastlines` 只在图里有高程 ≤ `waterLevel` 的区域时出现（163 张里 138 张有）；
- 水面由 `height <= waterLevel` 得到，`waterLevel` 为 `null` 时整图按陆地处；等于水线的像素算水；
- 高度图本身不入库（体积大），需要时用 `make_contours.py` 在本机生成 `*_heightmap.npz`。

## 5. 怎么用

### 5.1 地图名 → 取一张图

```
GET <base>/<map>_contours.json
```

`<map>` 取自关卡名，例如 `levels/avg_berlin.bin` → `avg_berlin`。

加载前请校验两件事，任何一件不符就把整层留空，不要混用别的地图：

```js
if (doc.meta.crs !== 'replay-world-meters') return;
if (doc.meta.map !== map) return;
```

### 5.2 世界坐标 ↔ 等高线像素

等高线直接就是世界米坐标，渲染时按当前视野做线性映射即可。若需要回到高度图像素：

```
col = (worldX - bounds.minX) / cellSizeM - 0.5
row = (worldY - bounds.minY) / cellSizeM - 0.5     // 行 0 = minY
```

### 5.3 贴图像素 ↔ 世界坐标

贴图（`out/textures/`，不入库）行 0 对应 `maxY`，与高度图相反：

```
worldX = bounds.minX + (px + 0.5) * mppX
worldY = bounds.maxY - (py + 0.5) * mppY
```

其中 `mppX = (maxX - minX) / width`，`mppY = (maxY - minY) / height`，亚像素不确定度约 ±0.5 px。贴图 `worldBounds` 对陆战图只覆盖 2–4 km 战斗区域，不是整关范围。

### 5.4 经纬度

`dataset/geo/<map>.json` 的 `bounds` 是 WGS84 经纬度框。**先读 `confidence`**：只有 `high` / `medium` 可以当坐标用，`low` 只能当提示，`unresolved` 没有位置。`air_grand_canyon` 已确认是错误匹配，`ADVISORY.json` 里单独列出。

## 6. 陆水掩膜 `landmask/<map>.json`

给前端做"陆地着色、水面透明"用的二值区域，按高度图网格生成，和等高线同一套 world 坐标。

```jsonc
{
  "schema": "wt-landmask/1",
  "map": "air_skyscraper_city",
  "crs": "replay-world-meters",
  "bounds": { "minX": -32768.0, "maxX": 32800.03,
              "minY": -32768.0, "maxY": 32800.03 },
  "waterLevel": 0.0,                 // null = 没有低于 0 的高度，整图陆地
  "landFraction": 0.773744,
  "land": {
    "encoding": "rle-rows",
    "width": 1024, "height": 1024,
    "rowZero": "maxY",               // 行 0 = 世界最大 Y（北）
    "startsWith": 0,                 // 第一个游程的值（1 = 陆地，0 = 水）
    "runs": [ ... ]                  // 行主序游程长度，1/0 交替
  },
  "landRings": {
    "fillRule": "evenodd",
    "rings": [ [x0,y0, x1,y1, ...] ], // 闭合环，世界米
    "pointCount": 12345
  },
  "gridNote": "...",
  "waterLevelNote": "..."
}
```

解码规则：

- `land.rle`：按行主序、从 `(0,0)` 开始，从 `startsWith` 起 1/0 交替展开，`runs` 加起来必须等于 `width * height`；
- `landRings`：用 **even-odd** 填充，洞就是额外的环，不需要判断外环/内环；
- 采样一个世界坐标 `(x, y)`：

```js
const px = Math.floor((x - bounds.minX) / (bounds.maxX - bounds.minX) * width);
const py = Math.floor((bounds.maxY - y) / (bounds.maxY - bounds.minY) * height);
```

- 网格是高度图按整数倍降采样（最大 2048），像素中心与贴图对齐；
- 水线规则：`h <= waterLevel` 是水，等于水线的像素算水；
- 掩膜只按地形高度判断，高于海平面的河流/湖泊算陆地。

索引 `landmask/index.json` 汇总每张图的 `grid`、`landFraction`、`waterLevel`、`rings`、`bytes`。另有栅格产品（`<map>_land.png`，255 = 陆地）由 `make_landmask.py` 生成，需要先导出贴图以取得网格尺寸。

## 7. 机场 / 跑道 / 重生点 `airfields/<map>.json`

```jsonc
{
  "map": "air_skyscraper_city",
  "crs": "replay-world-meters",
  "frame": "level-lmap",              // 机场几何取自关卡 lmap 的坐标系
  "levelBounds": { ... },             // 整关范围
  "baseMapBounds": { ... },           // 底图（战斗区）范围
  "airfields": [
    {
      "name": "t1_airfield",
      "kind": "runway",
      "unitClass": "dynaf_universal_1line_a",
      "level": "levels/air_skyscraper_city.bin",
      "mission": "gamedata/missions/cta/planes/arcade/gs/air_skyscraper_city_gs.blk",
      "center": [-9519.9, -8942.5],   // [x, z]，米
      "headingDeg": 181.4,            // ends[0] → ends[1]
      "reciprocalDeg": 1.4,
      "takeoffHeadingDeg": 1.4,
      "spawnEnd": 1,
      "ends": [[-9500.0, -8107.7], [-9539.7, -9777.2]],
      "lengthM": 1670.0,
      "widthM": 140.0,
      "altM": 117.7,                  // 绝对世界海拔
      "source": "mission",
      "confidence": "high",
      "geometryFile": "gamedata/objectgroups/dynaf_universal_1line_a.blk",
      "missionFile": "gamedata/missions/...blk",
      "note": "..."
    }
  ],
  "airfieldsOutsideBaseMap": [ ... ],   // 底图外的跑道，单独列出
  "respawns": [
    { "kind": "airfield", "pos": [-9576.5, -9661.3], "altM": 117.7,
      "name": "t1_airfield", "unitClass": "dynaf_universal_1line_a",
      "level": "levels/air_skyscraper_city.bin", "mission": "..." }
  ],
  "respawnsOutsideBaseMap": [ ... ],
  "missions": [ ... ],
  "note": "..."
}
```

要点：

- `center` / `ends` / `pos` 都是 `[x, z]`，单位米，与等高线同一世界坐标；
- `ends[1]` 是起飞端，`takeoffHeadingDeg` 才是起飞方向（`headingDeg + 180`）；
- `headingDeg` 的 0° 是正北，顺时针；
- `altM` 是绝对海拔，不是相对地形的高度；
- `spawnEnd` 表示任务实际从哪一端起飞；
- `airfieldsOutsideBaseMap` / `respawnsOutsideBaseMap` 不混进主列表，渲染时按需取。

索引 `airfields/index.json` 汇总：`levels`、`levelsWithAirfields`、`airfields`、`airfieldsOutsideBaseMap`、`respawns`、`respawnsOutsideBaseMap`、`runwaysTotal`、`maps`、`noAirfieldData`（17 关，含机库与纯海图）、`unattributableMissions`（22 个任务无法归属到任何地图，原因逐条记录，没有猜）。

## 8. 经纬度参考 `geo/<map>.json`

```jsonc
{
  "map": "air_afghan",
  "crs": "WGS84",
  "bounds": { "minLon": 68.545161, "maxLon": 70.088667,
              "minLat": 34.208128, "maxLat": 35.483808 },
  "corners": { "nw": [68.545161, 35.483808], "ne": [...], "sw": [...], "se": [...] },
  "method": "ocr+gazetteer+dem-match",   // dem-only / dem-coarse-only / ocr+gazetteer+lsq / none
  "residualKm": 2.51,                    // 地名标注到拟合位置的残差
  "anchorResidualKm": 2.51,
  "bboxUncertaintyKm": 1.0,              // DEM 相关峰半高宽，判断能不能当坐标用
  "anchors": [
    { "text": "Charikar", "lat": 35.01361, "lon": 69.17139,
      "px": 884.5, "py": 736.5, "score": 100.0,
      "gazetteer": "charikar", "offsetKm": 1.71 }
  ],
  "crosscheck": { ... },                 // 水陆比例、水体掩膜等交叉校验
  "confidence": "high",                  // high | medium | low | unresolved
  "demNcc": 0.93,                        // 高度图与 DEM 的归一化互相关峰值
  "textureBounds": { ... },
  "note": "..."
}
```

索引 `geo/index.json` 额外记录 `beforeVeto`（否决前的 87 个解）、`byConfidence`、`byMethod`、`byReliefBucket`、`anchorResidualKm`、`bboxUncertaintyKm`、`demNccMedian`、`unresolvedReasons`、`rejected`。

**只有 `high` / `medium` 可以当坐标**；`low` 当提示；`unresolved` 没有位置。`geo/ADVISORY.json` 单独列出已知不可用的结果（例如 `air_grand_canyon`）。

## 9. 未随仓库发布的贴图与高度图

贴图（`out/textures/`，约 1 GB）与高度图（`out/*_heightmap.npz`，约 1.1 GB）体积大，而且属于游戏素材，因此不入库。它们可以随时在本机重新生成：

```powershell
python export_textures.py      # -> out/textures/*.png + index.json（--dds 同时导出原始 DXT）
python make_contours.py <map>  # -> out/<map>_heightmap.npz + 等高线 + 预览图
```

贴图索引 `out/textures/index.json` 的结构：

```jsonc
{
  "crs": "replay-world-meters",
  "maps": {
    "air_afghan": {
      "level": "levels/air_afghan.bin",
      "originX": -65536.0, "originY": -65536.0,
      "textures": {
        "map":       { "asset": "air_afghan_map$hq", "fmt": "DXT1",
                       "width": 2048, "height": 2048, "hq": true,
                       "metersPerPixelX": 64.0, "metersPerPixelY": 64.0,
                       "pack": "hq_tex_locations_maps", "png": "air_afghan_map_hq.png" },
        "map_thumb": { "width": 256, "height": 256, ... }
      }
    }
  },
  "packs": { "<pack>": { "file": "...", "exists": true } }
}
```

覆盖：160 个关卡有贴图，其中 147 个有可用作底图的 `map` 影像（144 张 2048²、3 张 4096²），13 个只有 `tankmap`；另有 87 张 `tankmap`、1 张 `planemap`、1 张 `airmap`，主图合计 241 张，全部 ≥ 2048²。4 个 `hangar_field*` 没有贴图。

## 10. 重新生成

```powershell
$env:WT_GAME_DIR  = "D:\Games\War Thunder"          # 含 levels/ 的目录
$env:WT_OODLE_DLL = "D:\path\to\daKernel-dev.dll"   # Oodle 内核（见 README）

python extract_all.py                 # 全部关卡 -> out/*_contours.json + *.npz
python validate_index.py              # 校验并写 out/index.json
python fix_contour_artifacts.py       # 过滤栅格空边伪等高线（需要 npz）
python drop_subwater_contours.py      # 整图位于水面以下的关卡不再出线
python check_contour_jumps.py         # 折线跳变自检

python export_textures.py             # 贴图 -> out/textures/
python make_landmask_json.py --all    # 陆水掩膜 -> out/landmask/*.json + index.json
python validate_landmask.py           # 校验掩膜（RLE/环/一致性），写 proof
python wt_airfields.py --all          # 机场 -> out/airfields/
python wt_geo.py all                  # 经纬度 -> out/geo/（需要联网下 DEM）
python make_geo_advisory.py           # 写 out/geo/ADVISORY.json
```

`out/` 是工作目录，已在 `.gitignore` 中。确认产物无误后，把它们同步成发布快照：

```powershell
Copy-Item out\*_contours.json dataset\contours\ -Force
Copy-Item out\index.json      dataset\contours\index.json -Force
Copy-Item out\landmask\*      dataset\landmask\ -Force
Copy-Item out\geo\*           dataset\geo\ -Force
Copy-Item out\airfields\*     dataset\airfields\ -Force
```

`dataset/` 只是快照，不参与生成；下游仓库按发布标签取它，而不是直接跟分支。

## 11. 来源与许可

- 数据由本仓库的代码从**本机安装的战雷客户端**解析得到，仓库不再分发游戏原始资源（贴图、高度图、关卡文件均不入库）；
- 原始素材版权归 Gaijin Entertainment；派生数据仅用于互操作与研究；
- 经纬度部分用到了 GeoNames（CC BY 4.0）与 SRTM / terrarium 高程瓦片，详见 `THIRD-PARTY-NOTICES.md`。
