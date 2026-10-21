# War Thunder 地图分析 / WT-MAP-ANALYZE

从《战争雷霆》的 `levels/*.bin` 关卡文件离线解出地形，输出等高线 JSON、地图贴图、机场/跑道、真实经纬度参考。仓库同时附带一份已经算好的数据集（163 张地图的等高线等），可以直接被地图工具或回放播放器读取。

[English](#english) · [数据格式](docs/FORMAT.md) · [数据集](docs/DATASET.md) · [方法与验证](docs/METHOD-VALIDATION.md)

## 能做什么

| 能力 | 产物 | 入口 |
|---|---|---|
| 高度图解码（HM2 / LandMesh） | `out/<map>_heightmap.npz` | `wt_hmap.py`、`wt_landmesh.py`、`wt_maps.py` |
| 等高线 | `out/<map>_contours.json` + 预览 PNG | `make_contours.py`、`extract_all.py` |
| 陆水掩膜（海岸线 / 水体） | `out/landmask/<map>.json` | `make_landmask_json.py`、`validate_landmask.py` |
| 地图贴图 | `out/textures/*.png` | `export_textures.py`、`wt_textures.py` |
| 机场 / 跑道 / 重生点 | `out/airfields/*.json` | `wt_airfields.py` |
| 真实经纬度参考 | `out/geo/*.json` | `wt_geo.py` |
| 已发布数据集 | `dataset/contours`、`dataset/landmask`、`dataset/geo`、`dataset/airfields` | [数据集说明](docs/DATASET.md) |

坐标统一为 `replay-world-meters`：`x` 向东、`y` 向北、单位米，与回放/8111 工具的世界坐标一致。

## 目录

```
warthunder-maps/
  wt_*.py  datalm.py  make_contours.py  extract_all.py  …   解码与提取脚本
  docs/                                                     格式、数据集、方法与验证
  dataset/contours/                                         163 张等高线 + index.json
  dataset/landmask/                                         163 张陆水掩膜 + index.json
  dataset/geo/                                              经纬度参考
  dataset/airfields/                                        机场 / 跑道 / 重生点
  out/                                                      本机生成的工作目录（不提交）
```

## 快速开始

要求 Windows（Oodle 内核 DLL 是 Windows 库）、Python 3.10+。

```powershell
pip install -r requirements.txt

# 指向本机游戏目录（包含 levels/ 的那一层）
$env:WT_GAME_DIR  = "D:\Games\War Thunder"
# Oodle 内核 DLL，见下一节；也可以放到仓库的 vendor\daKernel-dev.dll
$env:WT_OODLE_DLL = "D:\path\to\daKernel-dev.dll"

python make_contours.py avg_berlin          # 单张图：等高线 + 高度图 + 预览图
python extract_all.py                       # 全部关卡（已有输出默认跳过，--force 重做）
python wt_airfields.py --all                # 机场/跑道/重生点
python wt_geo.py all                        # 经纬度（需要联网下 DEM 瓦片）
```

生成的原始产物在 `out/`（已在 `.gitignore` 中）；`dataset/` 是随仓库发布的快照，两者内容格式一致。

### Oodle 内核

高度图与贴图用 Oodle 压缩，Oodle 没有开源解码器。这里复用 [Dagor Asset Explorer](https://github.com/quentin-dh/Dagor-Asset-Explorer) 构建里带的 `daKernel-dev.dll`，通过导出序号 574 调用 `oodle_decompress`。该 DLL 不在本仓库分发，需要自行获取并用 `WT_OODLE_DLL` 指定路径。`max_size` 必须等于解压后大小，差一个字节都会失败，所以代码先按格式公式算出预期大小。

## 脚本

| 文件 | 作用 |
|---|---|
| `wt_hmap.py` | DBLD 容器 + HM2 高度图解码 |
| `datalm.py` | LandMesh（lmap）参考解码器 |
| `wt_landmesh.py` | LandMesh 生产封装：TIN → 规则高度网格 |
| `wt_maps.py` | 统一入口：HM2 优先，LandMesh 回退 |
| `make_contours.py` | 单张图生成等高线 JSON / npz / PNG |
| `extract_all.py` | 批量生成 |
| `validate_index.py` | 校验全部 JSON 并写 `out/index.json` |
| `fix_bounds.py`、`fix_contour_artifacts.py`、`drop_subwater_contours.py`、`check_contour_jumps.py` | 数据集修整与自检 |
| `make_landmask.py`、`make_landmask_json.py`、`validate_landmask.py` | 陆水掩膜（栅格 / JSON）与一致性校验 |
| `wt_textures.py`、`export_textures.py`、`validate_textures.py` | DxP2 贴图包读取、导出、对齐校验 |
| `wt_airfields.py` | vromfs / 任务 DataBlock → 机场、跑道、重生点 |
| `wt_geo.py`、`make_geo_advisory.py` | 地名 OCR + DEM 配准 → 经纬度参考 |
| `wt_paths.py` | 本机路径（游戏目录、Oodle DLL）的读取 |

## 数据覆盖

164 个关卡中 163 个有地形：90 个直接来自 HM2，73 个旧格式关卡由 LandMesh 重建。重建精度用 90 张同时含两种来源的图交叉验证，中位绝对误差 0.14 m。163 张图另有陆水掩膜，按高度图的 `waterLevel` 划出陆地与水面，供前端做海岸线着色。细节与限制见[方法与验证](docs/METHOD-VALIDATION.md)。

## 许可与来源

- 代码：MIT，见 [LICENSE](LICENSE)；
- 仓库里的数据是从本机安装的游戏解析出的派生数据，不再分发游戏原始资源（关卡文件、高度图、贴图均不入库）；
- 《战争雷霆》及其素材版权归 Gaijin Entertainment，本项目与 Gaijin 无关；
- 第三方依赖与数据来源见 [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)。

---

## English

Offline extraction of War Thunder level terrain. It decodes the `DBLD` level containers, `HM2` compressed heightmaps and legacy `lmap` LandMesh terrain, then produces contour JSON, minimap textures, airfield/runway geometry and a real-world georeference for every level.

- Coordinates: `replay-world-meters`, x east / y north, metres.
- Coverage: 164 levels, 163 with terrain (90 from HM2, 73 rebuilt from LandMesh; median error 0.14 m over 90 cross-checked levels).
- The `dataset/` directory ships the computed contour JSON (163 maps), georeference and airfield data so downstream tools can load them without the game.
- Land/water masks are exported per level as RLE bitmaps plus closed rings (even-odd fill), so a map view can shade land and keep water transparent without the full terrain mesh.
- Requirements: Windows, Python 3.10+, `pip install -r requirements.txt`, plus the Oodle kernel DLL (`daKernel-dev.dll`, not redistributed).
- Set `WT_GAME_DIR` to the War Thunder install folder and `WT_OODLE_DLL` to the DLL, then run `python make_contours.py avg_berlin` or `python extract_all.py`.

`docs/FORMAT.md` documents the container layouts, `docs/DATASET.md` the data format, and `docs/METHOD-VALIDATION.md` the validation results and limitations.

Code is MIT licensed. The derived data is produced from a locally installed copy of the game; no original game assets are redistributed. War Thunder and its assets are property of Gaijin Entertainment; this project is not affiliated with Gaijin.
