# 第三方组件与数据来源

本仓库的代码以 MIT 发布。下面列出运行或生成数据时用到的第三方组件与数据源；它们各自保留原许可。

## Python 依赖

| 组件 | 用途 | 许可 |
|---|---|---|
| NumPy | 数组与栅格运算 | BSD-3-Clause |
| SciPy | 形态学、优化、插值 | BSD-3-Clause |
| scikit-image | marching squares 等值线提取 | BSD-3-Clause |
| Pillow | 图像读写 | MIT-CMU（HPND） |
| Matplotlib | 预览图渲染 | Matplotlib License（PSF 风格） |
| zstandard | ZSTD 解压 | BSD-3-Clause |
| opencv-python | 图像处理 | Apache-2.0 |
| rapidocr-onnxruntime | 贴图地名 OCR | Apache-2.0 |
| rapidfuzz | 地名模糊匹配 | MIT |

## 格式参考

- [GaijinEntertainment/DagorEngine](https://github.com/GaijinEntertainment/DagorEngine)：容器、高度图、LandMesh、DxP2、vromfs 的公开源码。只读源码做格式对齐，没有复制代码。
- [quentin-dh/Dagor-Asset-Explorer](https://github.com/quentin-dh/Dagor-Asset-Explorer)：Oodle 调用的参考实现。其构建产物里的 `daKernel-dev.dll` 是 Oodle 内核，**不在本仓库分发**，需要自行获取。

## 地理数据

| 数据 | 用途 | 来源与许可 |
|---|---|---|
| GeoNames `allCountries` | 地名匹配 | https://www.geonames.org/ ，CC BY 4.0 |
| SRTM 1″（NASA / USGS） | 地形互相关 | 公有领域 |
| AWS Terrain Tiles（Mapzen terrarium） | 全球粗筛 | https://registry.opendata.aws/terrain-tiles/ ，CC BY 4.0 |

## 游戏素材

《战争雷霆》（War Thunder）的关卡、高度图、贴图、任务文件版权归 Gaijin Entertainment 所有，本项目与 Gaijin 无关。

仓库只包含从本机安装的游戏解析出的**派生数据**（等高线、机场坐标、经纬度参考），不包含关卡文件、高度图、贴图或任何游戏原始资源。贴图导出功能只在本机运行并写入本地 `out/` 目录，这些文件不会进入仓库。
