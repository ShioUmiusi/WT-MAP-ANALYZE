"""Export every War Thunder level's minimap/base-map texture offline.

The running game serves these through 8111 `/map.img`; the same images ship in the
`DxP2` texture packs, so they can be exported without the game running.

    content/base/res/locations_maps.dxp.bin              <asset>_map        1024x1024 (4 at 4096)
                                                        <asset>_map_thumb   256x256
    content.hq/hq_tex/res/hq_tex_locations_maps.dxp.bin  <asset>_map$hq     2048x2048
    content/base/res/locations_maps-hq.dxp.bin           two extra 2048x2048
    patch/content/base/res/locations_maps.dxp.bin        patched overrides (4096)
    content/pkg_infantry/res/pkg_infantry_level_maps.dxp.bin        1024x1024
    content.hq/pkg_infantry_hq/res/pkg_infantry_hq_level_maps.dxp.bin  4096/2048

Usage:
    python export_textures.py                 # everything missing (resumable)
    python export_textures.py --force         # re-render all
    python export_textures.py --only avg_berlin air_afghan
    python export_textures.py --max-dim 1024  # downscale output
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wt_textures import TexturePack
from wt_paths import game_dir

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "textures")


def packs():
    """DxP2 packs holding map images: (label, path) in override order."""
    game = game_dir()
    return [
        ("locations_maps", os.path.join(game, "content", "base", "res",
                                        "locations_maps.dxp.bin")),
        ("locations_maps_hq_extra", os.path.join(game, "content", "base", "res",
                                                 "locations_maps-hq.dxp.bin")),
        ("hq_tex_locations_maps", os.path.join(game, "content.hq", "hq_tex", "res",
                                               "hq_tex_locations_maps.dxp.bin")),
        ("pkg_infantry_level_maps", os.path.join(game, "content", "pkg_infantry", "res",
                                                 "pkg_infantry_level_maps.dxp.bin")),
        ("pkg_infantry_hq_level_maps", os.path.join(game, "content.hq", "pkg_infantry_hq",
                                                    "res", "pkg_infantry_hq_level_maps.dxp.bin")),
        ("locations_maps_patch", os.path.join(game, "patch", "content", "base", "res",
                                              "locations_maps.dxp.bin")),
    ]

# suffixes that denote a playable top-down map image, most specific first
SUFFIXES = ("_map_thumb", "_tankmap", "_planemap", "_airmap", "_minimap", "_colormap", "_map")

# level stem -> asset stem, for levels whose texture asset is named differently
ALIASES = {
    "avg_maginot_rework": "avg_maginot",
    "avn_new_zeland_cape": "avn_new_zealand",
    "avg_lazzaro_italy_new_city": "avg_lazzaro_italy",
    "avg_hurtgen": "avg_hurtgen",
    "avg_korea_lake": "avg_korea",
    "avg_karpaty_passage": "avg_karpaty",
    "avg_port_novorossiysk": "avg_nvrsk",
    "avg_nuclear_incident": "avg_nuclear",
    "tunisia": "avg_tunisia_desert",
    "sector_montmedy": "avg_sector_montmedy",
    "hvg_granitograd": "modern_soviet_granitograd",
    "hvg_al_massa": "hvg_al_massa",
}


def level_stems():
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(game_dir(), "levels", "*.bin")))


def build_asset_index(verbose=True):
    """asset_name -> {pack, index, header, TexturePack} for the best (largest) variant."""
    packs = {}
    best = {}
    # a later pack in packs() overrides, but resolution wins over order when strictly larger
    for pack_name, path in packs():
        if not os.path.exists(path):
            if verbose:
                print(f"  [skip] missing pack {pack_name}: {path}")
            continue
        t0 = time.time()
        pk = TexturePack(path)
        packs[pack_name] = pk
        n_new = 0
        for i in range(len(pk)):
            asset = pk.asset_name(i)
            h = pk.headers[i]
            if h.w == 0 or h.h == 0:
                continue
            cur = best.get(asset)
            if cur is None or h.w * h.h >= cur["w"] * cur["h"]:
                best[asset] = dict(asset=asset, pack_name=pack_name, pack=pack_name,
                                   index=i, pack_obj=pk, w=h.w, h=h.h,
                                   fmt=h.fmt_name.decode("latin1"),
                                   compression=h.compression)
                n_new += 1
        if verbose:
            print(f"  loaded {pack_name}: {len(pk)} textures, {n_new} kept, "
                  f"{os.path.getsize(path)/1e6:.0f} MB, {time.time()-t0:.2f}s")
    return best, packs


def split_suffix(asset):
    for suf in SUFFIXES:
        if asset.endswith(suf) and len(asset) > len(suf):
            return asset[: -len(suf)], suf
    # $hq suffix on a map family, e.g. air_afghan_map$hq
    if asset.endswith("$hq"):
        base, suf = split_suffix(asset[: -len("$hq")])
        if base:
            return base, suf
    return None, None


def resolve_assets(stem, by_stem):
    """Return {suffix: [asset names sorted best-first]} for one level stem."""
    cands = [stem]
    if stem in ALIASES:
        cands.append(ALIASES[stem])
    # try progressively looser forms
    for pre in ("avg_", "air_", "avn_", "arcade_", "hvg_"):
        if stem.startswith(pre):
            cands.append(stem[len(pre):])
    res = {}
    for c in cands:
        if c in by_stem:
            for suf, assets in by_stem[c].items():
                res.setdefault(suf, []).extend(a for a in assets if a not in res.get(suf, []))
            break
    return res


def safe_name(asset):
    """Filename-safe form of a Dagor asset name ('$hq' would break many shells/URLs)."""
    return asset.replace("$", "_")


def _write_dds(rec, out_png, force=False):
    """Emit the raw DXT .dds next to the PNG (exact source bytes, ~8x smaller)."""
    out_dds = os.path.splitext(out_png)[0] + ".dds"
    if os.path.exists(out_dds) and not force:
        return
    try:
        data = rec["pack_obj"].to_dds(rec["index"])
        with open(out_dds, "wb") as f:
            f.write(data)
    except Exception as e:
        print(f"    dds failed {rec['asset']}: {e}")


def world_bounds(stem):
    """(minX, maxX, minY, maxY) in replay-world meters, from the contour JSON we already ship."""
    p = os.path.join(HERE, "out", f"{stem}_contours.json")
    if os.path.exists(p):
        try:
            meta = json.load(open(p, "r", encoding="utf-8"))["meta"]
            b = meta["bounds"]
            return (b["minX"], b["maxX"], b["minY"], b["maxY"]), meta
        except Exception:
            pass
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--max-dim", type=int, default=0)
    ap.add_argument("--no-thumb", action="store_true")
    ap.add_argument("--dds", action="store_true",
                    help="also emit the raw DXT .dds (8x smaller than decoded PNG, exact source)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("indexing texture packs ...")
    best, packs = build_asset_index()

    by_stem = {}
    for asset, rec in best.items():
        base, suf = split_suffix(asset)
        if base is None:
            continue
        by_stem.setdefault(base, {}).setdefault(suf, []).append(asset)
    for base in by_stem:
        for suf in by_stem[base]:
            by_stem[base][suf].sort(key=lambda a: -best[a]["w"] * best[a]["h"])

    levels = level_stems()
    if args.only:
        levels = [l for l in levels if l in args.only]

    index_path = os.path.join(OUT, "index.json")
    index = {}
    if os.path.exists(index_path):
        try:
            index = json.load(open(index_path, "r", encoding="utf-8")).get("maps", {})
        except Exception:
            index = {}

    t_all = time.time()
    nmade = nskip = nfail = 0
    unresolved = []
    for k, stem in enumerate(levels, 1):
        assets = resolve_assets(stem, by_stem)
        if not assets:
            unresolved.append(stem)
            print(f"[{k}/{len(levels)}] {stem}: no texture asset")
            continue
        entry = {"map": stem, "level": f"levels/{stem}.bin", "textures": {}}
        bounds, meta = world_bounds(stem)
        if bounds:
            entry["worldBounds"] = dict(minX=bounds[0], maxX=bounds[1], minY=bounds[2], maxY=bounds[3])
            if meta:
                entry["originX"] = meta.get("originX")
                entry["originY"] = meta.get("originY")
        for suf, names in assets.items():
            if args.no_thumb and suf in ("_map_thumb",):
                continue
            asset = names[0]
            rec = best[asset]
            out_png = os.path.join(OUT, safe_name(asset) + ".png")
            key = suf.lstrip("_")
            info = {"asset": asset, "png": os.path.basename(out_png),
                    "width": rec["w"], "height": rec["h"], "fmt": rec["fmt"],
                    "pack": rec["pack_name"],
                    "hq": asset.endswith("$hq") or rec["w"] > 1024}
            if bounds:
                info["metersPerPixelX"] = round((bounds[1] - bounds[0]) / rec["w"], 4)
                info["metersPerPixelY"] = round((bounds[3] - bounds[2]) / rec["h"], 4)
            entry["textures"][key] = info
            if os.path.exists(out_png) and not args.force:
                nskip += 1
                if args.dds:
                    _write_dds(rec, out_png, args.force)
                continue
            try:
                t0 = time.time()
                im = rec["pack_obj"].to_pil(rec["index"])
                if args.max_dim and max(im.size) > args.max_dim:
                    im = im.resize((args.max_dim, args.max_dim))
                im.save(out_png)
                nmade += 1
                print(f"[{k}/{len(levels)}] {stem}: {key} <- {asset} "
                      f"{rec['w']}x{rec['h']} {os.path.getsize(out_png)/1e6:.2f} MB "
                      f"{time.time()-t0:.2f}s")
                if args.dds:
                    _write_dds(rec, out_png, args.force)
            except Exception as e:
                nfail += 1
                print(f"[{k}/{len(levels)}] {stem}: FAILED {asset}: {e}")
        index[stem] = entry

    doc = {
        "crs": "replay-world-meters",
        "note": (
            "Pixel grid is top-left origin, north up. Verified against out/<map>_contours.json "
            "for avg_berlin / air_afghan / air_denmark / air_grand_canyon / avg_football_field / "
            "avg_training_ground: the texture covers worldBounds exactly, so with "
            "mppX=(maxX-minX)/width, mppY=(maxY-minY)/height: "
            "worldX = minX + (px+0.5)*mppX, worldY = maxY - (py+0.5)*mppY. "
            "Sub-pixel uncertainty is +-0.5 px. `_map` is used for air maps, `_tankmap` for "
            "ground maps; `$hq` variants are the high-resolution ones (2048 or 4096 px)."
        ),
        "packs": {name: {"file": os.path.basename(path), "exists": os.path.exists(path)}
                  for name, path in packs()},
        "maps": index,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(f"\nwrote {index_path}: {len(index)} maps")
    print(f"png written {nmade}, skipped {nskip}, failed {nfail}, "
          f"unresolved {len(unresolved)} {unresolved}")
    print(f"total {time.time()-t_all:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
