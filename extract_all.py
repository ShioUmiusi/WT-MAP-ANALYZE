"""Batch-extract contours for every War Thunder level.

Delegates to make_contours.py (the tested CLI path) so there is only one implementation.

Both height sources are supported:
  * HM2 block          -> exact heightmap
  * no HM2 (legacy)    -> terrain rebuilt from the lmap LandMesh TIN

Levels with neither (e.g. the all-water `water.bin`) are reported as skipped, not errors.
Already-generated outputs are kept unless --force is passed.
"""
import os
import subprocess
import sys
import time

from wt_paths import levels_dir

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def main():
    os.makedirs(OUT, exist_ok=True)
    force = "--force" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("-")]

    names = sorted(f for f in os.listdir(levels_dir()) if f.endswith(".bin"))
    if only:
        names = [n for n in names if os.path.splitext(n)[0] in only]

    ok = skip = fail = 0
    t_all = time.time()
    for i, name in enumerate(names, 1):
        base = os.path.splitext(name)[0]
        target = os.path.join(OUT, base + "_contours.json")
        if os.path.exists(target) and not force:
            skip += 1
            continue
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, os.path.join(HERE, "make_contours.py"), base],
            cwd=HERE, capture_output=True, text=True,
        )
        dt = time.time() - t0
        if r.returncode == 0:
            ok += 1
            line = (r.stdout or "").strip().splitlines()
            info = line[-1] if line else ""
            print(f"[{i:3d}/{len(names)}] {base:30s} OK   {dt:5.1f}s  {info[:90]}", flush=True)
        elif r.returncode == 2:
            skip += 1
            msg = (r.stderr or "").strip().splitlines()
            print(f"[{i:3d}/{len(names)}] {base:30s} SKIP {msg[0][:70] if msg else ''}", flush=True)
        else:
            fail += 1
            print(f"[{i:3d}/{len(names)}] {base:30s} FAIL rc={r.returncode}", flush=True)
            err = (r.stderr or "").strip().splitlines()
            for e in err[-4:]:
                print(f"        {e}", flush=True)

    print(f"\ndone: ok={ok} skipped={skip} fail={fail} in {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
