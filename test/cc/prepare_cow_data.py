#!/usr/bin/env python3
"""Builds the fine (960x1280) and coarse (240x320) rho fields for the paper's cow-image
continuous-cuts problem (examples/compare_variants.py in RMO-continuous-cuts), so the C++
port (src/main_cc_cow.cc) can run on the actual problem from Sec. 6.3 instead of only the
41x41 synthetic disk.

This replicates bernoulli_multilevel.problem.prepare_multilevel_cc's preprocessing exactly
(grayscale, 2x bilinear enlarge, Gaussian blur sigma=1, foreground/background seed-rectangle
means, two composed factor-2 average-pool steps for the coarse level) using numpy/skimage
directly rather than importing that function, since only the raw rho arrays are needed here,
not the torch objective closures prepare_multilevel_cc builds around them. Checked against
the reference's actual torch call (avg_pool2d, which pools in float32 -- problem.py casts to
.float() before pooling and back to .double() after) on the real cow image: max abs diff
1.2e-7, well below anything that matters for a double-precision solve.

The C++ side has no JPEG decoder, resizer, or Gaussian blur of its own -- decode/resize/blur
are delegated here to skimage, the same code path the paper's own figures were produced
with (RMO-continuous-cuts/src/bernoulli_multilevel/problem.py), rather than risking a
fresh reimplementation silently diverging from it. What runs in C++ is the actual
multilevel/single-level optimization under test (src/main_cc_cow.cc), on real data.

Usage:
    python3 prepare_cow_data.py /path/to/RMO-continuous-cuts [--out-dir DIR]

Writes raw, row-major float64 binary files (no header, native byte order) to --out-dir
(default: test/cc/data next to this script): rho_fine_960x1280.bin (the finest level, used
by every hierarchy depth), rho_coarse_240x320.bin (the 2-level hierarchy's single composed
coarse level, n_pools=2), and rho_480x640.bin / rho_120x160.bin (the intermediate levels the
3- and 4-level hierarchies insert, n_pools=1 and 3 respectively -- examples/levels_2_3_4.py's
own "best configurations", one factor-2 average-pool step apart from their neighbors).
"""
import argparse
import pathlib
import sys

import numpy as np


# Reproduces examples/compare_variants.py's configuration for the paper's cow experiment.
ENLARGE = 2
FG_SEED = (180, 210, 180, 300)   # (r0, r1, c0, c1) in ORIGINAL-image pixel coordinates
BG_SEED = (9, 86, 448, 614)


def avg_pool2(a: np.ndarray) -> np.ndarray:
    """Disjoint 2x2 block mean (kernel=2, stride=2); a's dimensions must be even."""
    h, w = a.shape
    assert h % 2 == 0 and w % 2 == 0, f"avg_pool2 needs even dimensions, got {a.shape}"
    return a.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="path to the RMO-continuous-cuts checkout")
    ap.add_argument("--out-dir", default=None, help="output directory (default: <this file's dir>/data)")
    args = ap.parse_args()

    sys.path.insert(0, args.reference.rstrip("/") + "/src")   # for bernoulli_multilevel, if needed later
    from skimage import color, filters, io
    from skimage.transform import resize as sk_resize

    image_path = args.reference.rstrip("/") + "/examples/data/cowdata.jpg"

    img_orig = io.imread(image_path)
    if img_orig.ndim == 3:
        img_orig = color.rgb2gray(img_orig)
    h0, w0 = img_orig.shape
    img_fine = sk_resize(img_orig, (h0 * ENLARGE, w0 * ENLARGE),
                         order=1, anti_aliasing=False, preserve_range=True)
    h_fine, w_fine = img_fine.shape
    scale = float(ENLARGE)

    fg_r0, fg_r1, fg_c0, fg_c1 = FG_SEED
    bg_r0, bg_r1, bg_c0, bg_c1 = BG_SEED
    fg_mask = np.zeros((h_fine, w_fine), dtype=bool)
    fg_mask[int(fg_r0 * scale):int(fg_r1 * scale), int(fg_c0 * scale):int(fg_c1 * scale)] = True
    bg_mask = np.zeros((h_fine, w_fine), dtype=bool)
    bg_mask[int(bg_r0 * scale):int(bg_r1 * scale), int(bg_c0 * scale):int(bg_c1 * scale)] = True

    f_img = filters.gaussian(img_fine, sigma=1.0)
    u_f = np.mean(f_img[fg_mask])
    u_b = np.mean(f_img[bg_mask])
    rho_fine = ((u_f - f_img) ** 2 - (u_b - f_img) ** 2).astype(np.float64)

    # Strict pyramid: each level is one more factor-2 average-pool step than the last
    # (associative, so pooling the previous level's rho by one more step is exactly the
    # same as pooling rho_fine by the cumulative count directly, as problem.py does).
    # n_pools=1 -> 480x640 (examples/levels_2_3_4.py's "+ Original" level, coinciding with
    # the un-enlarged image's own resolution), n_pools=2 -> 240x320 (the 2-level hierarchy's
    # single composed coarse level), n_pools=3 -> 120x160 (4-level's "Coarse-2 deeper").
    rho_480 = avg_pool2(rho_fine)
    rho_240 = avg_pool2(rho_480)
    rho_120 = avg_pool2(rho_240)

    for name, rho in [("rho_fine", rho_fine), ("rho_480", rho_480), ("rho_240", rho_240), ("rho_120", rho_120)]:
        print(f"{name:9s} {rho.shape}   min={rho.min(): .6f}  max={rho.max(): .6f}")

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = {
        f"rho_fine_{h_fine}x{w_fine}.bin": rho_fine,
        "rho_480x640.bin": rho_480,
        "rho_coarse_240x320.bin": rho_240,   # kept for main_cc_cow.cc's existing filename
        "rho_120x160.bin": rho_120,
    }
    for filename, rho in files.items():
        path = out_dir / filename
        rho.astype("<f8").tofile(path)
        print(f"wrote {path}  ({rho.nbytes} bytes)")


if __name__ == "__main__":
    main()
