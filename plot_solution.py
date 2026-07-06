#!/usr/bin/env python3
"""Rasterize and plot solution vectors serialized by write_support_points()/
write_solution() (include/gpe/util/serialize.h), instead of relying on
deal.II's SVG output which writes one vector path per mesh cell and does not
scale to large DoF counts.

The DoF coordinates and values (1D, 2D or 3D) are interpolated onto a
fixed-resolution regular grid -- this, not the DoF count, is what bounds the
plot size. 1D solutions are drawn as a line, 2D as a seaborn heatmap, and 3D
as three orthogonal mid-plane heatmap slices (xy/xz/yz), since there is no
native 3D array support in seaborn. Two solutions (e.g. from different mesh
levels or iterations, at possibly different resolutions/dimensions) can be
compared via a log10 |difference| plot on the same target grid.

Examples
--------
Plot a single solution:
    python3 plot_solution.py solution_2d_sl_b100_lvl1_coords.bin \\
        solution_2d_sl_b100_lvl1_iter3.bin --out iter3.png

Compare against a reference solution (possibly from a different level, hence
a separate coordinates file):
    python3 plot_solution.py solution_2d_ml_b100_lvl4_coords.bin \\
        solution_2d_ml_b100_lvl4_iter5.bin \\
        --reference solution_2d_sl_b100_lvl6_iter25.bin \\
        --reference-coords solution_2d_sl_b100_lvl6_coords.bin \\
        --out iter5.png

Fix the colorbar range (e.g. to [0, 0.3]) so several solutions plotted this
way, or a run's own diff plots, share the same color scale and are directly
comparable -- otherwise each plot auto-scales to its own min/max:
    python3 plot_solution.py solution_2d_ml_b100_lvl4_coords.bin \\
        solution_2d_ml_b100_lvl4_iter5.bin --vmin 0 --vmax 0.3 \\
        --diff-vmin -6 --diff-vmax 0 --out iter5.png

Replace the default plot title, and floor the diff plot above discretization
noise instead of showing its full dynamic range down to ~1e-300 (--title
applies to both the main plot and, if --reference is given, the diff plot,
unless --diff-title is also given to override it just for the latter):
    python3 plot_solution.py solution_2d_ml_b100_lvl4_coords.bin \\
        solution_2d_ml_b100_lvl4_iter5.bin --reference solution_2d_ml_b100_lvl4_iter0.bin \\
        --title "GP ground state (beta=100, lvl=4, iter=5)" \\
        --diff-title "Error vs. iter=0" --eps 1e-8 --out iter5.png
"""
import argparse
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import griddata

# File-signature constants, mirroring include/gpe/util/serialize.h.
#
# Each is a 4-byte ASCII tag ("GEPC" / "GEPS", short for "GPE Coords" /
# "GPE Solution" with the middle two letters swapped -- purely a mnemonic,
# not an acronym) packed as a little-endian uint32. The C++ side writes the
# integer's raw bytes with reinterpret_cast, so on a little-endian machine
# the bytes that actually land in the file are, in order:
#     struct.pack("<I", 0x43504547) == b"GEPC"
#     struct.pack("<I", 0x53504547) == b"GEPS"
# This is the same trick as PNG's byte-signature or ELF's 0x7F 'E' 'L' 'F':
# a fixed first few bytes that let a reader immediately reject the wrong
# file type (e.g. a coords file passed where a solution file was expected,
# or an unrelated/corrupted file) instead of misinterpreting its bytes as
# floating-point data.
MAGIC_COORDS = 0x43504547
MAGIC_SOLUTION = 0x53504547

# griddata's 3D path builds a Qhull Delaunay tessellation of the source point
# cloud, which gets expensive fast; default to a much coarser grid than 1D/2D
# unless the caller overrides --resolution explicitly.
DEFAULT_RESOLUTION = {1: 1024, 2: 512, 3: 64}


def load_coords(path):
    """Read a *_coords.bin file written by write_support_points().

    Layout: uint32 magic | uint32 dim | uint64 n_dofs | n_dofs*dim doubles,
    row-major (x0,y0,[z0,] x1,y1,[z1,] ...).

    Returns an (n_dofs, dim) array, one row per DoF support point.
    """
    with open(path, "rb") as f:
        magic, dim = struct.unpack("<II", f.read(8))
        if magic != MAGIC_COORDS:
            raise ValueError(f"{path}: not a coordinates file (bad magic)")
        (n_dofs,) = struct.unpack("<Q", f.read(8))
        data = np.fromfile(f, dtype="<f8", count=n_dofs * dim)
    return data.reshape(n_dofs, dim)


def load_solution(path):
    """Read a *_iterN.bin file written by write_solution().

    Layout: uint32 magic | uint64 n_dofs | n_dofs doubles.

    Returns a 1D array of length n_dofs, in the same DoF order as the
    corresponding load_coords() array (same level/mesh).
    """
    with open(path, "rb") as f:
        (magic,) = struct.unpack("<I", f.read(4))
        if magic != MAGIC_SOLUTION:
            raise ValueError(f"{path}: not a solution file (bad magic)")
        (n_dofs,) = struct.unpack("<Q", f.read(8))
        data = np.fromfile(f, dtype="<f8", count=n_dofs)
    return data


def _fill_nan_nearest(coords, values, query_points, grid_z):
    """griddata's default 'linear' method leaves points outside the convex
    hull of coords as NaN; fill those in with nearest-neighbor values so the
    whole grid is covered."""
    nan_mask = np.isnan(grid_z)
    if nan_mask.any():
        grid_z[nan_mask] = griddata(coords, values, query_points[nan_mask], method="nearest")
    return grid_z


def rasterize_1d(coords, values, resolution):
    """Interpolate scattered 1D (x,) points onto `resolution` equally-spaced
    points spanning [min(x), max(x)].

    coords: (n, 1) array. values: (n,) array.
    Returns {"dim": 1, "x": (resolution,) grid coordinates, "z": (resolution,)
    interpolated values}, consumed by plot_line()/plot_field().
    """
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    x_grid = np.linspace(x_min, x_max, resolution)
    z_grid = griddata(coords[:, 0], values, x_grid, method="linear")
    z_grid = _fill_nan_nearest(coords[:, 0], values, x_grid, z_grid)
    return {"dim": 1, "x": x_grid, "z": z_grid}


def rasterize_2d(coords, values, resolution):
    """Interpolate scattered (x, y) points onto a resolution x resolution
    regular grid spanning the bounding box of coords.

    coords: (n, 2) array. values: (n,) array.
    Returns {"dim": 2, "grid": (resolution, resolution) array}, where
    grid[i, j] corresponds to (x=grid_x[i,j], y=grid_y[i,j]) with row index
    increasing along y and column index along x (see plot_heatmap() for the
    orientation fix-up this implies when displaying it as an image).
    """
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0)
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid_z = griddata(coords, values, (grid_x, grid_y), method="linear")
    grid_z = _fill_nan_nearest(coords, values, (grid_x, grid_y), grid_z)
    return {"dim": 2, "grid": grid_z}


def rasterize_3d(coords, values, resolution):
    """Interpolate scattered (x, y, z) points onto a resolution^3 regular
    grid spanning the bounding box of coords, via a Delaunay tessellation of
    coords (scipy.interpolate.griddata, method="linear"). This is the most
    expensive of the three rasterize_*() functions -- cost grows with both
    the number of source points and resolution^3 -- hence the low default
    resolution picked for 3D in DEFAULT_RESOLUTION.

    coords: (n, 3) array. values: (n,) array.
    Returns {"dim": 3, "grid": (resolution,)*3 array, "axes": the three 1D
    per-dimension coordinate arrays used to build the grid (indexing="ij", so
    grid[i, j, k] corresponds to (axes[0][i], axes[1][j], axes[2][k]))}.
    Consumed by plot_slices()/plot_field(), which slices this cube along each
    axis at its midpoint since there is no direct way to plot a volume with
    seaborn.
    """
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    axes = [np.linspace(mins[d], maxs[d], resolution) for d in range(3)]
    grid_x, grid_y, grid_z_coord = np.meshgrid(*axes, indexing="ij")
    query_points = (grid_x, grid_y, grid_z_coord)

    grid_z = griddata(coords, values, query_points, method="linear")
    grid_z = _fill_nan_nearest(coords, values, query_points, grid_z)
    return {"dim": 3, "grid": grid_z, "axes": axes}


def rasterize(coords, values, resolution):
    """Interpolate a (possibly unstructured) point cloud onto a fixed
    resolution regular grid. This is what bounds the output size instead of
    the DoF count.

    Dispatches on coords.shape[1] to rasterize_1d/2d/3d(); see those for the
    exact shape of the returned dict (always has a "dim" key, plus "x"/"z"
    for 1D or "grid" [+ "axes" for 3D] otherwise).
    """
    dim = coords.shape[1]
    if dim == 1:
        return rasterize_1d(coords, values, resolution)
    elif dim == 2:
        return rasterize_2d(coords, values, resolution)
    elif dim == 3:
        return rasterize_3d(coords, values, resolution)
    else:
        raise ValueError(f"unsupported coordinate dimension {dim}")


def plot_line(x, z, title, out_path, vmin=None, vmax=None):
    """Save a 1D line plot of z(x) to out_path. Counterpart of plot_heatmap()
    for 1D data, as dispatched by plot_field().

    vmin/vmax, when given, fix the y-axis range instead of auto-scaling to
    z's own min/max -- the 1D analogue of a fixed colorbar range, so that
    line plots from different solutions remain visually comparable.
    """
    plt.figure(figsize=(7, 4))
    ax = sns.lineplot(x=x, y=z)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel(title)
    ax.set_ylim(vmin, vmax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_heatmap(grid_z, title, out_path, cmap="viridis", center=None, vmin=None, vmax=None, ax=None):
    """Draw a single 2D field as a seaborn heatmap.

    If ax is None (the default), creates its own figure and saves it to
    out_path -- this is the standalone 2D case. If ax is given (as used by
    plot_slices() to lay out three slices side by side), draws into that
    existing axes instead and leaves saving/closing the figure to the caller;
    out_path is ignored in that case.

    center, when set, is forwarded to seaborn to fix the colormap's midpoint
    (e.g. 0 for a signed diverging field) instead of auto-scaling to the
    data's own min/max.

    vmin/vmax, when given, fix the colorbar range instead of auto-scaling to
    grid_z's own min/max -- this is what makes heatmaps from different
    solutions (or different diff plots) directly comparable by color.
    """
    standalone = ax is None
    if standalone:
        plt.figure(figsize=(6, 5))
        ax = plt.gca()

    sns.heatmap(
        grid_z, cmap=cmap, center=center, vmin=vmin, vmax=vmax, square=True,
        xticklabels=False, yticklabels=False, cbar=True, ax=ax,
    )
    ax.invert_yaxis()  # row 0 of grid_z is the minimum coordinate value
    ax.set_title(title)

    if standalone:
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()


def plot_slices(grid_z, title, out_path, cmap="viridis", center=None, vmin=None, vmax=None):
    """Three orthogonal mid-plane slices through a 3D field, laid out side by
    side in one figure and saved to out_path. There is no native 3D array
    support in seaborn, so this is the volumetric analogue of plot_heatmap().

    grid_z: (resolution,)*3 array as returned in rasterize_3d()'s "grid" key.
    Slice titles report the grid *index* at which each cut was taken (not the
    physical coordinate) since this function has no access to the "axes"
    array rasterize_3d() also returns.

    vmin/vmax fix a shared colorbar range across all three slices (and, by
    passing the same values elsewhere, across different plots entirely);
    see plot_heatmap().
    """
    n = grid_z.shape[0]
    mid = n // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_heatmap(grid_z[mid, :, :], f"{title} (x={mid}, yz-slice)", None, cmap, center, vmin, vmax, ax=axes[0])
    plot_heatmap(grid_z[:, mid, :], f"{title} (y={mid}, xz-slice)", None, cmap, center, vmin, vmax, ax=axes[1])
    plot_heatmap(grid_z[:, :, mid], f"{title} (z={mid}, xy-slice)", None, cmap, center, vmin, vmax, ax=axes[2])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_field(result, title, out_path, cmap="viridis", center=None, vmin=None, vmax=None):
    """Dispatch a rasterize()/rasterize_1d/2d/3d() result to the matching
    plot_line()/plot_heatmap()/plot_slices() function, based on its "dim"
    key. cmap/center/vmin/vmax are only used for the 2D/3D cases, except
    vmin/vmax which also fix plot_line()'s y-axis range in the 1D case."""
    if result["dim"] == 1:
        plot_line(result["x"], result["z"], title, out_path, vmin=vmin, vmax=vmax)
    elif result["dim"] == 2:
        plot_heatmap(result["grid"], title, out_path, cmap=cmap, center=center, vmin=vmin, vmax=vmax)
    elif result["dim"] == 3:
        plot_slices(result["grid"], title, out_path, cmap=cmap, center=center, vmin=vmin, vmax=vmax)


def main():
    """CLI entry point: load one (coords, solution) pair, rasterize and plot
    it, and -- if --reference is given -- load a second pair (against
    --reference-coords, or the same coords if omitted), rasterize it onto the
    same grid, and additionally plot log10|difference| to *_logdiff.png."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("coords", help="path to *_coords.bin (support point coordinates)")
    parser.add_argument("solution", help="path to *_iterN.bin (solution vector)")
    parser.add_argument("--reference", help="optional reference *_iterN.bin for a log10|diff| plot")
    parser.add_argument("--reference-coords",
                        help="coords file for --reference, if from a different mesh/level "
                             "(defaults to the 'coords' argument)")
    parser.add_argument("--resolution", type=int, default=None,
                        help="grid resolution in each dimension (default: 1024 for 1D, "
                             "512 for 2D, 64 for 3D -- 3D interpolation cost grows as "
                             "resolution^3, so raise it with care)")
    parser.add_argument("--vmin", type=float, default=None,
                        help="fix the lower end of the colorbar/y-axis range for the solution "
                             "plot (default: auto-scale to this solution's own min), so that "
                             "plots of different solutions use the same color scale and are "
                             "directly comparable")
    parser.add_argument("--vmax", type=float, default=None,
                        help="fix the upper end of the colorbar/y-axis range for the solution "
                             "plot (default: auto-scale to this solution's own max)")
    parser.add_argument("--diff-vmin", type=float, default=None,
                        help="fix the lower end of the colorbar range for the log10|diff| plot "
                             "(default: auto-scale). Independent of --vmin, since the diff plot "
                             "is on a log10 scale, not the solution's own scale")
    parser.add_argument("--diff-vmax", type=float, default=None,
                        help="fix the upper end of the colorbar range for the log10|diff| plot "
                             "(default: auto-scale)")
    parser.add_argument("--eps", type=float, default=1e-300,
                        help="tolerance added before taking the log in the diff plot, i.e. "
                             "log10(|psi - psi_ref| + eps) instead of plain log10|psi - psi_ref| "
                             "(default: 1e-300, i.e. just enough to avoid log10(0) -- raise it "
                             "(e.g. to 1e-8) to floor the plot above discretization/roundoff "
                             "noise instead of showing the full dynamic range down to that noise)")
    parser.add_argument("--title", default=None,
                        help="replaces the default plot title ('psi', or 'log10 |psi - psi_ref|' "
                             "for the diff plot if --diff-title is not also given), e.g. to "
                             "label a run's parameters")
    parser.add_argument("--diff-title", default=None,
                        help="replaces the default diff plot title, overriding --title for that "
                             "plot specifically (default: falls back to --title, then to "
                             "'log10 |psi - psi_ref|')")
    parser.add_argument("--out", default="solution.png", help="output image path")
    args = parser.parse_args()

    coords = load_coords(args.coords)
    values = load_solution(args.solution)
    if coords.shape[0] != values.shape[0]:
        raise ValueError(
            f"coords ({coords.shape[0]} points) and solution ({values.shape[0]} values) "
            "sizes do not match -- did you mix up files from different levels?"
        )

    dim = coords.shape[1]
    resolution = args.resolution or DEFAULT_RESOLUTION[dim]
    if dim == 3 and resolution > 128:
        print(f"warning: 3D resolution {resolution} means a {resolution}^3 grid; "
              "this can be slow/memory-heavy to interpolate", file=sys.stderr)

    title = args.title or "psi"

    result = rasterize(coords, values, resolution)
    plot_field(result, title, args.out, vmin=args.vmin, vmax=args.vmax)
    print(f"wrote {args.out}")

    if args.reference:
        ref_coords_path = args.reference_coords or args.coords
        ref_coords = load_coords(ref_coords_path)
        ref_values = load_solution(args.reference)
        if ref_coords.shape[1] != dim:
            raise ValueError(
                f"reference dimension ({ref_coords.shape[1]}D) does not match "
                f"the solution's dimension ({dim}D)"
            )

        ref_result = rasterize(ref_coords, ref_values, resolution)
        diff_title = args.diff_title or args.title or "log10 |psi - psi_ref|"

        if dim == 1:
            log_diff = np.log10(np.abs(result["z"] - ref_result["z"]) + args.eps)
            diff_result = {"dim": 1, "x": result["x"], "z": log_diff}
        else:
            log_diff = np.log10(np.abs(result["grid"] - ref_result["grid"]) + args.eps)
            diff_result = {"dim": dim, "grid": log_diff, **{k: v for k, v in result.items() if k == "axes"}}

        diff_out = args.out.rsplit(".", 1)[0] + "_logdiff.png"
        plot_field(diff_result, diff_title, diff_out, cmap="magma",
                   vmin=args.diff_vmin, vmax=args.diff_vmax)
        print(f"wrote {diff_out}")


if __name__ == "__main__":
    main()
