# Overview

See [ARCHITECTURE.md](ARCHITECTURE.md).


# Installation

The following section explains how to install the deal.ii finite element library for macOS, Ubuntu and Windows.
After completing these instructions, the [CMake](#cmake) section explains how to build the `gpe` programs.


## macOS

Download the latest deal.ii `.dmg` from GitHub:

https://github.com/dealii/dealii/releases

Either download the Sequoia (macOS 15.x) or Tahoe (macOS 26.x) version. Install the file
like any other macOS application, by dragging it to `Applications`. 

> *Note:*
> The latest available version for Intel CPUs is version 9.5.2.

Depending on security settings, you may need to approve the `.dmg` manually. See the 
[Mac User Guide](https://support.apple.com/de-de/guide/mac-help/mh40616/mac)
for details.

Opening the deal.ii application results in a terminal with a correctly set environment for building deal.ii programs.
If not done before, ensure `XCode` is installed:

```bash
xcode-select --install
```

For more information, see the [deal.ii Wiki](https://github.com/dealii/dealii/wiki/MacOSX).


## Ubuntu

Install the deal.ii packages:

```bash
sudo apt install libdeal.ii-dev libdeal.ii-doc
```

This installs both the library files, and the deal.ii examples.

Depending on the Ubuntu version, these packages may be out-of-date. A 
[personal package archive](https://launchpad.net/~ginggs/+archive/ubuntu/deal.ii-9.7.1-backports) 
for 9.7.1 is available.

After installation, all libraries are available in the system path.

For more information, see the [deal.ii Wiki](https://github.com/dealii/dealii/wiki/Debian-and-Ubuntu).


## Windows

Install [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) and follow the instructions for [Ubuntu](#ubuntu).


# CMake

After installing `deal.ii`, the project can be built with `CMake`. Download the git repository and initialize the submodules:

```bash
git clone --recurse-submodules https://github.com/Riemannian-Multilevel/gpe-multilevel-dealii
cd gpe-multilevel-dealii
```

Navigate to the source directory and build the applications:

```bash
rm -rf build
mkdir build
cd build

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j
```

> *Note:*
> The *Release* and *Debug* build types are optimized for performance and debugging, respectively.

After the process is completed, all `gpe` programs are available in the `build` directory. 


# Usage

Each program has command-line parameters that can be listed with the `--help` flag. For example:

```bash
./main_coarse --help
Allowed options:
  --help                                produce help message

General problem options:
  --degree arg (=1)                     polynomial degree for finite element
  --dimension arg (=2)                  problem dimension
  --order arg (=default)                ordering for degrees of freedom
                                        (default|random|cuthill_mckee|king|min_
                                        deg)
  --boundary arg (=neumann)             boundary constraints
                                        (neumann|dirichlet)
  --radius arg (=10)                    default radius of the cube domain
  --beta arg (=100)                     non-linearity factor
  --mesh arg (=quadrilateral)           type of mesh elements used
                                        (quadrilateral|simplex)
  --potential arg (=square)             used potential (square|optical_lattice)

RGD options:
  --max-iter arg (=25)                  maximum number of iterations
  --tol-residual arg (=0.0001)          tolerance for M-residual
  --step-size arg (=1)                  step size for RGD
  --line-search [=arg(=1)] (=0)         use armijo line search
  --ls-max-iter arg (=3)                maximum number of iterations for line
                                        search
  --ls-alpha arg (=1)                   alpha for armijo line search
  --ls-beta arg (=0.59999999999999998)  beta for armijo line search
  --ls-sigma arg (=0.20000000000000001) sigma for armijo line search
  --ls-min arg (=0.10000000000000001)   minimal step size for armijo line
                                        search

multilevel options:
  --levels arg                          number of times to globally refine the
                                        mesh
  --multilevel arg                      levels for the multilevel hierarchy
                                        (consecutive)

Inner solver options:
  --solver arg (=cg)                    sparse solver (gmres|minres|cg)
  --precond arg (=none)                 preconditioner (none|diagonal|sparse_il
                                        u|amg)
  --max-inner arg (=500)                maximum number of iterations for sparse
                                        solver
  --tol-inner arg (=9.9999999999999995e-07)
                                        tolerance for sparse solver, relative
                                        to right-hand side
  --tol-inner-res arg (=0.01)           tolerance for sparse solver, relative
                                        to residual

FAS options:
  --kappa arg (=0.80000000000000004)    weight for ratio of restricted and
                                        coarse gradient
  --eps arg (=0.0001)                   minimum norm of restricted gradient
  --coarse-every arg (=2)               minimum number of fine steps before
                                        coarse step is taken
  --metric arg (=mass)                  metric for coarse model
                                        (none|frobenius|mass)
  --transport arg (=mass)               vector transport operator
                                        (frobenius|mass|differential|adjoint_re
                                        striction|adjoint_differential|adjoint_
                                        restriction_frobenius|adjoint_different
                                        ial_frobenius|differential_frobenius)
  --interpolate arg (=none)             galerkin condition on linear
                                        interpolation (none|mass)

Output options:
  --output-bin [=arg(=)]                write all iterates in binary format,
                                        optionally with base name
                                        (--output-bin=<name>)
  --output-vtk [=arg(=)]                write the solution in VTK format,
                                        optionally with base name
                                        (--output-vtk=<name>)
  --output-every arg (=0)               with --output-vtk, also write every
                                        k-th and the final iterate as a series
                                        (0: final iterate only)
  --simple-table [=arg(=1)] (=0)        write the iterates of all levels as
                                        rows of one table as they are computed,
                                        without reduction rates (default: one
                                        table per level when its cycle
                                        finishes)
```

To replicate the paper results, run the study from the `build` directory:

```bash
cd build
python3 ../study.py
```

This runs three stages, which can also be run one at a time (`python3 ../study.py study|median|figures`):

- `study`: every configuration runs `--reps` times (default 3). Each study gets a new directory
  `runs/l<level>_<potential>_<date>_<time>_<commit>/` (`-dirty` is added for uncommitted changes), with the
  tables of the repetitions in `rep<k>/`, the source state (`commit.txt`, `dirty.txt`, `dirty.patch`), the
  parameters (`params.txt`) and the binary used. `runs/latest` links to the last study. An existing directory
  is never overwritten.
- `median`: `plot/median_tables.py` writes the tables with the median CPU time over the repetitions.
- `figures`: `plot/plot_convergence.py` writes Figures 7/8/9 of the paper for the mass and the Frobenius metric
  to `figures/b<beta>/`, for every beta. A figure whose tables are missing is skipped.

`median` and `figures` use the latest study, or another one with `--out runs/<study>`.

The parameters and their defaults are listed by `python3 ../study.py --help`. Pass the same parameters to
`median` and `figures`:

```bash
python3 ../study.py --beta-range 100 1000 10000 --iter-range 20 40 60 --top-level 10 --reps 1
python3 ../study.py figures --plot-args='--format png --coarse-steps'
```

`--beta-range` and `--iter-range` are paired and must have the same length.

This requires the Python packages `seaborn`, `pandas` and `orgparse`.


# AI disclosure

Claude (Fable 5, Opus 5.5) was used for the following:

* Generation of prototypes in `staging/` and `plot/`
  * `plot/plot_*.py`: plotting code
  * `staging/test/`: numerical tests
* Code review and debugging
* Doxygen documentation

Commits, which are exclusively generated by Claude, have their authorship set accordingly.