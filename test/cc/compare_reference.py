#!/usr/bin/env python3
"""Run the paper's reference implementation (RMO-continuous-cuts, package
bernoulli_multilevel) on the same synthetic disk image that src/main_cc.cc and
src/main_cc_coarse.cc use, so that the C++ port can be compared to the reference
trajectory-by-trajectory instead of primitive-by-primitive.

Prints, per fine-level iteration: the single-level RGD energy, the 2-level
(Option 1) energy, whether a coarse correction was taken, and the stationarity
measure in both norms (Euclidean, as the reference reports it, and Fisher-Rao,
as the C++ residual column reports it).

Usage:
    python3 compare_reference.py /path/to/RMO-continuous-cuts [options]

Only torch and numpy are needed. bernoulli_multilevel's package __init__ imports
problem.py, which imports scikit-image; a stub is installed for that module so
the import succeeds without scikit-image (nothing from problem.py is used).

Parameter conventions (see --help):
  --eps-mode ref    eps enters the reference objective unsquared, i.e. the TV
                    term is sqrt(g^2 + eps). This is what objective.py does and
                    therefore what the paper's Figures 13-15 were produced with.
  --eps-mode paper  eps is squared before being handed to the reference, so the
                    TV term is sqrt(g^2 + eps^2) as in eq. (42) -- the convention
                    the C++ port implements (main_cc.cc / main_cc_coarse.cc now
                    pass sqrt(eps_ref) so the two conventions agree numerically;
                    see CONTINUOUS_CUTS.md \\S1's "eps convention").

This script and derive_golden_values.py are the two differential tests against
the Python reference (algorithm trajectory here; primitives there) -- run both
whenever cc/ or the reference implementation changes.
"""
import argparse
import math
import sys
import types


def install_skimage_stub():
    if "skimage" in sys.modules:
        return
    try:
        import skimage  # noqa: F401
        return
    except ImportError:
        pass
    stub = types.ModuleType("skimage")
    for name in ("color", "filters", "io", "transform"):
        sub = types.ModuleType(f"skimage.{name}")
        setattr(stub, name, sub)
        sys.modules[f"skimage.{name}"] = sub
    sys.modules["skimage"] = stub


def synthetic_disk(n, patch=5):
    """Same recipe as include/rmo/cc/synthetic.h: bright disk (0.8) of radius
    min(rows,cols)/4 on a dark (0.2) background, foreground seed patch at the
    centre, background seed patch at the top-left corner, no smoothing."""
    import torch
    cy = cx = (n - 1) / 2.0
    radius = n / 4.0
    r = torch.arange(n, dtype=torch.float64).view(-1, 1).expand(n, n)
    c = torch.arange(n, dtype=torch.float64).view(1, -1).expand(n, n)
    inside = torch.sqrt((r - cy) ** 2 + (c - cx) ** 2) <= radius
    g = torch.where(inside, torch.tensor(0.8, dtype=torch.float64), torch.tensor(0.2, dtype=torch.float64))

    fg0 = n // 2 - patch // 2
    c_f = g[fg0:fg0 + patch, fg0:fg0 + patch].mean()
    c_b = g[0:patch, 0:patch].mean()
    rho = (c_f - g) ** 2 - (c_b - g) ** 2
    return g, rho


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="path to the RMO-continuous-cuts checkout")
    ap.add_argument("--n", type=int, default=41, help="fine grid size (n x n); coarse is ceil(n/2)")
    ap.add_argument("--iters", type=int, default=300, help="fine-level iterations for both solvers")
    ap.add_argument("--coarse-iters", type=int, default=10, help="coarse-level iterations per correction")
    ap.add_argument("--alpha-fine", type=float, default=0.1)
    ap.add_argument("--eps-fine", type=float, default=1e-4)
    ap.add_argument("--alpha-coarse", type=float, default=0.4)
    ap.add_argument("--eps-coarse", type=float, default=1e-3)
    ap.add_argument("--eps-mode", choices=["ref", "paper"], default="ref",
                    help="ref: sqrt(g^2+eps) (objective.py); paper: sqrt(g^2+eps^2) (eq. 42, C++ port)")
    ap.add_argument("--eta", type=float, default=0.6)
    ap.add_argument("--gnorm-threshold", type=float, default=0.5,
                    help="reference's gnorm_c_threshold (Euclidean norm of the fine Riemannian gradient)")
    ap.add_argument("--rho-coarse", choices=["avgpool", "fullweight"], default="avgpool",
                    help="avgpool: problem.py's avg_pool2d(ceil_mode); fullweight: 0.25*(B^T rho) as main_cc_coarse.cc")
    ap.add_argument("--every", type=int, default=25, help="print every k-th iteration (0: only the summary)")
    args = ap.parse_args()

    sys.path.insert(0, args.reference.rstrip("/") + "/src")
    install_skimage_stub()

    import torch
    import torch.nn.functional as F
    from bernoulli_multilevel.manifold import FR_metric, apply_G_inv
    from bernoulli_multilevel.multilevel import run_ml_experiment
    from bernoulli_multilevel.objective import create_cc_objective
    from bernoulli_multilevel.primitives import restrict_adjoint_bilinear
    from bernoulli_multilevel.single_level import solve_single_level

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    n = args.n
    _, rho_fine = synthetic_disk(n)

    if args.rho_coarse == "avgpool":
        rho_coarse = F.avg_pool2d(rho_fine[None, None], kernel_size=2, stride=2, ceil_mode=True)[0, 0]
    else:
        rho_coarse = 0.25 * restrict_adjoint_bilinear(rho_fine)

    def eps_arg(eps):
        return eps if args.eps_mode == "ref" else eps * eps

    f_fine = create_cc_objective(rho_fine, args.alpha_fine, eps=eps_arg(args.eps_fine))
    f_coarse = create_cc_objective(rho_coarse, args.alpha_coarse, eps=eps_arg(args.eps_coarse))
    f_list = [f_coarse, f_fine]

    phi0 = torch.full((n, n), 0.5, dtype=torch.float64, requires_grad=True)

    def stationarity(phi):
        p = phi.detach().clone().requires_grad_(True)
        f_fine(p).backward()
        g_riem = apply_G_inv(p.detach(), p.grad)
        euclid = torch.norm(g_riem).item()
        fisher = math.sqrt(FR_metric(g_riem, g_riem, p.detach()).item())
        return euclid, fisher

    hparams = {
        "max_levels": 2,
        "level_delta_pools": [1],
        "maxIter": [args.coarse_iters, 1],
        "tau": [0.1, 0.1],
        "eta": args.eta,
        "gnorm_c_threshold": args.gnorm_threshold,
    }

    E0 = f_fine(phi0).item()
    g0 = stationarity(phi0)
    print(f"# reference={args.reference} n={n} coarse={rho_coarse.shape[0]} eps-mode={args.eps_mode} "
          f"rho-coarse={args.rho_coarse} eta={args.eta} gnorm-threshold={args.gnorm_threshold} "
          f"coarse-iters={args.coarse_iters}")
    print(f"# E0={E0:.6f} |grad|_2={g0[0]:.4e} |grad|_FR={g0[1]:.4e}")

    phi_sl, e_sl, gn_sl, t_sl, hist_sl = solve_single_level(f_fine, phi0, n_iter=args.iters, verbose=False)
    phi_ml, e_ml, gn_ml, t_ml, hist_ml, c_ml = run_ml_experiment(
        f_list, hparams, n, n, args.iters, approach="Option 1", verbose=False)

    print(f"{'iter':>5} {'E_sl':>12} {'E_ml':>12} {'coarse':>6} {'|g_sl|_2':>10} {'|g_ml|_2':>10} {'|g_ml|_FR':>10}")
    for k in range(args.iters):
        if args.every and (k % args.every == 0 or k == args.iters - 1):
            fr = stationarity(hist_ml[k])[1]
            print(f"{k + 1:5d} {e_sl[k]:12.6f} {e_ml[k]:12.6f} {'*' if c_ml[k] else ' ':>6} "
                  f"{gn_sl[k]:10.3e} {gn_ml[k]:10.3e} {fr:10.3e}")

    e_sl_final = e_sl[-1]
    reached = next((k + 1 for k, e in enumerate(e_ml) if e <= e_sl_final), None)
    print(f"# single-level: E({args.iters})={e_sl_final:.6f}  cpu={t_sl[-1]:.2f}s")
    print(f"# multilevel:   E({args.iters})={e_ml[-1]:.6f}  cpu={t_ml[-1]:.2f}s  "
          f"coarse corrections={sum(c_ml)}  first iteration with E_ml <= E_sl({args.iters}): {reached}")


if __name__ == "__main__":
    main()
