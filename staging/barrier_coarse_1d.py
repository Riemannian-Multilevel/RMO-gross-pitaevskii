#!/usr/bin/env python3
"""Two-level prototype in 1D: the Nash (first-order coherent) coarse model on the positive mass sphere with log
barriers, in density variables with the Fisher-Rao geometry (the mass metric, see tex/BlackboardS.tex).

Usage: staging/barrier_coarse_1d.py <output dir for data> <output dir for figures and tables>

Checks (table tab_barrier_checks.tex) and a two-level iteration (fig_barrier_twolevel.pdf) for
tex/pos_sphere_variants.tex, Section "Barrier and the Nash coarse model".

Discretization: lumped P1 on [-L, L] with Neumann boundary, f(x) = 1/2 x^T S x + 1/2 sum d V x^2 + kappa/4 sum d x^4,
V = x^2, density p = d x^2 on the simplex. Coarse level: every other node. Transfers:
  restriction of points      rho = P^T p            (full weighting, keeps the total mass)
  prolongation of increments P sigma = p * P(sigma / rho)   ("relative" prolongation)
  restriction of gradients   R xi = P^T xi
with P the linear interpolation (nonnegative, rows summing to one); P maps tangent vectors (sum zero) to tangent
vectors, and R is the Fisher-Rao adjoint of P (Galerkin condition).
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

L, KAPPA, N_COARSE = 10.0, 100.0, 41
THETA = 0.99


class Level:
    """Lumped P1 discretization on n nodes; functions of the density p (sum p = 1)."""

    def __init__(self, n):
        self.n = n
        self.X = np.linspace(-L, L, n)
        h = self.X[1] - self.X[0]
        self.d = np.full(n, h)
        self.d[[0, -1]] = h / 2
        main = np.full(n, 2 / h)
        main[[0, -1]] = 1 / h
        self.S = sp.diags([main, np.full(n - 1, -1 / h), np.full(n - 1, -1 / h)], [0, 1, -1], format="csr")
        self.V = self.X ** 2

    def x(self, p):
        return np.sqrt(np.maximum(p, 0) / self.d)

    def energy(self, p, mu=0.0, w=None):
        x = self.x(p)
        e = 0.5 * x @ (self.S @ x) + 0.5 * np.sum(self.V * p) + 0.25 * KAPPA * np.sum(p ** 2 / self.d)
        if mu > 0:
            e -= mu * np.sum(w * np.log(p))
        return e

    def hessian(self, p, mu=0.0, w=None):
        """Hessian of F in the density p (dense): S_ij x_i x_j / (4 p_i p_j) + diag(-(S x)_i x_i / (4 p_i^2)
        + kappa / (2 d_i)) + barrier mu w_i / p_i^2."""
        x = self.x(p)
        Sx = self.S @ x
        a = x / (2 * p)
        Hm = (self.S.toarray() * np.outer(a, a))
        Hm[np.diag_indices(self.n)] += -Sx * x / (4 * p ** 2) + KAPPA / (2 * self.d)
        if mu > 0:
            Hm[np.diag_indices(self.n)] += mu * w / p ** 2
        return Hm

    def g4(self, p, mu=0.0, w=None):
        """4 p * grad_p F: the Fisher-Rao gradient before the projection onto the tangent space."""
        x = self.x(p)
        g = 2 * x * (self.S @ x) + 2 * self.V * p + 2 * KAPPA * p ** 2 / self.d
        if mu > 0:
            g -= 4 * mu * w
        return g

    def grad(self, p, mu=0.0, w=None):
        """Fisher-Rao Riemannian gradient on the simplex: g - (sum g) p."""
        g = self.g4(p, mu, w)
        return g - g.sum() * p


def fr(a, b, p):
    """Fisher-Rao inner product at p (one quarter of it, matching the mass metric)."""
    return np.sum(a * b / (4 * p))


def interpolation(nc):
    nf = 2 * nc - 1
    rows, cols, vals = [], [], []
    for i in range(nf):
        if i % 2 == 0:
            rows.append(i); cols.append(i // 2); vals.append(1.0)
        else:
            rows += [i, i]; cols += [i // 2, i // 2 + 1]; vals += [0.5, 0.5]
    return sp.csr_matrix((vals, (rows, cols)), shape=(nf, nc))


class TwoLevel:
    def __init__(self, nc):
        self.H, self.h = Level(nc), Level(2 * nc - 1)
        self.P = interpolation(nc)

    def restrict(self, p):
        return self.P.T @ p

    def prolong_rel(self, p, rho, sigma):
        return p * (self.P @ (sigma / rho))

    def prolong_add(self, p, sigma):
        """Additive prolongation of density increments, projected to sum zero (comparison only)."""
        v = self.P @ sigma
        return v - v.sum() * p


class CoarseModel:
    """Q(s) = F_H(s; mu_H) - <w, s - rho>_rho,  w = grad F_H(rho; mu_H) - R grad F_h(p; mu_h)."""

    def __init__(self, tl, p, mu_h=0.0, mu_H=0.0, weights="mass"):
        self.tl, self.p = tl, p
        self.mu_h, self.mu_H = mu_h, mu_H
        self.w_h = tl.h.d if weights == "mass" else np.ones(tl.h.n)
        self.w_H = tl.H.d if weights == "mass" else np.ones(tl.H.n)
        self.rho = tl.restrict(p)
        self.Rg = tl.P.T @ tl.h.grad(p, mu_h, self.w_h)
        self.w = tl.H.grad(self.rho, mu_H, self.w_H) - self.Rg

    def value(self, s):
        return self.tl.H.energy(s, self.mu_H, self.w_H) - fr(self.w, s - self.rho, self.rho)

    def euclid_grad(self, s):
        """Gradient of Q in R^n (for the optimizer)."""
        g = self.tl.H.g4(s, self.mu_H, self.w_H) / (4 * s)
        return g - self.w / (4 * self.rho)

    def grad(self, s):
        g = 4 * s * self.euclid_grad(s)
        return g - g.sum() * s

    def minimize(self, tol=1e-14, max_iter=500):
        """Newton's method on the simplex {sum s = 1}: KKT step with the analytic Hessian, fraction-to-the-boundary
        damping (0.99) and Armijo backtracking on Q; stops at a Newton decrement below tol. Returns the minimizer and
        the number of iterations."""
        n = self.tl.H.n
        s = self.rho.copy()
        ones = np.ones(n)
        for it in range(max_iter):
            g = self.euclid_grad(s)
            Hm = self.tl.H.hessian(s, self.mu_H, self.w_H)
            K = np.block([[Hm, ones[:, None]], [ones[None, :], np.zeros((1, 1))]])
            sol = np.linalg.solve(K, np.concatenate([-g, [0.0]]))
            ds = sol[:n]
            decrement = -g @ ds
            if decrement < tol:
                return s, it
            neg = ds < 0
            t = min(1.0, 0.99 * np.min(s[neg] / -ds[neg])) if neg.any() else 1.0
            q0 = self.value(s)
            while self.value(s + t * ds) > q0 - 1e-4 * t * decrement + 1e-15 * abs(q0) and t > 1e-20:
                t *= 0.5
            s = s + t * ds
        return s, max_iter


def ground_state(level, tol=1e-14):
    """Ground state density by the self-consistent iteration x <- A(x)^{-1} D x (positive for this M-matrix)."""
    x = np.ones(level.n)
    x /= np.sqrt(level.d @ x ** 2)
    for _ in range(2000):
        A = level.S + sp.diags(level.d * (level.V + KAPPA * x ** 2))
        y = spla.spsolve(A.tocsc(), level.d * x)
        y /= np.sqrt(level.d @ y ** 2)
        if np.max(np.abs(y - x)) < tol:
            x = y
            break
        x = y
    return level.d * x ** 2


def armijo(level, p, eta, f0, slope, alpha0, mu=0.0, w=None, beta=0.5, sigma=1e-4, max_trials=60, trial=None):
    """Armijo backtracking along p + alpha eta, or along trial(alpha) if given (same points, other formula)."""
    noise = 10 * np.finfo(float).eps * max(1.0, abs(f0))
    alpha = alpha0
    for _ in range(max_trials):
        q = trial(alpha) if trial is not None else p + alpha * eta
        # Normalize, as PosUnitMassSphere::retract: the gradient is tangent only on the simplex, so that rounding
        # errors in sum p would grow by a factor 1 + alpha sum(4 p grad F) per step
        q = q / q.sum()
        if np.all(q > 0) and level.energy(q, mu, w) <= f0 + sigma * alpha * slope + noise:
            return alpha, q
        alpha *= beta
    return 0.0, p


def tau_max(p, eta):
    neg = eta < 0
    return np.min(p[neg] / -eta[neg]) if neg.any() else np.inf


def smooth(level, p, mu=0.0, w=None):
    """One Fisher-Rao gradient step with Armijo line search, starting at min(1, theta tau_max) (option A)."""
    g = level.grad(p, mu, w)
    eta = -g
    alpha0 = min(1.0, THETA * tau_max(p, eta))
    _, q = armijo(level, p, eta, level.energy(p, mu, w), -fr(g, g, p), alpha0, mu, w)
    return q


def two_level(tl, p0, cycles, coarse=True, mu_H=0.0, prolongation="rel", nu=2, cap=False):
    """nu smoothing steps per cycle and, if coarse, one Nash coarse correction with a coarse barrier mu_H; cap: the
    coarse step starts at min(1, theta tau_max) (option A) also for the relative prolongation."""
    p = p0.copy()
    hist = {"grad": [], "min": [], "alpha": [], "coarse_min": [], "energy": []}
    for _ in range(cycles):
        for _ in range(nu):
            p = smooth(tl.h, p)
        if coarse:
            model = CoarseModel(tl, p, 0.0, mu_H)
            s, _ = model.minimize()
            d = tl.prolong_rel(p, model.rho, s - model.rho) if prolongation == "rel" \
                else tl.prolong_add(p, s - model.rho)
            g = tl.h.grad(p)
            slope = fr(g, d, p)
            alpha0 = min(1.0, THETA * tau_max(p, d)) if prolongation == "add" or cap else 1.0
            # Relative prolongation: p + alpha d = p (1 - alpha + alpha P(s / rho)), evaluated in this form, since
            # p + d cancels to zero in floating point where s / rho is below the rounding level
            ratio = tl.P @ (s / model.rho)
            trial = (lambda a: p * (1 - a + a * ratio)) if prolongation == "rel" else None
            if slope < 0 and alpha0 > 0:
                alpha, p_new = armijo(tl.h, p, d, tl.h.energy(p), slope, alpha0, trial=trial)
            else:
                alpha, p_new = 0.0, p
            hist["alpha"].append(alpha)
            hist["coarse_min"].append(float(s.min()))
            if np.any(p_new <= 0):  # cannot happen: armijo() accepts positive points only
                hist["grad"].append(np.nan)
                hist["min"].append(float(p_new.min()))
                hist["energy"].append(np.nan)
                break
            p = p_new
        g = tl.h.grad(p)
        hist["grad"].append(float(np.sqrt(fr(g, g, p))))
        hist["min"].append(float(p.min()))
        hist["energy"].append(float(tl.h.energy(p)))
    return p, hist


def sci(v):
    if v == 0:
        return "0"
    m, e = f"{v:.1e}".split("e")
    return rf"{m}\cdot10^{{{int(e)}}}"


def sci_mu(v):
    return "0" if v == 0 else rf"10^{{{int(round(np.log10(v)))}}}"


def checks(tl, p, p_star, rng):
    rows = []
    n_h, n_H = tl.h.n, tl.H.n
    rho = tl.restrict(p)

    # Galerkin condition and tangent spaces
    sig = rng.standard_normal(n_H)
    sig -= sig.mean()
    sig -= sig.mean()  # again, to remove the rounding error of the first pass from sum(sig)
    xi = rng.standard_normal(n_h); xi -= xi.sum() / n_h
    lhs = fr(tl.prolong_rel(p, rho, sig), xi, p)
    rhs = fr(sig, tl.P.T @ xi, rho)
    rows.append(("Galerkin $\\langle P\\sigma,\\xi\\rangle_p=\\langle\\sigma,R\\xi\\rangle_\\rho$",
                 abs(lhs - rhs) / abs(lhs)))
    rows.append(("$P$ maps tangent vectors to tangent vectors", abs(tl.prolong_rel(p, rho, sig).sum())))

    # Coherence by finite differences, for barrier configurations
    for name, mu_h, mu_H in [("no barriers", 0, 0), ("fine barrier only", 1e-3, 0),
                             ("coarse barrier only", 0, 1e-3), ("both, same $\\mu$", 1e-3, 1e-3)]:
        model = CoarseModel(tl, p, mu_h, mu_H)
        err = 0.0
        for _ in range(5):
            z = rng.standard_normal(n_H)
            sig = rho * (z - rho @ z)  # tangent, bounded relative perturbation
            t = 1e-5
            fd = (model.value(rho + t * sig) - model.value(rho - t * sig)) / (2 * t)
            exact = fr(model.Rg, sig, rho)
            err = max(err, abs(fd - exact) / max(abs(exact), 1e-300))
        rows.append((f"coherence $\\grad Q(\\rho)=R\\grad F_{{h}}(p)$, {name}", err))

    # Barrier transparency: the barrier parts of w cancel for mass weights and equal mu
    w0 = CoarseModel(tl, p).w
    for weights in ["mass", "unit"]:
        wb = CoarseModel(tl, p, 1e-3, 1e-3, weights).w
        rows.append((f"barrier part of $w$, $\\mu_h=\\mu_H$, {weights} weights",
                     np.sqrt(fr(wb - w0, wb - w0, rho) / fr(w0, w0, rho))))

    # Convexity of Q along segments in the simplex
    for name, mu_H in [("no barrier", 0.0), ("coarse barrier", 1e-3)]:
        model = CoarseModel(tl, p, 0.0, mu_H)
        worst = np.inf
        for _ in range(20):
            a = rng.dirichlet(np.ones(n_H)); b = rng.dirichlet(np.ones(n_H))
            t = np.linspace(0, 1, 41)
            q = np.array([model.value((1 - ti) * a + ti * b) for ti in t])
            second = q[:-2] - 2 * q[1:-1] + q[2:]
            worst = min(worst, second.min() / max(1e-300, np.abs(q).max()))
        rows.append((f"convexity of $Q$ (min.\\ relative second difference), {name}", worst))

    # Descent and positivity of the coarse correction, and the unbiased fixed point
    for name, mu_H in [("no barrier", 0.0), ("coarse barrier $10^{-6}$", 1e-6)]:
        model = CoarseModel(tl, p, 0.0, mu_H)
        s, newton_it = model.minimize()
        rows.append((f"coarse minimizer (Newton, {newton_it} iterations): $\\min_j\\tilde s_j$, {name}", s.min()))
        d = tl.prolong_rel(p, model.rho, s - model.rho)
        g = tl.h.grad(p)
        slope = fr(g, d, p)
        decrease = model.value(s) - model.value(model.rho)
        rows.append((f"descent $\\langle\\grad F_h,d\\rangle\\leq Q(\\tilde s)-Q(\\rho)<0$, {name}: slope",
                     slope))
        rows.append((f"\\quad $Q(\\tilde s)-Q(\\rho)$, {name}", decrease))
        rows.append((f"\\quad $\\min_i(p+d)_i/p_i=\\min_i(\\tilde P(\\tilde s/\\rho))_i$ (relative prolongation), {name}",
                     np.min(tl.P @ (s / model.rho))))
        dadd = tl.prolong_add(p, s - model.rho)
        rows.append((f"\\quad $\\min_i(p+d)_i/p_i$ (additive prolongation), {name}", np.min((p + dadd) / p)))

    # Fixed point: at the fine ground state, the coarse model with a barrier gives no correction
    for mu_H in [0.0, 1e-6, 1e-2]:
        model = CoarseModel(tl, p_star, 0.0, mu_H)
        s, _ = model.minimize()
        rows.append((f"fixed point: $\\|\\tilde s-\\rho^*\\|_1$ at the ground state, $\\mu_H={sci_mu(mu_H)}$",
                     np.abs(s - model.rho).sum()))
    return rows


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    data.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    tl = TwoLevel(N_COARSE)
    p_star = ground_state(tl.h)

    # A state far from the solution: constant start and a few capped smoothing steps
    p0 = tl.h.d / tl.h.d.sum()
    p = p0.copy()
    for _ in range(3):
        p = smooth(tl.h, p)

    rows = checks(tl, p, p_star, rng)
    lines = [r"\begin{tabular}{@{}p{0.74\textwidth}r@{}}", r"\toprule", r"check & value \\", r"\midrule"]
    for name, value in rows:
        lines.append(f"{name} & ${sci(value)}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_barrier_checks.tex").write_text("\n".join(lines) + "\n")

    # Two-level iteration
    runs = {
        "smoother only": dict(coarse=False),
        "Nash, no barrier": dict(mu_H=0.0),
        "Nash, coarse barrier $10^{-8}$": dict(mu_H=1e-8),
        "Nash, coarse barrier $10^{-4}$": dict(mu_H=1e-4),
        "Nash, no barrier, additive prolongation": dict(mu_H=0.0, prolongation="add"),
        "Nash, no barrier, capped coarse step": dict(mu_H=0.0, cap=True),
    }
    results = {}
    for name, kw in runs.items():
        _, hist = two_level(tl, p0, 60, **kw)
        results[name] = hist
    with open(data / "results.json", "w") as f:
        json.dump({"rows": [(n, float(v)) for n, v in rows], "twolevel": results,
                   "ground_state_min_density": float(p_star.min())}, f, indent=1)

    F_star = tl.h.energy(p_star)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.3), layout="constrained")
    for name, hist in results.items():
        k = np.arange(1, len(hist["grad"]) + 1)
        axes[0].semilogy(k, np.maximum(np.array(hist["energy"]) - F_star, 1e-17), label=name)
        axes[1].semilogy(k, hist["grad"], label=name)
        axes[2].semilogy(k, np.abs(hist["min"]), label=name)
        if hist["alpha"]:
            axes[3].plot(np.arange(1, len(hist["alpha"]) + 1), hist["alpha"], ".", ms=3, label=name)
    axes[0].set_title(r"$F_h(p_k)-F_h(p^*)$ (floored at $10^{-17}$)")
    axes[1].set_title(r"$\|\mathrm{grad}\,F_h(p_k)\|$ (Fisher--Rao)")
    axes[2].set_title(r"$\min_i p_{k,i}$ (dotted: ground state)")
    axes[3].set_title(r"accepted coarse step $\alpha$")
    for ax in axes:
        ax.set_xlabel("cycle")
    axes[2].axhline(p_star.min(), color="k", ls=":", lw=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    fig.savefig(out / "fig_barrier_twolevel.pdf")
    plt.close(fig)

    for name, hist in results.items():
        g = [v for v in hist["grad"] if np.isfinite(v)]
        print(f"{name:45s} cycles {len(hist['grad']):3d} final grad {g[-1] if g else float('nan'):.2e} "
              f"min p {hist['min'][-1]:.2e} min coarse s {min(hist['coarse_min']) if hist['coarse_min'] else float('nan'):.2e}")
    for name, value in rows:
        print(f"  {value: .2e}  {name}")
    print(f"ground state: min density {p_star.min():.2e}")


if __name__ == "__main__":
    main()
