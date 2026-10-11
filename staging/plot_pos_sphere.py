#!/usr/bin/env python3
"""Figures and tables of tex/pos_sphere_domain.tex from the output of staging/pos_sphere_study.sh.

Usage: staging/plot_pos_sphere.py <data dir> <output dir>

<data dir> is the output directory of pos_sphere_study.sh (groups levels, radius, beta, metric, tol). Writes
fig_*.pdf and tab_*.tex to <output dir>.
"""
import math

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import LogNorm

from pos_sphere_plot import data_and_out_dirs, read_csv, read_rows, sci_tex, write_table

CASES = ["unit", "positive_theta0.5", "positive_theta0.9", "positive_theta0.99"]
LABELS = {
    "unit": "unit sphere (norm retraction)",
    "positive_theta0.5": r"positive sphere, $\theta=0.5$",
    "positive_theta0.9": r"positive sphere, $\theta=0.9$",
    "positive_theta0.99": r"positive sphere, $\theta=0.99$",
}
COLORS = {
    "unit": "#222222",
    "positive_theta0.5": "#1f77b4",
    "positive_theta0.9": "#2ca02c",
    "positive_theta0.99": "#d62728",
}


def grid(fields, key):
    """Values of a nodal field on the tensor grid of support points (Q1 on a square)."""
    xs, ys = np.unique(fields["px"]), np.unique(fields["py"])
    ix = np.searchsorted(xs, fields["px"])
    iy = np.searchsorted(ys, fields["py"])
    z = np.full((len(ys), len(xs)), np.nan)
    z[iy, ix] = fields[key]
    return xs, ys, z


# ---------------------------------------------------------------------------------------------------------------
# Figure 1: why option A is needed (first step from the constant start)
# ---------------------------------------------------------------------------------------------------------------
def fig_first_step(data, out):
    f = read_csv(data / "levels" / "l6" / "fields.csv")
    x0, eta0, limit = f["x0"], f["eta0"], f["limit0"]
    ratio = eta0 / x0

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3), layout="constrained")

    # (a) ground state, log scale
    xs, ys, z = grid(f, "x_unit")
    im = axes[0].pcolormesh(xs, ys, np.abs(z), norm=LogNorm(), shading="nearest", cmap="viridis")
    fig.colorbar(im, ax=axes[0])
    axes[0].set_title(r"(a) ground state $x^*$ (log scale)")
    axes[0].set_aspect("equal")

    # (b) step at which each density reaches zero along eta_0
    xs, ys, z = grid(f, "limit0")
    z_plot = np.where(np.isfinite(z), z, np.nan)
    im = axes[1].pcolormesh(xs, ys, z_plot, norm=LogNorm(vmin=0.3, vmax=30), shading="nearest", cmap="magma")
    fig.colorbar(im, ax=axes[1])
    cs = axes[1].contour(xs, ys, np.where(np.isfinite(z), z, 1e3), levels=[1.0], colors="cyan", linewidths=1.0)
    axes[1].clabel(cs, fmt={1.0: r"$\tau=1$"}, fontsize=7)
    axes[1].set_title(r"(b) step at which $p_i$ reaches 0 (white: grows)")
    axes[1].set_aspect("equal")

    # (c) density along the step, relative to the start: p_i(tau)/p_i(0) = 1 + 2 tau eta_i/x_i
    tau = np.linspace(0, 1.2, 200)
    i_min = int(np.argmin(np.where(np.isfinite(limit), limit, np.inf)))
    r = np.hypot(f["px"], f["py"])
    picks = {
        "centre": int(np.argmin(r)),
        "half radius": int(np.argmin(np.abs(r - 0.5 * r.max()))),
        r"node of $\tau_{\max}$": i_min,
    }
    for name, i in picks.items():
        axes[2].plot(tau, 1 + 2 * tau * ratio[i], label=name)
    tau_max = limit[i_min]
    axes[2].axhline(0, color="k", lw=0.6)
    axes[2].axvline(tau_max, color="gray", ls="--", lw=0.8)
    axes[2].axvline(0.99 * tau_max, color="#d62728", ls=":", lw=1.0)
    axes[2].text(tau_max, 1.6, r" $\tau_{\max}$", color="gray", fontsize=7)
    axes[2].text(0.99 * tau_max, 1.85, r"$0.99\,\tau_{\max}$ ", color="#d62728", fontsize=7, ha="right")
    axes[2].set_xlabel(r"step size $\tau$")
    axes[2].set_ylabel(r"$p_i(\tau)/p_i(0)$")
    axes[2].set_ylim(-0.6, 2.2)
    axes[2].set_title("(c) density along the first step")
    axes[2].legend(loc="lower left")

    fig.savefig(out / "fig_first_step.pdf")
    plt.close(fig)
    return tau_max


# ---------------------------------------------------------------------------------------------------------------
# Figure 2: the first step along the diagonal, with and without the cap
# ---------------------------------------------------------------------------------------------------------------
def fig_first_iterate(data, out):
    f = read_csv(data / "levels" / "l6" / "fields.csv")
    on_diag = np.isclose(f["px"], f["py"])
    order = np.argsort(f["px"][on_diag])
    s = f["px"][on_diag][order] * math.sqrt(2)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), layout="constrained")
    for ax, log in zip(axes, [False, True]):
        ax.plot(s, f["x0"][on_diag][order], color="gray", ls="--", label=r"start $x_0$")
        ax.plot(s, f["x_unit"][on_diag][order], color="k", lw=1.5, label=r"ground state $x^*$")
        ax.plot(s, f["x1_unit"][on_diag][order], color=COLORS["unit"], ls="-.", label=r"$x_1$, norm retraction, $\tau=1$")
        for case in ["positive_theta0.5", "positive_theta0.99"]:
            ax.plot(s, f["x1_" + case][on_diag][order], color=COLORS[case],
                    label=r"$x_1$, density retraction, " + LABELS[case].split(", ")[1])
        ax.set_xlabel("position along the diagonal")
        if log:
            ax.set_yscale("log")
            ax.set_title("log scale")
        else:
            ax.set_title("linear scale")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7)
    fig.savefig(out / "fig_first_iterate.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Figures 3 and 4: iteration histories
# ---------------------------------------------------------------------------------------------------------------
def fig_history(run_dir, out, name, title, max_k=None, log_step=False, dots=True):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), layout="constrained")
    for case in CASES:
        it = read_csv(run_dir / f"iter_{case}.csv")
        k = it["k"]
        sel = np.ones_like(k, dtype=bool) if max_k is None else k <= max_k
        kw = dict(color=COLORS[case], label=LABELS[case], lw=1.0)
        axes[0].semilogy(k[sel], it["residual"][sel], **kw)
        (axes[1].semilogy if log_step else axes[1].plot)(k[sel], it["step"][sel], **kw)
        if dots and case != "unit":
            capped = (it["capped"] > 0) & sel
            axes[1].plot(k[capped], it["step"][capped], "o", ms=2.5, color=COLORS[case])
        axes[2].semilogy(k[sel], np.abs(it["min_value"][sel]), **kw)
    axes[0].set_title("residual")
    axes[1].set_title(r"step size $\tau_k$" + (r" (dots: capped at $\theta\,\tau_{\max}$)" if dots else
                                              r" (all positive-sphere steps capped)"))
    axes[2].set_title(r"$|\min_i x_{k,i}|$")
    for ax in axes:
        ax.set_xlabel("iteration $k$")
        if ax.get_yscale() == "log":
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=6))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7)
    fig.suptitle(title)
    fig.savefig(out / name)
    plt.close(fig)


def fig_limit(data, out):
    """Step limit tau_max along the iterations on the small and the large domain (theta = 0.99)."""
    fig, ax = plt.subplots(figsize=(5.0, 3.2), layout="constrained")
    for radius, color in [(4, "#2ca02c"), (6, "#1f77b4"), (10, "#d62728")]:
        it = read_csv(data / "radius" / f"r{radius}" / "iter_positive_theta0.99.csv")
        ax.semilogy(it["k"], it["tau_max"], color=color, label=f"radius {radius}")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.text(ax.get_xlim()[1], 1e3, r"$\tau_{\max}\geq 1/2$ in exact arithmetic ", fontsize=7, color="gray",
            ha="right")
    ax.set_ylim(1e-170, 1e20)
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(numticks=8))
    ax.set_xlabel("iteration $k$")
    ax.set_ylabel(r"$\tau_{\max}(x_k,\eta_k)$")
    ax.set_title(r"step limit, positive sphere, $\theta=0.99$")
    ax.legend()
    fig.savefig(out / "fig_limit.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------
def cell(r):
    it = int(r["iterations"])
    if r["failed"] == "1":
        return rf"undef.\ at {it}"
    if r["converged"] == "1":
        return f"{it}" if r["case"] == "unit" else f"{it} ({r['n_capped']})"
    return rf"$r={sci_tex(float(r['residual']))}$"  # not converged within the maximum number of iterations


def table(data, out, group, key, header):
    rows = read_rows(data / group / "summary.csv")
    by_param = {}
    for r in rows:
        by_param.setdefault(key(r), {})[r["case"]] = r
    lines = [r"\begin{tabular}{@{}l" + "r" * len(CASES) + r"@{}}", r"\toprule",
             header + r" & unit & $\theta=0.5$ & $\theta=0.9$ & $\theta=0.99$ \\", r"\midrule"]
    for param, cases in by_param.items():
        lines.append(param + " & " + " & ".join(cell(cases[c]) for c in CASES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(out / f"tab_{group}.tex", lines)
    return rows


def main():
    data, out = data_and_out_dirs(__doc__)

    tau_max = fig_first_step(data, out)
    fig_first_iterate(data, out)
    fig_history(data / "levels" / "l6", out, "fig_history_r4.pdf",
                "radius 4, level 6: option A converges", max_k=60)
    fig_history(data / "radius" / "r10", out, "fig_history_r10.pdf",
                "radius 10, level 5: the step limit collapses", max_k=200, log_step=True, dots=False)
    fig_limit(data, out)

    all_rows = []
    all_rows += table(data, out, "levels", lambda r: f"{r['level']} ({r['n_dofs']})", "level (DoFs)")
    all_rows += table(data, out, "radius", lambda r: r["radius"], "radius")
    all_rows += table(data, out, "beta", lambda r: r["beta"], r"$\kappa$")
    metric_names = {("ENERGY_ADAPTIVE", "0"): "energy-adaptive, $\\tau=1$",
                    ("ENERGY_ADAPTIVE", "1"): "energy-adaptive, Armijo",
                    ("MASS", "1"): "mass, Armijo", ("FROBENIUS", "1"): "Frobenius, Armijo"}
    all_rows += table(data, out, "metric", lambda r: metric_names[(r["metric"], r["line_search"])], "metric, step")
    all_rows += table(data, out, "tol", lambda r: f"${sci_tex(float(r['tol_inner_res']))}$", "inner tolerance")

    converged = [r for r in all_rows if r["case"] != "unit" and r["converged"] == "1"]
    max_diff = max(float(r["energy_diff"]) for r in converged)
    print(f"first-step limit tau_max (radius 4, level 6): {tau_max:.3f}")
    print(f"converged positive-sphere runs: {len(converged)}, max |E - E_unit| = {max_diff:.1e}")


if __name__ == "__main__":
    main()
