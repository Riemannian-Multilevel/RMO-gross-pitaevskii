#!/usr/bin/env python3
"""Figure of tex/pos_sphere_variants.tex on the mass and Frobenius metrics, from staging/pos_sphere_massfrob.sh.

Usage: staging/plot_pos_sphere_massfrob.py <data dir> <output dir>
"""
import matplotlib.pyplot as plt
import numpy as np

from pos_sphere_plot import data_and_out_dirs, read_csv, read_summary

plt.rcParams["legend.fontsize"] = 7


def main():
    data, out = data_and_out_dirs(__doc__)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), layout="constrained")

    # (a) unit sphere: starting step of the line search; positive sphere with option A
    ax = axes[0]
    for alpha, color in [("1", "#d62728"), ("1e-1", "#ff7f0e"), ("1e-2", "#2ca02c"), ("3e-3", "#9467bd")]:
        it = read_csv(data / "ls" / f"mass_a{alpha}" / "iter_unit.csv")
        ax.semilogy(it["k"], it["residual"], color=color, lw=1, label=rf"unit sphere, $\alpha_0={alpha}$")
    it = read_csv(data / "ls" / "mass_a1" / "iter_positive_theta0.99.csv")
    ax.semilogy(it["k"], it["residual"], color="k", lw=1.5, label=r"positive sphere, A, $\alpha_0=1$")
    ax.axhline(1e-9, color="gray", ls="--", lw=0.6)
    ax.set_xlabel("iteration $k$")
    ax.set_title("(a) mass metric, Armijo: residual (Q1, radius 10)")
    ax.legend()

    # (b) fixed steps: final residual against the step size
    ax = axes[1]
    for metric, marker in [("mass", "o"), ("frobenius", "s")]:
        taus, unit, pos = [], [], []
        for f in (data / "fixed").glob(f"{metric}_t*/summary.csv"):
            s = read_summary(f)
            taus.append(float(s["unit"]["step_size"]))
            unit.append(float(s["unit"]["residual"]))
            pos.append(float(s["positive_theta0.99"]["residual"]))
        order = np.argsort(taus)
        taus, unit, pos = np.array(taus)[order], np.array(unit)[order], np.array(pos)[order]
        ax.loglog(taus, unit, marker + "-", color="#1f77b4", label=f"unit sphere, {metric}")
        ax.loglog(taus, pos, marker + "--", color="#d62728", label=f"positive sphere (cap), {metric}")
        bound = np.nanmedian(read_csv(data / "fixed" / f"{metric}_t1" / "iter_unit.csv")["tau_bound"])
        ax.axvline(bound, color="#1f77b4" if metric == "mass" else "#9467bd", ls=":", lw=1)
        ax.text(bound, 3e-10 if metric == "mass" else 3e-9, rf" $\tau_{{\mathrm{{bound}}}}$ ({metric})", fontsize=6.5,
                rotation=90, va="bottom")
    ax.axhline(1e-9, color="gray", ls="--", lw=0.6)
    ax.set_xlabel(r"fixed step $\tau$")
    ax.set_ylabel("residual after 1000 iterations or at convergence")
    ax.set_title("(b) fixed steps (Q1, radius 10)")
    ax.legend(fontsize=6)

    # (c) step limit relative to its Z-matrix bound, option A, mass metric, radius 6
    ax = axes[2]
    for d, name, color in [("q1", "Q1 (Z)", "#2ca02c"), ("p1", "P1 (Z)", "#1f77b4"), ("p2", "P2 (non-Z)", "#d62728")]:
        it = read_csv(data / "bound" / f"{d}_r6_mass" / "iter_positive_theta0.99.csv")
        ok = np.isfinite(it["tau_max"])
        ax.semilogy(it["k"][ok], it["tau_max"][ok] / it["tau_bound"][ok], color=color, lw=1, label=name)
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_xlabel("iteration $k$")
    ax.set_title(r"(c) $\tau_{\max}/\tau_{\mathrm{bound}}$, option A, mass metric, radius 6")
    ax.legend()

    fig.savefig(out / "fig_massfrob.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
