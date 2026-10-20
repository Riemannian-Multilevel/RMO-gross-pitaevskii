#!/usr/bin/env python3
"""Figures of tex/pos_sphere_compare.tex, from staging/pos_sphere_compare.sh.

fig_divergence.pdf needs the runs of the variant nonorm, which pos_sphere_study does not provide (see
staging/pos_sphere_compare.sh); fig_compare_{ea,mass}.pdf do not.

Usage: staging/plot_pos_sphere_compare.py <data dir> <output dir>
"""
import matplotlib.pyplot as plt
import numpy as np

from pos_sphere_plot import data_and_out_dirs, read_csv, read_summary, status

RADII = [4, 6, 8, 10]
# case of pos_sphere_study, label, colour, line style
# (names as in fig_analysis_overview_*.pdf, staging/plot_pos_sphere_analysis.py)
CASES = [("unit", "unit", "k", "-"),
         ("positive_theta0.99", "A", "#d62728", "--"),
         ("positive_clip_theta0.99", "clip", "#2ca02c", "-."),
         ("positive_floor-nocap", "floor", "#1f77b4", ":")]
TITLES = {"ea": r"energy-adaptive, $\tau=1$", "mass": r"mass, Armijo, $\alpha_0=1$"}
TINY = 1e-17  # plotted in place of zero on log axes


def compare(data, out, metric):
    """Residual and accepted step against the iteration, one column per radius."""
    fig, axes = plt.subplots(2, len(RADII), figsize=(12.5, 5), sharex="col", layout="constrained")
    for col, radius in enumerate(RADII):
        run = data / "compare" / f"{metric}_r{radius}"
        summary = read_summary(run / "summary.csv")
        for case, label, color, ls in CASES:
            if case not in summary:
                continue
            it = read_csv(run / f"iter_{case}.csv")
            failed = status(summary[case]) == "failed"
            axes[0, col].semilogy(it["k"], it["residual"], color=color, ls=ls, lw=1.2, label=label)
            # steps oscillate under Armijo: dots instead of lines
            axes[1, col].semilogy(it["k"], it["step"], ".", color=color, ms=1.5)
            for ax, key in zip(axes[:, col], ["residual", "step"]):
                if failed:
                    ok = np.isfinite(it[key])
                    ax.plot(it["k"][ok][-1], it[key][ok][-1], "x", color=color, ms=7)
        axes[0, col].axhline(1e-9, color="gray", ls="--", lw=0.6)
        axes[0, col].set_title(f"radius {radius}")
        axes[1, col].set_xlabel("iteration $k$")
    axes[0, 0].set_ylabel(r"residual $r(x_k)$")
    axes[1, 0].set_ylabel(r"accepted step $\tau_k$")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle(rf"{TITLES[metric]}, $\beta=1000$, $Q_1$, level 6; $\times$: retraction undefined (underflow)")
    fig.savefig(out / f"fig_compare_{metric}.pdf")
    plt.close(fig)


def divergence(data, out):
    """Drift from the sphere without the normalization, radius 6."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), layout="constrained")
    run = data / "divergence" / "mass_r6"
    curves = [(run, "unit", "mass, unit", "k", "-"),
              (run, "positive_theta0.99", "mass, A", "#1f77b4", "-"),
              (run, "positive_nonorm_theta0.99", "mass, A without normalization", "#d62728", "--")]
    for ax, key, title in zip(axes, ["norm_dev", "energy", "residual"],
                              [r"(a) $|x_k^\top D x_k-1|$", r"(b) energy $f(x_k)$", r"(c) residual $r(x_k)$"]):
        for path, case, label, color, ls in curves:
            it = read_csv(path / f"iter_{case}.csv")
            y = np.maximum(it[key], TINY) if key == "norm_dev" else it[key]
            ax.semilogy(it["k"], y, color=color, ls=ls, lw=1.2, label=label)
        ax.set_xlabel("iteration $k$")
        ax.set_title(title)
        ax.set_xlim(0, 300)
    # other metrics without normalization: no drift (Frobenius) or damped drift (energy-adaptive)
    for metric, label, color in [("frobenius", "Frobenius, A without normalization", "#9467bd"),
                                 ("ea", "energy-adaptive, A without normalization", "#2ca02c")]:
        it = read_csv(data / "divergence" / f"{metric}_r6" / "iter_positive_nonorm_theta0.99.csv")
        axes[0].semilogy(it["k"], np.maximum(it["norm_dev"], TINY), color=color, ls=":", lw=1.2, label=label)
    axes[0].axhline(TINY, color="gray", lw=0.4)
    axes[0].text(300, TINY, "exactly $0$ ", ha="right", va="bottom", fontsize=7, color="gray")
    axes[2].axhline(1e-9, color="gray", ls="--", lw=0.6)
    axes[0].legend(fontsize=6.5, loc="center right")
    fig.suptitle(r"A without normalization: radius 6, $\beta=1000$, $Q_1$, level 6, $\theta=0.99$")
    fig.savefig(out / "fig_divergence.pdf")
    plt.close(fig)


def main():
    data, out = data_and_out_dirs(__doc__)
    for metric in ["ea", "mass"]:
        compare(data, out, metric)
    divergence(data, out)


if __name__ == "__main__":
    main()
