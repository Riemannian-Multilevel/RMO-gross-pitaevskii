#!/usr/bin/env python3
"""Step sizes of the single-level versions per metric, from the iteration files of the parameter sweep
(staging/pos_sphere_sweep_cpu.sh).

Usage: staging/plot_pos_sphere_steps.py <data dir> <output dir> <disc> <beta> <potential> <radius>

<data dir> contains rep1/matrix/b<beta>-<potential>/<disc>_r<radius>_<panel>/iter_<case>.csv. For one problem, writes
fig_steps_<disc>_b<beta>_<potential>_r<radius>.pdf: one column per metric (EA, mass, Frobenius), with
  (a) the step size tau_k (comparable within a metric only; dots: capped steps),
  (b) tau_k max_i |eta_i| / x_i, the largest first-order relative change of a component in step k (comparable
      across metrics: the gradients of the metrics differ in scale, e.g. by the lumped masses ~ h^2 between mass and
      Frobenius),
  (c) the residual,
for the best version per rule and retraction of profiles_sweep_q1p1_all.pdf (and the unit sphere, Armijo/fixed step).
"""
import sys
from pathlib import Path

import numpy as np

from pos_sphere_plot import plt, read_csv

# (panel, case, label): the versions of profiles_sweep_q1p1_all.pdf, n: normalization, m: density, e: e-retraction
VERSIONS = {
    "EA": [("ea", "unit", "τ=1, n: unit"), ("ea", "positive_clip-floor_theta0.99", "τ=1, m: clip+floor"),
           ("ea", "positive_clip-e-cap-up_theta0.99", "τ=1, e: clip-e-cap-up"),
           ("ea-armijo", "unit", "Armijo, n: unit"), ("ea-armijo", "positive_floor-nocap", "Armijo, m: floor"),
           ("ea-armijo", "positive_clip-e-cap-up_theta0.99", "Armijo, e: clip-e-cap-up"),
           ("ea-exact", "positive_floor-cont-cap_theta0.99", "exact, m: cont.+A"),
           ("ea-exact", "positive_clip-e-cap-up_theta0.99", "exact, e: clip-e-cap-up")],
    "mass": [("mass", "unit", "Armijo, n: unit"), ("mass", "positive_theta0.99", "Armijo, m: A"),
             ("mass", "positive_e-cap-up_theta0.99", "Armijo, e: e-cap-up"),
             ("mass-exact", "positive_floor_theta0.99", "exact, m: floor+A"),
             ("mass-exact", "positive_e-cap-up_theta0.99", "exact, e: e-cap-up")],
    "Frobenius": [("frobenius", "unit", "Armijo, n: unit"), ("frobenius", "positive_theta0.99", "Armijo, m: A"),
                  ("frobenius", "positive_e-cap_theta0.99", "Armijo, e: e-cap"),
                  ("frobenius-exact", "positive_floor_theta0.99", "exact, m: floor+A"),
                  ("frobenius-exact", "positive_e-cap-up_theta0.99", "exact, e: e-cap-up")],
}
FAMILY_COLOR = {"n": "#555555", "m": "#1f77b4", "e": "#d62728"}
RULE_STYLE = {"τ=1": ":", "Armijo": "--", "exact": "-"}


def main():
    if len(sys.argv) != 7:
        sys.exit(__doc__)
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    disc, beta, potential, radius = sys.argv[3:7]
    out.mkdir(parents=True, exist_ok=True)
    base = data / "rep1" / "matrix" / f"b{beta}-{potential}"
    fig, axes = plt.subplots(3, 3, figsize=(11, 8.2), sharex="col", layout="constrained")
    for j, (metric, versions) in enumerate(VERSIONS.items()):
        for panel, case, label in versions:
            f = base / f"{disc}_r{radius}_{panel}" / f"iter_{case}.csv"
            if not f.exists():
                print(f"missing: {f}")
                continue
            it = read_csv(f)
            k, step = it["k"], it["step"]
            sel = np.isfinite(step) & (step > 0)
            rule, family = label.split(", ")[0], label.split(", ")[1][0]
            kw = dict(color=FAMILY_COLOR[family], ls=RULE_STYLE[rule], lw=1.1, label=label)
            axes[0, j].semilogy(k[sel], step[sel], **kw)
            capped = sel & (it["capped"] == 1)
            axes[0, j].plot(k[capped], step[capped], "o", ms=2, color=FAMILY_COLOR[family])
            change = step * it["rel_rate"]
            axes[1, j].semilogy(k[sel], change[sel], **kw)
            axes[2, j].semilogy(k, it["residual"], **kw)
        axes[0, j].set_title(f"{metric}\n(a) step size $\\tau_k$ (dots: capped)", fontsize=9)
        axes[1, j].set_title(r"(b) $\tau_k\max_i|\eta_i|/x_i$ (relative change)", fontsize=9)
        axes[2, j].set_title("(c) residual", fontsize=9)
        axes[2, j].set_xlabel("iteration $k$")
        axes[0, j].legend(fontsize=6, loc="best")
    ymin = min(ax.get_ylim()[0] for ax in axes[1])
    ymax = max(ax.get_ylim()[1] for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(ymin, ymax)  # (b) is comparable across the metrics: common scale
    fig.suptitle(f"{disc.upper()}, β = {beta}, V = {potential.replace('_', ' ')}, radius {radius}, level 6", fontsize=10)
    name = f"fig_steps_{disc}_b{beta}_{potential}_r{radius}.pdf"
    fig.savefig(out / name)
    plt.close(fig)
    print(name)


if __name__ == "__main__":
    main()
