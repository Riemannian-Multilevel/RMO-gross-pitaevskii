#!/usr/bin/env python3
"""Figure and tables of tex/pos_sphere_variants.tex on the nonmonotone line search and Barzilai-Borwein steps, from
staging/pos_sphere_nonmonotone.sh.

Usage: staging/plot_pos_sphere_nonmonotone.py <data dir> <output dir>

Outputs: fig_nonmonotone.pdf, tab_nonmonotone.tex (iterations per discretization, metric and variant, fixed start
against Barzilai-Borwein), tab_nonmonotone_reference.tex (reference value of the Armijo condition).
"""
import matplotlib.pyplot as plt

from pos_sphere_plot import data_and_out_dirs, read_csv, read_summary, write_table

plt.rcParams["legend.fontsize"] = 7

TOL = 1e-9
DISCS = [("q1_r10", r"$Q_1$, $R=10$"), ("p1_r10", r"$P_1$, $R=10$"), ("p2_r6", r"$P_2$, $R=6$"),
         ("q2_r6", r"$Q_2$, $R=6$")]
METRICS = [("mass", "mass"), ("frobenius", "Frob."), ("ea", "EA")]
CASES = [("unit", "unit"), ("positive_theta0.99", "A"), ("positive_floor_theta0.99", "floor+A"),
         ("positive_floor-nocap", "floor"), ("positive_clip_theta0.99", "clip")]


def kkt_iteration(it):
    """First iterate with free residual and negative part of the multipliers below the tolerance."""
    ok = (it["free_residual"] < TOL) & (it["nu_min"] > -TOL)
    return int(it["k"][ok][0]) if ok.any() else None


def outcome(data, disc, run, case):
    """Iterations to the tolerance (Q2: to the KKT tolerance), or a marker: F failed (collapse), s stalled."""
    run_dir = data / disc / run
    s = read_summary(run_dir / "summary.csv")
    if case not in s:
        return None
    r = s[case]
    constrained = disc == "q2_r6" and case != "unit"
    if r["failed"] == "1":
        return "F"
    if constrained:
        k = kkt_iteration(read_csv(run_dir / f"iter_{case}.csv"))
        return k if k is not None else "s"
    return int(r["iterations"]) if r["converged"] == "1" else "s"


def cell(v):
    if v is None:
        return ""
    if v == "F":
        return r"\textsf{F}"
    if v == "s":
        return r"--"
    return str(v)


def table_main(data, out):
    lines = [r"\begin{tabular}{@{}ll" + "rr" * len(CASES) + "@{}}", r"\toprule",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in CASES) + r" \\",
             " ".join(rf"\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(CASES))),
             "discretization & metric & " + " & ".join(["fix.", "BB"] * len(CASES)) + r" \\", r"\midrule"]
    for disc, disc_name in DISCS:
        for m, (metric, metric_name) in enumerate(METRICS):
            cells = []
            for case, _ in CASES:
                for run in (f"{metric}_fixed_monotone", f"{metric}_bb_max"):
                    cells.append(cell(outcome(data, disc, run, case)))
            lines.append((disc_name if m == 0 else "") + f" & {metric_name} & " + " & ".join(cells) + r" \\")
        if disc != DISCS[-1][0]:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(out / "tab_nonmonotone.tex", lines)


def table_reference(data, out):
    """Iterations for every line search, mass metric, unit sphere and floor (no cap)."""
    runs = [("fixed_monotone", r"$\alpha_0=1$, monotone"), ("fixed_max", r"$\alpha_0=1$, max"),
            ("fixed_average", r"$\alpha_0=1$, average"), ("bb_monotone", "ABB, monotone"), ("bb_max", "ABB, max"),
            ("bb_average", "ABB, average"), ("bb1_max", "BB1, max"), ("bb2_max", "BB2, max")]
    cols = [(d, c) for d, _ in DISCS for c in ("unit", "positive_floor-nocap")]
    lines = [r"\begin{tabular}{@{}l" + "rr" * len(DISCS) + "@{}}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{name}}}" for _, name in DISCS) + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DISCS))),
             "line search & " + " & ".join(["unit", "floor"] * len(DISCS)) + r" \\", r"\midrule"]
    for run, name in runs:
        cells = [cell(outcome(data, d, f"mass_{run}", c)) for d, c in cols]
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table(out / "tab_nonmonotone_reference.tex", lines)


def figure(data, out):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), layout="constrained")

    # (a) unit sphere, mass metric: the starting step matters, the reference value does not
    ax = axes[0]
    for run, label, color, ls in [("mass_fixed_monotone", r"$\alpha_0=1$, monotone", "#d62728", "-"),
                                  ("mass_fixed_max", r"$\alpha_0=1$, max ($m=10$)", "#ff7f0e", "--"),
                                  ("mass_bb_monotone", "ABB, monotone", "#1f77b4", "-"),
                                  ("mass_bb_max", "ABB, max ($m=10$)", "#2ca02c", "--"),
                                  ("mass_bb_average", r"ABB, average ($\eta=0.85$)", "#9467bd", ":")]:
        it = read_csv(data / "q1_r10" / run / "iter_unit.csv")
        ax.semilogy(it["k"], it["residual"], color=color, ls=ls, lw=1, label=label)
    ax.axhline(TOL, color="gray", ls="--", lw=0.6)
    ax.set_xlim(0, 400)
    ax.set_xlabel("iteration $k$")
    ax.set_title(r"(a) unit sphere, mass metric: residual ($Q_1$, $R=10$)")
    ax.legend()

    # (b) constrained solution: free residual
    ax = axes[1]
    for run, case, label, color, ls in [
            ("mass_fixed_monotone", "positive_floor-nocap", r"mass, floor, $\alpha_0=1$", "#d62728", "-"),
            ("mass_bb_monotone", "positive_floor-nocap", "mass, floor, ABB monotone", "#1f77b4", "-"),
            ("mass_bb_max", "positive_floor-nocap", "mass, floor, ABB max", "#2ca02c", "--"),
            ("mass_bb_max", "positive_floor_theta0.99", "mass, floor+A, ABB max", "#17becf", ":"),
            ("frobenius_bb_max", "positive_floor-nocap", "Frobenius, floor, ABB max", "#8c564b", "--"),
            ("ea_bb_max", "positive_floor-nocap", "EA, floor, ABB max", "#9467bd", "-"),
            ("ea_step1", "positive_clip_theta0.99", r"EA, clip, $\tau=1$", "#e377c2", ":")]:
        it = read_csv(data / "q2_r6" / run / f"iter_{case}.csv")
        ax.semilogy(it["k"], it["free_residual"], color=color, ls=ls, lw=1, label=label)
    ax.axhline(TOL, color="gray", ls="--", lw=0.6)
    ax.set_xlim(0, 300)
    ax.set_xlabel("iteration $k$")
    ax.set_title(r"(b) constrained solution: free residual ($Q_2$, $R=6$)")
    ax.legend(fontsize=6.5)

    # (c) step sizes on the positive sphere: Barzilai-Borwein proposal, cap and accepted step
    ax = axes[2]
    it = read_csv(data / "q1_r10" / "mass_bb_max" / "iter_positive_theta0.99.csv")
    ax.semilogy(it["k"], it["initial_step"], color="#2ca02c", lw=1, label="A: ABB step")
    ax.semilogy(it["k"], it["cap"], color="k", lw=0.8, ls=":", label=r"A: cap $\theta\tau_{\max}$")
    ax.semilogy(it["k"], it["step"], ".", color="#d62728", ms=2.5, label="A: accepted step")
    ax.semilogy(it["k"], it["tau_bound"], color="#7f7f7f", lw=0.8, ls="--", label=r"$\tau_{\mathrm{bound}}$")
    it = read_csv(data / "q1_r10" / "mass_bb_max" / "iter_positive_floor-nocap.csv")
    ax.semilogy(it["k"], it["step"], "s", color="#1f77b4", ms=2, label="floor (no cap): accepted ABB step")
    ax.set_xlabel("iteration $k$")
    ax.set_title(r"(c) step sizes, mass metric, ABB max ($Q_1$, $R=10$)")
    ax.legend()

    fig.savefig(out / "fig_nonmonotone.pdf")
    plt.close(fig)


def main():
    data, out = data_and_out_dirs(__doc__)
    table_main(data, out)
    table_reference(data, out)
    figure(data, out)


if __name__ == "__main__":
    main()
