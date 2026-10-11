"""Helpers of the plot_pos_sphere*.py scripts: command line, matplotlib setup and the CSV output of pos_sphere_study."""
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def data_and_out_dirs(doc):
    """<data dir> and <output dir> from the command line (usage: the docstring @p doc); creates the output dir."""
    if len(sys.argv) != 3:
        sys.exit(doc)
    data, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    return data, out


def read_rows(path):
    """Rows of a CSV file as dicts of strings."""
    with open(path) as f:
        return list(csv.DictReader(f))


def read_csv(path):
    """Columns of a CSV file of numbers (fields.csv, iter_*.csv) as arrays."""
    rows = read_rows(path)
    return {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}


def read_summary(path):
    """Rows of a summary.csv by case."""
    return {r["case"]: r for r in read_rows(path)}


def status(r):
    """Outcome of a case of summary.csv: failed (retraction undefined), converged or stopped."""
    if r["failed"] == "1":
        return "failed"
    return "converged" if r["converged"] == "1" else "stopped"


def sci_tex(a):
    """@p a as m\\cdot10^{e} with one decimal (math mode)."""
    if a == 0:
        return "0"
    if not math.isfinite(a):
        return r"\infty"
    m, e = f"{a:.1e}".split("e")
    return rf"{m}\cdot10^{{{int(e)}}}"


def write_table(path, lines):
    """Writes the lines of a tabular environment."""
    Path(path).write_text("\n".join(lines) + "\n")
