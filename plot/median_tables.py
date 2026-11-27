#!/usr/bin/env python3
"""Median of the CPU times over repeated runs of study.sh.

usage: median_tables.py OUT_DIR REP_DIR [REP_DIR ...]

Each REP_DIR holds the .org convergence tables of one repetition. The tables of the repetitions must
agree in every cell except the last column (`elapsed`, the CPU time); a difference is an error, since
the runs are deterministic. OUT_DIR gets the tables with `elapsed` replaced by its row-wise median,
written with 4 digits as in the tables.
"""
import statistics
import sys
from pathlib import Path


def median_table(name, tables):
    """Lines of the table `name`, given its lines in each repetition."""
    if len({len(t) for t in tables}) != 1:
        sys.exit(f"error: {name}: the repetitions differ in the number of lines")

    out = []
    for i, lines in enumerate(zip(*tables)):
        rows = [line.rstrip().rstrip("|").split("|") for line in lines]
        if len({tuple(r[:-1]) for r in rows}) != 1:
            sys.exit(f"error: {name}: the repetitions differ besides the CPU time:\n" + "\n".join(lines))
        if i == 0 or not lines[0].startswith("|"):  # header, or not a row of the table
            if len(set(lines)) != 1:
                sys.exit(f"error: {name}: the repetitions differ:\n" + "\n".join(lines))
            out.append(lines[0])
            continue
        try:
            times = [float(r[-1]) for r in rows]
        except ValueError:
            sys.exit(f"error: {name}: no CPU time in the last column:\n" + "\n".join(lines))

        old = rows[0][-1].strip()
        new = f"{statistics.median(times):.4e}"
        head, _, tail = lines[0].rpartition(old)
        out.append(head + new.rjust(len(old)) + tail)
    return out


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out_dir, *rep_dirs = map(Path, sys.argv[1:])

    names = sorted(p.name for p in rep_dirs[0].glob("*.org"))
    if not names:
        sys.exit(f"error: no .org tables in {rep_dirs[0]}")
    for d in rep_dirs:
        for name in names:
            if not (d / name).is_file():
                sys.exit(f"error: {name} is missing in {d}")

    # All tables are checked before the first one is written
    tables = {name: median_table(name, [(d / name).read_text().splitlines() for d in rep_dirs]) for name in names}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, lines in tables.items():
        (out_dir / name).write_text("\n".join(lines) + "\n")
    print(f"median of {len(rep_dirs)} repetition(s): {len(names)} tables written to {out_dir}")


if __name__ == "__main__":
    main()
