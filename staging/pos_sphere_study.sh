#!/usr/bin/env bash
# Runs of pos_sphere_study for tex/pos_sphere_domain.tex: gradient descent on the positive unit-mass sphere
# (density retraction with fraction-to-the-boundary steps) against the unit-mass sphere (norm retraction), for lumped
# Q1 elements. Figures and tables: staging/plot_pos_sphere.py.
#
# Usage: staging/pos_sphere_study.sh <build dir> <output dir> [parallel jobs, default 6]
# Writes one directory per run (summary.csv, fields.csv, iter_*.csv) and one summary.csv per group of runs.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.5,0.9,0.99)

# Mesh refinement: radius 4, energy-adaptive, step 1
for level in 4 5 6 7; do
    add "levels/l$level" "${common[@]}" --levels "$level" --radius 4
done

# Domain size: level 5; the ground state decays like exp(-|x|^2/2) towards the boundary
for radius in 4 6 8 10; do
    add "radius/r$radius" "${common[@]}" --levels 5 --radius "$radius"
done

# Interaction strength: radius 4, level 6
for beta in 10 100 1000; do
    add "beta/b$beta" "${common[@]}" --levels 6 --radius 4 --beta "$beta"
done

# Metric and step rule: radius 4, level 6
add metric/ea_step "${common[@]}" --levels 6 --radius 4 --metric-smooth energy_adaptive
add metric/ea_ls "${common[@]}" --levels 6 --radius 4 --metric-smooth energy_adaptive "${LINE_SEARCH[@]}" --ls-alpha 1
add metric/mass_ls "${common[@]}" --levels 6 --radius 4 --metric-smooth mass "${LINE_SEARCH[@]}" --ls-alpha 1
add metric/frobenius_ls "${common[@]}" --levels 6 --radius 4 --metric-smooth frobenius "${LINE_SEARCH[@]}" --ls-alpha 1

# Inner solver accuracy on the large domain: radius 10, level 5
for tol in 1e-2 1e-4 1e-6; do
    add "tol/t$tol" "${common[@]}" --levels 5 --radius 10 --tol-inner-res "$tol"
done

study_run

# Summaries per group, in the order of the rows of the tables
merge_summaries levels l4 l5 l6 l7
merge_summaries radius r4 r6 r8 r10
merge_summaries beta b10 b100 b1000
merge_summaries metric ea_step ea_ls mass_ls frobenius_ls
merge_summaries tol t1e-2 t1e-4 t1e-6
