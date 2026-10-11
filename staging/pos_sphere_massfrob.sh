#!/usr/bin/env bash
# Runs of pos_sphere_study for the mass and Frobenius metrics (tex/pos_sphere_variants.tex): step limit against its
# Z-matrix lower bound, fixed steps on the unit sphere, and the starting step of the line search. Figure:
# staging/plot_pos_sphere_massfrob.py.
#
# Usage: staging/pos_sphere_massfrob.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99 --floor 1e-12)
args q1 "${DISC[q1]}"

# Step limit against its bound: Z (q1, p1) and non-Z (p2) discretizations with a positive ground state
for d in q1 p1 p2; do
    args disc "${DISC[$d]}"
    for radius in 4 6; do
        for m in mass frobenius; do
            add "bound/${d}_r${radius}_$m" "${common[@]}" "${disc[@]}" --radius "$radius" --metric-smooth "$m" \
                "${LINE_SEARCH[@]}" --ls-alpha 1 --variants plain,floor
        done
    done
done

# Fixed steps on the unit sphere (and the positive sphere, where the cap limits them), Q1, radius 10
for m in mass frobenius; do
    for tau in 1e-3 2e-3 4e-3 8e-3 1.6e-2 3.2e-2 1; do
        add "fixed/${m}_t$tau" "${common[@]}" "${q1[@]}" --radius 10 --metric-smooth "$m" --step-size "$tau" \
            --variants plain
    done
done

# Starting step of the line search on the unit sphere, Q1, radius 10
for m in mass frobenius; do
    for alpha in 1 1e-1 1e-2 3e-3; do
        add "ls/${m}_a$alpha" "${common[@]}" "${q1[@]}" --radius 10 --metric-smooth "$m" "${LINE_SEARCH[@]}" \
            --ls-alpha "$alpha" --variants plain
    done
done

study_run
