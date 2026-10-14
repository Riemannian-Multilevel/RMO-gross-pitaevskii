#!/usr/bin/env bash
# CPU-time study of the e-retraction with the exact line search (--ls-exact): every discretization x radius 4, 6, 8,
# 10 x energy-adaptive, mass and Frobenius, with the unit sphere (closed-form exact step) and the variants e, e-cap,
# e-cap-up (and clip-e-cap-up for the energy-adaptive metric). The step rules Armijo and 1 are in
# staging/pos_sphere_econn_cpu.sh. Run it through staging/pos_sphere_cpu.sh; figures: staging/plot_pos_sphere_cpu.py
# econn. Findings: staging/pos_sphere_cpu.md.
#
# Usage: staging/pos_sphere_econn_exact_cpu.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99 "${LINE_SEARCH[@]}" --ls-alpha 1 --ls-exact)
declare -A metric=(
    [ea-exact]="--metric-smooth energy_adaptive --variants e,e-cap,e-cap-up,clip-e-cap-up"
    [mass-exact]="--metric-smooth mass --variants e,e-cap,e-cap-up"
    [frobenius-exact]="--metric-smooth frobenius --variants e,e-cap,e-cap-up"
)

for d in "${!DISC[@]}"; do
    args disc "${DISC[$d]}"
    for radius in 4 6 8 10; do
        for m in "${!metric[@]}"; do
            args metric_args "${metric[$m]}"
            add "matrix/${d}_r${radius}_${m}" "${common[@]}" "${disc[@]}" --radius "$radius" "${metric_args[@]}"
        done
    done
done

study_run
