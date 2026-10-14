#!/usr/bin/env bash
# CPU-time study of the e-retraction (variants e, e-cap, e-cap-up, and clip-e-cap-up for the energy-adaptive metric)
# against the unit sphere with the normalization retraction: every discretization x radius 4, 6, 8, 10 x
# energy-adaptive (step 1, Armijo), mass (Armijo) and Frobenius (Armijo). Run it through staging/pos_sphere_cpu.sh;
# figures: staging/plot_pos_sphere_cpu.py econn. Findings: staging/pos_sphere_cpu.md.
#
# Usage: staging/pos_sphere_econn_cpu.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99)
declare -A metric=(
    [ea]="--metric-smooth energy_adaptive --step-size 1 --variants e,e-cap,e-cap-up,clip-e-cap-up"
    [ea-armijo]="--metric-smooth energy_adaptive ${LINE_SEARCH[*]} --ls-alpha 1 --variants e,e-cap,e-cap-up,clip-e-cap-up"
    [mass]="--metric-smooth mass ${LINE_SEARCH[*]} --ls-alpha 1 --variants e,e-cap,e-cap-up"
    [frobenius]="--metric-smooth frobenius ${LINE_SEARCH[*]} --ls-alpha 1 --variants e,e-cap,e-cap-up"
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
