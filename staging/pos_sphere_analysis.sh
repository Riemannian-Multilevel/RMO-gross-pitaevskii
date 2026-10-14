#!/usr/bin/env bash
# Runs of pos_sphere_study for the analysis of the large-domain variants (tex/pos_sphere_variants.tex): every
# discretization (mesh, degree) x radius x metric with the applicable variants, and sweeps of the floor.
# Figures and tables: staging/plot_pos_sphere_analysis.py; CPU times: staging/pos_sphere_cpu.sh (MATRIX_ONLY=1).
#
# Usage: staging/pos_sphere_analysis.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99)
declare -A metric=(
    [ea]="--metric-smooth energy_adaptive --variants plain,clip,floor,floor-nocap,clip-floor,floor-cont"
    [ea-armijo]="--metric-smooth energy_adaptive ${LINE_SEARCH[*]} --ls-alpha 1 --variants plain,clip,floor,floor-nocap,clip-floor,floor-cont"
    [mass]="--metric-smooth mass ${LINE_SEARCH[*]} --ls-alpha 1 --variants plain,floor,floor-nocap,floor-cont"
    [frobenius]="--metric-smooth frobenius ${LINE_SEARCH[*]} --ls-alpha 1 --variants plain,floor,floor-nocap,floor-cont"
)

# Main matrix, with the default floor and continuation factor
for d in "${!DISC[@]}"; do
    args disc "${DISC[$d]}"
    for radius in 4 6 8 10; do
        for m in "${!metric[@]}"; do
            args metric_args "${metric[$m]}"
            add "matrix/${d}_r${radius}_${m}" "${common[@]}" --floor 1e-12 --floor-factor 1e-3 "${disc[@]}" \
                --radius "$radius" "${metric_args[@]}"
        done
    done
done

# Size of the fixed floor and of the continuation factor: radius 10, energy-adaptive (skipped with MATRIX_ONLY=1)
[[ ${MATRIX_ONLY:-0} == 1 ]] && { study_run; exit; }
for d in q1 p1; do
    args disc "${DISC[$d]}"
    for floor in 1e-6 1e-9 1e-12 1e-15 1e-20 1e-50; do
        add "floor/${d}_f$floor" "${common[@]}" "${disc[@]}" --radius 10 --metric-smooth energy_adaptive \
            --variants floor-nocap --floor "$floor"
    done
    for factor in 1e-1 1e-3 1e-5; do
        add "factor/${d}_c$factor" "${common[@]}" "${disc[@]}" --radius 10 --metric-smooth energy_adaptive \
            --variants floor-cont --floor-factor "$factor"
    done
done

study_run
