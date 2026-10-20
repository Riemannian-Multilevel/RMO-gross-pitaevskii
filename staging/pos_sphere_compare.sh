#!/usr/bin/env bash
# Runs of pos_sphere_study for tex/pos_sphere_compare.tex: the unit sphere against the positive sphere variants A, clip
# and floor (names of tex/pos_sphere_variants.tex), radius 4 to 10, beta 1000, mass and energy-adaptive metric; and the
# drift of A without normalization, radius 6. Figures: staging/plot_pos_sphere_compare.py.
#
# Note: the drift runs use the variant nonorm (A without normalization), which pos_sphere_study does not provide; they
# fail with "unknown variant nonorm". The comparison runs (plain, clip, floor-nocap) are not affected.
#
# Usage: staging/pos_sphere_compare.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99 --floor 1e-12 --beta 1000)
args q1 "${DISC[q1]}"
declare -A metric=(
    [ea]="--metric-smooth energy_adaptive"
    [mass]="--metric-smooth mass ${LINE_SEARCH[*]} --ls-alpha 1"
    [frobenius]="--metric-smooth frobenius ${LINE_SEARCH[*]} --ls-alpha 1"
)
declare -A variants=([ea]=plain,clip,floor-nocap [mass]=plain,floor-nocap)

# A, clip (energy-adaptive only) and floor against the unit sphere
for m in ea mass; do
    args metric_args "${metric[$m]}"
    for radius in 4 6 8 10; do
        add "compare/${m}_r$radius" "${common[@]}" "${q1[@]}" --radius "$radius" "${metric_args[@]}" \
            --variants "${variants[$m]}"
    done
done

# A with and without normalization, radius 6
for m in ea mass frobenius; do
    args metric_args "${metric[$m]}"
    add "divergence/${m}_r6" "${common[@]}" "${q1[@]}" --radius 6 "${metric_args[@]}" --variants plain,nonorm
done

study_run
