#!/usr/bin/env bash
# CPU-time study of the density retraction with the exact line search (--ls-exact, tex/exact_line_search.tex) against
# Armijo and, for the energy-adaptive metric, the fixed step 1: every discretization x radius 4, 6, 8, 10 x metric x
# step rule, each with the unit sphere (normalization retraction; exact: closed-form step) and the capped variants of
# the density retraction (plain = A, floor = floor+A, floor-cont-cap = cont+A; energy-adaptive also clip and
# clip-floor). The variants without cap have no exact step (the energy along the clamped curve can decrease towards
# t -> infinity) and are left out. Run it through staging/pos_sphere_cpu.sh; figures: staging/plot_pos_sphere_cpu.py
# exact. Findings: staging/pos_sphere_cpu.md.
#
# Usage: [RULES="exact armijo step1"] staging/pos_sphere_exact_cpu.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99)
declare -A variants=(
    [ea]="--metric-smooth energy_adaptive --variants plain,clip,floor,clip-floor,floor-cont-cap"
    [mass]="--metric-smooth mass --variants plain,floor,floor-cont-cap"
    [frobenius]="--metric-smooth frobenius --variants plain,floor,floor-cont-cap"
)
declare -A rules=(
    [exact]="${LINE_SEARCH[*]} --ls-alpha 1 --ls-exact"
    [armijo]="${LINE_SEARCH[*]} --ls-alpha 1"
    [step1]="--step-size 1"
)

for d in "${!DISC[@]}"; do
    args disc "${DISC[$d]}"
    for radius in 4 6 8 10; do
        for m in ea mass frobenius; do
            args variant_args "${variants[$m]}"
            for rule in ${RULES:-exact armijo step1}; do
                [[ $rule == step1 && $m != ea ]] && continue
                args rule_args "${rules[$rule]}"
                add "matrix/${d}_r${radius}_${m}-${rule}" "${common[@]}" "${disc[@]}" --radius "$radius" \
                    "${variant_args[@]}" "${rule_args[@]}"
            done
        done
    done
done

study_run
