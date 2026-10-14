#!/usr/bin/env bash
# CPU-time study of the normalization retraction on the positive sphere with the positivity cap (variant norm-cap,
# NormPosSphere): every discretization x radius 4, 6, 8, 10 x metric x step rule (exact: the closed-form ray step,
# capped; Armijo; step 1 for the energy-adaptive metric). The unit sphere is run as reference. Run it through
# staging/pos_sphere_cpu.sh; the run directories end in +norm, so that staging/plot_pos_sphere_cpu.py exact can merge
# them with staging/pos_sphere_exact_cpu.sh. Findings: staging/pos_sphere_cpu.md.
#
# Usage: [RULES="exact armijo step1"] staging/pos_sphere_norm_cpu.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99 --variants norm-cap)
declare -A metrics=([ea]=energy_adaptive [mass]=mass [frobenius]=frobenius)
declare -A rules=(
    [exact]="${LINE_SEARCH[*]} --ls-alpha 1 --ls-exact"
    [armijo]="${LINE_SEARCH[*]} --ls-alpha 1"
    [step1]="--step-size 1"
)

for d in "${!DISC[@]}"; do
    args disc "${DISC[$d]}"
    for radius in 4 6 8 10; do
        for m in ea mass frobenius; do
            for rule in ${RULES:-exact armijo step1}; do
                [[ $rule == step1 && $m != ea ]] && continue
                args rule_args "${rules[$rule]}"
                add "matrix/${d}_r${radius}_${m}-${rule}+norm" "${common[@]}" "${disc[@]}" --radius "$radius" \
                    --metric-smooth "${metrics[$m]}" "${rule_args[@]}"
            done
        done
    done
done

study_run
