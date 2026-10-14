#!/usr/bin/env bash
# Runs of pos_sphere_study for the e-connection: the e-retraction of the positive sphere (variant e, the e-geodesic of
# the densities; e-cap, e-cap-up with a relative step limit) against the density retraction (m-geodesic, plain) and, on the unit
# sphere, the norm, orthographic (ortho) and exponential map (exp), for every metric and step rule; for the
# energy-adaptive metric also with the clipped direction (clip-e, clip-e-cap, clip-e-cap-up). Findings: staging/pos_sphere_econn.md.
#
# Usage: staging/pos_sphere_econn.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99)
declare -A variants=(
    [energy_adaptive]="--variants plain,e,e-cap,e-cap-up,clip-e,clip-e-cap,clip-e-cap-up,ortho,exp"
    [mass]="--variants plain,e,e-cap,e-cap-up,ortho,exp"
    [frobenius]="--variants plain,e,e-cap,e-cap-up,ortho,exp"
)
# Step rules per metric: <name> <options>
ea_rules=(
    "step1      --step-size 1"
    "armijo     ${LINE_SEARCH[*]} --ls-alpha 1"
    "bb_max     ${LINE_SEARCH[*]} --ls-alpha 1 --ls-initial alternating --ls-reference max"
)
mf_rules=(
    "armijo       ${LINE_SEARCH[*]} --ls-alpha 1"
    "bb_monotone  ${LINE_SEARCH[*]} --ls-alpha 1 --ls-initial alternating --ls-reference monotone"
    "bb_max       ${LINE_SEARCH[*]} --ls-alpha 1 --ls-initial alternating --ls-reference max"
)

# Main matrix: discretization x radius x metric x step rule
for d in q1 p1 q2 p2; do
    args disc "${DISC[$d]}"
    for radius in 4 6 10; do
        for m in energy_adaptive mass frobenius; do
            if [[ $m == energy_adaptive ]]; then rules=("${ea_rules[@]}"); else rules=("${mf_rules[@]}"); fi
            args variant_args "${variants[$m]}"
            for entry in "${rules[@]}"; do
                read -r name options <<< "$entry"
                args rule_args "$options"
                add "matrix/${d}_r${radius}_${m}_$name" "${common[@]}" "${disc[@]}" --radius "$radius" \
                    --metric-smooth "$m" "${variant_args[@]}" "${rule_args[@]}"
            done
        done
    done
done

# Fixed steps for the mass and Frobenius metrics, Q1, radius 4 and 10
args q1 "${DISC[q1]}"
for radius in 4 10; do
    for m in mass frobenius; do
        args variant_args "${variants[$m]}"
        for tau in 1e-3 4e-3 1.6e-2 6.4e-2 2.5e-1 1; do
            add "fixed/r${radius}_${m}_t$tau" "${common[@]}" "${q1[@]}" --radius "$radius" --metric-smooth "$m" \
                "${variant_args[@]}" --step-size "$tau"
        done
    done
done

# Fixed steps for the energy-adaptive metric, Q1 and Q2, radius 10
for d in q1 q2; do
    args disc "${DISC[$d]}"
    args variant_args "${variants[energy_adaptive]}"
    for tau in 0.25 0.5 1 1.5; do
        add "fixed/${d}_r10_energy_adaptive_t$tau" "${common[@]}" "${disc[@]}" --radius 10 \
            --metric-smooth energy_adaptive "${variant_args[@]}" --step-size "$tau"
    done
done

# Long runs of the e-retraction (sublinear rate?): Q1 and P1, radius 10, mass and Frobenius, 5000 iterations
long=(--mass-lumping --max-iter 5000 --tol-residual 1e-9 --thetas 0.99)
for d in q1 p1; do
    args disc "${DISC[$d]}"
    for m in mass frobenius; do
        for entry in "${mf_rules[@]:1}"; do
            read -r name options <<< "$entry"
            args rule_args "$options"
            add "long/${d}_r10_${m}_$name" "${long[@]}" "${disc[@]}" --radius 10 --metric-smooth "$m" \
                --variants e,e-cap,e-cap-up "${rule_args[@]}"
        done
    done
done

study_run
