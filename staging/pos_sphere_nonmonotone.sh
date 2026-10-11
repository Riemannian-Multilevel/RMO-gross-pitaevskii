#!/usr/bin/env bash
# Runs of pos_sphere_study for the nonmonotone line search and Barzilai-Borwein steps (tex/pos_sphere_variants.tex):
# discretizations with tiny (Q1, P1, radius 10) or vanishing (Q2, a constrained solution) ground-state tails, and
# a non-Z discretization with a positive ground state (P2), for every metric, against the reference value of the Armijo condition
# (--ls-reference) and the starting step (--ls-initial). Figure and tables: staging/plot_pos_sphere_nonmonotone.py.
#
# Usage: staging/pos_sphere_nonmonotone.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99 --floor 1e-12)
# Discretization and radius: <disc>_r<radius>
cases=(q1_r10 p1_r10 q2_r6 p2_r6)
declare -A metric=(
    [ea]="--metric-smooth energy_adaptive --variants plain,clip,floor,floor-nocap"
    [mass]="--metric-smooth mass --variants plain,floor,floor-nocap"
    [frobenius]="--metric-smooth frobenius --variants plain,floor,floor-nocap"
)
# Line searches: <name> <options>
lsearch=(
    "fixed_monotone  --ls-initial fixed --ls-reference monotone"
    "fixed_max       --ls-initial fixed --ls-reference max"
    "fixed_average   --ls-initial fixed --ls-reference average"
    "bb_monotone     --ls-initial alternating --ls-reference monotone"
    "bb_max          --ls-initial alternating --ls-reference max"
    "bb_average      --ls-initial alternating --ls-reference average"
    "bb1_max         --ls-initial bb1 --ls-reference max"
    "bb2_max         --ls-initial bb2 --ls-reference max"
)

for c in "${cases[@]}"; do
    args disc "${DISC[${c%_r*}]} --radius ${c##*_r}"
    for m in "${!metric[@]}"; do
        args metric_args "${metric[$m]}"
        for entry in "${lsearch[@]}"; do
            read -r name options <<< "$entry"
            args ls_args "$options"
            add "$c/${m}_$name" "${common[@]}" "${disc[@]}" "${metric_args[@]}" "${LINE_SEARCH[@]}" --ls-alpha 1 \
                "${ls_args[@]}"
        done
    done
    # Energy-adaptive metric with the fixed step 1, the default of the earlier studies
    args metric_args "${metric[ea]}"
    add "$c/ea_step1" "${common[@]}" "${disc[@]}" "${metric_args[@]}"
done

study_run
