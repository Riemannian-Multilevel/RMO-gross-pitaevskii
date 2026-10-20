#!/usr/bin/env bash
# CPU-time sweep over the problem parameters: discretization x beta 100, 1000, 10000 x potential square,
# optical_lattice x radius 4, 6, 8, 10, all at --levels $LEVEL (default 6), with the versions of the CPU-time studies
# (pos_sphere_analysis.sh MATRIX_ONLY=1, pos_sphere_exact_cpu.sh, pos_sphere_econn_cpu.sh,
# pos_sphere_econn_exact_cpu.sh, pos_sphere_norm_cpu.sh) merged into one run per metric and step rule. The exact step
# is run only for the density and e-retractions (norm-cap exact is left out; the unit sphere is the reference case of
# every run). Run it through staging/pos_sphere_cpu.sh; profiles: staging/profiles_pos_sphere_sweep.py.
#
# Run directories: matrix/b<beta>-<potential>/<disc>_r<radius>_<panel>, panel as in staging/profiles_pos_sphere_cpu.py.
#
# MAX_ITER (default 1000) replaces the iteration limit of COMMON, e.g. 2000 for P2 with the optical lattice at
# beta 100, where the unit sphere needs 1272-1732 iterations with the exact step (radius 6-10).
#
# LEVEL 5 for Q2, P2 gives about the degrees of freedom of Q1, P1 at level 6 (Q2: 1089 as Q1).
#
# Usage: [DISCS="q1 p1"] [MAX_ITER=1000] [LEVEL=6] staging/pos_sphere_sweep_cpu.sh <build dir> <output dir> [parallel jobs, default 6]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
study_init "$@"

common=("${COMMON[@]}" --thetas 0.99 --floor 1e-12 --floor-factor 1e-3)
for i in "${!common[@]}"; do [[ ${common[i]} == --max-iter ]] && common[i + 1]=${MAX_ITER:-1000}; done
density_ea="plain,clip,floor,floor-nocap,clip-floor,floor-cont,floor-cont-cap"
density_mf="plain,floor,floor-nocap,floor-cont,floor-cont-cap"
density_ea_exact="plain,clip,floor,clip-floor,floor-cont-cap"  # without cap there is no exact step
density_mf_exact="plain,floor,floor-cont-cap"
e_ea="e,e-cap,e-cap-up,clip-e-cap-up"
e_mf="e,e-cap,e-cap-up"
declare -A panel=(
    [ea]="--metric-smooth energy_adaptive --step-size 1 --variants $density_ea,$e_ea,norm-cap"
    [ea-armijo]="--metric-smooth energy_adaptive ${LINE_SEARCH[*]} --ls-alpha 1 --variants $density_ea,$e_ea,norm-cap"
    [ea-exact]="--metric-smooth energy_adaptive ${LINE_SEARCH[*]} --ls-alpha 1 --ls-exact --variants $density_ea_exact,$e_ea"
    [mass]="--metric-smooth mass ${LINE_SEARCH[*]} --ls-alpha 1 --variants $density_mf,$e_mf,norm-cap"
    [mass-exact]="--metric-smooth mass ${LINE_SEARCH[*]} --ls-alpha 1 --ls-exact --variants $density_mf_exact,$e_mf"
    [frobenius]="--metric-smooth frobenius ${LINE_SEARCH[*]} --ls-alpha 1 --variants $density_mf,$e_mf,norm-cap"
    [frobenius-exact]="--metric-smooth frobenius ${LINE_SEARCH[*]} --ls-alpha 1 --ls-exact --variants $density_mf_exact,$e_mf"
)
declare -A mesh=([q]=quadrilateral [p]=simplex)

for d in ${DISCS:-q1 p1}; do
    disc=(--mesh "${mesh[${d:0:1}]}" --degree "${d:1:1}" --levels "${LEVEL:-6}")
    for beta in 100 1000 10000; do
        for potential in square optical_lattice; do
            for radius in 4 6 8 10; do
                for p in ea ea-armijo ea-exact mass mass-exact frobenius frobenius-exact; do
                    args panel_args "${panel[$p]}"
                    add "matrix/b${beta}-${potential}/${d}_r${radius}_${p}" "${common[@]}" "${disc[@]}" \
                        --beta "$beta" --potential "$potential" --radius "$radius" "${panel_args[@]}"
                done
            done
        done
    done
done

study_run
