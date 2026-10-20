#!/usr/bin/env bash
# Runs of pos_sphere_study for the large-domain variants of tex/pos_sphere_domain.tex: clipping w = A(x)^{-1} D x at
# zero against a floor (threshold) in the density retraction.
#
# Usage: staging/pos_sphere_variants.sh <build dir> <output dir>
set -euo pipefail

build=${1:?build directory}
out=${2:?output directory}
study="$build/pos_sphere_study"

common=(--mass-lumping --max-iter 1000 --tol-residual 1e-9)
variants=(--variants plain,clip,floor,floor-nocap,clip-floor --thetas 0.99)
ls=(--line-search --ls-max-iter 40 --ls-beta 0.5 --ls-min 1e-12 --ls-alpha 1)

run() {  # run <group> <name> <args...>
    local group=$1 name=$2
    shift 2
    mkdir -p "$out/$group"
    "$study" "${common[@]}" "$@" --out "$out/$group/$name" | grep -v '^Number'
    tail -n +2 "$out/$group/$name/summary.csv" >> "$out/$group/summary.csv.tmp"
    head -1 "$out/$group/$name/summary.csv" > "$out/$group/header.tmp"
}

finish() {  # finish <group>: summary.csv of the group
    local group=$1
    cat "$out/$group/header.tmp" "$out/$group/summary.csv.tmp" > "$out/$group/summary.csv"
    rm "$out/$group/header.tmp" "$out/$group/summary.csv.tmp"
}

rm -rf "$out"

# Domain size, level 5 and 6
for level in 5 6; do
    for radius in 4 6 8 10; do
        run radius "l${level}_r${radius}" --levels "$level" --radius "$radius" "${variants[@]}"
    done
done
finish radius

# Mesh refinement on the large domain
for level in 5 6 7; do
    run levels "l$level" --levels "$level" --radius 10 "${variants[@]}"
done
finish levels

# Size of the floor: radius 10, level 6
for floor in 1e-6 1e-9 1e-12 1e-15 1e-20; do
    run floor "f$floor" --levels 6 --radius 10 --floor "$floor" --variants floor-nocap,clip-floor --thetas 0.99
done
finish floor

# Other metrics (no clipping): radius 10, level 5, Armijo line search
for metric in mass frobenius; do
    run metric "$metric" --levels 5 --radius 10 --metric-smooth "$metric" "${ls[@]}" \
        --variants plain,floor,floor-nocap --thetas 0.99
done
finish metric
