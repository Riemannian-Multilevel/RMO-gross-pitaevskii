#!/usr/bin/env bash
# The staged benchmark of multilevel FAS on the positive sphere (md/positive_multilevel_benchmark.md, section 3 and
# step 6): the runs of one stage with pos_sphere_ml (and main_coarse through main_coarse_summary.py for the unit
# sphere). Run it through staging/pos_sphere_cpu.sh for the repeats (one deal.II thread, 4 jobs, WAIT_LOAD=25):
#
#   STAGE=0 staging/pos_sphere_cpu.sh staging/pos_sphere_ml.sh <build dir> <out>/stage0 3 4
#   STAGE=A staging/pos_sphere_cpu.sh staging/pos_sphere_ml.sh <build dir> <out>/stageA 1 4
#   STAGE=B CONFIGS=<file> ...   (staging/pos_sphere_ml_stageB.sh runs the rounds of stage B)
#   STAGE=C CONFIGS=<file> staging/pos_sphere_cpu.sh staging/pos_sphere_ml.sh <build dir> <out>/stageC 3 4
#
# Stages (section 3): 0 the baseline (blocks fas, sl, size, unit; BLOCKS selects some), A one factor at a time from the
# baseline on every smoother, 2 and 3 levels, and the RATIO diagnostic; B and C the configurations of CONFIGS, one per
# line: `<config id> <level counts> [<size>]`, level counts as `2,3` (1: single level), size 4225 (default) or 16641.
#
# A configuration is `<smoother>[__<factor>=<value>...]`, e.g. ea-armijo-floor__cg=mass__kappa=0.5. Smoothers:
# ea-armijo-floor, ea-armijo-clip, ea-exact-floor, mass-exact-floor. Factor values: pt=log, cg=<coarse gradient>,
# cr=plain, cmi=<coarse iterations>, inner=default, kappa=<value>, ctrl=ratio. Fixed settings: those of section 3.
#
# Sizes: level counts are compared at the same number of DoFs on the finest level. 4225 DoFs: Q1 fine level 7, P1
# fine level 6; 16641 DoFs: Q1 8, P1 7 (P1 at level l has the DoFs of Q1 at l + 1). With n levels the levels are the n
# finest down from the fine level.
#
# Run directories: <stage output>/<config id>/<disc>-<levels>/b<beta>-<potential>-r<radius> (unit sphere: config `unit`),
# merged into <stage output>/runs.csv (merge_tree). SMOKE=1: the problems 1000/square/6 and 100/optical_lattice/6 only.
#
# Usage: [STAGE=0|A|B|C] [CONFIGS=<file>] [BLOCKS="fas sl size unit"] [SMOKE=1] [DRY_RUN=1]
#        staging/pos_sphere_ml.sh <build dir> <output dir> [parallel jobs, default 4]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_BIN=pos_sphere_ml
source "$here/pos_sphere_lib.sh"
study_init "$@"
(($# < 3)) && JOBS=4
build=$1
stage=${STAGE:-0}

# Fixed settings (section 3); the inner tolerance is separate, for the factor value inner=default
fixed=("${COMMON[@]}" --dimension 2 "${LINE_SEARCH[@]}" --ls-alpha 1)
inner=(--tol-inner-res 1e-6 --tol-inner 1e-12 --max-inner 5000)
declare -A disc_args=([q1]="--mesh quadrilateral --degree 1" [p1]="--mesh simplex --degree 1")
declare -A fine_level=([q1-4225]=7 [p1-4225]=6 [q1-16641]=8 [p1-16641]=7)
declare -A smoother_args=(
    [ea-armijo-floor]="--smoother floor"
    [ea-armijo-clip]="--smoother clip-e-cap-up"
    [ea-exact-floor]="--smoother floor --ls-exact"
    [mass-exact-floor]="--smoother floor --metric-smooth mass --ls-exact"
)
smoothers=(ea-armijo-floor ea-armijo-clip ea-exact-floor mass-exact-floor)
# Stage A: the factor values of the screening (section 3)
factor_values=(pt=log cg=model-shift cg=fine cg=mass cg=frobenius cr=plain cmi=2 cmi=8 inner=default kappa=0.5
               kappa=0.95)

if [[ ${SMOKE:-0} == 1 ]]; then
    problems=("1000 square 6" "100 optical_lattice 6")
else
    problems=()
    for beta in 100 1000 10000; do
        for potential in square optical_lattice; do
            for radius in 4 6 8 10; do
                problems+=("$beta $potential $radius")
            done
        done
    done
fi

# Levels of <disc> at <size> with <n> levels, comma-separated, ascending
levels_of() {
    local f=${fine_level[$1-$2]} n=$3 l s=""
    for ((l = f - n + 1; l <= f; l++)); do
        s+="${s:+,}$l"
    done
    echo "$s"
}

# Arguments of a configuration id into the array config_args; with inner=default the inner tolerance is left out
config_args() {
    local id=$1 word
    local -a words
    IFS=' ' read -ra words <<< "${id//__/ }"
    [[ -n ${smoother_args[${words[0]}]:-} ]] || { echo "error: unknown smoother in $id" >&2; exit 2; }
    read -ra config_args <<< "${smoother_args[${words[0]}]}"
    local keep_inner=1
    for word in "${words[@]:1}"; do
        case $word in
            pt=log) config_args+=(--point-transfer log-threshold) ;;
            cg=*) config_args+=(--coarse-grad "${word#cg=}") ;;
            cr=plain) config_args+=(--coarse-retraction plain) ;;
            cmi=*) config_args+=(--coarse-max-iter "${word#cmi=}") ;;
            inner=default) keep_inner=0 ;;
            kappa=*) config_args+=(--kappa "${word#kappa=}") ;;
            ctrl=ratio) config_args+=(--correction-control ratio) ;;
            *) echo "error: unknown factor value $word in $id" >&2; exit 2 ;;
        esac
    done
    ((keep_inner)) && config_args+=("${inner[@]}")
    return 0
}

# All problems of one configuration: add_config <config id> <level counts, comma-separated> [size]
add_config() {
    local id=$1 counts=$2 size=${3:-4225} d n levels p beta potential radius
    local -a counts_a disc_a
    config_args "$id"
    IFS=, read -ra counts_a <<< "$counts"
    for d in q1 p1; do
        read -ra disc_a <<< "${disc_args[$d]}"
        for n in "${counts_a[@]}"; do
            levels=$(levels_of "$d" "$size" "$n")
            for p in "${problems[@]}"; do
                read -r beta potential radius <<< "$p"
                add "$id/$d-${levels//,/-}/b$beta-$potential-r$radius" "${fixed[@]}" "${disc_a[@]}" \
                    "${config_args[@]}" --multilevel "$levels" --beta "$beta" --potential "$potential" \
                    --radius "$radius"
            done
        done
    done
}

# The unit-sphere reference (main_coarse, energy-adaptive, linear interpolation, mass model and transport)
add_unit() {
    local counts=$1 d n levels p beta potential radius
    local -a counts_a disc_a
    IFS=, read -ra counts_a <<< "$counts"
    for d in q1 p1; do
        read -ra disc_a <<< "${disc_args[$d]}"
        for n in "${counts_a[@]}"; do
            levels=$(levels_of "$d" 4225 "$n")
            for p in "${problems[@]}"; do
                read -r beta potential radius <<< "$p"
                add_cmd "unit/$d-${levels//,/-}/b$beta-$potential-r$radius" python3 "$here/main_coarse_summary.py" \
                    --binary "$build/main_coarse" "${fixed[@]}" "${disc_a[@]}" "${inner[@]}" --metric-coarse mass \
                    --transport mass --multilevel "$levels" --beta "$beta" --potential "$potential" --radius "$radius"
            done
        done
    done
}

blocks=" ${BLOCKS:-fas sl size unit} "
case $stage in
    0)
        for s in "${smoothers[@]}"; do
            [[ $blocks == *" fas "* ]] && add_config "$s" 2,3,4
            [[ $blocks == *" sl "* ]] && add_config "$s" 1
        done
        [[ $blocks == *" size "* ]] && add_config ea-armijo-floor 1,2,3,4 16641
        [[ $blocks == *" unit "* ]] && add_unit 1,2,3,4
        ;;
    A)
        for s in "${smoothers[@]}"; do
            for v in "${factor_values[@]}"; do
                add_config "${s}__$v" 2,3
            done
        done
        add_config ea-armijo-floor__ctrl=ratio 3
        add_config ea-armijo-floor__pt=log__ctrl=ratio 3
        ;;
    B | C)
        [[ -f ${CONFIGS:-} ]] || { echo "error: STAGE=$stage needs CONFIGS=<file>" >&2; exit 2; }
        while read -r id counts size; do
            [[ -z $id || $id == \#* ]] && continue
            add_config "$id" "$counts" "${size:-4225}"
        done < "$CONFIGS"
        ;;
    *)
        echo "error: unknown STAGE $stage" >&2
        exit 2
        ;;
esac

study_run
merge_tree "$OUT"
