# Shared driver of the pos_sphere_*.sh studies (source it, do not run it): collects the runs of pos_sphere_study in
# <output dir>/commands.txt and runs them in parallel, each with its output in <output dir>/<run>/log.txt.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/pos_sphere_lib.sh"
#   study_init "$@"                 # arguments of the study: <build dir> <output dir> [parallel jobs, default 6]
#   add <run> <args...>             # one run of pos_sphere_study, written to <output dir>/<run>
#   study_run                       # runs all; exits with 1 if a run failed
#   merge_summaries <group> <run>...  # <output dir>/<group>/summary.csv from the runs <group>/<run>, in this order
#
# With DRY_RUN=1, study_run only writes commands.txt.

# Settings shared by the studies: lumped mass, stop at residual 1e-9 or after 1000 iterations
COMMON=(--mass-lumping --max-iter 1000 --tol-residual 1e-9)
# Armijo line search for the mass and Frobenius metrics (the starting step --ls-alpha is set by the studies)
LINE_SEARCH=(--line-search --ls-max-iter 40 --ls-beta 0.5 --ls-min 1e-12)

# Discretizations: <mesh><degree> (q: quadrilateral, p: simplex with bubbles for degree 2), at the level of the studies
declare -A DISC=(
    [q1]="--mesh quadrilateral --degree 1 --levels 6"
    [q2]="--mesh quadrilateral --degree 2 --levels 5"
    [p1]="--mesh simplex --degree 1 --levels 6"
    [p2]="--mesh simplex --degree 2 --levels 5"
)

study_init() {
    if (($# < 2 || $# > 3)); then
        echo "usage: $(basename "$0") <build dir> <output dir> [parallel jobs, default 6]" >&2
        exit 2
    fi
    STUDY="$1/pos_sphere_study"
    OUT=$2
    JOBS=${3:-6}
    COMMANDS="$OUT/commands.txt"

    if [[ ! -x $STUDY ]]; then
        echo "error: $STUDY not found; build the target pos_sphere_study" >&2
        exit 1
    fi
    # Replace the output of an earlier run of a study, but never another directory
    if [[ -e $OUT ]]; then
        if [[ ! -f $COMMANDS ]]; then
            echo "error: $OUT exists and is not the output of a study (no commands.txt)" >&2
            exit 1
        fi
        rm -rf "$OUT"
    fi
    mkdir -p "$OUT"
    : > "$COMMANDS"
}

add() {
    local dir="$OUT/$1"
    shift
    printf 'mkdir -p %q && %q' "$dir" "$STUDY" >> "$COMMANDS"
    printf ' %q' "$@" --out "$dir" >> "$COMMANDS"
    printf ' > %q 2>&1\n' "$dir/log.txt" >> "$COMMANDS"
}

study_run() {
    local n_runs failed
    n_runs=$(wc -l < "$COMMANDS")
    if [[ ${DRY_RUN:-0} == 1 ]]; then
        echo "$n_runs runs in $COMMANDS (dry run)"
        return
    fi
    failed=$(xargs -d '\n' -P "$JOBS" -I{} bash -c '{} || echo "failed: {}"' < "$COMMANDS")
    echo "$n_runs runs in $OUT"
    if [[ -n $failed ]]; then
        echo "$failed" >&2
        exit 1
    fi
}

merge_summaries() {
    local group=$1 run
    shift
    [[ ${DRY_RUN:-0} == 1 ]] && return
    head -1 "$OUT/$group/$1/summary.csv" > "$OUT/$group/summary.csv"
    for run in "$@"; do
        tail -n +2 "$OUT/$group/$run/summary.csv" >> "$OUT/$group/summary.csv"
    done
}

# Words of a string, for the entries of DISC and similar tables: args <array name> <string>
args() {
    read -ra "$1" <<< "$2"
}
