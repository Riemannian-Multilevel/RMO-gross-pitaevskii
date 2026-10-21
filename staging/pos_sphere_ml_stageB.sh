#!/usr/bin/env bash
# Stage B of the staged multilevel benchmark (md/positive_multilevel_benchmark.md, section 3 and step 6.2): rounds of
# ml_stages.py next -> the configurations of the round, 3 repeats (staging/pos_sphere_cpu.sh) -> ml_stages.py decide,
# until ml_stages.py next reports that stage B is done. Needs <output dir>/state.json from ml_stages.py noise and
# screen. Round r in <output dir>/B/round<r> (config file round<r>.txt); a round without new configurations (all
# measured before) only decides.
#
# Usage: [WAIT_LOAD=25] [REPEATS=3] [JOBS=4] staging/pos_sphere_ml_stageB.sh <build dir> <output dir>
set -euo pipefail
if (($# != 2)); then
    echo "usage: $(basename "$0") <build dir> <output dir>" >&2
    exit 2
fi
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build=$1 out=$2
state="$out/state.json"
[[ -f $state ]] || { echo "error: $state not found (ml_stages.py noise, screen)" >&2; exit 1; }
mkdir -p "$out/B"

r=$(find "$out/B" -maxdepth 1 -name 'round*.txt' | wc -l | tr -d ' ')
while true; do
    r=$((r + 1))
    configs="$out/B/round$r.txt"
    status=0
    python3 "$here/ml_stages.py" next "$state" "$configs" || status=$?
    if ((status == 3)); then
        break
    elif ((status != 0)); then
        exit "$status"
    fi
    if [[ -s $configs ]]; then
        STAGE=B CONFIGS="$configs" "$here/pos_sphere_cpu.sh" "$here/pos_sphere_ml.sh" "$build" "$out/B/round$r" \
            "${REPEATS:-3}" "${JOBS:-4}"
        python3 "$here/ml_stages.py" decide "$state" "$out/B/round$r"
    else
        python3 "$here/ml_stages.py" decide "$state"
    fi
done
echo "stage B done after $((r - 1)) rounds; traces in $out/trace_*.csv"
