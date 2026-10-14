#!/usr/bin/env bash
# CPU-time runs of a pos_sphere_*.sh study: the study is run <repeats> times, by default one run at a time, with one
# deal.II thread, into <output dir>/rep<k>. The CPU time of a case is the column `elapsed` of summary.csv (the CPU time of
# GradientDescent::cycle); staging/plot_pos_sphere_cpu.py takes the median over the repeats.
#
# With several deal.II threads the CPU time includes the time of waiting worker threads: in a test (Q1, level 6,
# radius 10, energy-adaptive) it varied by a factor of 3 between identical runs and was 3 to 10 times the
# single-threaded time, which repeats to within 2 %.
#
# Parallel runs (one deal.II thread each): on an Apple M3 Pro (5 performance, 6 efficiency cores) 4 jobs stay within
# the repeat noise (CPU time +4 % median, ratios to the unit sphere within 0.96-1.05) at 3.6 times the throughput; from
# 5 jobs on, runs land on efficiency cores and take up to 2.5 times the CPU time (staging/pos_sphere_cpu.md, section 3).
# Use at most one job fewer than the number of performance cores.
#
# Concurrent load: a parallel compile (load average 46-51) biases CPU times; background work (Spotlight indexing,
# antivirus; load average about 17) runs on efficiency cores and did not (reference cases within 1-6 %). WAIT_LOAD=25
# waits for the former and tolerates the latter.
#
# Usage: [WAIT_LOAD=<load>] staging/pos_sphere_cpu.sh <study script> <build dir> <output dir> [repeats, default 3]
#        [jobs, default 1]
# Example: MATRIX_ONLY=1 staging/pos_sphere_cpu.sh staging/pos_sphere_analysis.sh <build> <out>
set -euo pipefail
if (($# < 3 || $# > 5)); then
    echo "usage: $(basename "$0") <study script> <build dir> <output dir> [repeats, default 3] [jobs, default 1]" >&2
    exit 2
fi
study=$1 build=$2 out=$3 repeats=${4:-3} jobs=${5:-1}
export DEAL_II_NUM_THREADS=1
mkdir -p "$out"
for ((k = 1; k <= repeats; k++)); do
    # With WAIT_LOAD set, wait until the 1-minute load average is below it (other jobs on the machine bias CPU times)
    if [[ -n ${WAIT_LOAD:-} ]]; then
        while (( $(sysctl -n vm.loadavg | awk '{printf "%d", $2 * 100}') >= ${WAIT_LOAD} * 100 )); do
            sleep 60
        done
    fi
    echo "rep$k: start $(date '+%F %T'), load $(sysctl -n vm.loadavg)"
    "$study" "$build" "$out/rep$k" "$jobs"
done
