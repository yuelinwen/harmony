#!/bin/bash
#
# QPS-recall trade-off -- the paper's Fig. 6 and Fig. 7, section 6.2.1.
#
# nprobe is the knob between accuracy and speed, so one value gives a point
# and a sweep gives a curve. The question the curve answers is not "how fast
# is this" but "at a given recall, which layout is fastest", which is the only
# fair way to compare systems that can each be made faster by being wronger.
#
# The paper draws Faiss, Harmony, Harmony-vector and Harmony-dimension over
# ten datasets. Here:
#
#   8x1   the paper's Harmony-vector, and what --mode harmony picks on this
#         cluster, so those two lines would coincide
#   4x2   hybrid
#   2x4   hybrid
#   1x8   the paper's Harmony-dimension
#   one machine   in place of Faiss, which is not wired up as a baseline yet
#
# on Sift1M, the one dataset here that the paper also uses.
#
# Writes results.csv beside this script. Every row carries the five
# measurements as well as recall and QPS, so a curve's shape can be explained
# and not only drawn.
#
#   scripts/build.sh                            first, on the cluster master
#   experiment/qps-recall-trade-off/run.sh

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."

OUT="$HERE/results.csv"
rm -f "$OUT"

# 1 to 256 covers recall from about 0.4 to 1.0. 256 is nlist, where the index
# degenerates to brute force and recall stops improving. Several values in one
# invocation share one index and one distribution, so only the search repeats.
PROBES="--nprobe 1 2 4 8 16 32 64 128 256"

# --loop 3: a single pass on these VMs scatters by about 10%, and the first of
# the four passes is an untimed warm-up.
#
# --check 0: verifying against the single machine costs ten times the search
# it checks. The layouts are verified once each at the end instead.
COMMON="--nq 1024 --loop 3 --cache 1 $PROBES --csv $OUT"

for grid in "8 1" "4 2" "2 4" "1 8"; do
    set -- $grid
    echo "===== ${1} x ${2} ====="
    scripts/run.sh 8 --bvec "$1" --bdim "$2" $COMMON --check 0 --baseline 0
done

# The single-machine curve. It does not depend on the grid, so it is measured
# once rather than on every line -- which is why --baseline is off above.
echo "===== single machine ====="
scripts/run.sh 8 --bvec 8 --bdim 1 $COMMON --check 0 --baseline 1

# One verified run per grid, so the timings above are known to be timings of
# the right answer. Deliberately not written to the CSV: same settings,
# different purpose, and `differing` is a verdict rather than a measurement.
echo "===== correctness ====="
for grid in "8 1" "4 2" "2 4" "1 8"; do
    set -- $grid
    printf '%s x %s  ' "$1" "$2"
    scripts/run.sh 8 --bvec "$1" --bdim "$2" --nprobe 32 --nq 1024 --cache 1 \
        --check 1 2>&1 | grep "differing from single"
done

echo
echo "wrote $OUT"
