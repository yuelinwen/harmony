#!/bin/bash
#
# Pruning ratio breakdown -- the paper's Table 3, section 6.3.3.
#
# The paper cuts the dimensions into four and reports, per slice, the share of
# candidates that never reached it. Its Sift1M row is
#
#     0.00  41.76  85.04  98.40      average 57.05
#
# and it does not say at what nprobe or k. That turns out to matter more than
# anything else: the same code gives 3.10% or 55.30% on the second slice
# depending on where it is measured. So this sweeps the operating point rather
# than reporting one number, and the question it answers is not "how much does
# this prune" but "what did the paper have to have been measuring".
#
# Three knobs, in the order they matter:
#
#   k        a smaller k is a much tighter threshold: the 10th neighbour is
#            far closer than the 100th
#   nprobe   a larger nprobe drags in distant clusters whose candidates fail
#            on the first slice
#   prewarm  how good the starting threshold is, which has a ceiling -- the
#            perfect-threshold row below is that ceiling
#
# Four nodes and a 1x4 grid, matching the paper's "dimensional split of size
# 4" on four nodes.
#
#   scripts/build.sh                               on the cluster master
#   experiment/pruning-ratio-breakdown/run.sh

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."

OUT="$HERE/results.csv"
rm -f "$OUT"

# One row per configuration. --check 0 throughout: this measures pruning, and
# correctness is established in experiment/qps-recall-trade-off.
COMMON="--bvec 1 --bdim 4 --nq 1024 --cache 1 --check 0 --csv $OUT"

row() {
    printf '%-42s' "$1"
    shift
    scripts/run.sh 4 $COMMON "$@" 2>&1 | awk '
        /^pruning, by/      { f = 1; next }
        f && /^ +[0-9]/     { printf "%8s", $4 }
        /^  distance ops/   { printf "   ops %s\n", $NF; f = 0 }'
}

echo "=== k, at nprobe 32 with the default prewarm ==="
row "k=100  nprobe=32  prewarm=500"   --nprobe 32  --k 100 --prewarm 500
row "k=10   nprobe=32  prewarm=500"   --nprobe 32  --k 10  --prewarm 500
row "k=1    nprobe=32  prewarm=500"   --nprobe 32  --k 1   --prewarm 500

echo
echo "=== nprobe, at k=10 ==="
row "k=10   nprobe=32   prewarm=2000" --nprobe 32  --k 10  --prewarm 2000
row "k=10   nprobe=128  prewarm=2000" --nprobe 128 --k 10  --prewarm 2000
row "k=10   nprobe=160  prewarm=2000" --nprobe 160 --k 10  --prewarm 2000
row "k=10   nprobe=192  prewarm=2000" --nprobe 192 --k 10  --prewarm 2000
row "k=10   nprobe=256  prewarm=2000" --nprobe 256 --k 10  --prewarm 2000

echo
echo "=== prewarm, at nprobe 32 k=100 ==="
row "prewarm=500   lists=1"           --nprobe 32 --k 100 --prewarm 500
row "prewarm=2000  lists=1"           --nprobe 32 --k 100 --prewarm 2000
row "prewarm=8000  lists=1"           --nprobe 32 --k 100 --prewarm 8000
row "prewarm=2000  lists=3"           --nprobe 32 --k 100 --prewarm 2000 --prewarmlists 3
row "prewarm=500   lists=10"          --nprobe 32 --k 100 --prewarm 500  --prewarmlists 10

echo
echo "=== the ceiling: every probed cluster scanned before the search ==="
echo "    (not a usable setting -- it is the search done twice, and it is"
echo "     here to say how much of the gap a better threshold could close)"
row "perfect threshold  k=100"        --nprobe 32 --k 100 --prewarm 4000 --prewarmlists 32
row "perfect threshold  k=10"         --nprobe 32 --k 10  --prewarm 4000 --prewarmlists 32
row "perfect threshold  k=1"          --nprobe 32 --k 1   --prewarm 4000 --prewarmlists 32

echo
echo "wrote $OUT"
