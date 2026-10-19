#!/bin/bash
#
# Contribution of each optimization -- the paper's Fig. 10, section 6.3.2.
#
# Three features are switched on one at a time and the throughput measured
# after each, so each step's ratio is what that feature is worth on top of
# the ones before it. The paper's numbers, on four nodes:
#
#     Msong    1.00 -> 1.88 -> 2.62 -> 3.27
#     Sift1M   1.00 -> 1.63 -> 1.81 -> 3.21
#     average  balanced load 1.75x, pipeline and async 1.25x, pruning 1.51x
#
# The three, and the switches that turn each off here:
#
#   balanced load         --assign roundrobin, which deals clusters out by id
#                         instead of by how much work each brings
#   pipeline and async    --disablepipeline (one block open at a time) and
#                         --blocksend 1 (wait for each forward to land)
#   pruning               --disablepruning (dimension level) and
#                         --disablevectorpruning (between partitions)
#
# Run on 2x2 as well as 4x1 because pruning needs something to prune: with a
# single dimension slice the threshold is only tested after the whole
# distance has been computed, so it saves no arithmetic at all and the last
# step would measure nothing.
#
# And once more at nprobe 160, k 10. The pruning step is worth almost nothing
# at nprobe 32, k 100 -- not because it does not work, but because a top-100
# threshold is the 100th distance and stays loose. Reporting only the first
# operating point would say pruning does not pay, which is a statement about
# the operating point and not about pruning.
#
# Every run is under timeout. The first configuration deadlocked when this
# was first written -- a worker blocked for the next partition's threshold
# while the master was blocked waiting for blocks that worker had not opened
# yet -- and at 100% CPU on both sides it read as a slow run, not a hang.
# The guard is cheap and turns the next one of those into a visible failure.
#
#   scripts/build.sh                                  on the cluster master
#   experiment/optimization-contribution/run.sh

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."

OUT="$HERE/results.csv"
LOGS="$HERE/logs"
rm -f "$OUT"
mkdir -p "$LOGS"

LIMIT=900

NO_BALANCE="--assign roundrobin"
NO_PIPE="--disablepipeline --blocksend 1"
NO_PRUNE="--disablepruning --disablevectorpruning"

step() {
    local tag="$1"; shift
    printf '  %-22s' "$tag"
    local log="$LOGS/${GRID}_${tag// /_}.txt"
    set +e
    timeout "$LIMIT" scripts/run.sh 4 $BASE "$@" >"$log" 2>&1
    local rc=$?
    set -e
    tr -d '\000' <"$log" >"$log.clean" && mv "$log.clean" "$log"
    if [ $rc -eq 124 ]; then
        echo "TIMED OUT after ${LIMIT}s -- see $log"
        pkill -f "[m]pirun" 2>/dev/null || true
        for h in $(cat scripts/hosts.txt); do
            ssh -n -o ConnectTimeout=5 "$h" "pkill -9 -x main" 2>/dev/null || true
        done
        sleep 4
        return
    fi
    awk '
        /^  QPS:/                  { qps = $2 }
        /^  1\. dimensions/        { ops = $4 " " $5; pct = $6 }
        /^  3\. bytes per hop/     { bytes = $5 " " $6 }
        /^  4\. node busy-time/    { skew = $5 }
        END { printf "QPS %8.1f   ops %-8s %-12s bytes %-10s skew %s\n",
                     qps, ops, pct, bytes, skew }
    ' "$log"
}

# bVec bDim nprobe k
for case in "2 2 32 100" "4 1 32 100" "2 2 160 10"; do
    set -- $case
    GRID="${1}x${2}_nprobe${3}_k${4}"
    echo "===== ${1}x${2}, nprobe $3, k $4 ====="
    BASE="--bvec $1 --bdim $2 --nprobe $3 --k $4 --nq 1024 --loop 3 \
          --cache 1 --check 0 --baseline 0 --csv $OUT"

    step "1 none"             $NO_BALANCE $NO_PIPE $NO_PRUNE
    step "2 +balanced load"               $NO_PIPE $NO_PRUNE
    step "3 +pipeline async"                       $NO_PRUNE
    step "4 +pruning"
done

echo
echo "wrote $OUT and $LOGS"
