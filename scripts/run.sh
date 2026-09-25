#!/bin/bash
#
# Starts a run. The first argument is how many workers to use; everything
# after it goes straight to ./main.
#
#   scripts/run.sh                          4 workers, defaults
#   scripts/run.sh 8                        8 workers
#   scripts/run.sh 4 --mode dimension       any option ./main takes
#   scripts/run.sh 4 --nq 1000 --nprobe 64
#   ./main --help                           the full option list
#
# There is no separate command for the master and the workers. mpirun starts
# every process itself: rank 0 becomes the master, the rest become workers.
# It ssh's into the machines in scripts/hosts.txt to do it, which is why they
# need passwordless ssh between them.

set -e

# run from the project root whichever directory this was called from
cd "$(dirname "$0")/.."

WORKERS=${1:-4}
[ $# -gt 0 ] && shift

if [ ! -x ./main ]; then
    echo "./main not built - run scripts/build.sh first" >&2
    exit 1
fi

# One process per machine, and one OpenMP thread per core on it: the layout
# the paper runs (§5 puts MPI between nodes and OpenMP inside one).
#
# -N 1 is what enforces the one-per-machine part. hosts.txt lists addresses
# only, so mpirun would otherwise assume as many slots as the machine has
# cores and pack eight processes onto the first one before touching the
# second. -N 1 also refuses to start if more processes are asked for than
# there are machines, rather than quietly wrapping round.
HOSTS=scripts/hosts.txt

if [ ! -f "$HOSTS" ]; then
    echo "$HOSTS is missing - run scripts/setup_cluster.sh first" >&2
    exit 1
fi

MACHINES=$(grep -c '[^[:space:]]' "$HOSTS")
if [ $((WORKERS + 1)) -gt "$MACHINES" ]; then
    echo "$WORKERS workers needs $((WORKERS + 1)) machines, $HOSTS has $MACHINES" >&2
    exit 1
fi

THREADS=$(nproc)
echo "cluster: $WORKERS workers, $THREADS threads each"

set -x
mpirun -n $((WORKERS + 1)) --hostfile "$HOSTS" -N 1 ./main --threads $THREADS "$@"
