#!/bin/bash
#
# Starts a run. The first argument is how many workers to use; everything
# after it goes straight to ./main.
#
#   scripts/run.sh                          4 workers, defaults
#   scripts/run.sh 8                        8 workers
#   scripts/run.sh 4 --mode dimension       any option ./main takes
#   scripts/run.sh 4 --nq 1000 --nprobe 64
#   ./main --help                     the full option list
#
# There is no separate command for the master and the workers. mpirun starts
# every process itself: rank 0 becomes the master, the rest become workers.
# On a cluster it ssh's into the machines in hosts.txt to do it, which is why
# they need passwordless ssh between them.
#
# With hosts.txt present this spreads one process per machine. Without it,
# everything runs locally, which is the normal way to work on a laptop.

set -e

# run from the project root whichever directory this was called from
cd "$(dirname "$0")/.."

WORKERS=${1:-4}
[ $# -gt 0 ] && shift

if [ ! -x ./main ]; then
    echo "./main not built - run scripts/build.sh first" >&2
    exit 1
fi

# one process per machine, and one OpenMP thread per core on it: the layout
# the paper runs. Locally the processes already share the cores, so one thread
# each avoids them fighting over them.
if [ -f hosts.txt ]; then
    THREADS=$(nproc 2>/dev/null || echo 1)
    PLACEMENT="--hostfile hosts.txt --map-by node"
    echo "cluster: $WORKERS workers, $THREADS threads each"
else
    THREADS=1
    PLACEMENT="--oversubscribe"
    echo "local: $WORKERS workers on this machine"
fi

set -x
mpirun -n $((WORKERS + 1)) $PLACEMENT ./main --threads $THREADS "$@"
