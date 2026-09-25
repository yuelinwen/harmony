#!/bin/bash
#
# Builds ./main on the cluster master, and copies it to the other machines.
#
#   scripts/build.sh
#
# mpirun needs the executable at the same path on every machine it starts a
# process on, so the copy has to happen after every build -- which is why it
# lives here rather than in a script of its own. The data files are not
# copied: only rank 0 opens a file, and the workers get their blocks over MPI.
#
# The cluster machines are one distro on one architecture, so a binary built
# on the master runs on all of them and one build is enough. This is the only
# supported place to build: MKL and GCC's OpenMP are both required.

set -e

# run from the project root whichever directory this was called from
cd "$(dirname "$0")/.."

# ---- compiler flags ----------------------------------------------------
#
# -fassociative-math is what makes the distance loop fast. Without it the
# compiler has to keep the additions in source order, so each one waits on the
# previous and the FMA pipeline sits idle. Allowing it to reassociate splits
# the sum across several accumulators: 4.8 -> 12.7 GFLOP/s measured on Xeon
# Sapphire Rapids, 1.65x end to end.
#
# The two companion flags are what -fassociative-math requires. Unlike
# -ffast-math this does not assume the absence of NaN or infinities, and the
# results do not change -- the acceptance check still reports 0 differing
# queries.
#
# Widening to AVX-512 (-mprefer-vector-width=512) measured 35% *slower* and is
# deliberately left out.

FLAGS="-std=c++20 -O3 -march=native -Wall -Wextra -I."
FLAGS="$FLAGS -fassociative-math -fno-signed-zeros -fno-trapping-math"

# ---- which source this binary was built from ---------------------------
#
# Stamped into the binary and written to every CSV row, so a measurement can
# be traced back to the code that produced it.
#
# Both halves are needed. The commit alone is not enough: measurements are
# normally taken before the change is committed, and on the cluster the tree
# is updated by tar, which leaves .git pointing at whatever was last
# committed -- so the hash would label a week of different runs identically.
# The digest of the sources actually handed to the compiler cannot.

COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
SRC=$(cat main.cpp src/*.h src/*/*.h src/*/*.cpp | sha1sum | cut -c1-8)
FLAGS="$FLAGS -DHARMONY_COMMIT=\"$COMMIT+$SRC\""
echo "build $COMMIT+$SRC"

# GCC ships OpenMP, so this is all it takes.
FLAGS="$FLAGS -fopenmp"

# ---- MKL, which is required --------------------------------------------
#
# Used for one thing: the first dimension slice of a batch, where nothing can
# be pruned yet and the whole block has to be computed regardless. That case
# is a dense matrix multiply, which is what MKL is good at -- measured up to
# 1.49x end to end on this cluster, the gain falling as the slices get
# narrower.
#
# Later slices never take this path: their purpose is to skip most candidates,
# which a dense multiply cannot do.
#
# There is no fallback. There used to be one, for building on a laptop, and it
# hid a real bug for a day: the gemm path was switched off in the code and the
# scalar numbers looked like MKL simply not helping.

if [ ! -f /usr/include/mkl/mkl.h ]; then
    echo "MKL is required and /usr/include/mkl/mkl.h is missing." >&2
    echo "  sudo apt-get install libmkl-dev" >&2
    exit 1
fi
FLAGS="$FLAGS -I/usr/include/mkl"
LIBS="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm"

# mpicxx is a wrapper around the system compiler that adds the MPI include and
# library paths, so no MPI flags are needed here.
mpicxx $FLAGS main.cpp src/node/*.cpp src/index/*.cpp src/test/*.cpp -o main $LIBS
echo "built ./main"

# ---- copy to the workers, if this is the master ------------------------
#
# The test is whether one of this machine's own addresses is in hosts.txt.
#
# What does *not* travel with the binary are its shared libraries. Linking
# something new means installing it on the workers too, or they fail at
# startup with "cannot open shared object file".

HOSTS=scripts/hosts.txt
[ -f "$HOSTS" ] || exit 0

MY_IPS=" $(hostname -I 2>/dev/null || true) "
MASTER=""
while read -r host _; do
    [ -z "$host" ] && continue
    case "$MY_IPS" in *" $host "*) MASTER=$host ;; esac
done < "$HOSTS"

[ -n "$MASTER" ] || exit 0

# -n on ssh matters: without it ssh swallows stdin, which here is hosts.txt,
# and the loop stops after the first machine.
while read -r host _; do
    [ -z "$host" ] && continue
    [ "$host" = "$MASTER" ] && continue      # this is where it was built
    ssh -n -o BatchMode=yes "$host" "mkdir -p ~/harmony"
    scp -q main "$host:~/harmony/main"
    echo "  copied to $host"
done < "$HOSTS"
