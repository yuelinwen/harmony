#!/bin/bash
#
# Builds ./main. Run it on whichever machine you are on -- it works out the
# platform itself.
#
#   scripts/build.sh
#
# Two platforms are in play and they need different OpenMP flags:
#
#   macOS (the laptop)   Apple's clang has no OpenMP of its own, so the flags
#                        have to point at Homebrew's libomp. Install it with
#                        `brew install libomp` if the build says it is missing.
#
#   Linux (the VMs)      GCC ships OpenMP, so -fopenmp is all it takes.
#
# Everything else is identical on both, and the binary produced on one Linux
# VM runs on all of them -- same distro, same architecture -- so it is enough
# to build on the master and scp ./main to the others.

set -e

# run from the project root whichever directory this was called from
cd "$(dirname "$0")/.."

# ---- flags used on both platforms --------------------------------------
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

# ---- OpenMP, which is where the platforms differ -----------------------

if [ "$(uname)" = "Darwin" ] && [ -d "$(brew --prefix libomp 2>/dev/null)" ]; then
    # macOS: clang needs -Xpreprocessor to accept -fopenmp, and libomp has to
    # be found and linked by hand.
    OMP_PREFIX=$(brew --prefix libomp)
    FLAGS="$FLAGS -Xpreprocessor -fopenmp -I$OMP_PREFIX/include -L$OMP_PREFIX/lib -lomp"
    echo "macOS build (libomp at $OMP_PREFIX)"
else
    # Linux: GCC handles it on its own.
    FLAGS="$FLAGS -fopenmp"
    echo "Linux build (-fopenmp)"
fi

# ---- MKL, if it is installed -------------------------------------------
#
# Used for one thing: the first dimension slice of a batch, where nothing can
# be pruned yet and the whole block has to be computed regardless. That case
# is a dense matrix multiply, which is what MKL is good at -- measured 3.16x
# over the scalar loop at batch 32, but 0.64x at batch 1, so it only pays off
# because queries are batched.
#
# Later slices never take this path: their purpose is to skip most candidates,
# which a dense multiply cannot do.
#
# x86 only, so a macOS/ARM build simply goes without it.

if [ -f /usr/include/mkl/mkl.h ]; then
    FLAGS="$FLAGS -DHARMONY_USE_MKL -I/usr/include/mkl"
    LIBS="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm"
    echo "  with MKL (gemm for the first slice)"
elif [ -n "$MKLROOT" ] && [ -f "$MKLROOT/include/mkl.h" ]; then
    FLAGS="$FLAGS -DHARMONY_USE_MKL -I$MKLROOT/include -L$MKLROOT/lib/intel64"
    LIBS="-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm"
    echo "  with MKL from \$MKLROOT"
else
    LIBS=""
    echo "  without MKL (scalar loop everywhere)"
fi

# mpicxx is a wrapper around the system compiler that adds the MPI include and
# library paths, so no MPI flags are needed here.
mpicxx $FLAGS main.cpp src/node/*.cpp src/index/*.cpp -o main $LIBS
echo "built ./main"
