#!/bin/bash
# Builds ./main with MPI and, if it can be found, OpenMP.
#
# Apple's clang does not ship OpenMP, so on macOS the flags have to point at
# Homebrew's libomp. On Linux -fopenmp is enough.

set -e

FLAGS="-std=c++20 -O2 -march=native -Wall -Wextra -I."

if [ "$(uname)" = "Darwin" ] && [ -d "$(brew --prefix libomp 2>/dev/null)" ]; then
    OMP_PREFIX=$(brew --prefix libomp)
    FLAGS="$FLAGS -Xpreprocessor -fopenmp -I$OMP_PREFIX/include -L$OMP_PREFIX/lib -lomp"
    echo "building with OpenMP (libomp at $OMP_PREFIX)"
else
    FLAGS="$FLAGS -fopenmp"
    echo "building with OpenMP (-fopenmp)"
fi

mpicxx $FLAGS main.cpp src/node/*.cpp src/index/*.cpp -o main
echo "built ./main"
