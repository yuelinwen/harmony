#!/bin/bash
#
# Index memory -- the paper's Table 4, section 6.4.2.
#
# What a distributed index costs in space against the single-machine one it
# replaces. The question is whether cutting the data up means storing more of
# it: the vectors themselves are only divided, but a dimension-sliced layout
# has every column of a row holding the same id list, and a norm per vector
# per slice that a single machine never keeps at all.
#
# Four nodes, matching the paper, in the three layouts it reports:
#
#   4x1   Harmony-vector      no dimension slicing, so no duplication
#   2x2   Harmony             the hybrid
#   1x4   Harmony-dimension   every machine holds every cluster
#
# On Sift1M, which the paper also uses, and on two 784-dimensional datasets
# it does not -- because the overhead is n * 8 * bDim against data of
# n * d * 4, so its share depends on the dimension and not on the dataset
# size, and one dataset cannot show that.
#
# Nothing is searched. --nq 64 is the smallest batch that still exercises the
# whole startup path; the memory is settled once the data has been handed out.
#
#   scripts/build.sh                    on the cluster master
#   experiment/index-memory/run.sh

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."

for data in Data/sift/sift Data/fashion/fashion Data/mnist/mnist; do
    echo "===== $data ====="
    for grid in "4 1" "2 2" "1 4"; do
        set -- $grid
        printf '  %sx%s  ' "$1" "$2"
        scripts/run.sh 4 --data "$data" --bvec "$1" --bdim "$2" \
            --nq 64 --cache 1 --check 0 --baseline 0 2>&1 \
            | grep -E "index memory|% of the single" | tr -s ' ' | tr '\n' ' '
        echo
    done
done

# The paper's baseline column. The on-disk index is what there is to measure
# without linking faiss into something that reports its resident size; for a
# flat IVF the two are within the file header of each other.
echo
echo "===== faiss index on disk ====="
for f in Data/*/*_faissivf_nlist*.index; do
    [ -f "$f" ] || continue
    ls -l "$f" | awk '{printf "  %-52s %6.0f MB (%4.0f MiB)\n", $NF, $5/1e6, $5/1048576}'
done
