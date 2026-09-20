#!/bin/bash
#
# Converts an ann-benchmarks .hdf5 into the three flat files the program reads.
# It does not download anything -- put the .hdf5 in Data/ yourself.
#
#   scripts/data.sh Data/sift-128-euclidean.hdf5
#   scripts/data.sh Data/gist-960-euclidean.hdf5 gist
#
# The name defaults to everything before the first dash, so the example above
# writes Data/sift_base.bin, Data/sift_query.bin and Data/sift_gt.bin. Pass a
# second argument to choose it yourself. Run with --data Data/sift.
#
# The hdf5 holds four arrays and all four are used:
#   train      -> _base.bin    the vectors being searched
#   test       -> _query.bin   the queries, from different images than train
#   neighbors  -> _gt.bin      the true top-k of each query, brute-forced
#   distances  -> _gtd.bin     how far away each of those is, squared
#
# The file stores plain Euclidean distance (checked: 232.87 where the squared
# distance is 54229), and everything in this project is squared L2, so the
# export squares it. Converting here rather than in the program keeps one
# convention: every distance in a .bin file is squared.
#
# Binary format, little-endian:
#   int32 n, int32 dim, then n*dim values row-major -- float32 for the
#   vectors, int32 for the groundtruth ids. Two freads and it is in memory.

set -e

cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "usage: scripts/data.sh <file.hdf5> [name]" >&2
    echo >&2
    echo "hdf5 files in Data/:" >&2
    ls Data/*.hdf5 2>/dev/null | sed 's/^/  /' >&2 || echo "  (none)" >&2
    exit 1
fi

HDF5=$1
NAME=${2:-$(basename "$HDF5" .hdf5 | cut -d- -f1)}

if [ ! -f "$HDF5" ]; then
    echo "no such file: $HDF5" >&2
    exit 1
fi
# _gtd.bin was added after the first datasets were converted, so a tree that
# already has the other three still needs this one. Only that file is written
# in that case; the rest are left alone.
if [ -f "Data/${NAME}_base.bin" ] && [ -f "Data/${NAME}_gtd.bin" ]; then
    echo "Data/${NAME}_base.bin already exists - delete it to redo"
    exit 0
fi
if [ -f "Data/${NAME}_base.bin" ]; then
    echo "Data/${NAME}_base.bin exists; adding the missing _gtd.bin only"
    ONLY_GTD=1
fi

# Squared L2 is what makes dimension pruning work: every term is non-negative,
# so a partial sum only grows and one that has passed the threshold can never
# come back. Under cosine that does not hold, and an angular file's
# groundtruth is ranked by angle anyway, so recall measured against it would
# be meaningless. Better to stop here than to convert it and leave the low
# recall to be explained later.
case "$HDF5" in
    *angular*)
        echo "$HDF5 is an angular dataset; this project is squared L2 only," >&2
        echo "and dimension pruning depends on it" >&2
        exit 1 ;;
esac

# Checked here so a missing library is one line rather than a traceback. Only
# the machine that converts needs these; the cluster reads the .bin files.
if ! python3 -c "import h5py, numpy" 2>/dev/null; then
    echo "needs h5py and numpy:  pip3 install h5py numpy" >&2
    exit 1
fi

python3 - "$HDF5" "Data/$NAME" "${ONLY_GTD:-0}" <<'PY'
import struct
import sys

import h5py
import numpy as np

src, prefix = sys.argv[1], sys.argv[2]
only_gtd = sys.argv[3] == "1"

with h5py.File(src, "r") as f:
    # the name is checked by the shell above; this catches a file whose name
    # does not say what it holds
    metric = f.attrs.get("distance", b"")
    if isinstance(metric, bytes):
        metric = metric.decode()
    if metric and "euclidean" not in str(metric).lower():
        sys.exit(f"{src} uses {metric} distance; this project is squared L2 only")

    wanted = [("base", "train", np.float32),
              ("query", "test", np.float32),
              ("gt", "neighbors", np.int32),
              ("gtd", "distances", np.float32)]
    if only_gtd:
        wanted = [w for w in wanted if w[0] == "gtd"]

    for name, field, dtype in wanted:
        a = np.ascontiguousarray(f[field][:], dtype=dtype)
        if field == "distances":
            a = a * a        # stored as distance, used as squared distance
        n, dim = a.shape
        with open(f"{prefix}_{name}.bin", "wb") as out:
            out.write(struct.pack("<ii", n, dim))
            a.tofile(out)
        print(f"  {prefix}_{name}.bin   n={n} dim={dim}")
PY

echo "run it with: --data Data/$NAME"
