#!/usr/bin/env python3
"""
Convert an ann-benchmarks .hdf5 dataset into flat binary files that C++
can read with two freads.

Output format (little-endian):
    int32 n        number of rows
    int32 dim      number of columns
    then n * dim values, row-major:
        float32 for base/query
        int32   for groundtruth

Usage:
    python3 util/convert_hdf5.py Data/sift-128-euclidean.hdf5 Data/sift
Produces:
    Data/sift_base.bin    (train)
    Data/sift_query.bin   (test)
    Data/sift_gt.bin      (neighbors)
"""

import struct
import sys

import h5py
import numpy as np


def write_bin(path, array, dtype):
    array = np.ascontiguousarray(array, dtype=dtype)
    n, dim = array.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, dim))
        array.tofile(f)
    print(f"wrote {path}: n={n} dim={dim} dtype={dtype.__name__}")


def main():
    if len(sys.argv) != 3:
        print("usage: convert_hdf5.py <input.hdf5> <output_prefix>")
        return 1

    in_path, prefix = sys.argv[1], sys.argv[2]

    with h5py.File(in_path, "r") as f:
        write_bin(prefix + "_base.bin", f["train"][:], np.float32)
        write_bin(prefix + "_query.bin", f["test"][:], np.float32)
        write_bin(prefix + "_gt.bin", f["neighbors"][:], np.int32)

    return 0


if __name__ == "__main__":
    sys.exit(main())
