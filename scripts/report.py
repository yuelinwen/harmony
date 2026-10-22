#!/usr/bin/env python3
"""Read back the rows a sweep just wrote, and print them as a table.

The run binary appends to test/*.csv on every run, one file per measurement
(see src/test/metrics.cpp). A sweep records how many rows were already there
before it started and passes that in as --from, so what it prints is exactly
its own runs and not the whole history.

This exists so the experiment scripts stop parsing terminal output. Every awk
one-liner that did it before was its own small bug: a regex that stopped
matching when a column was renamed, a divide by zero on a run that failed, a
`set --` that silently ate the arguments.
"""

import argparse
import csv
import os
import sys

TEST_DIR = "test"


def load(name):
    path = os.path.join(TEST_DIR, name + ".csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fmt(v, width):
    return str(v)[:width].ljust(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0,
                    help="skip this many rows of runs.csv (the count taken "
                         "before the sweep began)")
    ap.add_argument("--cols", default="",
                    help="comma-separated columns of runs.csv to print")
    ap.add_argument("--pivot", default="",
                    help="<file>:<column>, one column per hop_id, joined on "
                         "run_id -- e.g. candidates_per_hop:skipped_pct")
    ap.add_argument("--label", default="",
                    help="comma-separated columns to use as the row label")
    ap.add_argument("--width", type=int, default=13)
    args = ap.parse_args()

    runs = load("runs")[args.start:]
    if not runs:
        print("(no rows)", file=sys.stderr)
        return 1

    label = [c for c in args.label.split(",") if c]
    cols = [c for c in args.cols.split(",") if c]

    pivot_rows = {}
    hops = []
    if args.pivot:
        fname, field = args.pivot.split(":")
        wanted = {r["run_id"] for r in runs}
        for r in load(fname):
            if r["run_id"] in wanted:
                pivot_rows.setdefault(r["run_id"], {})[int(r["hop_id"])] = r[field]
        for d in pivot_rows.values():
            hops = sorted(set(hops) | set(d))

    head = [fmt(c, args.width) for c in label + cols]
    head += [fmt("hop%d" % h, args.width) for h in hops]
    print("  " + "".join(head))

    for r in runs:
        line = [fmt(r.get(c, ""), args.width) for c in label + cols]
        d = pivot_rows.get(r["run_id"], {})
        line += [fmt(d.get(h, ""), args.width) for h in hops]
        print("  " + "".join(line))
    return 0


if __name__ == "__main__":
    sys.exit(main())
