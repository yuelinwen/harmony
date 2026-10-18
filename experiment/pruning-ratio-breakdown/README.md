# Pruning ratio breakdown

Paper section 6.3.3, Table 3.

## What it measures

The dimensions are cut into four. For each slice, what share of all
candidates never reached it, because an earlier slice had already pushed
their partial sum past the threshold.

The paper's Sift1M row:

```
0.00    41.76    85.04    98.40        average 57.05
```

The first slice is always 0.00 -- everyone computes it, there is nothing to
prune against yet.

## Running it

```bash
scripts/build.sh                               # on the cluster master
experiment/pruning-ratio-breakdown/run.sh
```

Four nodes, 1x4, matching the paper's "dimensional split of size 4" on four
nodes. Writes `results.csv`.

## The finding

**The pruning ratio is a property of the operating point, not of the
implementation.** The paper does not say what nprobe or k produced Table 3,
and the same code gives anything from 3.1% to 55.3% on the second slice
depending on where it is measured:

| setting | slice 1 | 2 | 3 | 4 | distance ops |
| --- | ---: | ---: | ---: | ---: | ---: |
| **paper, Sift1M** | 0.00 | **41.76** | **85.04** | **98.40** | -- |
| nprobe 160, k 10 | 0.0 | **40.4** | 88.3 | 99.4 | 42.95% |
| nprobe 32, k 100 (defaults here) | 0.0 | 3.1 | 25.3 | 71.2 | 75.09% |

At nprobe 160 with k = 10 the numbers land on the paper's. Read against this
project's defaults they look six times weaker, which is what a first reading
suggested and which was wrong.

## The three knobs, in the order they matter

### k -- a smaller k is a far tighter threshold

nprobe 32, prewarm 500:

| k | slice 2 | 3 | 4 |
| ---: | ---: | ---: | ---: |
| 100 | 3.1% | 25.3% | 71.2% |
| 10 | 11.0% | 56.5% | 93.9% |
| 1 | 30.2% | 81.2% | 98.9% |

The 100th neighbour is much further away than the 10th, so the threshold it
sets is looser by about an order of magnitude.

### nprobe -- a larger one drags in clusters that fail immediately

k 10, prewarm 2000:

| nprobe | slice 2 | 3 | 4 | distance ops |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 16.8% | 70.2% | 97.6% | 53.83% |
| 128 | 33.5% | 85.9% | 99.3% | 45.33% |
| 160 | 40.4% | 88.3% | 99.4% | 42.95% |
| 192 | 46.1% | 90.1% | 99.5% | 41.06% |
| 256 | 55.3% | 92.9% | 99.6% | 38.05% |

Past the few nearest clusters, nprobe buys mostly distant candidates, and a
distant candidate is one the first slice already disqualifies.

### prewarm -- real, but it has a ceiling

nprobe 32, k 100:

| prewarm | slice 2 | 3 | 4 | distance ops |
| --- | ---: | ---: | ---: | ---: |
| 500 x 1 (default) | 3.1% | 25.3% | 71.2% | 75.09% |
| 500 x 10 | 5.1% | 41.7% | 88.3% | 66.24% |
| 2000 x 3 | 8.8% | 53.2% | 93.2% | 61.19% |
| 8000 x 1 | 9.8% | 53.0% | 92.6% | 61.17% |
| **perfect threshold** | **12.1%** | **64.8%** | **96.9%** | **56.54%** |

The last row scans every probed cluster before the search starts, so the
threshold is exactly the true k-th distance. It is not a usable setting --
it is the search done twice -- and it is here to bound what a better
threshold could ever buy.

**At k = 100 even a perfect threshold reaches only 12.1% on the second
slice.** The paper's 41.76% is therefore not a k = 100 measurement, whatever
its prewarm was. That is what rules out "the threshold is worse here" as the
explanation.

At equal budget the concentrated prewarm still wins or ties (8000 x 1 against
2000 x 3), which is the behaviour `--prewarmlists 1` was chosen for. Note the
budget is **per cluster**: `--prewarmlists 10 --prewarm 500` scans 5000
vectors, not 500.

## Why this matters beyond Table 3

Any comparison of pruning methods has to fix nprobe and k across all of them.
A method measured at nprobe 160 / k 10 will look an order of magnitude better
than the same method at nprobe 32 / k 100, and nothing about the method
changed.

More prewarm prunes more but is not free, and on this cluster it does not pay:
at 2x4, raising prewarm from 500 to 2000 took distance ops from 67.35% to
62.13% and QPS from 1403 to 1360. The arithmetic saved is smaller than the
master-side cost of saving it, because communication is what bounds this
cluster, not compute.

## Notes

- Slice 1 is 0.00% by construction, in the paper and here.
- The grid is 1x4 on four nodes. More vector partitions prune more, because a
  later round starts from a threshold an earlier one tightened: at nprobe 32,
  k 100, prewarm 500, 2x4 on eight nodes gives 6.4 / 41.3 / 82.9 against
  1x4's 3.1 / 25.3 / 71.2.
- `distance ops` is the exact count of one-dimension operations, from the
  harness. `work_pct` in the CSV weights each slice equally instead and only
  agrees with it when the slices are the same width.
