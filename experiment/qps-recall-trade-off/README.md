# QPS-recall trade-off

Paper section 6.2.1, Fig. 6 and Fig. 7.

## What it measures

`nprobe` trades accuracy for speed, so one value gives a point and a sweep
gives a curve. The curve answers the only question worth asking of systems
that can each be made faster by being wronger: **at a given recall, which
layout is fastest?**

## Running it

```bash
scripts/build.sh                            # on the cluster master
experiment/qps-recall-trade-off/run.sh
```

Writes `results.csv` beside the script: 45 rows, one per (layout, nprobe),
each carrying recall, QPS and the five measurements.

## Setup

| | |
| --- | --- |
| Dataset | Sift1M (1,000,000 x 128), the one dataset here the paper also uses |
| Cluster | 1 master + 8 workers, 8 cores each, OpenMPI |
| Index | nlist 256, 25 kmeans rounds, 256 training points per centroid |
| Queries | 1024, k = 100 |
| Sweep | nprobe 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| Timing | `--loop 3`, after one untimed warm-up |

Lines, against the paper's:

| Here | Paper |
| --- | --- |
| 8x1 | Harmony-vector. Also what `--mode harmony` picks on this cluster, so those two lines would coincide. |
| 4x2, 2x4 | hybrids, which the paper does not plot separately |
| 1x8 | Harmony-dimension |
| single-node IVF | this program's own index on one machine. It stands in for Faiss, which is not wired up as a baseline yet, so the speedups here are **not** comparable with the paper's 4.93x |

## Results

QPS, higher is better:

| nprobe | recall | 8x1 | 4x2 | 2x4 | 1x8 | single-node IVF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.40268 | 16257 | 16434 | 12825 | 12372 | **19845** |
| 2 | 0.58335 | 14460 | 13952 | 10022 | 8749 | **11130** |
| 4 | 0.75687 | **12972** | 11274 | 7402 | 5873 | 5872 |
| 8 | 0.89403 | **11552** | 8950 | 4999 | 3202 | 3018 |
| 16 | 0.96904 | **8558** | 6147 | 3126 | 1780 | 1542 |
| 32 | 0.99555 | **7366** | 4263 | 1904 | 965 | 767 |
| 64 | 0.99974 | **3957** | 2347 | 1022 | 454 | 384 |
| 128 | 0.99987 | **2487** | 1101 | 497 | 221 | 191 |
| 256 | 0.99987 | **1019** | 573 | 285 | 120 | 96 |

Correctness, one verified run per layout at nprobe 32:

```
8x1  0/1024    4x2  0/1024    2x4  0/1024    1x8  0/1024
```

The sweep rows themselves are unchecked (`--check 0`, `differing = -1` in the
CSV) because verifying costs ten times the search it checks.

## What the curves say

**Distribution does not pay until there is enough work to distribute.** At
nprobe 1 a query touches about 3,900 candidates, and single-node IVF beats
all four layouts. The crossover is at nprobe 4, recall 0.76. Above it 8x1
pulls away, reaching 10.6x single-node IVF at nprobe 256.

**More dimension slices is strictly worse here, at every recall.** The
ordering 8x1 > 4x2 > 2x4 > 1x8 never changes over the whole sweep.

**1x8 lands on top of single-node IVF.** At nprobe 256 it is 120 QPS against
96 -- eight machines producing one machine's throughput.

## Why, from the five measurements

At nprobe 128:

| layout | distance ops | bytes forwarded |
| --- | ---: | ---: |
| 8x1 | 100.00% | 6.6 MB |
| 4x2 | 64.43% | 2,024 MB |
| 2x4 | 56.32% | 6,065 MB |
| 1x8 | 56.26% | 14,148 MB |

Slicing the dimensions removes **44%** of the arithmetic and multiplies the
traffic by **2,144x**. On this interconnect the second term decides it.

`tools/netbench` measured the interconnect at about 856 MB/s for the 2 MB
messages a chain hop carries, and **395-414 us** of one-way latency -- an
order of magnitude worse than a physical 10GbE link, because the network is
virtual.

## Against the paper

| Paper's finding | Here |
| --- | --- |
| Harmony-vector is optimal below 99% recall | **Holds**, and extends: 8x1 is optimal at every recall above 0.6 |
| Harmony (hybrid) is best overall, 4.93x over Faiss | **Does not hold.** The hybrids lose to 8x1 everywhere |
| All variants beat the single-node baseline | **Holds above recall 0.76**; below it single-node IVF wins |

The disagreement is the interconnect, not the implementation. The paper's
cluster has 100 Gb/s links; dimension slicing pays there and does not pay
here. The cost model reaches the same conclusion from its own measurements
and picks 8x1 on this cluster, which is why `--mode harmony` and 8x1 are the
same line.

## Notes

- Recall saturates at 0.99987, not 1.0, even at nprobe 256 where the index
  degenerates to brute force. The residue is ties at rank 100 against the
  groundtruth file, not missed neighbours.
- 4x2 edges out 8x1 at nprobe 1 (16434 against 16257). That is inside the
  run-to-run spread on these VMs, which is about 10%; it is not a crossover.
- QPS covers the distributed search only. Index build and distribution are
  timed separately and are in the CSV as `train_time`, `add_time` and
  `distribute_time`.
