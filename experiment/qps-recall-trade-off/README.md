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

Writes two files beside the script:

- `results.csv` -- 45 rows, one per (layout, nprobe), each carrying recall,
  QPS and the five measurements
- `faiss.csv` -- 9 rows, the single-node faiss baseline

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
| faiss | the paper's baseline: `IndexIVFFlat`, same nlist, same 8 threads one worker gets |
| single-node IVF | this program's own index on one machine. Not a baseline -- it shares the clustering and the distance kernel with the distributed side. Kept because the gap to faiss is the correction any speedup against it needs. |

## Results

QPS, higher is better:

| nprobe | recall | 8x1 | 4x2 | 2x4 | 1x8 | own IVF | faiss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.40268 | 16257 | 16433 | 12825 | 12372 | 19845 | **26815** |
| 2 | 0.58335 | 14460 | 13952 | 10022 | 8749 | 11130 | **17718** |
| 4 | 0.75687 | **12971** | 11274 | 7402 | 5873 | 5872 | 10372 |
| 8 | 0.89403 | **11552** | 8950 | 4999 | 3202 | 3018 | 4656 |
| 16 | 0.96904 | **8558** | 6147 | 3126 | 1780 | 1541 | 2839 |
| 32 | 0.99555 | **7366** | 4263 | 1904 | 965 | 767 | 1568 |
| 64 | 0.99974 | **3957** | 2347 | 1022 | 454 | 384 | 815 |
| 128 | 0.99987 | **2487** | 1101 | 497 | 221 | 191 | 389 |
| 256 | 0.99987 | **1019** | 573 | 285 | 120 | 96 | 178 |

Speedup over single-node faiss:

| nprobe | recall | 8x1 | 4x2 | 2x4 | 1x8 | own IVF |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.40268 | 0.61x | 0.61x | 0.48x | 0.46x | 0.74x |
| 2 | 0.58335 | 0.82x | 0.79x | 0.57x | 0.49x | 0.63x |
| 4 | 0.75687 | 1.25x | 1.09x | 0.71x | 0.57x | 0.57x |
| 8 | 0.89403 | 2.48x | 1.92x | 1.07x | 0.69x | 0.65x |
| 16 | 0.96904 | 3.01x | 2.17x | 1.10x | 0.63x | 0.54x |
| 32 | 0.99555 | **4.70x** | 2.72x | 1.21x | 0.62x | 0.49x |
| 64 | 0.99974 | **4.86x** | 2.88x | 1.26x | 0.56x | 0.47x |
| 128 | 0.99987 | **6.40x** | 2.83x | 1.28x | 0.57x | 0.49x |
| 256 | 0.99987 | 5.71x | 3.21x | 1.60x | 0.67x | 0.54x |

Correctness, one verified run per layout at nprobe 32:

```
8x1  0/1024    4x2  0/1024    2x4  0/1024    1x8  0/1024
```

The sweep rows themselves are unchecked (`--check 0`, `differing = -1` in the
CSV) because verifying costs ten times the search it checks.

## What the curves say

**The comparison is fair.** The two IVF implementations agree on recall to
within 0.005 at every nprobe, so neither is buying speed with accuracy.

**Distribution does not pay until there is enough work to distribute.** At
nprobe 1 a query touches about 3,900 candidates and faiss on one node beats
all four layouts. The crossover is at nprobe 4, recall 0.76.

**4.70x over faiss at recall 0.9955, 4.86x at 0.9997.** The paper reports
4.93x, averaged over ten datasets on four nodes. Same range -- but this is
eight nodes, so per node it is well short: 59% of the worker count against
the paper's 123%.

**More dimension slices is strictly worse here, at every recall.** The
ordering 8x1 > 4x2 > 2x4 > 1x8 never changes over the whole sweep, and 1x8
never beats one node of faiss.

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
| Harmony (hybrid) is best overall | **Does not hold.** The hybrids lose to 8x1 everywhere |
| 4.93x over Faiss | **Same range** (4.70-4.86x at high recall), but on twice the nodes |
| All variants beat single-node Faiss | **Only 8x1 and 4x2, and only above recall 0.76.** 1x8 never does |

Two reasons the hybrids lose here, and they compound:

1. **Communication is expensive.** 4x2 saves 24% of the arithmetic at
   nprobe 32 and pays 82x the bytes for it. `--netprobe` prices a byte at
   about 21 multiply-adds on this cluster.
2. **Pruning is weaker than the paper reports.** On Sift1M with four slices
   the paper's Table 3 gives 0.00 / 41.76 / 85.04 / 98.40; this run gives
   0.00 / 6.40 / 41.30 / 82.90. The second slice prunes six times less.
   Unexplained -- the paper does not state the nprobe, k or nlist behind
   Table 3. Worth chasing before any experiment that leans on dimension
   slicing.

## Notes

- Recall saturates at 0.99987, not 1.0, even at nprobe 256 where the index
  degenerates to brute force. The residue is ties at rank 100 against the
  groundtruth file, not missed neighbours. faiss shows the same, at 0.99988.
- **faiss is about 2x this program's own single-node IVF** (1568 against 767
  QPS at nprobe 32). Speedups measured against the latter are inflated by
  that much, which is why the baseline had to be faiss.
- 4x2 edges out 8x1 at nprobe 1 (16433 against 16257). That is inside the
  run-to-run spread on these VMs, about 10%; it is not a crossover.
- QPS covers the distributed search only. Index build and distribution are
  timed separately and are in the CSV as `train_time`, `add_time` and
  `distribute_time`.
