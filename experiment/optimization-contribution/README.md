# Contribution of each optimization

Paper section 6.3.2, Fig. 10.

## What it measures

Three features are switched on one at a time and the throughput measured
after each, so each step's ratio is what that feature is worth *on top of the
ones before it*. The three:

| feature | what it does | switched off by |
| --- | --- | --- |
| balanced load | deals clusters to rows by how much work each brings, not by id | `--assign roundrobin` |
| pipeline, async | many blocks open at once, and a forward rides alongside the next block's arithmetic | `--disablepipeline --blocksend 1` |
| pruning | drops a candidate as soon as its partial distance passes the threshold | `--disablepruning --disablevectorpruning` |

## Running it

```bash
scripts/build.sh                                  # on the cluster master
experiment/optimization-contribution/run.sh
```

Four nodes, matching the paper. Sift1M, 1024 queries, three repeats.
`results.csv` has every column; `logs/` has each run's whole output.

Three operating points. 2x2 and 4x1 because pruning needs something to
prune -- with one dimension slice the threshold is only tested after the
whole distance has been computed. And 2x2 again at nprobe 160, k 10, because
a top-100 threshold is the 100th distance and stays loose, so reporting only
nprobe 32, k 100 would say pruning does not pay when what it says is that
*that operating point* does not pay.

## Results

### 2x2, nprobe 32, k 100

| step | QPS | gain | cumulative | distance ops | busy-time skew |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 none | 620.3 | -- | 1.00x | 17.20 G (100%) | 1.14x |
| 2 + balanced load | 616.0 | **0.99x** | 0.99x | 17.20 G (100%) | 1.18x |
| 3 + pipeline, async | 1585.2 | **2.57x** | 2.56x | 17.20 G (100%) | 1.09x |
| 4 + pruning | 1671.1 | **1.05x** | **2.69x** | 13.62 G (79.2%) | 1.09x |

### 4x1, nprobe 32, k 100

| step | QPS | gain | cumulative | distance ops | busy-time skew |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 none | 1755.6 | -- | 1.00x | 17.20 G (100%) | 1.10x |
| 2 + balanced load | 1760.8 | **1.00x** | 1.00x | 17.20 G (100%) | 1.05x |
| 3 + pipeline, async | 4672.4 | **2.65x** | 2.66x | 17.20 G (100%) | 1.22x |
| 4 + pruning | 4274.7 | **0.91x** | **2.43x** | 17.20 G (100%) | 1.15x |

### 2x2, nprobe 160, k 10

| step | QPS | gain | cumulative | distance ops | busy-time skew |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 none | 123.7 | -- | 1.00x | 78.84 G (100%) | 1.07x |
| 2 + balanced load | 126.1 | **1.02x** | 1.02x | 78.84 G (100%) | 1.08x |
| 3 + pipeline, async | 324.5 | **2.57x** | 2.62x | 78.84 G (100%) | 1.19x |
| 4 + pruning | 354.0 | **1.09x** | **2.86x** | 44.69 G (56.7%) | 1.10x |

### Against the paper

| | balanced load | pipeline, async | pruning | total |
| --- | ---: | ---: | ---: | ---: |
| paper, Sift1M | 1.63x | 1.11x | 1.77x | 3.21x |
| paper, Msong | 1.88x | 1.39x | 1.25x | 3.27x |
| here, 2x2 nprobe 32 | 0.99x | 2.57x | 1.05x | 2.69x |
| here, 4x1 nprobe 32 | 1.00x | 2.65x | 0.91x | 2.43x |
| here, 2x2 nprobe 160 | 1.02x | 2.57x | 1.09x | 2.86x |

**The total reproduces; the split does not.** 2.4x to 2.9x against the paper's
3.2x, but the paper's three roughly equal contributions come out here as one
feature worth 2.6x and two worth nothing.

## Why the split differs

**This cluster is bound by network latency, not by arithmetic.** `tools/netbench`
measures 395--414 us one way between these VMs. The master spends 78% to 99%
of a batch blocked in `MPI_Waitany`, and a worker's own busy time is 0.095s
out of a 0.24s batch at 4x1 and 0.363s out of 2.89s at nprobe 160. The
workers are mostly waiting. Everything below follows from that.

**1. Overlapping communication is worth 2.6x, not 1.25x.** With one block
open at a time and a blocking wait on each forward, every block pays the full
latency in series. Letting them overlap is the single largest thing measured
here, in all three operating points, and it is large precisely because the
latency it hides is large.

**2. Pruning is worth 1.05x even when it removes 43% of the arithmetic.** At
nprobe 160 the distance ops drop from 78.84 G to 44.69 G and mean worker busy
time drops 0.591s to 0.363s, a 39% cut that tracks the arithmetic almost
exactly -- and throughput moves 9%, because the time saved was not the time
the batch was waiting on. Arithmetic is not what the clock is spent on.

At 4x1 it is worse than nothing, 0.91x: with a single dimension slice the
threshold is only reached after the whole distance has been computed, the
distance ops stay at 17.20 G of 17.20 G, and `candidates scanned` is
*identical* between step 3 and step 4 -- 134,344,065 either way. The
vector-level barrier it adds is real work and it buys nothing. This is the
layout's property, not a defect: it is why the grid search picks 8x1 for
throughput and the pruning only pays once there is more than one slice.

**3. Balanced load is worth nothing here, for two separate reasons.**

The first is the step order. At step 2 the pipeline is still off, so the
master drains one group before dispatching the next and exactly one block is
open in the whole system. There is no concurrency for a balanced assignment
to exploit. The paper measures the same order, so this is not a difference in
method -- but on a system where step 2 has no parallelism to balance, the
feature cannot show up until step 3, where it is folded into the pipeline's
number.

The second is the workload. Round-robin already lands within 1.05x--1.18x of
even here, so LPT has almost nothing to fix. That is Sift1M's doing: the
queries are drawn from the data's own distribution, the clusters come out
close to the same size, and dealing them out by id is nearly as good as
weighing them. Measured separately in the skewed-workload runs, the same LPT
is worth a great deal once the hot clusters move away from the ones the index
was built around -- partition load goes from 1.01x to 7.52x imbalance. The
paper's 1.75x is a statement about a skewed workload; 1.00x here is a
statement about an unskewed one.

## A deadlock this experiment found

Step 1 hung the first time it was run: 24 minutes at 99.9% CPU on all five
ranks, which reads as a slow run rather than a hang because MPI busy-waits.

`--disablevectorpruning` and `--disablepipeline` deadlocked against each
other. Without the vector-level barrier a worker takes the next partition's
thresholds as soon as it has finished one block, with a blocking receive, and
before opening the blocks still in its queue. With the pipeline off the master
does not send those thresholds until the group's blocks have all come back.
Each side was waiting for the other.

Fixed in `src/node/worker_node.cpp`: with work in hand the worker now probes
for the thresholds instead of blocking on them, the same way it already
probed for jobs. With vector-level pruning on -- the default -- the gate is
unchanged.

Every run here is under `timeout`, so the next one of those fails visibly.

## Notes

- `candidates scanned` is identical between steps 3 and 4 at every operating
  point. Vector-level pruning never skips a candidate on its own; all it does
  is tighten the threshold that dimension-level pruning spends.
- Step 4 is the default build, so its QPS is comparable with the other
  experiments at the same settings.
- `--check 0` throughout: correctness is not what this measures, and the
  single-machine comparison would dominate the run time.
