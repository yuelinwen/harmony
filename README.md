# Harmony

A from-scratch C++ reproduction of the SIGMOD 2025 paper *HARMONY: A Scalable
Distributed Vector Database for High-Throughput Approximate Nearest Neighbor
Search* (Xu et al., Proc. ACM Manag. Data 3(4), Article 249).

The goal is a 1:1 reproduction of the system. Absolute performance numbers
cannot match — the paper runs 20 Intel Xeon nodes with 100Gb/s links — but the
mechanisms and the relative behaviour should.

## Build and run

```bash
./build.sh                                  # handles the macOS/Linux OpenMP split
mpirun -n 5 ./main                          # 1 master + 4 workers
mpirun -n 5 ./main --mode auto              # cost model picks the grid
mpirun -n 5 --hostfile hosts.txt ./main     # across machines
```

Any bad option prints the full list. Run from the project root: `Data/` paths
are relative. Only rank 0 reads the data files; workers receive their blocks
over MPI.

Useful for quick iteration: `--nlist 16 --iters 2 --nq 20` builds the index in
seconds instead of two minutes.

## Layout

```
main.cpp              MPI_Init, dispatch by rank
src/config.h          command-line options, grid resolution
src/comm/messages.h   MPI tags, job codes, the PRUNED sentinel
src/index/            dataset I/O, squared L2, global IVF (also the reference
                      implementation the distributed answer is checked against)
src/engine/           top-K heap (its worst() is the pruning threshold), slice plan
src/node/             master (plan, distribute, route, merge) and worker
                      (hold a block, accumulate, prune, forward)
```

## How a query flows

```
queryPipeline     centroids -> nprobe clusters -> group by row
  vectorPipeline    drive all rows at once, refill each row as it reports
    dispatchBatch     hand a row's clusters to its workers
      worker chain      each adds its dimensions, drops what passed the threshold
```

Workers are a `bVec x bDim` grid. A row owns a set of clusters; the columns of
that row each hold a slice of the dimensions. `bVec=N,bDim=1` is the paper's
Harmony-vector, `bVec=1,bDim=N` is Harmony-dimension, both > 1 is Harmony.

Parallelism comes from the two outer levels — rows run concurrently, and each
row keeps several clusters in flight. The innermost level is deliberately
serial: a cluster must visit its workers one at a time or there is no running
total to prune against (paper §3.2, Challenge 3).

## State

Done: Algorithm 1 (all four functions), the `bVec x bDim` grid and all three
modes, MPI with worker-to-worker forwarding, rotation of chain entry points,
non-blocking `Isend`/`Irecv`, OpenMP inside workers, command-line options,
recall measurement, and the cost model with load-aware plan selection.

That covers Fig. 3 (1-4) and §5.

Not done: the skewed workload generator of §6.2.1 (variance=500/1000), which
is only needed for Fig. 7/8; and calibration on real hardware.

Every change is checked with `queries differing from single machine: 0/N`,
which compares the distributed answer against `IvfIndex::search()`. Ties are
counted separately — squared distances on SIFT are integers and two candidates
at the same distance may legitimately swap.

## Where this departs from the paper

The paper leaves a number of things unspecified or, in a few places, states
something that does not hold. These are the deliberate deviations.

**Algorithm 1 line 12, `UpdatePruning(q, partialDist)`** — not implemented.
The text says it lowers the threshold from a partial result, but a partial sum
is a lower bound on the true distance, so using it to tighten a top-K
threshold would drop candidates that belong in the answer.

**Algorithm 1 line 20** — the pseudocode keeps one heap for a whole query set.
Each query needs its own top-K, so there is one heap per query here.

**`prune(q)`** — the pseudocode prunes queries; what actually gets pruned is
candidate vectors.

**Algorithm 1 line 22, `filterQueries`** — the paper selects the queries that
need a partition. Queries are processed one at a time here, so the same step
selects the clusters of that query which live in the partition.

**Algorithm 1 line 18** — the paper updates the threshold after a whole vector
partition. It is updated after each cluster here, which prunes strictly more
and changes nothing else.

**PrewarmHeap** — the paper samples `randomVectors` plus the centroid. Samples
come from the nearest cluster here, which gives a much tighter starting
threshold. The paper's own description (§4.3 prose vs the pseudocode) is not
consistent on this point.

**Cost model, computation term** — §4.3 states that computation per machine
does not vary significantly across layouts, and the cost model has no term for
it. Measured on Sift1M that does not hold once pruning is on: distance work
runs 100% at one dimension slice, 62% at two, 51% at four. Following the paper
here makes the model choose the layout that measures slowest. `pruneFactor()`
adds the missing term from measured values; the paper offers no way to predict
a pruning ratio.

**Rotation of chain entry points** — §4.3 says a machine that becomes
overloaded should have its dimension moved later, and Fig. 5b sketches a 3x3
example, but gives no scheduling rule. A round-robin start offset is used
here. It is static, not the dynamic adjustment the paper describes.

**Threshold distribution** — §5 says the master broadcasts tau^2 periodically.
It is carried in each cluster's job message here, which updates more often.

**Search over partition plans** — §4.2 says "iteratively searching". The
factorisations of the worker count are simply enumerated; there are only a
handful.

**alpha** — described as learned by offline profiling, with no method or
range given (0.3 appears in one example). It is a command-line option.

**variance = 500** — §6.2.1 gives no units and no generator. §4.2.1 defines
I(pi) as a standard deviation while the text calls it variance.

**§4.3 communication formula** — `n_probe * B_vec / n_list` yields less than
one partition for realistic parameters and is not used.

**Intel MKL** — §5 uses it to accelerate distance computation. Compiler flags
are used instead, which measured better and cost nothing.

On a Xeon Sapphire Rapids the distance loop runs at 4.8 GFLOP/s by default.
The limit is not SIMD width or memory bandwidth — cache-resident and
out-of-cache data run at the same speed, and forcing AVX-512 is 35% *slower*.
It is the accumulator: `sum = sum + d*d` makes every addition wait on the one
before it. Allowing the compiler to reassociate the sum splits it across
several accumulators and gives 12.7 GFLOP/s, 1.65x end to end, with the
acceptance check still at 0 differing queries.

MKL would not help here anyway. Its advantage comes from turning many
distances into one matrix multiply, which needs either data reuse or batching.
There is no reuse — each base vector is read once per query — and batching
conflicts with pruning, which exists precisely to avoid computing distances in
full. The paper acknowledges the same tension in §6.6 when it discusses PDX
"eliminating the SIMD performance bottleneck in dimension-pruning workloads".

**Appendices A, B, C** — referenced by the text (TopK=100 results, index build
time, peak memory) but absent from the PDF, so there is nothing to compare
those experiments against.

**Datasets** — the paper uses ten. Only Sift1M is set up. SpaceV1B and Sift1B
are out of reach (16 nodes, 400GB+), but they appear only in Fig. 6/7/8;
Table 3, Table 4 and Fig. 9-12 use the eight smaller ones. The paper also
expands StarLightCurves and HandOutlines without saying how.

## The test cluster

Ten Ubuntu 25.04 VMs, `yw-vdb-1` through `-10`, 8 cores and 30GB each, Xeon
Sapphire Rapids. `yw-vdb-1` (192.168.73.200, public 134.87.10.158) is the
master: it is the only one that reads the data files, since workers receive
their blocks over MPI. `hosts.txt` lists all ten at `slots=1`.

```bash
mpirun -n 5 --hostfile hosts.txt --map-by node ./main --threads 8
```

`--map-by node` puts one process per machine rather than packing them onto
the first; `--threads 8` then fills each machine's cores, which is the layout
the paper runs.

Inter-VM bandwidth measures about 109 MB/s -- roughly 1 Gb/s, two orders of
magnitude below the paper's 100 Gb/s. That difference dominates every timing
comparison, and it flips which layout wins:

| grid | shared memory | this cluster |
|------|--------------:|-------------:|
| 4x1 vector    | 112.7 QPS | **51.8 QPS** |
| 2x2 harmony   | **153.1** | 36.1 |
| 1x4 dimension | 131.9 | 20.9 |

On one machine, splitting by dimension is nearly free and pays for itself in
pruning. Across a 1 Gb/s network it is the worst thing to do. The cost model
picks correctly in both cases, which is the point of having one -- but it also
means the absolute QPS here, and the ranking of the three modes, cannot be
compared against the paper's numbers directly.

`--commcost` was calibrated two ways that agreed: 12.7 GFLOP/s over 109 MB/s
gives about 116, and sweeping the value until the model's ranking matches the
measured one gives anything from 10 up. The default is 100.

## Remaining work

1. Re-measure `pruneFactor()` per dataset.
2. The remaining datasets, for Table 3 and Table 4.
3. The skewed workload generator, for Fig. 7/8.
4. Batched query processing. Algorithm 1 line 13 takes a `QueryBatch`, and
   Fig. 5a shows queries moving through the pipeline in batches; queries are
   processed one at a time here. Batching is also what would make MKL worth
   having: the first dimension slice prunes nothing, so for a batch it is one
   dense matrix multiply. This is the largest remaining gap to the paper.
