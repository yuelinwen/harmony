# Harmony

A reproduction of *HARMONY: A Scalable Distributed Vector Database for
High-Throughput Approximate Nearest Neighbor Search* (SIGMOD 2025) in C++/MPI.

## Usage

### Building

Build on the cluster master. The script compiles there and copies the binary to
every machine in `scripts/hosts.txt`, because `mpirun` needs the executable at
the same path on all of them.

```bash
scripts/build.sh
```

MKL and GCC's OpenMP are both required. `./maketruth` is built as well when
`third_party/faiss` has been compiled; it is skipped otherwise.

### Preparing a dataset

`scripts/data.sh` turns an ann-benchmarks `.hdf5` into the four flat files the
program reads. It does not download anything -- put the `.hdf5` in `Data/`
yourself.

```bash
scripts/data.sh Data/sift/sift-128-euclidean.hdf5 sift/sift
```

```
<prefix>_base.bin    the vectors being searched
<prefix>_query.bin   the queries
<prefix>_gt.bin      the true top-k of each query, for recall
<prefix>_gtd.bin     their squared distances, for r2
```

A dataset that ships no groundtruth gets one from `./maketruth`, which brute
forces it with faiss:

```bash
./maketruth --data Data/xxx/xxx --k 100
```

### Running

`scripts/run.sh` takes the worker count and passes everything after it straight
to `./main`. One process per machine, one OpenMP thread per core.

```bash
scripts/run.sh 8 --nq 1024 --cache 1
```

```bash
# a grid chosen by hand: 4 vector partitions x 2 dimension slices
scripts/run.sh 8 --bvec 4 --bdim 2 --nq 1024 --cache 1

# let the cost model pick the grid
scripts/run.sh 8 --mode auto --nq 1024 --cache 1

# another dataset
scripts/run.sh 8 --data Data/fashion/fashion --bvec 2 --bdim 4 --nq 8192 --cache 1

# several nprobe values against one index and one distribution
scripts/run.sh 8 --nprobe 8 16 32 64 --nq 1024 --cache 1

# average over three timed passes, after one untimed warm-up
scripts/run.sh 8 --loop 3 --nq 1024 --cache 1

# append one row per run to a file
scripts/run.sh 8 --nq 1024 --cache 1 --csv results.csv
```

### Reading the output

```
===== 1. data =====       base, query and groundtruth sizes
===== 2. index =====      kmeans, or the cached index it loaded instead
===== 3. layout =====     which worker holds which clusters and which dimensions
===== 4. distribute =====
===== 5. results =====
  queries differing from single machine: 0/1024
                          the verdict -- anything but 0 is a bug
  recall@100, r2          how good the answer is against the groundtruth
  QPS, speedup            how fast
===== where each worker's time went =====
                          idle / recv / compute / send, per machine
===== where the master's time went =====
===== harness =====       candidates, dimension operations and bytes per hop;
                          then the totals and the threshold staleness
```

`differing` is the acceptance criterion: the distributed answer must equal what
rank 0 (the master) alone would have returned for the same queries and the same
clusters, compared as sets. Recall measures how good the index is; `differing`
measures whether the distribution is correct, and they are not the same
question.

### Further Usage

Any unrecognised option prints the full list, so `--help` works:

```
./main --help
usage: mpirun -n <N+1> ./main [options]

data and index
  --data <prefix>    reads <prefix>_base/_query/_gt.bin        (Data/sift/sift)
  --nlist <int>      clusters in the index              (256)
  --iters <int>      kmeans rounds                      (25)
  --trainpoints <int>  kmeans training points per centroid (256)
                     0 uses every base vector, which is much slower
  --cache <0|1>      reuse the index file beside the data       (0)
                     writes it on the first run with these settings

how the workers are laid out
  --mode <name>      harmony | vector | dimension       (harmony)
                       harmony   = the cost model picks the grid
                       vector    = N x 1, no dimension pipeline
                       dimension = 1 x N, no vector partitions
  --bvec <int>       vector partitions, overrides --mode
  --bdim <int>       dimension slices,  overrides --mode

the search
  --nq <int>         queries to run                     (100)
  --nprobe <int>...  clusters visited per query         (32)
                     several values run one after another, sharing
                     one index: --nprobe 8 16 32
  --k <int>          neighbours returned                (100)

speed
  --threads <int>    OpenMP threads per worker          (1)
  --batch <int>      queries processed together        (1024)
  --prewarm <int>    heap seed vectors per cluster      (500)
  --prewarmlists <int>  clusters seeded per query         (1)
                     either set to 0 turns seeding off
  --disablepruning   turn off dimension-level pruning
                     (drop a candidate past the threshold)
  --disablevectorpruning  turn off vector-level pruning
                     (tighten the threshold between partitions)
  --block <int>      query blocks per group        (from bDim)
                     default keeps about 256/bDim^2 queries in a
                     block: 4 at 8x1, 16 at 4x2, 64 at 2x4,
                     128 at 1x8, which is what measured fastest
  --blocksend <0|1>  wait for each forward to land        (0)
  --disablepipeline  one block out at a time, no overlap
                     otherwise all of a partition's blocks fly
                     at once
  --check <0|1>      verify against a single machine    (1)
                     costs more than the search it checks
  --baseline <0|1>   time that pass, report the speedup  (1)
  --loop <int>       timed passes, averaged               (1)
                     above 1 adds an untimed warm-up pass
  --csv <path>       append one row per run to this file

skewed workload (paper 6.6)
  --assign <name>    lpt | roundrobin                    (lpt)
                     roundrobin is the arm to measure lpt against
  --skew <float>     share of probes aimed at a hot eighth  (0)
                     0 searches for real; above it the probe lists
                     are synthetic, so recall stops meaning anything
                     while differing still does
  --indexskew <f>    the skew the layout is built from (--skew)
  --skewshift <f>    share of the test hot set that moved     (0)
                     0 = the same clusters are hot in both, which
                     LPT absorbs; raise it to make the layout miss
                     the clusters the queries want (Fig. 8)

cost model, only read when --mode harmony
  --commcost <float> a transferred byte, in multiply-adds (100)
  --alpha <float>    weight on the imbalance term        (0.3)
  --warmup <int>     queries used to find hot clusters   (1000)
```

### Ablation arms

The switches above cover the comparisons the paper draws:

| Arm | Flag | What it removes |
| --- | --- | --- |
| Harmony-vector | `--mode vector` | the dimension pipeline (N x 1) |
| Harmony-dimension | `--mode dimension` | the vector partitions (1 x N) |
| no dimension pruning | `--disablepruning` | dropping a candidate past the threshold |
| no vector pruning | `--disablevectorpruning` | tightening the threshold between partitions |
| blocking communication | `--blocksend 1` | overlapping a send with the next block (Fig. 2b) |
| no pipeline | `--disablepipeline` | several blocks of a partition in flight at once |
| no load-aware assignment | `--assign roundrobin` | LPT; clusters go to rows by id instead |
| skewed workload | `--skew`, `--skewshift` | a balanced workload (Fig. 7, Fig. 8) |
