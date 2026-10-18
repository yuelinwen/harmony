# Index memory

Paper section 6.4.2, Table 4.

## What it measures

What a distributed index costs in space against the single-machine index it
replaces. The vectors are only divided, so the total should barely move --
but a dimension-sliced layout has every column of a row holding the same id
list, and a norm per vector per slice that a single machine never keeps.

## Running it

```bash
scripts/build.sh                    # on the cluster master
experiment/index-memory/run.sh
```

Four nodes, matching the paper. Nothing is searched; the memory is settled
once the data has been handed out.

## Results

### Sift1M (1,000,000 x 128), four nodes

| layout | total | per node | vs one machine | share of faiss |
| --- | ---: | ---: | ---: | ---: |
| one machine | 492 MiB | -- | -- | -- |
| **faiss index** | **496 MiB** | -- | -- | -- |
| 4x1 (Harmony-vector) | 495 MiB | **123 MB** | +0.75% | 24.8% |
| 2x2 (Harmony) | 503 MiB | **125 MB** | +2.30% | 25.2% |
| 1x4 (Harmony-dimension) | 518 MiB | **129 MB** | +5.40% | 26.0% |

Against the paper's Sift1M row:

| | Faiss | Harmony-vector | Harmony-dimension | Harmony |
| --- | ---: | ---: | ---: | ---: |
| paper | 497 MB | 126 MB | 131 MB | 128 MB |
| here | 496 MiB | 123 MB | 129 MB | 125 MB |
| difference | −0.2% | −2.4% | −1.5% | −2.3% |

### Two 784-dimensional datasets, four nodes

| dataset | one machine | 4x1 | 2x2 | 1x4 | 1x4 overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sift1M (128 dim) | 492 MiB | 495 | 503 | 518 | **+5.40%** |
| fashion-mnist (784 dim) | 180 MiB | 179 | 180 | 181 | **+0.6%** |
| MNIST (784 dim) | 180 MiB | 179 | 180 | 181 | **+0.6%** |

## Against the paper

**1. Index overhead is proportional to dataset size and dimension.** Holds.
Sift1M is 492 MiB and fashion-mnist 180 MiB, a ratio of 2.73; their `n x d`
ratio is 2.72.

**2. All three schemes occupy about 1/4 of the Faiss index on four nodes.**
Holds. Measured 24.8% / 25.2% / 26.0% per node against the paper's 25.4% /
25.8% / 26.4%. Distributing the index costs no significant duplication.

**3. Harmony and Harmony-dimension carry some extra, about 2%.** Holds for
the hybrid -- 2x2 measures +2.30% -- but 1x4 is +5.40%, and the paper's own
numbers say the same (131/126 = +4.0%). **What the paper does not say is that
the share depends on the dimension**: the same 1x4 layout costs +0.6% on a
784-dimensional dataset, a ninth of what it costs on Sift1M.

## Where the overhead comes from

```
vector data   divided, total unchanged        n * d * 4
ids           one copy per column             n * 4 * bDim
||v||^2       one copy per column             n * 4 * bDim     single machine keeps none
```

So the overhead is `n * 8 * bDim` against data of `n * d * 4`, and its share
is `2 * bDim / d` -- set by the dimension and the number of slices, not by
how many vectors there are.

| | predicted | measured |
| --- | ---: | ---: |
| Sift1M, bDim 4, d 128 | 6.25% | 5.40% |
| fashion, bDim 4, d 784 | 1.02% | 0.60% |
| Sift1M, bDim 1, d 128 | 1.56% | 0.75% |

Measured runs below predicted because the ids and norms are counted against
the data alone above, while the denominator reported is the whole
single-machine index, which also holds the centroids. At bDim 1 there is no
id duplication at all and the whole 0.75% is the norms array.

**The higher the dimension, the less dimension slicing costs in space.**

## Notes

- The faiss figure is the index file on disk. For a flat IVF that is within
  a header of the resident size.
- `||v||^2` is precomputed so the chain head can use `‖q‖² + ‖v‖² − 2q·v` and
  hand the dot products to MKL. It is the entire overhead at bDim 1.
- fashion-mnist and MNIST give identical numbers because they have the same
  shape, 60,000 x 784. That is a coincidence of the datasets, not a bug.
