# scripts

Run them from anywhere; each one finds the project root itself.

| | |
|---|---|
| `setup_cluster.sh` | prepare fresh machines: ssh keys, packages, `hosts.txt`. Once. |
| `data.sh` | convert an `.hdf5` in `Data/` into the three files the program reads. |
| `build.sh` | compile `./main`, and copy it to the workers when run on the master. |
| `run.sh` | start a run. Cluster if `hosts.txt` exists, local otherwise. |

`hosts.txt` is one address per line, this machine first. Addresses only —
how many processes go on each machine is `run.sh`'s business, not the file's.
It is gitignored, being particular to one cluster.

## First time on a new cluster

```bash
scripts/setup_cluster.sh 192.168.73.188 192.168.73.115 192.168.73.116
scripts/build.sh
scripts/run.sh 3
```

## Every time after that

```bash
scripts/build.sh && scripts/run.sh 4
```

## A new dataset

Put the `.hdf5` in `Data/` yourself — nothing downloads it — then:

```bash
scripts/data.sh Data/gist-960-euclidean.hdf5
scripts/run.sh 4 --data Data/gist
```

Euclidean datasets only. An angular one is refused: dimension pruning depends
on squared L2, where a partial sum can only grow, and an angular file's
groundtruth is ranked by angle anyway.

## On a laptop

No `hosts.txt` there, so `run.sh` keeps everything local and single-threaded —
the processes would otherwise fight over the same cores. `build.sh` skips the
copy step for the same reason.

```bash
brew install open-mpi libomp        # once
scripts/build.sh && scripts/run.sh 4 --nlist 16 --iters 2 --nq 32
```

Small `--nlist` and `--iters` build the index in seconds instead of minutes,
which is what makes it usable for checking a change.

## Options

```bash
./main --help
```

Or any option it does not recognise. They are grouped by what they affect:
data, layout, search, speed, cost model.
