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

## On the cluster

Which machine a step runs on is half of it — only the transfer runs on the
laptop, everything else on the master.

```bash
KEY=~/.ssh/key.pem scripts/setup_cluster.sh <worker-ip> <worker-ip> ...
```

```bash
scripts/data.sh Data/sift-128-euclidean.hdf5        # once per dataset
```

```bash
tar cf - src main.cpp scripts | ssh <master> 'cd ~/harmony && tar xf -'
```

```bash
scripts/build.sh
```

```bash
scripts/run.sh 4
```

The first two are once, both on the master. `setup_cluster.sh` is the only
command with addresses written out, because it is the one that writes
`hosts.txt`; everything after it reads the file. It takes the workers only —
it puts itself first — and `KEY=` is just for the first contact, before its
own key is installed. Only rank 0 opens a data file, so `Data/` is needed on
the master alone; the workers get their blocks over MPI.

`<master>` is whatever address the laptop reaches the master at. That is not
`hosts.txt`'s first line: the file holds the addresses the machines use among
themselves, which from outside the cluster are not routable.

The last three are the edit-and-check loop. The transfer is the one that runs
on the laptop, and it is not optional: the master has no copy of an edit until
it is sent, so skipping it means the next run tests the old code without
saying so. Check that the files actually landed before building — a copy step
that silently skips a directory does not fail, it just leaves the old binary
in place. `build.sh` then scp's the new binary to every worker, because mpirun
needs it at the same path on all of them.

`run.sh N` runs N workers on N+1 machines, one process per machine, one OpenMP
thread per core.

## A new dataset

Put the `.hdf5` in `Data/` yourself — nothing downloads it — then:

```bash
scripts/data.sh Data/gist-960-euclidean.hdf5
scripts/run.sh 4 --data Data/gist
```

Euclidean datasets only. An angular one is refused: dimension pruning depends
on squared L2, where a partial sum can only grow, and an angular file's
groundtruth is ranked by angle anyway.

Converting needs `h5py` and `numpy` (`pip3 install h5py numpy`), on whichever
machine does the converting. Nothing else does: the cluster reads the `.bin`
files.

## On a laptop

Do not use `run.sh` here. It picks its branch by whether `hosts.txt` exists,
and that file stays behind on a laptop that has ever been pointed at a cluster,
so it would ssh into addresses this machine cannot reach. Call `mpirun`
directly: without `--hostfile` every process starts locally, and
`--oversubscribe` lets all N+1 of them share the one machine.

```bash
brew install open-mpi libomp        # once
```

```bash
scripts/data.sh Data/sift-128-euclidean.hdf5        # once per dataset
```

```bash
scripts/build.sh
```

```bash
mpirun -n 5 --oversubscribe ./main --nlist 16 --iters 2 --nprobe 4 --nq 32
```

The first two are once; the last two are the edit-and-check loop. `data.sh`
stops early if the `.bin` files are already there, so it is safe to leave in a
script. `-n 5` is one master and four workers.

Small `--nlist` and `--iters` build the index in half a second instead of half
a minute, which is what makes it usable for checking a change; `--nprobe` has
to come down with `--nlist`, or it asks for more clusters than exist and every
query scans the whole index.

`build.sh` needs no such care — it looks for one of this machine's own
addresses in `hosts.txt`, finds none, and skips the copy step by itself.

A run is correct when it prints `differing: 0`. QPS is not comparable to the
cluster: there is no MKL on arm64, and the processes are sharing cores.

## Options

```bash
./main --help
```

Or any option it does not recognise. They are grouped by what they affect:
data, layout, search, speed, cost model.
