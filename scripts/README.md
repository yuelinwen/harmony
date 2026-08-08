# scripts

Run them from anywhere; each one finds the project root itself.

| | |
|---|---|
| `setup_cluster.sh` | prepare fresh machines: ssh keys, packages, `hosts.txt`. Once. |
| `build.sh` | compile `./main`. macOS and Linux, MKL if present. |
| `deploy.sh` | copy `./main` to the machines in `hosts.txt`. After every build. |
| `run.sh` | start a run. Cluster if `hosts.txt` exists, local otherwise. |

## First time on a new cluster

```bash
scripts/setup_cluster.sh 192.168.73.188 192.168.73.115 192.168.73.116 192.168.73.254
scripts/build.sh
scripts/deploy.sh
scripts/run.sh 4
```

## Every time after that

```bash
scripts/build.sh && scripts/deploy.sh && scripts/run.sh 4
```

## On a laptop

No `hosts.txt`, so `run.sh` keeps everything local and single-threaded --
the processes would otherwise fight over the same cores. `deploy.sh` is not
needed.

```bash
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
