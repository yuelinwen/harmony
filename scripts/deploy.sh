#!/bin/bash
#
# Copies the freshly built ./main to every machine in hosts.txt.
#
#   scripts/deploy.sh
#
# mpirun expects the executable at the same path on every machine it starts a
# process on, so this has to be rerun after every build. The data files are
# not copied: only rank 0 reads them, and the workers get their blocks over
# MPI.
#
# The machines all run the same distro on the same architecture, so one binary
# works everywhere. What does *not* travel with it are its shared libraries --
# if a build starts linking something new, install it on the workers too, or
# they will fail at startup with "cannot open shared object file".

set -e

cd "$(dirname "$0")/.."

if [ ! -f hosts.txt ]; then
    echo "no hosts.txt - nothing to deploy to" >&2
    exit 1
fi
if [ ! -x ./main ]; then
    echo "./main not built - run scripts/build.sh first" >&2
    exit 1
fi

ME=$(hostname -I 2>/dev/null | awk '{print $1}')

# -n on ssh matters: without it ssh swallows stdin, which here is hosts.txt,
# and the loop stops after the first machine.
while read -r host _; do
    [ -z "$host" ] && continue
    [ "$host" = "$ME" ] && continue          # this is where it was built
    ssh -n -o BatchMode=yes "$host" "mkdir -p ~/harmony"
    scp -q main "$host:~/harmony/main"
    echo "  $host"
done < hosts.txt

echo "deployed to the workers"
