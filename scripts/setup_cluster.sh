#!/bin/bash
#
# Prepares a fresh set of machines to run as a cluster. Run it once, on the
# machine that will be the master.
#
#   scripts/setup_cluster.sh <worker-ip> <worker-ip> ...
#
# It expects to already be able to reach each of them -- either passwordless
# already, or through a key named on the command line:
#
#   KEY=~/.ssh/key.pem scripts/setup_cluster.sh <worker-ip> ...
#
# What it does, per machine:
#   - adds this machine's public key to authorized_keys, because mpirun
#     starts remote processes over ssh and cannot answer a password prompt
#   - installs OpenMPI, and MKL if this machine has it, since a binary built
#     here will expect the same libraries there
#   - writes scripts/hosts.txt, one address per line with this machine first
#
# The file holds addresses and nothing else. How many processes go on each
# machine is decided by run.sh, which passes -N 1.

set -e

cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    echo "usage: scripts/setup_cluster.sh <worker-ip> [worker-ip ...]" >&2
    exit 1
fi

SSH_KEY=${KEY:+-i $KEY}
SSH_OPTS="-o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=10"

# a key of our own for the cluster, kept separate from however we got in
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C harmony-mpi >/dev/null
PUB=$(cat ~/.ssh/id_ed25519.pub)

WANT="libopenmpi-dev openmpi-bin"
[ -f /usr/include/mkl/mkl.h ] && WANT="$WANT libmkl-dev"

for ip in "$@"; do
    echo "== $ip =="
    ssh $SSH_KEY $SSH_OPTS "ubuntu@$ip" \
        "mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
         grep -qF '$PUB' ~/.ssh/authorized_keys 2>/dev/null || echo '$PUB' >> ~/.ssh/authorized_keys"

    ssh $SSH_OPTS "ubuntu@$ip" \
        "sudo apt-get update -qq >/dev/null 2>&1; \
         sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq $WANT >/dev/null 2>&1; \
         echo -n '  '; hostname; echo -n '  '; mpirun --version | head -1"
done

{
    hostname -I | awk '{print $1}'      # this machine, the master, first
    for ip in "$@"; do echo "$ip"; done
} > scripts/hosts.txt

echo
echo "scripts/hosts.txt:"
sed 's/^/  /' scripts/hosts.txt
echo
echo "next: scripts/build.sh && scripts/run.sh $#"
