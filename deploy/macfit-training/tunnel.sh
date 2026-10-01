#!/bin/sh
set -eu
: "${MACFIT_GPU_HOST:?MACFIT_GPU_HOST is required}"
umask 077
mkdir -p /tmp/training-ssh
cp /run/training-secrets/ssh-key /tmp/training-ssh/id_ed25519
cp /run/training-secrets/known-hosts /tmp/training-ssh/known_hosts
chmod 600 /tmp/training-ssh/id_ed25519 /tmp/training-ssh/known_hosts
while :; do
  ssh -N -T \
    -i /tmp/training-ssh/id_ed25519 \
    -o IdentitiesOnly=yes -o BatchMode=yes \
    -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/tmp/training-ssh/known_hosts \
    -o ExitOnForwardFailure=yes -o ConnectTimeout=15 \
    -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    -L 127.0.0.1:8792:127.0.0.1:8791 \
    root@"${MACFIT_GPU_HOST}" || true
  sleep 3
done
