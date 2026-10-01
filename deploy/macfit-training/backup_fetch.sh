#!/bin/sh
set -eu
: "${MACFIT_GPU_HOST:?MACFIT_GPU_HOST is required}"
set -o pipefail
umask 077
mkdir -p /tmp/backup-ssh /archive/incoming
test ! -L /archive/incoming
cp /run/backup-secrets/backup-key /tmp/backup-ssh/id_ed25519
cp /run/backup-secrets/known-hosts /tmp/backup-ssh/known_hosts
chmod 600 /tmp/backup-ssh/id_ed25519 /tmp/backup-ssh/known_hosts
part=/archive/incoming/snapshot.tgz.part
trap 'rm -f "$part"' EXIT HUP INT TERM
rm -f "$part"
set -C
# head bounds writes during transfer. pipefail rejects SSH failure or SIGPIPE at the cap.
timeout 480 ssh -T -i /tmp/backup-ssh/id_ed25519 \
  -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
  -o UserKnownHostsFile=/tmp/backup-ssh/known_hosts \
  -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
  root@"${MACFIT_GPU_HOST}" | head -c 26843545600 > "$part"
size=$(wc -c < "$part")
test "$size" -gt 0
test "$size" -lt 26843545600
mv "$part" /archive/incoming/snapshot.tgz
