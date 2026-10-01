#!/bin/sh
# A failed pull must still allow the finalizer to run and suspend expired schedules.
set -u
umask 077
status=1
case "${POD_UID:-}" in
  ''|*[!a-f0-9-]*) printf '1\n' > /final-state/fetch-status; exit 0 ;;
esac
if (
  set -eu
  : "${MACFIT_GPU_HOST:?MACFIT_GPU_HOST is required}"
  set -o pipefail
  directory=/archive/final-incoming/$POD_UID
  mkdir -p /tmp/final-ssh "$directory"
  test ! -L "$directory"
  cp /run/backup-secrets/backup-key /tmp/final-ssh/key
  cp /run/backup-secrets/known-hosts /tmp/final-ssh/known_hosts
  chmod 600 /tmp/final-ssh/key /tmp/final-ssh/known_hosts
  part=$directory/snapshot.tgz.part
  trap 'rm -f "$part"' EXIT HUP INT TERM
  rm -f "$part"
  set -C
  timeout 480 ssh -T -i /tmp/final-ssh/key \
    -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=yes \
    -o UserKnownHostsFile=/tmp/final-ssh/known_hosts \
    -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 \
    root@"${MACFIT_GPU_HOST}" | head -c 26843545600 > "$part"
  size=$(wc -c < "$part")
  test "$size" -gt 0
  test "$size" -lt 26843545600
  mv "$part" "$directory/snapshot.tgz"
); then
  status=0
fi
printf '%s\n' "$status" > /final-state/fetch-status
exit 0
