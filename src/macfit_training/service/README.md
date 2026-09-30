# MacFit private training service

Run one service process as an unprivileged account on the GPU host:

```sh
python -m macfit_training.service
```

The server binds only `127.0.0.1:8791`. Configure the private SSH bridge to supply
`X-Training-Gateway`, and keep the systemd unit's `KillMode=control-group` so an
unexpected service exit also stops child workers. Every job route verifies a Firebase
ID token for the configured `MACFIT_FIREBASE_PROJECT_ID`; browser-supplied user IDs
are never identities. Set this explicitly for a deployment; the reusable default
`macfit-example` does not grant access to another Firebase project.

The verifier checks RS256 signatures, issuer, audience, required identity/time
claims and expiration, with a 30-second clock tolerance. It does not query
Firebase token revocation or disabled-account state. Previously issued tokens
can remain usable until expiration plus that tolerance after revocation; this
service does not provide immediate per-user revocation. Signing certificates
are cached only for their bounded lifetime and refreshed from Google's public
endpoint. An expired cache cannot authenticate requests when refresh fails.
The same ownership and gateway checks apply to the CPU archive service.

Required settings are `MACFIT_TRAINING_DATA` (default `/srv/macfit-training/data`) and
`MACFIT_TRAINING_GATEWAY_SECRET` (at least 32 random bytes). After an actual ROCm tensor
check, write `MACFIT_GPU_READY_FILE` (default `/srv/macfit-training/gpu-ready.json`) with
`{"ready":true,"boot_id":"the current /proc/sys/kernel/random/boot_id"}`. Readiness
expires on reboot and never substitutes for the hardware check.

`MACFIT_GPU_DEADLINE=2026-10-01T03:45:00Z` is this temporary deployment's hard GPU
work cutoff (**2026-09-30 at 23:45 America/Detroit**);
`MACFIT_STOP_ACCEPTING_AT=2026-10-01T03:00:00Z` also stops new submissions. Admission
reserves the maximum wall time for each active/queued job plus 120 seconds for
cleanup and final backup. Consequently admission can close earlier than the nominal
stop time. The supervisor stops remaining worker processes before the hard deadline.
Historical owner-scoped jobs and verified downloads remain available.
The separate OpenShift final backup is scheduled at 23:55 Detroit (`03:55:00Z`).
These settings do not delete the GPU server. Use the
[deployment guide](../../../deploy/macfit-training/README.md) to render the
bridge, backup/archive resources and one-use finalizer, and verify their actual
completion before retiring storage. A three-job backup was verified during the
2026-09-30 deployment checkpoint; the final backup remained scheduled at that
checkpoint and must be checked independently.

## Backup and restore

Use a dedicated forced-command SSH key to run this export. The command writes only
gzip-compressed tar bytes to stdout, and prints a sanitized error to stderr on failure:

```sh
python -m macfit_training.service.backup export \
  --data-dir /srv/macfit-training/data \
  --source-dir /srv/macfit-training/repo \
  --evidence-file /srv/macfit-training/runtime-evidence.json
```

`--source-dir` is optional and permits only `src`, `pyproject.toml`, `README[.md]`,
`docs/training`, `examples/training`, `deploy/macfit-training`, `tests/training`, and
the full `web/macfit` static frontend and browser tests, using a small set of source,
document, and web extensions. The frontend's exact `src/nginx.conf` path is included.
Hidden files, known credential files, and private/virtual-environment/node_modules
folders are excluded. `--evidence-file` may repeat; `data/evidence/*.json` is included
automatically. Evidence JSON must not contain credentials or private keys. Do not
pass credential files to the evidence arguments. Model caches are omitted; base
model repository IDs and immutable revisions remain in each saved input/result.

The export uses SQLite's online backup API for a consistent database snapshot and
copies only allowlisted job files. Registered completed artifacts must match their
stored SHA-256 and size. Running events are a bounded prefix; a running job's result
and unregistered artifacts are not completed evidence. Snapshot rows for unfinished
jobs become `failed/archive_interrupted` without changing the live database.
The job's `model-snapshot.json`, `runtime-environment.json`, and `training-source.json`
are included with 8 MiB, 16 MiB, and 1 MiB limits so offline adapter export can verify
the pinned model snapshot and provenance. Worker logs and credential files are excluded.

Each archive has `manifest.json` with file paths, lengths, and SHA-256 hashes. Capture
stdout to a temporary file, require a zero exit status, and validate by restoring to
a temporary directory before atomically publishing a backup as latest. Retain the
previous verified backup until the new one is verified. Backups contain users' private
training inputs and should remain on the private backup PVC with restricted access.

Restore to a path that does not exist:

```sh
python -m macfit_training.service.backup restore \
  --archive /backup/latest.tar.gz \
  --data-dir /srv/macfit-archive/data
```

Restore rejects traversal, links, duplicates, excess size, hash mismatches, invalid
SQLite data, and runnable jobs before atomically publishing the destination. Source
and evidence are under `source/` and `evidence/`. The automatic `archive-mode.json`
marker makes this directory permanently read-only for GPU work. You may also set
`MACFIT_ARCHIVE_ONLY=1`. Run the ordinary API with fresh gateway credentials and its
Firebase verifier; it needs the package's service dependencies but no torch/ROCm.
It serves jobs and downloads while capabilities report unavailable for new work.
Do not remove the archive marker to resume old jobs; submit a new request to a
separately validated GPU service if training should be repeated.
