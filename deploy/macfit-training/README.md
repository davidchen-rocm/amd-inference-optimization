# Private training bridge and persistent backup

`backup_pull.py` verifies a GPU service archive on the OpenShift backup PVC and
publishes it only after all content and database checks pass. It uses Python 3.11+
and the standard library; `macfit_training` must be available on `PYTHONPATH` from
the deployed source bundle. It does not import GPU libraries, use Firebase login
tokens, read an SSH key, or create Kubernetes Secrets.

See the [training guide](../../docs/training/README.md) for editable job configs,
generation versus training model selection, human approval and adapter use.
The GPU service, continuous backups and CPU archive serve different roles:
archive reads preserve historical results but cannot run or host a trained model.

## Pull and verify

The deployment provides the PVC at `/archive`, an SSH init container, and a Python
verification container. Configure `concurrencyPolicy: Forbid` on the CronJob so
two init containers cannot overwrite the shared incoming path. The verifier also
holds an exclusive filesystem lock, which protects publication and retention.

The dedicated SSH key must be restricted by the GPU server to this command:

```sh
PYTHONPATH=/srv/macfit-training/repo/src /srv/macfit-training/venv/bin/python \
  -m macfit_training.service.backup export \
  --data-dir /srv/macfit-training/data \
  --source-dir /srv/macfit-training/repo \
  --evidence-file /srv/macfit-training/runtime-evidence.json
```

The deployment supplies the actual Python interpreter path. Use a forced command
with no shell, forwarding, or PTY access for this key. Pin the SSH host key.
The export writes gzip tar bytes to stdout; do not merge stderr into stdout.

The SSH init container creates `/archive/incoming` with restrictive permissions,
uses `umask 077`, pulls to `/archive/incoming/snapshot.tgz.part`, and renames that
file to `snapshot.tgz` only if SSH exits successfully. The init container must
enforce a **25 GiB incoming-file limit during the transfer**, for example with its
shell's supported file-size resource limit and a bounded execution deadline.
Checking size only in the subsequent Python container cannot prevent a failed
or oversized SSH transfer from first filling the PVC. Neither the key nor known
hosts belongs inside the archive PVC.

The main container then runs:

```sh
PYTHONPATH=/opt/training-source/src python /opt/backup/backup_pull.py verify \
  --archive /archive/incoming/snapshot.tgz --archive-root /archive \
  --max-archive-gib 25 --max-unpacked-gib 32 --keep 3
```

The input defaults are the same as this command. The main container can use a
read-only root filesystem; all backup writes are confined to the PVC. The source
module bundle must already be mounted/extracted at the chosen `PYTHONPATH`.
Successful verification emits a small JSON receipt to stdout. Errors return a
nonzero exit status and a generic message without logging dataset content or
exception strings. The deployment must surface failed jobs to the operator.

For larger data volumes, the verifier accepts `--max-archive-gib`,
`--max-unpacked-gib`, `--max-entry-gib`, and `--free-reserve-gib` (integer GiB).
For example, `--max-archive-gib 24 --max-unpacked-gib 24 --max-entry-gib 8`
allows larger incompressible archives and databases. The service restore ceiling
is 32 GiB; the wrapper cannot exceed it. Increase the SSH init container's
transfer file limit at the same time. Larger snapshots can reduce retention below
four copies to preserve the current successful copy and the configured free
space reserve on a 100 GiB PVC.

## What is verified and retained

Every regular tar member must have an allowed normalized relative path. Absolute
paths, traversal, duplicates, directories, sparse files, symbolic links, hard
links, devices, and FIFOs are rejected. Every manifest file entry must match its
actual SHA-256 and byte length, with no unlisted or missing files. The complete
gzip stream and trailer are checked; hidden content after the tar end marker is
rejected. The rendered CronJob uses 25 GiB compressed, 32 GiB total unpacked, 2 GiB per file,
8 MiB for the manifest, and 20,000 members. The exported source/evidence is
additionally restricted by the service's backup allowlists.

The archive is copied into a private staging version, and its exact saved bytes
are revalidated before `service.backup.restore_archive` restores to a destination
that **does not yet exist**. That restore checks SQLite integrity and rejects
active jobs. The wrapper additionally runs SQLite `quick_check` and compares the
job count to the manifest. Files and directories are flushed before publication.

The layout after success is:

```text
/archive/
  .backup.lock
  incoming/snapshot.tgz
  latest -> versions/20261001T120000.000000Z-1234abcd
  versions/<UTC verification timestamp>-<random suffix>/
    archive.tgz          # exact pulled bytes, preserved for independent restore
    manifest.json        # original archive manifest
    metadata.json        # verification receipt and archive SHA-256
    data/
      jobs.sqlite3
      archive-mode.json
      jobs/...
      source/...
      evidence/...
```

`latest` is replaced atomically only after a complete verified version exists.
The incoming manifest's `created_at` must not precede the current receipt's
`source_created_at`. This check runs under the publication lock, preventing an
older periodic transfer that finishes late from replacing a newer final snapshot.
Read `/archive/latest/metadata.json` for the matching receipt; metadata lives
inside that version so it changes with the same atomic pointer. For a multi-file
inspection, resolve `latest` once and read through that version path. Never read
the mutable `incoming` file as the latest successful backup.

The rendered CronJob keeps the newest three verified snapshots by verification timestamp. Before a
large restore, older verified snapshots may be pruned to maintain a 1 GiB free
space reserve. The current `latest` is always protected; if it and the new
snapshot cannot fit, verification fails while preserving the latest. Invalid or
incomplete incoming archives never replace `latest`. An interrupted private
staging directory is removed on the next invocation under the lock. A crash
between final-directory creation and pointer publication may leave an additional
verified version; a later successful run includes it in retention. No automatic
pruning occurs when the latest pointer or its metadata needs operator recovery.

## Evidence and recovery

The receipt records `verified_at`, `source_created_at`, job/file counts, compressed
and unpacked sizes, archive SHA-256, and `sqlite_quick_check: "ok"`. The preserved
`data/source` and `data/evidence` identify the source/runtime associated with the
archive. Review these timestamps: a retained archive does not prove that newer
GPU jobs have been backed up. Source and public evidence are evidence copies,
not a restored production deployment or a credential backup.

For an independent CPU-only restore, first compare the saved archive SHA-256
with its receipt, then use a fresh destination:

```sh
PYTHONPATH=/opt/training-source/src python -m macfit_training.service.backup restore \
  --archive /archive/latest/archive.tgz --data-dir /recovery/new-restored-data
```

The destination must not exist. Restores remain `archive_only`: unfinished jobs
are preserved as failed/archive-interrupted records and are not resumed on a GPU.
The PVC contains private user datasets and artifacts and must remain private.

CPU regression checks:

```sh
PYTHONPATH=src python -m pytest -q \
  tests/training/test_backup_pull.py tests/training/test_service_backup.py
```

## Render the OpenShift deployment

This command generates a Kubernetes `List` for review. It never connects to the
cluster, creates a Secret, or applies a resource. The output file must be new.

```sh
python3 deploy/macfit-training/render_cluster.py \
  --gpu-host gpu.example.net --node example-openshift-node \
  --uid 1000740000 --firebase-project-id macfit-example \
  --output /private/path/training-cluster.json
```

After reviewing the generated resources and preparing the prerequisites below,
apply with **server-side apply**:

```sh
oc apply --server-side --field-manager=macfit-training \
  -f /private/path/training-cluster.json
```

The compressed source ConfigMap exceeds Kubernetes' 262,144-byte annotation
limit when client-side apply duplicates it in the `last-applied-configuration`
annotation. Server-side apply avoids this annotation; it is required for this
bundle. Review field ownership conflicts if reported, without automatically
forcing them.

Defaults target namespace `mac-fit`, UID/GID `1000740000`, and SNO node
`example-openshift-node`. Supply your own GPU hostname, node and Firebase project
when rendering deployment files kept outside the repository. The CLI requires
`--gpu-host`; IPv4/DNS validation rejects commands, ports and SSH options. Images
are pinned to the reviewed official Python and
Alpine/Git SHA-256 digests in the renderer; tag-only overrides are rejected.
The generated resources are:

- A service account without an automounted API token.
- A 200 GiB `Retain` local PV/PVC at `/var/mnt/macfit-training-backups` on that node.
- Immutable, digest-named ConfigMaps for a compressed Python source ZIP and scripts.
- One `Recreate` bridge Deployment with Python bridge, SSH tunnel and CPU archive containers.
- An internal ClusterIP Service on port 8080; there is no public Route or GPU port.
- An ingress NetworkPolicy allowing only same-namespace Pods labeled
  `app=mac-fit` to reach the bridge on TCP 8080. It does not restrict egress or
  expose the archive's port 8793. Normal node-originated kubelet probes remain
  permitted without a broad node CIDR exception.
- A backup CronJob every ten minutes with `Forbid` concurrency, a twenty-minute
  overall deadline, eight-minute SSH deadline, bounded transfer and retention of three.

The cluster operator must first prepare the local directory with ownership and
SELinux labeling that permit the namespace UID to write. Confirm actual backing
storage capacity: declaring a 200 GiB local PV does not impose a directory quota
or allocate another disk. PV creation and namespace-specific SCC validation are
operator actions, not actions taken by the renderer.

The renderer expects a separately created `macfit-training-connection` Secret
with these exact keys: `gateway-token`, `bridge-key`, `backup-key`, `known-hosts`.
No secret value is read while rendering. Each container receives only the keys
it needs. SSH private files are copied to its own writable `/tmp` with mode 0600;
the key volumes and other container filesystems remain read-only. The SSH images
receive a minimal ConfigMap `/etc/passwd` containing the arbitrary namespace UID.
All containers drop capabilities, deny privilege escalation, run non-root and
use RuntimeDefault seccomp. The GPU-side bridge key must permit only forwarding
to loopback port 8791. The backup key must allow only the forced export command
described above. Neither key belongs in the source ZIP, backup PVC or logs.

`backup_fetch.sh` uses an SSH/head pipeline with `pipefail`, a transfer deadline,
and a 25-GiB output cap before the Python verifier runs. Exactly-at-cap
transfers are rejected too, so truncation cannot masquerade as success. It never
replaces the incoming completed file on a failed SSH transfer. The verifier then
validates tar contents, hashes, database and limits before publishing `latest`.

The code ZIP includes only repository `src/**/*.py` files; its hash is checked
before extraction and its base64 payload must remain below 900 KiB. It is
extracted into a temporary directory in a shared `emptyDir`. A digest marker and
the exact file inventory/bytes are verified before atomic publication; ordinary
containers mount it read-only. Init-container retries reuse a complete verified
publication and clean abandoned staging directories. They reject changed output
instead of trusting the mere existence of the final directory. Render
again after source or script changes; the new content-addressed ConfigMaps roll
the Deployment. Old unused ConfigMaps may be removed later after confirming no
Deployment or historical Job references them.

## Continuous CPU archive reads

At first startup, `bootstrap.py vendor` installs the fully pinned CPU dependency
lockfile into `/archive/vendor/<requirements-and-runtime-hash>` and atomically
publishes `/archive/vendor/current`. It installs binary wheels only, without
dependency resolution beyond the checked-in complete lockfile. An import check
runs before publication. There is no PyTorch, Transformers or ROCm installation.
The first installation needs access to PyPI; subsequent matching pod restarts
reuse the persistent vendor directory. Change and revalidate the lockfile when
changing the pinned Python runtime or service dependencies.

`archive_server.py watch` waits for a verified backup and copies its database and
job files into a separate `/archive/serving/v<version>-<suffix>` directory. Copying
holds the publisher's archive lock, preventing concurrent retention from pruning
the source. A database integrity check and archive-only marker check run before
the new API starts on port 8793. SQLite WAL/schema setup modifies only this
serving copy, never the original retained snapshot. The archive API uses the
normal Firebase ownership verifier and gateway check, rejects all mutations,
and always reports `available: false`. A read response adds `X-MacFit-Archive: true`.

When a new validated backup arrives, the watcher prepares its copy while the old
API remains usable, then briefly stops and replaces the child process. If startup
fails, the previous copy is restarted. Serving copies are protected independently
of backup retention; only obsolete copies created by this watcher are removed.
On pod restart, the last checked serving copy can start even if the latest
pointer needs recovery. Interrupted temporary copies are cleaned under a unique
watcher lock. No GPU workers resume from restored data.

The archive container deliberately has no TCP readiness gate: before the first
backup it must not make an otherwise healthy GPU bridge disappear from the
Service. The bridge owns Pod readiness and falls back to archive **reads only**
when the GPU upstream cannot be reached. It never replays a POST to the archive.
The archive is only as recent as its `source_created_at` receipt; verify a fresh
successful snapshot before the GPU server is retired. Firebase certificate
refresh still needs Google's public certificate endpoint after the GPU is gone.
The verifier validates signed Firebase ID tokens and expiration but does not
query revocation or disabled-account state. An already issued token can remain
usable until expiration plus the 30-second clock tolerance. Immediate per-user
revocation needs additional server-side verification. The archive applies the
same authentication limitation; neither VPN access nor possession of a job ID
bypasses owner checks.

The writable serving copies need additional disk space; with the configured
per-snapshot limits, plan for retained archives, expanded data, old/new serving
copies, incoming transfer and dependency cache together. If there is insufficient
space, a new copy is rejected while the last good serving copy remains available.

Additional CPU deployment checks:

```sh
PYTHONPATH=src python -m pytest -q tests/training/test_cluster_runtime.py
```

The checks exercise manifest policy, source hash/path boundaries, independent
serving copies and the real archive API's read-only/authentication behavior.
Cluster scheduling, UID/SELinux access, image pulls, SSH credentials and the
initial live backup still require deployment verification by the operator.
At the 2026-09-30 deployment checkpoint, the verified archive contained three job
records. This is a historical backup checkpoint, not a claim that all three jobs
were successful or that the later final backup has completed. See the separate
[validation scope and evidence](../../docs/training/README.md#deployment-status-and-verification-scope)
before interpreting CPU test counts or operational checks.

## Final backup before a scheduled GPU retirement

The optional finalizer is scheduled for **2026-09-30 at 23:55 America/Detroit**
(`2026-10-01T03:55:00Z`). It runs in OpenShift independently of the MacBook. The
deployed GPU service is configured to stop work at **23:45 America/Detroit**
(`2026-10-01T03:45:00Z`), ten minutes earlier. Admission can stop earlier to fit
bounded jobs and cleanup. This finalizer does not terminate GPU workers or
delete the rented server. At the checkpoint above, the final run was still
scheduled and had not yet been verified.

Render it from the exact reviewed base manifest that was deployed, after
including the monotonic backup publication update:

```sh
python3 deploy/macfit-training/render_finalizer.py \
  --base-manifest /private/path/training-cluster.json \
  --output /private/path/training-finalizer.json
```

After review, apply it with the same server-side field manager:

```sh
oc apply --server-side --field-manager=macfit-training \
  -f /private/path/training-finalizer.json
```

This emits only an additional scripts ConfigMap, dedicated ServiceAccount,
Role/RoleBinding and finalizer CronJob. It copies the existing backup pod template,
image digests, source/scripts ConfigMap names and supplied `MACFIT_GPU_HOST`;
it neither regenerates nor changes the periodic CronJob. Review the date before
applying these one-use resources. Cron syntax has no year field, so the finalizer
self-suspends and its runtime also rejects a start outside the intended window.

The final transfer writes a separate directory keyed by Pod UID, so it cannot
overwrite a periodic transfer's incoming file. The publisher rejects a snapshot
whose source timestamp is older than the current latest receipt, preventing a
slow earlier periodic job from replacing the final snapshot. SSH failure is
recorded in a shared status file and does not prevent the main finalizer from
running. The source/bootstrap and image preparation still must succeed first.

Once the pull finishes, the main container first suspends both
`macfit-training-backup` and `macfit-training-finalizer`. This happens even if the
pull failed: the GPU window is ending, so leaving another failing schedule active
would not repair the backup. A long verifier operation cannot delay suspension
until after the Job deadline. Existing running Jobs are not deleted.

The main container then verifies/publishes the archive with the normal 25/32 GiB
compressed/unpacked bounds and retention of three. Success additionally requires
the published receipt to match the source version, a source timestamp no older
than sixty seconds before this run began, SQLite integrity, matching job count,
and terminal jobs without worker PIDs. `archive_interrupted` errors are rejected:
exporting an unfinished job as a restored failure is not evidence that its source
work completed before retirement. A stale receipt is never treated as final
backup success.

`/archive/finalization.json` and the Job log record whether the final snapshot was
verified and which schedules were confirmed suspended. Exit status 0 means both
verification and both suspensions succeeded. Backup failure exits 1 while
preserving available verified archives; failure to confirm suspension exits 2 and
requires operator action. The Job has `backoffLimit: 0`, an overall ten-minute
deadline, and keeps at most three failed Job records. **Suspended schedules alone
are not proof of a successful final backup.** Check a successful finalizer Job,
`verified: true`, and the final snapshot's timestamp together.
The main container's early schedule pause does not mark the snapshot verified;
that state is written only after every verification check succeeds. A killed,
timed-out or incomplete finalizer remains unverified even if both schedules are
already suspended. An older completion receipt cannot establish this run's
success; check its timestamp and the matching Job result.

The dedicated Role permits only `get` and `patch` on these exact two named
CronJobs. Its short-lived projected service-account token and cluster CA are
mounted only in the main finalizer container; ordinary bridge/backup pods retain
`automountServiceAccountToken: false`. Kubernetes API calls use standard-library
HTTPS with certificate validation and bounded retries. No Kubernetes client
package, gateway token, browser token or GPU key is stored in the completion
receipt. If the cluster/API is unavailable or the Pod cannot start, no in-cluster
job can guarantee final backup or schedule suspension; inspect the failed/pending
Job and the last existing receipt rather than assuming retirement was completed.
