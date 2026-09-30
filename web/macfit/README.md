# MacFit frontend

This directory versions the complete public frontend source and CPU-only Node
behavior tests for MacFit. The initial source was recovered from the existing
OpenShift frontend ConfigMap, then updated to use the authenticated training
service. No kubeconfig, login token, private deployment snapshot, browser profile,
or user dataset is included. `firebase-config.js` is a deployment template with
placeholder Firebase project values. Before deployment, replace every placeholder
with the target Firebase project's public web application configuration and remove
the `MACFIT_FIREBASE_CONFIG_REQUIRED` marker. Never place a Firebase Admin service
account or private key here. The deployment helper rejects the unchanged template.
For an existing installation, pass `--source` pointing to its already configured
private working copy; do not overwrite that working copy with this template.

Run the tests with a recent Node.js release:

```sh
cd web/macfit
node --test tests/*.test.mjs
```

`src/` is served by the existing Nginx frontend deployment. Training calls use
same-origin `/api/training/`; the more-specific Nginx location forwards them to
`macfit-training-bridge:8080`. The existing catalog/database `/api/` proxy, SPA
routing, security headers, asset rules, and health endpoint are preserved.

Prepare a reviewed immutable ConfigMap and guarded Deployment patch from the live
cluster using the repository deployment helper. Preparation only reads cluster
resources and saves rollback data locally:

```sh
python deploy/macfit-training/deploy_frontend_training.py \
  --kubeconfig /path/to/existing/kubeconfig prepare
```

Run that command from the repository root. Its default source is this directory's
`src`; snapshots are written beneath ignored repository `artifacts/`. An alternate
source can be supplied as `prepare --source /path/to/frontend/src`. A fresh checkout
requires the Firebase configuration replacement described above before prepare
will succeed.

The resulting `state.json`, `configmap.new.json`, and `patch.json` form a reviewable
release. The new ConfigMap retains all old keys and mounts all reviewed public
source files, including `training-client.js`; Nginx configuration stays out of the
public content mount. The patch changes only the existing content/nginx ConfigMap
references/items and guards the captured Deployment UID, resourceVersion, and old
volume contents. Apply also checks the old ConfigMap UID, resourceVersion, and data.

After review, an operator can explicitly apply:

```sh
python deploy/macfit-training/deploy_frontend_training.py \
  --kubeconfig /path/to/existing/kubeconfig apply /path/to/release
```

If anything changed concurrently, prepare and review a fresh release. Apply waits
for rollout; it does not claim browser or HTTP verification. Verify the actual
site assets, `/api/catalog`, `/api/training/capabilities`, account requirements,
and training flows after rollout. For rollback, use the same command with
`rollback` instead of `apply`. Rollback tests that the reviewed new volume entries
are still current and restores both captured old entries. Historical ConfigMaps
are retained.
