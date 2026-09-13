# Publication privacy

Publish source code, tests, portable examples, and general usage documentation.
Keep personal research reports, raw experiment stores, machine snapshots,
profiler traces, model caches, virtual environments, and local execution
configurations private. These can include hostnames, home paths, network
addresses, environment variables, prompts, and unpublished research results.

## Preparing a commit

Review and stage an explicit file set, then run:

```bash
python tools/check_publication.py
git diff --cached --check
```

The publication check reads the **Git index**, including partially staged files.
It rejects known local artifact roots, credential filenames, personal paths,
non-example email addresses, non-example IPv4 endpoints, and unscannable content.
It prints locations and categories without echoing the matched value. This is a
targeted privacy check, not a comprehensive secret detector or a history audit.

CI also runs [Gitleaks](https://github.com/gitleaks/gitleaks) on an export of the
committed files, plus Ruff and the CPU test suite. The scanner download is pinned
to a release and checksum. To scan locally without reading ignored private data:

```bash
scan_dir="$(mktemp -d)"
git checkout-index --all --prefix="$scan_dir/"
gitleaks dir --redact=100 --no-banner "$scan_dir"
```

Scan reachable history separately:

```bash
gitleaks git --log-opts=--all --redact=100 --no-banner
```

Passing a scan means no configured detector found a match in that scope. It does
not establish that every possible secret, personal detail, or unpublished
research result is absent. Review the intended publication scope as well.

## Local configuration and evidence

`/workspace`, `/path/to`, and `/home/USER` in examples are placeholders. Replace
them in a local copy, recapture runtime and model evidence, and recalculate the
associated command and protocol hashes before a live run.

Ignored files are not automatically included in Git commits, but may still be
included in a filesystem archive. Review archives separately. Keep private
research backups outside the public checkout where practical.

## History and metadata

Deleting a file in a later commit does not remove its historical versions.
Review all published branches, tags, commit metadata, release assets, and CI logs
before changing repository visibility. After rewriting history, verify that old
references and hosted cached objects no longer expose private content.

Use the account's GitHub noreply email for commits when email privacy is required.
Do not embed credentials in remote URLs, examples, or reports.
