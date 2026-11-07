> Historical preparation snapshot. For subsequent approved cleanup and publication status, see [LOCAL_HISTORY_VERIFICATION.md](LOCAL_HISTORY_VERIFICATION.md). Deployment integration remains deferred.

# Privacy remediation - preparation only, awaiting approval

No project commits or branch heads have been rewritten. No force-push, publication,
repository deletion, Actions deployment or Modal change has occurred. The live Git history
still contains the original paths. The clean result below is an **in-memory projection**,
not an actual cleaned Git repository and not proof that GitHub caches have been purged.

## 1. Backup completed before release-preparation changes

A complete private copy outside the checkout contains **44,229 files / 16,022,602,146 bytes**
(14.92 GiB), including .git, ignored data, checkpoints, environments and local review files.
Every copied file has a matching SHA256. Git fsck passed on the backup, and an additional
all-refs history bundle passed git bundle verify. File-by-file manifest and validation records
are stored alongside the private backup. Backup folder name:
`TRIDENT-Depth-T69_PRIVACY_BACKUP_20261107T091318Z`.
The absolute owner-only location is recorded in ignored local review storage and supplied
directly to the owner, rather than inserted into public documentation.

Private history bundle SHA256:
`0cece32abf53f4c3e0ab355b7a1566855374055af6b4b432e0ef0971c06f9ae1`.

Backups retain the original sensitive records deliberately. Keep them private and never
push or mirror their refs. Backup copying reads file bytes, including ignored data; no dataset
sample was decoded for evaluation, no model training ran and no inference used test samples.

## 2. Exact exposure inventory

Both advertised remote branches, main and jathin, point to
`633b55b95c636f6b39ec413cd847b3c3dbd7c8a7`. That commit contains these eight records:

| Historical record | Exact structural location | Exposure |
| --- | --- | --- |
| docs/corrective_design/verification.json | preservation.sha256, dictionary key #98 | One local archive path |
| docs/corrective_v1/verification.json | historical_archives_verified, dictionary key #1 | One local archive path |
| docs/custom_diagnostics/analysis.json | protected_files_sha256, dictionary keys #84 and #85 | Two local archive paths |
| docs/custom_diagnostics/report.md | line 208 | Local diagnostic artifact/archive location |
| docs/nyu/baseline_results_dense_sr.json | source_manifest | Local dataset-manifest location |
| docs/nyu/baseline_results_sparse_completion.json | source_manifest | Local dataset-manifest location |
| docs/nyu/source_inspection.json | source | Local dataset-source location |
| docs/pilot_dense4/report.md | line 33 | Local pilot archive location |

Dictionary indices are zero-based in the original serialized record. All contain machine-home
prefixes, exposing the user profile name and folder layout. Exact unredacted strings, line/field
locations, Git blob IDs, original/sanitized SHA256 values and referring roots are documented in
`release_review/privacy_remediation_20261107/final_projection/private_exact_exposure_and_projection.json`.
That file is private and excluded from the export. The public summary omits exact values.

A ninth record, docs/hybrid_v1_dense4/implementation.md, appears in a local-only tool capture
tree, not in the advertised public branch commits. The projection covers it too. A separate
non-sensitive absolute placeholder in historical README is changed to a relative checkpoint
example as hygiene; it is not counted as an additional personal-data exposure.

## 3. Proposed transformation and exact projected impact

Only documentation with detected home paths is transformed. Replace private Windows/Linux/macOS
home prefixes with ${USER_HOME}, including JSON dictionary keys and values. Preserve the useful
relative suffix. JSON comparison verifies that numeric values, metric fields, counts, hashes and
all unrelated values remain unchanged. Private originals remain byte-identical in the original
checkout and backup. Public current files are prepared as separate sanitized export copies.

The four withheld dataset-derived images are excluded from current exports and proposed history:

- docs/nyu/registration_1.png
- docs/nyu/registration_2.png
- docs/training_preparation/prediction_review_dense_sr.png
- docs/training_preparation/prediction_review_sparse_completion.png

Their original files are not deleted. Historical removal is simulated both by path and by their
four blob IDs, covering identical-content aliases. .gitignore and the exporter now exclude these
paths. Ignore rules alone do not remove tracked files or old history.

The read-only planner reconstructs original trees and checks their existing Git IDs before
projecting replacements. It computes potential object IDs with hashlib only. Its Git command
allowlist cannot execute hash-object, update-ref, commit-tree, filter-repo, reset or push.

| Item | Original | Projected, not created |
| --- | --- | --- |
| Signed initial commit | 1ee87ec21fc023d21adb46b4d617d4a331fc46be | Unchanged, including signature bytes |
| Affected main/jathin tip | 633b55b95c636f6b39ec413cd847b3c3dbd7c8a7 | 58bf70da990e0fbc74cc3f29cc268645b022e651 |
| Public branch history | Two commits | Two commits, same parent topology |
| All locally reachable commits | Five | Only one commit ID changes; local-only commit records retained |

Authors, committers, timestamps and messages are preserved in this projection. Source files,
configuration and notebook bytes are unchanged. A future reviewed release commit containing
the latest working implementation is separate; its SHA does not yet exist. Stop for review if
the actual approved rewrite differs from the projected bytes or encounters other signed commits.

Session-host refs/codex capture trees rotated after the owner's continuation message. This was
observed and recorded separately; the planner did not write those refs. Project branches and
remote-tracking heads are unchanged. Private captures must never be included in a public push.

## 4. Verification results and their limits

The final projection covers five commits plus local-only tree refs. It finds zero remaining
private-home paths and zero known credential signatures after the proposed transforms. A broader
decoded-string scan of 270 text blobs finds zero remaining absolute Windows drive-path literals.
JSON/notebook strings are decoded first to avoid treating escaped Python newlines as paths.
187 protected historical source/config/notebook file versions retain their original bytes.
The current 104 protected files, nine original/sanitized evidence hash pairs, approved checkpoint
and reviewed evaluation file are separately checked against their preserved hashes/backups.

**Actual current/remote history is not clean yet.** It retains the eight records until an
approved rewrite and publication. Signature scanning is not an exhaustive proof that every
possible secret is absent. No credentials were found, and no secret values are printed.

All original 289 tests and the added privacy-safety tests are run on the working tree and the
sanitized, image-free release copy. The verification record alongside this plan reports exact
counts. No test is weakened or deleted. No seven-frame benchmark continuation is performed:
scientific status remains 854/861, with sparse_noisy_holes 116/123.

## 5. After explicit approval: isolated local rewrite, still no public change

1. Freeze collaborator pushes and refresh all advertised heads/tags/PR refs. Abort if either
   branch no longer has the reviewed old tip. At inspection there were no tags or advertised
   PR refs, and GitHub reported zero forks; this is not a guarantee about hidden caches/clones.
2. Create a fresh isolated local cleanup clone, excluding private agent/capture refs. Leave the
   original working repository and full backup untouched. Keep the existing app and Volume intact.
3. Implement the reviewed minimal object transformation in that clone: write sanitized blobs,
   rebuild affected trees/unsigned commits, reuse unaffected objects and the signed initial
   commit, then update only the reviewed local refs. A generic rewrite tool must not silently
   strip the retained initial signature or prune commits. This writing stage is not implemented
   or executed by the read-only planner.
4. Cover both main and jathin, plus any newly discovered public refs. Clean/remove tainted local
   remote-tracking or temporary refs in the isolated clone so a full reachable-history scan is
   meaningful. Do not create old-to-new replace refs that keep tainted history reachable.
5. Run actual reachable-object/path/credential/image scans, Git fsck, source/numeric preservation
   comparisons and the full test suite. Overlay the separately reviewed sanitized current release
   without touching private originals, then review its exact staged diff. Record actual commit maps.
6. Stop again and show the verified local result. Local-rewrite approval does **not** authorize
   force-pushing, release publication, repository visibility changes or Actions deployment.

## 6. Separate future public-update approval and implications

A normal fast-forward push cannot replace the affected history. Any eventual approved update
must explicitly cover both main and jathin with expected-old SHA leases, not a broad --mirror
push or an unguarded force. Recheck branch protection and collaborator work first; any temporary
protection changes require owner approval and restoration. No such command runs in this phase.
Updating only main would leave the paths reachable from jathin. Do not publish backup, agent,
capture, replace or stale remote refs. Keep GitHub-to-Modal deployment opt-in disabled throughout
the privacy update; enabling it and the first production run require separate approval.

History changes invalidate old revision links and may affect reviews and collaborator branches.
Old commits can remain accessible in existing clones or GitHub caches/PR references even after
a branch update. GitHub Support may decline removal of data it does not classify as sensitive;
complete public erasure cannot be promised. The owner decides on any Support request or temporary
visibility change separately. The existing GitHub repository will not be deleted.
[GitHub's removal guidance](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## 7. Collaborator instructions for the later approved cutover

- Stop pushes to both affected branches while the owner coordinates cleanup.
- Privately back up unpushed work; do not upload old .git directories or backup refs.
- After the owner confirms the cleaned remote tips, use a fresh clone. Do not merge the old
  main/jathin history into it, and do not restore old refs with pull/push/mirror operations.
- Transfer unpushed changes as reviewed file diffs or carefully reviewed commits on a clean base;
  rescan transferred content. Recreate local environments and configure private data/checkpoint
  paths locally. Do not commit dataset payloads, images under hold or credentials.
- Confirm new heads, expected scientific source hashes and clean scans before resuming pushes.
  Original scientific provenance stays in private backup; retain the approved old-to-new map.

## Approval requested next

Review this plan, the owner-only exact exposure record and verification results. The next
possible authorization is **isolated local history rewrite and verification only**. Publication,
force-push and deployment remain separate, unapproved steps.
