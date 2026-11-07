# Public release checklist - review before any Git write

Update 2026-11-07: [Modal public deployment is verified](modal_deployment_20261107/report.md).
The preparation findings below are retained; their pending-hosting statements describe
the earlier review. No Git staging, commit or push has occurred. Final scientific results remain pending.

## Current verified state

The supplied epoch-12 archive passed SHA256, strict model, full Adam/scheduler/scaler/RNG,
history, best-epoch and saved validation-metadata checks. All 66 training-source hashes match.
The checkpoint was copied byte-for-byte into ignored private review storage, never changed.
This was checkpoint inspection, not a new scientific benchmark.

277 tests passed. An isolated headless Chrome session exercised real uploads, loading states,
all six classical methods and exact-checkpoint deep/hybrid on synthetic dense4 inputs. It checked
NPZ downloads, rendered previews, server errors, missing-RGB prevention, completion scale locking,
keyboard focus, accessible names and desktop/mobile emulation. The browser checked 1440, 390
and 360 px layouts without horizontal overflow. These are smoke checks, not scientific accuracy,
a full WCAG audit or physical-device verification. See public_release_verification.json.

## Proposed public release, not staged or published

- [ ] Review the exact inventory in PUBLIC_RELEASE_FILE_LIST.md.
- [ ] Review every source/doc change and approve the proposed file list before staging.
- [ ] Use a clean export under ignored release_review/public_tree for publication review.
      The exporter does not stage, commit, push, initialize Git or rewrite history.
- [ ] Keep original path-bearing evidence untouched in private storage. Nine separate
      sanitized copies and original/copy hashes are in public_evidence/manifest.json.
      JSON numeric metrics, sample IDs, configurations and content hashes are retained.
- [ ] The public export substitutes redacted documentation at original relative locations,
      preserving links; mirror copies are for review. They are not byte-identical scientific
      artifacts and must not replace originals in provenance/hash audits.
- [ ] Existing Git history can still contain original paths. Adding sanitized files to this
      repository does not erase history. Choose an approved clean-history publication plan
      rather than force-pushing or deleting evidence without authorization.
- [ ] Keep .gitattributes byte-preservation rules for scientific source/config/notebooks.
- [ ] Review licensing for the project, third-party dependencies and dataset-derived PNGs.
      No project LICENSE file was found; no licence is invented by this preparation.
- [ ] Review the final export for credentials and personal paths. Signature scanning cannot
      prove absence of secrets, especially in existing Git history.

## Safe-to-publish classes after review

Application/scientific source, tests, relative experiment configs, output-free notebooks,
requirements, Docker recipe, active submission docs and sanitized evidence copies.
The complete list includes unchanged dependencies, not only dirty files. Dataset-derived
images are separately marked for licence/privacy review. Team names/USNs are intentional
submission information supplied by the team.

## Keep private

Dataset payloads and archives; original weights and training/run ZIPs; unsanitized historical
path-bearing records; release_review profiles, fixture copies, checkpoint copies, logs and
screenshots; credentials/environment files; .git history unless separately reviewed. Raw model
weights need an explicit separate distribution decision, not accidental Git inclusion.
trident_depth/data/*.py is source code; the top-level data/ folder is private dataset storage.

## Exact deployment gates

1. Do not touch the currently running Colab benchmark. After its own completion, review
   replay/isolation status and import actual final metrics/panels in a separate authorized step.
2. Approve the clean release inventory and licence/image decisions. The review export has
   no Git history; do not blindly push the existing history containing private paths.
3. On a Docker-capable machine run the documented build and container smoke checks. The
   Docker configuration has only been reviewed statically here; no successful build is claimed.
4. Serve classical mode first or mount the exact checkpoint read-only for learned mode.
   Use PORT, one worker, HTTPS/reverse-proxy request limits and a health probe. Do not use
   a live Colab training/benchmark runtime as production hosting.
5. Verify /, /static/app.js, /api/health, /docs, real-format upload errors and numeric
   downloads. Confirm the loaded checkpoint SHA. Test cold starts and actual host latency.
6. Test on physical phone/laptop and review keyboard/screen-reader usability. Current
   mobile checks are browser emulation only. Inspect deployment logs without publishing
   user uploads or private filesystem paths.
7. Only after the hosted app passes these checks, record its actual URL in README/eval.
   No public URL has been created or verified by this task.

## Local commands (do not rerun the Colab job)

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m pip check
$env:TRIDENT_HOST='127.0.0.1'
.venv\Scripts\python.exe -m backend.serve
# Prepare another NEW export directory; an existing export is never overwritten.
.venv\Scripts\python.exe scripts/prepare_public_release.py --output release_review/public_tree_review2
```

Default execution of prepare_public_release.py prints candidate paths only. Staging, committing,
pushing, hosting and history rewriting remain separate approval-gated actions.
