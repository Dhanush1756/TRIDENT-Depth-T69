# Submission repository audit ? 7 November 2026

## Completed in this application/documentation phase

- Inspected actual backend/frontend, requirements, Docker recipe, tests and Git status.
- Added a backend-only extended checkpoint adapter; exact epoch-12 SHA256 is required before
  loading. It uses original model and hybrid implementations, with separate raw deep output.
- Preserved all scientific source/config/notebook files and previous experiment evidence.
  All 66 source hashes recorded by the actual training GPU preflight still match.
- Health reports real method/protocol/scale capabilities without local checkpoint paths. B0
  advertises only dense4 deep; unavailable hybrid is not advertised. Known legacy training
  pipelines constrain their learned options. Extended completion keeps resolution and uses
  conditioning 4. Learned serving requires RGB and rejects unqualified dense noise/holes.
- UI now uses capabilities, has a busy spinner/status, upload checks, backend/network errors,
  RGB preview and stale-result hiding. Numeric data downloads are preserved.
- Updated README, architecture, DESIGN, deployment instructions and submission evaluation
  structure. Final scientific benchmark values are deliberately pending.
- Added non-root container PORT startup/health check and excluded ZIP/credential context files.

## Verification actually run

Full suite: **273 passed**, one pre-existing Starlette/httpx deprecation warning, 36.45 s.
No existing tests were weakened. The added tests use temporary synthetic fixtures, never
real checkpoints or dataset training. They verify model-weight immutability, protocol/scale
restrictions, missing RGB, checkpoint hash refusal, malformed masks/object arrays, and raw
deep versus observed-point-preserving hybrid behavior.

A real local Uvicorn process served `/`, static JS/CSS, `/docs` and `/api/health`. Twelve
HTTP reconstructions (six classical methods x dense/completion) passed numeric-download
and supplied-GT checks on clearly synthetic constant-depth fixtures. The process was
terminated after its own checks; no other process or Colab runtime was contacted.
See [HTTP smoke record](submission_app_smoke.json). These are app correctness checks,
not scientific evaluation or model-quality results. `pip check`: no broken requirements.

## Publication and path review

No credential signatures were found in scanned current tracked/untracked text sources
(UTF-8/UTF-16; a signature scan is not proof of absence across Git history). No tracked
credential/key files were found. Added ignore rules exclude local environment/agent/key files.
Application paths derive from repository root or explicit environment settings; new runtime
code contains no machine-specific Windows dataset/checkpoint default.

Existing historical evidence contains local machine paths in:
`docs/custom_diagnostics/{analysis.json,report.md}`, `docs/pilot_dense4/report.md`,
`docs/nyu/{baseline_results_dense_sr.json,baseline_results_sparse_completion.json,source_inspection.json}`,
`docs/corrective_design/verification.json`, `docs/corrective_v1/verification.json` and
`docs/hybrid_v1_dense4/implementation.md`. These were intentionally not rewritten under the
preserve-evidence rule. Review a separate public copy for privacy before publishing. Existing
notebook Drive paths are configurable runtime locations, not application defaults.

Git retains prior untracked scientific implementation/config/notebook files in addition to
this phase's app/documentation edits. No commit or push was performed. Review all staged
files and preserve pinned source bytes before publication; do not stage artifacts or ZIPs.

## Unresolved gates ? do not mark complete

1. The external final T4 benchmark is running. Wait for its own completion, replay PASS and
   isolation audit, then import final tables/panels without changing old evidence.
2. The real epoch-12 checkpoint is only on Drive. Actual-weight local app inference has not
   been run; only synthetic extended integration is verified. Copy the exact checkpoint
   separately after the active run is safely complete and configure the documented env vars.
3. Docker is unavailable locally, so container build/run and image dependency installation
   remain unverified. The existing pinned requirements were preserved.
4. Browser rendering, interactive upload workflow, mobile layout and accessibility require
   manual browser/phone checks; HTTP/static availability alone does not prove them.
5. No verified public deployment exists. Deploy, test endpoint behavior and numeric downloads,
   measure actual host latency, then record the real URL.
6. Review at least three actual independent failure cases and ablations, finish the official
   submission template, review/publish Git changes, and rehearse the demonstration.

## Exact local commands

```powershell
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
$env:TRIDENT_HOST='127.0.0.1'
.venv\Scripts\python.exe -m backend.serve
```

Open http://127.0.0.1:8000. Follow [README](../README.md) for optional exact-hash learned
serving and [deployment guide](deployment.md) for Docker. No new training, test-set access,
benchmark restart or automatic experiment execution is included in these commands.
