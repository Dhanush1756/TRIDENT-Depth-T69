# Final submission readiness - review required

The project is not yet fully submission-ready: engineering and live inference are verified,
but publication privacy, image rights and first Actions deployment approval remain open.

| Requirement | Status | Evidence / remaining work |
| --- | --- | --- |
| PS-04 / Team T69 and four USNs | Completed | README and eval_P4_T69.md |
| Correct evaluation filename | Completed | eval_P4_T69.md at repository root |
| Architecture/design | Completed | DESIGN.md, docs/architecture.md, docs/extended_v1.md |
| Classical / learned / hybrid implementations | Completed | Preserved scientific source; all eight public app methods checked |
| Dense 2x / 4x / 8x | Completed implementation and saved validation metrics | 123 frames each; frozen synthetic area-mean degradations |
| Sparse / noisy / incomplete completion | Completed implementation and saved validation metrics | 123 frames each; same-resolution synthetic completion |
| Combined sparse/noisy/holes | Partially completed evaluation | 116/123; seven records missing; independent benchmark total 854/861 |
| Comparative evaluation | Completed for saved records | 57 task/method aggregates; B0 dense4 only; other protocols N/A |
| RMSE/MAE/PSNR/SSIM/boundary/timing/memory | Completed for saved records | Pooled and per-frame definitions in partial independent report |
| Loss ablations / failure analysis | Partially completed | Verified A1/B0 evidence; regional failure statistics; independent panels and complete hybrid component ablations remain unverified |
| Scene-safe 672/123 development split | Completed metadata verification | Original split/reference and saved hashes preserved; no official test samples accessed |
| Exact trained checkpoint | Completed | Approved epoch-12 SHA verified; private Modal storage read-only |
| Public app | Completed | Existing Modal URL, public HTTP/browser checks; synthetic only |
| Full test suite / dependencies | Completed | 289 tests pass, one existing warning; pip check passes |
| Docker | Not verified | CLI installed; daemon unavailable; no successful build claimed |
| Mobile / accessibility | Partially verified | 1440/390/360 browser emulation, focus and accessible names; no physical-device/full-WCAG claim |
| Public GitHub source | Pending publication approval | Existing repository is public, but current source is not pushed |
| Credential audit | Signature scan passed | 260 blobs in five locally reachable commits; no detected credential signatures; not proof of all-secret absence |
| Historical machine paths | Blocking publication privacy issue | Nine historical blobs, eight paths present in main; redacting a new commit cannot erase old history |
| Dataset-derived image publication rights | Pending owner review | Four image paths listed below; no licence invented |
| GitHub-to-Modal automation | Prepared, not verified | YAML/preflight checked; no real Actions run; owner secrets/variable/environment and approval required |
| Official design template and viva | Not verified | Organizer template unavailable; three-minute rehearsal and physical demo remain owner tasks |
| External dataset / real LiDAR / official test evaluation | Not verified, not claimed | No such samples used; synthetic NYU protocols clearly labelled |

## Privacy and publication gates

Current main is 633b55b95c636f6b39ec413cd847b3c3dbd7c8a7; it is also the existing remote
main head observed during review. Nothing has been staged, committed or pushed. No history
was rewritten. Eight path-bearing records already exist in main; a ninth was found in other
locally reachable history. The owner must resolve the strict no-private-history publication
requirement before publication. A normal new commit cannot remove older sensitive paths.
No secret values were found or printed by the scan. Do not infer that historical paths are
credentials, or that credential-signature scanning is a complete security audit.

Original records remain private and unchanged; the clean export uses separately sanitized
copies. Review docs/public_evidence/manifest.json and PUBLIC_RELEASE_FILE_LIST.md. Do not
blindly stage the original records or publish the current Git history as a clean export.
No new repository, force-push, history deletion or rewrite is authorized.

Image publication decisions required:

- docs/nyu/registration_1.png
- docs/nyu/registration_2.png
- docs/training_preparation/prediction_review_dense_sr.png
- docs/training_preparation/prediction_review_sparse_completion.png

These are dataset-derived evidence, not invented marketing images. The audit preserves them;
their inclusion in a review export is not a licence/privacy approval to publish them. No project
LICENSE is provided; ask the owner to choose an appropriate licence rather than inventing one.

## Exact submission entry points

- eval_P4_T69.md - reviewed comparison, units, aggregation and limitations.
- README.md - project, team, setup, supported modes, split, training and live links.
- DESIGN.md and docs/architecture.md - implementation and diagram.
- docs/partial_independent_recovery_20261107/report.md, aggregates.csv, aggregation.json - partial saved-record evidence.
- configs/extended_v1/ and notebooks/colab_extended_v1_epoch012_validation_T69.ipynb - frozen experiment/protocol identity.
- docs/MODAL_DEPLOYMENT.md and docs/GITHUB_ACTIONS_SETUP.md - hosting and pending automation.
- .github/workflows/modal-deploy.yml - disabled-by-default production workflow.

Keep original datasets, weights, raw logs/JSONL, archives, local review storage, credentials
and unreviewed machine paths out of publication. Preserve originals privately for provenance.

## Approval sequence

1. Resolve historical-path privacy and image/licence decisions; approve the exact public list.
2. Only then stage reviewed files, create `Finalize TRIDENT-Depth T69 submission and deployment`,
   and push normally to the existing main branch. Verify the remote commit. No force-push.
3. Owner configures two Modal secrets, production environment protection and Actions opt-in.
4. Obtain separate first-production-run approval, then enable and dispatch the workflow.
5. Only after a real successful run record its commit/run/artifacts and final live checks.

Published new commit SHA: **none**. Actions run URL: **none**. Automatic deployment verified:
**no**. Existing public application remains verified and has not been redeployed in this phase.
