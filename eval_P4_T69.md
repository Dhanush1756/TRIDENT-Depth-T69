1RUA24CSE0125,1RUA24CSE0129,1RUA24CSE0140,1RUA24CSE0178

# P4 — TRIDENT-Depth 2.0 — Team T69

Final NYU/KITTI experiments: **Not run**. No trained checkpoints or public
deployment exist. Classical Redwood sensor-reference smoke checks validate I/O
and harness execution only; they are not hackathon benchmark results.

Metrics are ordered as in the official P4 PDF. Definitions, units, masks,
thresholds, timing and memory limits are in `docs/evaluation_protocol.md`.

| Evaluation item | Interpolation | Classical | Deep learning | Hybrid |
| --- | --- | --- | --- | --- |
| RMSE and MAE (m) | Not run | Not run | Not run | Not run |
| PSNR (dB) and SSIM | Not run | Not run | Not run | Not run |
| Edge preservation and boundary accuracy | Not run | Not run | Not run | Not run |
| Performance at 2× / 4× / 8× | Not run | Not run | Not run | Not run |
| Processing latency and memory consumption | Not run | Not run | Not run | Not run |

Maintain separate tables for Protocol A dense SR and Protocol B sparse/noisy
completion, each dataset, each scale and each named method when experiments run.
Record official split, manifest hash, checkpoint provenance and machine details.
NYU-to-KITTI zero-shot results must be separate from any KITTI adaptation.

Ablation study: **Not run**. Inference toggle tooling is implemented;
training ablations require separately trained variants.

Failure-case study: **Not run**. Final submission requires at least three
distinct real examples, saved outputs and explanations. One smoke scene at three
scales does not meet this requirement.

GitHub origin: https://github.com/Dhanush1756/TRIDENT-Depth-T69 (local changes
have not been published or the remote contents verified).

Live application URL: **Not deployed**.

Architecture: `docs/architecture.md`. Design: `docs/design_document.md`.
Requirement-by-requirement audit: `docs/pdf_requirements_audit.md`.

## Real NYU dataset preparation

The official labelled NYU MAT/HDF5 release is downloaded and converted.
The official partition remains 795 training / 654 test frames across 249 / 215
disjoint scenes. A separate training-derived development partition has 672
training / 123 validation / 654 untouched test frames.

All 1,449 samples and 7,245 exported files passed integrity checks. RGB and
valid float32 depth values were preserved exactly. Filled-reference invalid
fraction is 0%; raw sensor depth missing fraction is 31.4338%. Both degradation
protocols were verified on actual NYU validation frames at all three scales.
See `docs/nyu_preparation.md` and the dataset integrity reports. These checks do
not establish trained-model quality or full test-set performance.

## Measured preliminary NYU validation results

Actual six-method results are available in
[NYU validation tables](docs/nyu/validation_results.md),
[CSV summary](docs/nyu/baseline_summary.csv), and the per-image JSON reports
alongside them. These are 288 records: eight frozen, scene-diverse development
validation frames × three scales × six methods × two separate protocols.
Every saved prediction was reopened; independently recomputed RMSE and MAE
matched its report exactly. Official test samples were excluded from this run.

Dense SR uses full 480×640 references with area-mean degradation. Sparse/noisy
completion uses seeded synthetic sampling, noise and holes on actual NYU
references and retains 480×640 resolution. Metric order, definitions, masks,
normalization, thresholds and timing/memory scope are documented with the tables.
Each protocol's source hashes, manifest hashes, configuration and package
versions are saved with its measured results.

The current classical refinement does not uniformly outperform interpolation:
bicubic has lower average dense-SR RMSE at all scales on this subset, while
classical refinement has higher boundary F1 than bicubic at 4× and 8×. No
neural training, trained-model comparison or final held-out test evaluation has
been performed. Final-result rows above deliberately remain `Not run`.
