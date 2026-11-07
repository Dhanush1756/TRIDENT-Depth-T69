# hybrid_v1_dense4: implementation and validation plan

Status: implementation and CPU tests only. **The 123-frame benchmark has not been run and still requires explicit approval.** No accuracy gain is claimed. No training, fine-tuning, optimizer steps, official test payload access, checkpoint export or changes to historical files are part of this phase.

## Scope and additive files

The new engine is `trident_depth.hybrid_v1_dense4.HybridV1Dense4`. It supports dense depth SR at exactly 4x, registered uint8 target-resolution RGB and metric LR observations. It is a separate opt-in path: legacy `InferenceEngine`, fusion, API, UI, models, checkpoints, original configurations, manifests and verification reports are unchanged.

| New file | Responsibility |
| --- | --- |
| `trident_depth/hybrid_v1_dense4.py` | Frozen-checkpoint loader, heuristic reliability, candidate fusion, bounded area consistency, separate inference engine |
| `trident_depth/evaluation/hybrid_v1_data.py` | Fixed validation identity, payload allowlist, original-loader bridge using a new validation-only manifest |
| `trident_depth/evaluation/hybrid_v1_dense4.py` | Gated 123-frame evaluation, metrics, timing, memory, provenance, reports and optional predictions |
| `configs/hybrid_v1_dense4.json` | Versioned, explicit fusion constants; independent of A1/B0 training configurations |
| `tests/test_hybrid_v1_dense4.py` | Numerical, isolation, compatibility and unchanged-reference tests |
| `docs/hybrid_v1_dense4/implementation.md` | This design, limitations and commands |
| `docs/hybrid_v1_dense4/preparation_verification.json` | Actual test and preservation verification, explicitly not benchmark results |

There are no modifications to existing files. The FastAPI application continues to use the legacy dispatcher; this new path is not advertised through its old health endpoint. App integration is outside this phase. Do not remove the old B0 hybrid guard to reach this implementation.

## Frozen B0 and preprocessing

Only the known B0 epoch-two checkpoint SHA256 is accepted:

```text
a6416c47a16ec78e97c8b734ae7fb23cba4dbbeb5f5564a3e02d61a4a159c1b2
```

The loader accepts a standalone checkpoint or the supplied original ZIP. ZIP mode reads exactly `corrective_v1_B0_dense4_seed69/epoch_002.pt` into memory, without extraction. It hashes bytes before calling the existing strict `load_checkpoint`, verifies architecture/configuration/provenance, and uses eval/inference mode with FP32 and frozen parameters. No weights are reset after loading. Parameter count remains 473,059.

Only `model(...)["depth"]` is consumed. The model's unused boundary head remains in the original architecture but is not used as confidence. B0's uncertainty head remains disabled. No A1 output, GT-derived confidence or learned confidence substitute is used.

For real validation, the original `TorchDepthDataset` supplies RGB in [0,1], metric depth, bicubic initialization, nearest observed support and the unchanged dense area degradation. The engine accepts those exact tensors; the raw B0 prediction and bicubic reference are returned unchanged for comparison. Fusion uses separate bounded copies. Bare-array inference reuses the same original classical interpolation functions.

All-invalid input, nonfinite valid measurements, invalid mask/RGB shapes or dtypes, out-of-range valid measurements, unsupported scales/protocols and absent RGB fail explicitly. Partially valid LR input has a defined nearest-fill fallback for unsupported initialization regions, then preserves valid LR means. Its result is flagged `partial_lr_input_unvalidated`: this is numerical robustness, not evidence of a trained sparse/noisy completion capability. The approved benchmark requires fully valid dense LR data.

## Candidates and heuristic reliability

Let O be the LR measurement map and V its boolean validity. Let U repeat LR pixels into their corresponding 4x4 HR blocks; A computes a nonoverlapping 4x4 area mean.

1. **N:** the original B0 prediction, bounded to [1e-6,10] m for fusion only.
2. **C:** the existing reconstruction pipeline with joint bilateral filtering by default: bilinear initialization, valid-only median denoising, hole filling and RGB-guided joint bilateral filtering. Existing guided filtering is the configurable alternative. This is a complete existing pipeline, not an independently reimplemented filter.
3. **G:** the existing `boundary_refine(N, all_valid, RGB)` result. Its gate is:

```text
h = rgb_edges(RGB) * max_filter_3(depth_edges(N, threshold=0.05 m))
G = (1-h)*N + h*guided_filter(N, RGB)
```

Only RGB edges near a predicted depth discontinuity can change G. An RGB texture edge alone cannot change a constant depth field through this geometry branch. This gate is heuristic and does not guarantee texture-artifact rejection on all real scenes.

Define, for j in {N,C,G}:

```text
S   = uniform_filter_5(U(V))
M_j = U(V * exp(-abs(A(D_j)-O) / tau_measurement))
L_j = exp(-abs(D_j-median_filter_3(D_j)) / tau_local)
A_C = exp(-abs(C-N) / tau_agreement)
A_G = exp(-abs(G-N) / tau_agreement)

r_N = p_N * (epsilon + S*M_N*L_N)
r_C = p_C * (epsilon + S*M_C*L_C) * (a_floor+(1-a_floor)*A_C)
r_G = p_G * (epsilon + S*M_G*L_G) * h * (a_floor+(1-a_floor)*A_G)

w_j = r_j / sum_active(r)
D_mix = sum_active(w_j*D_j)
```

Invalid LR drift terms are zeroed rather than treated as observed zero depth. The positive neural score guarantees a nonzero denominator. Scores are finite, nonnegative **heuristic reliability**, not calibrated probabilities or learned uncertainty. Local median residuals and LR drift are imperfect reliability proxies, especially at real boundaries. Candidate agreement can reinforce a shared error.

Defaults fixed before benchmarking: priors (N,C,G)=(0.6,0.2,0.2); measurement tau=0.02 m; local/agreement tau=0.05 m; agreement floor=0.05; epsilon=1e-6; geometry threshold=0.05 m. These values are untuned engineering controls, not proven optimal settings. The JSON records every exposed constant. Internal score/projection arithmetic uses float64; delivered depth and weight maps are float32.

## Reliable measurement preservation

Dense SR measurements are **area means**, not HR point samples. Each valid LR pixel constrains the mean of its corresponding 16 HR pixels. Do not overwrite every HR pixel with nearest-upsampled LR depth.

After blending, solve independently in each observed block:

```text
minimize sum_i (x_i-y_i)^2
subject to mean(x_i)=O_block and lower<=x_i<=upper

x_i = clip(y_i+lambda_block, lower, upper)
```

The scalar lambda is solved by 48 bounded bisection iterations. Bounds round inward to FP32-representable values before final casting. At the decimal floor, FP32 rounding can cause a sub-ULP difference from the stored LR target; preservation is numerical, not exact real-number arithmetic. Tests and the evaluator require maximum LR area error <=2e-6 m. Invalid LR blocks are not projected to fabricated observations.

This projection can shift values outside the pre-projection candidate convex hull, while retaining physical bounds. It can create block artifacts or harm boundary quality; improvement must be measured. Raw B0 is not projected or clipped for its reference metric.

## Comparisons and interpretation

The evaluation records the five requested methods plus a control:

| Name | Definition |
| --- | --- |
| `bicubic` | Original dataset bicubic initializer, unchanged |
| `b0` | Original frozen B0 prediction, unchanged |
| `b0_plus_classical` | Normalize scores for N,C; blend; apply bounds/area consistency |
| `b0_plus_geometry` | Normalize scores for N,G; blend; apply bounds/area consistency |
| `hybrid_v1_dense4` | Normalize scores for N,C,G; blend; apply bounds/area consistency |
| `b0_consistency_only` | B0 with the same range bounds and area projection, without filtering/fusion |

All component blends share measurement consistency, so improvements versus raw B0 cannot automatically be attributed to filtering. The extra control exposes the combined range/consistency contribution; range clipping fraction/magnitude and projection corrections are logged separately. These are candidate-component comparisons, not a comprehensive factor-isolated ablation study.

## Validation isolation and aggregation

The workflow accepts no split selector and requires the original manifest SHA256 and original 123-frame reference SHA256. It verifies IDs/order, 30 scene assignments and scene separation, official-train membership, RGB/depth/mask file hashes, original GT mask hashes, valid counts and full 480x640 / 120x160 dimensions.

The shared manifest is read for split/scene metadata only. No non-validation sample paths are resolved, statted, opened, hashed or decoded. A Python audit hook blocks non-validation dataset opens and dataset enumeration while the evaluator runs. It is a scoped guard over the inspected NumPy/Pillow loader, not an OS-wide trace. A new validation-only manifest with absolute validation paths is written inside the new run directory; the original manifest is never changed. The existing dataset loader then receives only those validation rows. Existing all-split file audits and generic benchmark entry points are not called.

Use the existing `validation_errors` and `evaluate` implementations:

- Global RMSE: sqrt(total squared metric error / total valid pixels).
- Global MAE: total absolute metric error / total valid pixels.
- Global PSNR: 10*log10(10^2/global MSE); zero-error PSNR is JSON null with an explicit infinity flag.
- SSIM: mean of eligible frame scores, original 7x7 valid-window definition; eligibility counts saved.
- Boundary precision, recall and F1: separate arithmetic frame means, at 0.05 m / one-pixel tolerance. Mean F1 is not recomputed from mean precision/recall.
- Mean finite per-frame PSNR and edge-preservation scores are also saved separately with counts; global PSNR and frame-mean PSNR are never silently interchanged.

Unchanged B0 and bicubic must replay their recorded global RMSE/MAE within 1e-6 m; B0 boundary F1 tolerance is 1e-5. Failure produces `INVALID_BASELINE_REPLAY` and a failure record, not a successful result claim. No automatic parameter tuning follows.

## Timing, memory and artifacts

The production engine computes only the complete hybrid output, plus retained B0/bicubic references. Component comparison blends run afterward, outside the full-pipeline timer. The evaluator verifies that reconstructing the comparison hybrid is bit-identical to the timed production result.

Record warmup/repetition counts, per-frame timing samples and medians for input tensor bridging/H2D, neural forward, prediction D2H, classical refinement, geometry refinement, heuristic reliability, fusion/projection and total standalone hybrid pipeline. CUDA is synchronized around the neural stage. Dataset loading/original preprocessing, component-comparison blends, metrics and saving are reported separately or excluded explicitly. These are not independently timed per-method baseline latencies. No tracemalloc runs during latency measurements.

CUDA allocated/reserved peaks are recorded after warmup through pipeline/comparison processing. On CPU, the existing process-lifetime peak RSS helper is used; this is not isolated per-method native memory. Runtime, hardware, package versions and precision flags are saved. A CPU run may replay B0 within tolerance, but its timings cannot be compared directly with historical Colab GPU timings.

A new output directory contains `config.json`, `provenance.json`, `validation_only_manifest.json`, `report.json`, `per_frame.json`, `report.md` and `access_audit.json`. Optional NPZ files store all six depth maps, original references/masks/RGB, inputs and fusion weights. File creation is exclusive. The output must use a separate `hybrid_v1_dense4` namespace; existing directories are rejected. Partial failures retain their new directory and write `failure.json`; reruns need a different directory.

## Commands

Focused tests, from repository-root PowerShell (no benchmark):

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:TRIDENT_B0_AUDIT_ZIP = '${USER_HOME}\Downloads\corrective_v1_B0_dense4_seed69-20261106T163951Z-1-001.zip'
.\.venv\Scripts\python.exe -B -m pytest tests/test_hybrid_v1_dense4.py -q -p no:cacheprovider
```

The environment variable enables the optional genuine-checkpoint test using synthetic inputs only. Without the local archive it skips explicitly; no replacement checkpoint or result is fabricated. The normal portable tests still run.

**After separate approval only**, the exact local CPU validation command is:

```powershell
.\.venv\Scripts\python.exe -B -m trident_depth.evaluation.hybrid_v1_dense4 --execute-validation --manifest data/nyu/converted/manifest_development.json --checkpoint '${USER_HOME}\Downloads\corrective_v1_B0_dense4_seed69-20261106T163951Z-1-001.zip' --config configs/hybrid_v1_dense4.json --reference docs/pilot_dense4/per_frame.json --output artifacts/hybrid_v1_dense4/validation_123_run001 --device cpu --warmup 1 --repeats 3 --save-predictions
```

For Colab, use the same module from a checkout containing these new files, the original B0 checkpoint/ZIP, and the already-extracted dataset. Override manifest/checkpoint/output paths, choose `--device cuda`, and retain Colab's CUDA-enabled PyTorch. Do not run an older source archive that lacks this implementation. No training environment setup or dataset extraction is automatically performed by this workflow.

The command runs exactly the 123 validation frames; there is no test option. Check the output path is new. Optional predictions can occupy substantial disk space; exact storage/runtime/GPU memory have not been measured for this benchmark. Omit `--save-predictions` for JSON/Markdown-only evidence, or reserve sufficient disk space before approving saved maps.

## Remaining gates

The 123-frame benchmark, GPU execution, measured gains/regressions, physical inspection of hybrid failure cases and app/UI integration have not been performed in this phase. The new path is a candidate implementation. No other scale, sparse/noisy learned capability, calibration claim or longer training is introduced. See `preparation_verification.json` for tests actually run and historical-file preservation checks.
