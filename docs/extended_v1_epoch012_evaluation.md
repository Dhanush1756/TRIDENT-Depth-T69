# Epoch-12 validation protocol and preliminary log review

This prepares a new validation-only benchmark; it does not train or change any
existing model, configuration, checkpoint or result. The exact checkpoint is
`epoch_012.pt`, SHA256
`09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd`.
It is on Drive at
`TRIDENT-T69/runs/extended_v1/20261107T034428Z_aa3a61d441/epoch_012.pt`.
It is not available locally: its byte hash/state inspection must execute in
the notebook before the benchmark. No local checkpoint-verification success
is asserted.

## Evidence inspected locally

The supplied Downloads preflight ZIP
`20261107T034428Z_aa3a61d441-20261107T050021Z-1-001.zip` contains the actual
console log, GPU preflight, environment, launch record and package freeze.
The downloaded completed training notebook independently displays epoch 12
and its checkpoint digest. Console SHA256:
`ee444bba51413d23e0261cfa45f893faea6dd92f505b7c5823789ede228d646c`.
All 66 original training source files match the recorded preflight hashes.
The original files are preserved; the new evaluator adds files only.

These are **recorded training-validation results**, not new reproduced results:

| Task | Epoch-12 RMSE m | MAE m | Mean-frame boundary F1 |
| --- | ---: | ---: | ---: |
| dense2 | 0.0304760682 | 0.0072344027 | 0.8957203053 |
| dense4 | 0.0443449216 | 0.0122680887 | 0.8475324496 |
| dense8 | 0.0739585459 | 0.0236561169 | 0.7361620269 |
| sparse | 0.0998506338 | 0.0249085801 | 0.7153524389 |
| noisy | 0.0109445855 | 0.0080953132 | 0.9179959190 |
| incomplete | 0.4087900953 | 0.0979916219 | 0.7631434107 |
| sparse_noisy_holes | 0.4246235982 | 0.1200495270 | 0.5708804633 |

Epoch 12 has the lowest recorded equal-task selection score, 0.1561412069 m.
Incomplete and combined tasks account for 76.25% of the sum of task RMSEs;
this describes the selection score, not a measured training-gradient imbalance.
Their epoch-1 RMSEs were 0.4337610532 and 0.4511557797 m: they improve but
remain poor. Noisy-task RMSE increases from 0.0102520709 to 0.0109445855 m.
Epoch-12 runtime was 356.45 s, with 905,888,768 allocated / 1,151,336,448
reserved GPU peak bytes. Those are training-run measurements, not benchmark
latency or inference memory.

## Checkpoint audit gate

`evaluation_audit.py` hashes before loading through the original extended
checkpoint loader. It requires original `run.json`, `epoch_012_metrics.json`
and the pinned console log. It verifies strict architecture loading, 473,059
parameters, finite model tensors, complete Adam/scheduler/scaler/RNG state,
8,064 Adam steps, twelve history rows identical to the console, best epoch,
per-frame error sums, all 861 validation identities, original source hashes,
frozen training configuration and B0 lineage. Additive evaluation modules
do not invalidate the original training source identity. Missing or divergent
files stop evaluation; no metadata is reconstructed from guesses.

The GPU benchmark additionally requires the original Python/Torch/dependency
versions, CUDA Tesla T4, FP32, deterministic algorithms and TF32 disabled.
Colab CUDA PyTorch is preserved during dependency setup. CPU tests and metadata
inspection remain possible locally; no CPU performance substitute is run.

## Data and comparison contract

Use the existing **validation-only** archive containing 369 sample payloads
for exactly 123 original validation IDs from 30 scenes, plus the unchanged
mixed manifest. No train/test payloads are staged. The existing validation
allowlist checks paths, symlinks, payload hashes, scene separation and masks.
The original `ProtocolDataset`, `load_sample`, degradation and `tensor_sample`
functions are reused at seed 69. This is 861 task/frame pairs, not 861 distinct
images. Completion is target-grid spatial factor 1, not 4x upsampling.

The six classical baselines are nearest/bilinear/bicubic interpolation with
nearest valid hole filling, joint bilateral, guided filtering, and the
existing complete classical pipeline. Hole-filled interpolation is explicit
in method names. On target-grid completion the interpolation methods may be
identical after filling; they are not independent gains.

Compare raw extended depth and the unchanged extended hybrid adapter as
separate methods. Evaluate original B0 **only on dense4**; all six other
tasks report B0 as unsupported, not silently adapted or extrapolated. B0
checkpoint SHA256 remains
`a6416c47a16ec78e97c8b734ae7fb23cba4dbbeb5f5564a3e02d61a4a159c1b2`.

Global RMSE = sqrt(total squared error / total valid GT pixels); global
MAE = total absolute error / total valid GT pixels. Global PSNR uses global
MSE and a fixed 10 m range. Frame PSNR is separately averaged over finite
eligible values. SSIM and boundary precision/recall/F1 are arithmetic
frame means with eligible counts; F1 is not computed from averaged P/R.
The original masks, 0.05 m boundary threshold, 1 px tolerance and 7x7 SSIM
window are unchanged. Predictions never exclude difficult pixels from scoring.

Replayed extended RMSE/MAE must match training within 1e-6 m and SSIM/F1
within 1e-5 for every task. Dense4 B0 uses its prior 1e-6 m / 1e-5 F1
tolerances. Failure yields `INVALID_REPLAY`, retained predictions and reports;
the run is not declared successful or used to tune parameters.

## Timing, memory and saved outputs

The fixed evaluation configuration specifies one warm-up and three timed
repetitions per method/frame. CUDA is synchronized before/after measurements.
Timing covers each method independently, including interpolation, normalization,
H2D, model inference, D2H and required classical postprocessing. Dataset I/O,
degradation, metrics, figures and file writes are excluded. Classical methods
run on CPU; their GPU memory records reflect resident models, not GPU filtering.
Allocated/reserved peaks and incremental allocated peaks are distinguished.
CPU RSS is process-lifetime peak, not isolated method peak. Repeated outputs
must be bitwise equal. No model tensors may change.

All output goes to a new
`verification_results/extended_v1_epoch012/<UTC_UUID>/` directory:
`provenance.json`, `per_frame.jsonl`, `report.json`, `report.md`,
`access_audit.json`, per-task prediction NPZs and labelled PNG panels.
Allow approximately 8 GB free Drive space; actual compressed size depends
on predictions. Failure preserves completed frame records and `failure.json`.
Never rerun into an existing directory.

## Failure analysis without tuning

Global scores alone can hide missing-region failures. For each completion
method, collect observed/missing and central-hole/outside-hole error sums;
distance-from-measurement bins are fixed at [0,1), [1,4), [4,16), [16,64),
[64,128), [128,1000000) pixels. Dense LR block means are never treated as HR
point observations. Aggregate regions by pixels, not equally weighted frames.
Record correction magnitude/sign and fraction of pixels improved/worsened
relative to the identical filled-bicubic initializer. A fixed diagnostic
counts false predicted edges at RGB edges on flat GT depth; it is not a
complete perceptual texture-copying metric.

Panels include five different scenes chosen by manifest order **before
inference**, plus three explicitly labelled worst raw-extended-RMSE frames
per task after scoring. Save all frame scores and predictions, including
regressions. Use fixed depth [0,10] m and absolute-error [0,0.5] m display
ranges with saturation labels. The report quantifies central-hole share of
squared error and comparisons against the initializer, without claiming
causal proof.

Confirmed implementation facts: a central rectangle removes 25% of image
area; nearest valid filling provides the initializer; measurement consistency
has no observations inside that rectangle; each task gets only 96 training
frames/epoch (1,152 exposures over twelve epochs). The network uses local
convolutions, not an explicit global geometric/semantic mechanism. Nearest
filling itself can carry distant measurements into the initializer, so
local feature context must not be described as absolutely no depth information.

Hypotheses: large unsupported surfaces are ambiguous, initializer errors
propagate into the hole, and limited/diversity-poor hole supervision or
local context may constrain reconstruction. The new region/distance/correction
evidence can support or weaken these explanations. High SSIM does not establish
successful hole filling; clean hybrid retention outside holes can improve
global scores without reconstructing the missing interior. No fix is proposed
as effective, no threshold is tuned and no new training is performed here.

## Colab execution

Use `colab_extended_v1_epoch012_validation_T69.ipynb` with a T4 runtime.
It first runs setup/tests, audits the checkpoint and verifies the validation
protocol. `RUN_BENCHMARK=False` by default: inspect the audit before setting
it True and executing the benchmark cell. This is an execution control,
not a training or export flag. No training/export function is included.
Keep the checkpoint in its original Drive run beside the two original JSON
files. The notebook uses a fresh current-source ZIP and the existing
validation-only data pack; the larger development/full NYU ZIP is not used.
