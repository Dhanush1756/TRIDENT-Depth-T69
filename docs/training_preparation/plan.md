# Phase 5 preparation and proposed training plan

**Approval status: no training authorized or launched.** There have been zero
optimizer updates. This report separates completed CPU audits and diagnostics
from GPU forecasts and proposed experiments. `audit.json` contains measured
audit evidence; `configs/phase5/` contains proposed runnable configurations.

## Classical baselines actually inspected

All 288 saved records were read and their stored prediction arrays reopened.
RMSE was recomputed against the saved ground-truth mask; values match to 1e-12.
Configurations use full 480x640 images, area-mean dense degradation, 10 m metric
range, RGB /255, and no reference resizing or Eigen crop. Synthetic completion
uses Bernoulli 5% sampling, 0.01 m Gaussian noise and a central 10% area hole.
See the original per-image reports in `../nyu/` for every metric and timing.

| Protocol | Scale | Lowest macro mean RMSE | RMSE (m) | Highest macro boundary F1 | F1 |
| --- | --- | --- | ---: | --- | ---: |
| Dense SR | 2x | Bicubic | 0.035024 | Nearest | 0.938714 |
| Dense SR | 4x | Bicubic | 0.062137 | Nearest | 0.751768 |
| Dense SR | 8x | Bicubic | 0.102029 | Guided | 0.589286 |
| Synthetic completion | 2x conditioning | Guided | 0.213052 | Classical | 0.709483 |
| Synthetic completion | 4x conditioning | Guided | 0.213052 | Classical | 0.709483 |
| Synthetic completion | 8x conditioning | Guided | 0.213052 | Classical | 0.709483 |

There is no universally strongest method. Nearest/bilinear/bicubic remain
zero-filled in same-resolution sparse completion, so they are weak completion
controls rather than competitive hole fillers. They must not be the only
comparison for a learned completion model. Guided, joint bilateral and classical
completion are the meaningful stronger controls. Identical sparse results at
three scales reflect identical inputs and scale-independent classical methods;
they are not evidence of three spatial upsampling capabilities.

The generated prediction-review sheets show one fixed scene at dense 8x and
sparse 4x conditioning, all six saved methods and their actual per-image RMSE.
Visual inspection confirms nearest blocking in dense SR, invalid-pixel retention
by interpolation-only completion and filled but imperfect chair/occlusion
boundaries after filtering. This is illustrative inspection, not a new failure
study or statistical proof. No classical parameters were changed after scoring.

## Architecture and initialization

Both networks consume eight target-resolution channels: RGB (3), initial depth
in metres (1), observed support (1), scale/8 (1), sparse-condition flag (1), and
RGB-presence flag (1). Dense support is the nearest-expanded LR validity;
sparse support remains the original observed mask, never the filled mask.

- **LightweightCNN: 11,873 parameters.** Three 3x3 convolutions, 8→32→32→1,
  ReLU hidden activations. Predict a residual added to the initial depth.
- **ResidualCNN: 473,059 parameters.** Two-convolution blocks, encoder
  32/64/128, decoder 64/32 with concatenated skips, residual and Laplace log-scale
  heads. A separate optional boundary head exists but is not supervised by the
  current losses and is not used by routing; it must not be claimed as a trained
  boundary-confidence estimator.
- **Proposed deterministic GPU variant:** nearest decoder upsampling followed
  by convolutions. Existing model construction and older checkpoints retain
  bilinear defaults. New checkpoint metadata records the decoder configuration.
  This change is explicit because deterministic CUDA backward through bilinear
  interpolation is unsupported in the documented PyTorch behavior.
- **Initial estimates:** bicubic for dense SR and guided for completion, chosen
  from the current development baseline evidence. Both CNNs use the same
  initializer within a condition. Training and checkpoint-based inference now
  resolve the same initializer, avoiding an initialization domain shift.

The baseline is not DKN/FDKN. Hybrid TRIDENT reuses the custom residual model
and combines its estimate with classical and geometric candidates. Current
routing is heuristic, not a separately trained routing network. Different
candidate depths are kept distinct from the network's initial estimate.

## Dataset and test-set isolation

The audited development manifest SHA-256 is
`d1cdbcf0ff3ee96f2e0c173c3c4701ed3f5c526ab517ce3b988268c3c6a391f5`.

| Use | Frames | Scenes | Official membership |
| --- | ---: | ---: | --- |
| Training | 672 | 219 | Official train |
| Validation / model selection | 123 | 30 | Official train |
| Locked test | 654 | 215 | Official test |

The original official manifest still has 795 train / 654 test frames. The
development partition uses whole scenes; train/validation/test scene sets are
disjoint. The runner checks official membership and the audited manifest hash.
The Torch training adapter rejects a `test` split and the runner only constructs
train and validation loaders. A unit test rejects relabelled official test data.

The converter inspected test pixels solely for data-integrity checks. No
official NYU test accuracy was used for classical selection, training,
hyperparameter tuning or checkpoint selection. Keep the final test locked until
the experiment protocol and validation-based choices are frozen. The original
eight-scene subset helps inspect methods; select checkpoints on all 123
validation frames, not just those eight scenes.

## Units, masks, losses and stability

RGB alone is scaled from uint8 to [0,1]. Depth stays float32 metres with fixed
0<depth<=10 m validity; there is no per-image min/max normalization. The reference
is NYU's filled depth, not independently measured dense sensor ground truth.
Raw sensor masks are preserved separately. Invalid zero values never contribute
to supervised averages; NaNs outside the reference mask are excluded.

Proposed total loss coefficients are depth L1 1.0, gradient 0.2, edge-weighted L1
0.1, uncertainty 0.05 and measurement 0.2. The uncertainty term contributes zero
for the CNN baseline without that head. All weights are exploratory.

- Gradients use valid pairs, with separate horizontal/vertical masked means.
- Edge weighting comes from reference-depth jumps at 0.05 m, not RGB texture.
- Uncertainty is Laplace log **scale** s, with NLL `exp(-s)*abs(error)+s`, clipped
  to [-6,6]. It is not log variance. NLL can be negative; that is not itself an
  error and is why checkpoint selection uses RMSE rather than composite loss.
- Dense consistency area-pools the HR prediction by the exact degradation scale
  before comparing with LR observations. It does not compare arbitrary HR
  pixels with LR measurements.
- Completion consistency uses observed pixels only. For noisy observations the
  0.2 coefficient is softened by 0.25, giving an effective 0.05 coefficient.
- Empty masks produce a finite zero supervised term. Nonfinite predictions,
  invalid values under an active reference mask, malformed uncertainty tensors
  and nonfinite/negative weights fail explicitly. Finite loss and gradient-norm
  checks remain enabled; gradient clipping is 5.0.

Twelve forward/backward probes on actual NYU diagnostic crops covered two
models × two protocols × three scales, including measurement loss. All losses
and computed gradients were finite and every parameter stayed bit-identical.
These were 128x160 diagnostic crops, not training, GPU profiles or evaluation
results. Additional unit tests cover empty masks, masked NaNs and extreme
log-scale inputs. Production training retains the full 480x640 target size.

## Proposed hyperparameters and schedule

| Setting | Proposal |
| --- | --- |
| Optimizer | Adam, betas (0.9,0.999), epsilon 1e-8 |
| Learning rate | Constant 1e-4 initially |
| Weight decay | 0 for the controlled first comparison |
| Precision / batch / workers | FP32 / batch 1 / workers 0 |
| Seed | 69 for first controlled series |
| Reference and input geometry | Full 480x640; no random crop or resize |
| Degradations | Fixed per sample/protocol/seed for comparable inputs |
| Validation | All 123 frames after each epoch |
| Selection | Lowest global valid-pixel validation RMSE; earliest epoch wins exact ties |
| Main budget | At most 20 epochs per model/condition, subject to pilot review |
| Scheduler / automatic early stopping | None in the first controlled run; best checkpoint recorded separately |

**Proposed first run, requiring explicit approval:** two epochs of the baseline
CNN on dense SR 4x, full resolution, bicubic initializer, batch 1 and seed 69.
The config is `configs/phase5/pilot_dense4_lightweight.json`. It is not launched.
At batch 1 it schedules 1,344 optimizer updates and 246 validation sample
evaluations. Measure GPU peaks, epoch duration, every loss component and
validation RMSE/MAE before approving a longer run.

After pilot review: train the baseline/custom dense 4x pair; then dense 2x and
8x; then completion 4x. Completion 2x/8x are explicit conditioning-control runs
and can be deferred if budget is limited. Twelve main configs cover both models
and all six protocol/scale conditions, plus six separate pipeline configs. The
pilot is a thirteenth experiment config. Each output directory is unique.

Hyperparameter tuning is restricted to validation. Any learning-rate/loss/fusion
search should be small, declared and logged; do not tune repeatedly against the
final test. Repeat final selected conditions with seeds 70 and 71 when compute
permits; the current prepared matrix uses seed 69 only.

## GPU memory and time estimates

No GPU runtime is connected, so **no GPU peak or training duration has been
measured**. Saved-tensor hooks on CPU at 128x160 estimate retained activation
storage at 480x640, batch 1: about 85.5 MiB for the baseline and 439.5 MiB for the
custom network. These omit gradients, CUDA context, allocator reservations,
temporary convolution workspaces and loss intermediates. Model+gradient+Adam
moment storage is only about 0.18 MiB / 7.22 MiB; activations dominate.

| Full-resolution FP32 budget | Baseline CNN | Custom residual network |
| --- | --- | --- |
| Batch 1 planning allowance | 1–2 GiB | 2–4 GiB |
| Batch 2 planning allowance | 2–4 GiB | 4–8 GiB |

These are engineering allowances, not allocation measurements or guarantees.
Use an available GPU runtime with comfortable headroom, ideally 12–16 GiB, and
verify the batch-1 peak before scaling batch size. Start FP32; mixed precision
requires separate finite-gradient and metric checks and is not enabled here.

At batch 1, estimate epoch time with
`672*(input_preparation + GPU_training_step) + 123*(input_preparation + GPU_validation_step)`.
The earlier CPU benchmark measured dense bicubic initialization around 0.013–
0.018 s/frame and completion guided initialization around 0.65–0.77 s/frame
(tracemalloc overhead present; notebook CPU may differ).

| Assumed GPU train / validation seconds per sample | Dense epoch estimate | Completion epoch estimate |
| --- | --- | --- |
| 0.1 / 0.05 | ~1.4 min | ~10.5 min |
| 0.5 / 0.2 | ~6.2 min | ~15.3 min |
| 1.0 / 0.3 | ~12 min | ~21.1 min |

These are conditional scheduling scenarios using 0.013 s dense / 0.7 s sparse
CPU preparation, not predictions for a named GPU. Allow extra time for data I/O,
checkpoint writes, runtime contention and deterministic kernels. Twenty epochs
would be roughly 0.5–4 hours dense or 3.5–7 hours completion under these scenarios.
CPU guidance may dominate completion; a validated preprocessing cache could
reduce repeated overhead, but no cache is silently substituted in the first run.
Calibrate the schedule from the approved two-epoch pilot.

## Checkpoints, resuming and reproducibility

Checkpoints save model architecture and weights, Adam state, epoch number, best
RMSE/epoch, complete experiment/pipeline settings, dataset manifest hash and
Python/NumPy/Torch/CUDA/DataLoader RNG states. Tensor/state-only loading remains
enabled. Files are written through temporary files and renamed after success.
Existing checkpoints are never overwritten.

Resume is supported at completed epoch boundaries, not in the middle of an
epoch. It restores the optimizer moments, model tensors and next shuffle order;
the CPU state round-trip test uses synthetic untrained state and performs no
optimizer update. Resume rejects mismatched model/decoder/manifest/configuration,
checks PyTorch version and requires history to end at the checkpoint epoch.
Only the total epoch budget may increase. Interrupted incomplete epochs are
replayed from the last completed checkpoint; dangling files/history inconsistencies
fail explicitly rather than silently truncate experiment evidence.

The runner records validation RMSE/MAE, loss components, epoch seconds and CUDA
peak allocated/reserved bytes, with a separate best-checkpoint pointer. Validation
RMSE pools squared errors over valid reference pixels. It is not the arithmetic
mean of per-image RMSE used in the eight-scene baseline summary; use the benchmark
harness's common aggregation for final method comparisons.

Seeds, deterministic algorithms, cuDNN flags and CUBLAS workspace settings are
explicit. Bitwise reproducibility across hardware, PyTorch releases or CPU/GPU
is not promised. GPU checkpoint/resume and exact repeatability still require
verification on the selected runtime. PyTorch references:
[deterministic operations](https://docs.pytorch.org/docs/2.14/generated/torch.use_deterministic_algorithms.html),
[reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html).

## Experiment comparison and technical risks

Compare all six classical methods, the CNN baseline, direct custom residual
output and hybrid TRIDENT on identical inputs. Reuse each trained residual
checkpoint for direct versus hybrid comparison. Record initialization and
measurement-consistency postprocessing so a gain is not misattributed to the
network. Core data losses are shared; uncertainty adds a custom-model term and
must be ablated before claiming its contribution.

For each protocol/scale, report RMSE, MAE, PSNR, SSIM, boundary precision/recall/F1,
defined edge preservation, repeated runtime and clearly scoped memory. After
validation choices are frozen, run the official test once per finalized method
and report separately. KITTI generalization remains a later independent study.

Run inference ablations for confidence, geometry, residual, uncertainty, adaptive
fusion and measurement consistency. Loss/head training ablations need separately
trained variants; inference toggles alone cannot establish their training benefit.

Primary risks: only 672 labelled training frames; overfitting; filled-reference
bias; RGB texture copying; thin structures and large holes; uncertainty scale
miscalibration; positive-output clamping dead zones; no supervised boundary-head
loss; CPU preprocessing bottlenecks; CUDA determinism/workspace constraints;
different metric aggregation conventions; and inconsistent training/inference
initializers (now explicitly recorded and matched). None is hidden by a claimed
trained-model improvement. Existing default constructors and old inference
checkpoints remain compatible, and the existing tests are retained.

## Commands after review

Review `audit.json`, the prediction sheets, this plan and `configs/phase5/`.
Training requires a GPU-enabled PyTorch environment with mounted data; this local
installation is CPU-only. **Do not run the following until training is approved:**

```text
python -m trident_depth.training.train --experiment configs/phase5/pilot_dense4_lightweight.json --execute-training
```

After approval and a completed checkpoint, epoch-boundary resume uses the same
experiment/run directory and `--resume path/to/epoch_002.pt`, retaining the
explicit `--execute-training` flag. No such command has been executed.

## Final preparation verification

The complete suite passes **121 tests**, retaining the prior 112 and adding nine
training-preparation checks. These include optimizer/RNG/shuffle state restore,
decoder checkpoint round-trip, initializer parity, masking, numerical stress,
validation units and official-test rejection. The known upstream Starlette
test-client deprecation warning remains; there are no test failures.
`pip check` reports no broken requirements and `git diff --check` passes.
All 19 generated JSON files resolve correctly: six pipeline definitions,
twelve main model experiments and one two-epoch pilot. All 13 experiment output
directories are distinct. No optimizer step or training launch was performed.
