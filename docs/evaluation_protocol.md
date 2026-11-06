# Evaluation definitions

Report dense SR and sparse completion separately, with separate rows at each
scale and explicit source/split/degradation configuration. Ground-truth validity
is the evaluation mask. Zero/missing predictions at valid reference pixels
count as errors; prediction masks never remove hard pixels from scoring.
Predictions must be finite at valid reference pixels. Empty reference masks
raise an error rather than produce a fabricated zero score.

Metrics are implemented on float64 arrays:

1. RMSE: square root of masked mean squared error, metres.
2. MAE: masked mean absolute error, metres.
3. PSNR: `10*log10(R**2/MSE)`, where fixed metric depth range `R` is configured,
   not the per-image depth extent. Exact match returns JSON null with
   `psnr_infinite=true` (mathematical +infinity).
4. SSIM: standard luminance/contrast covariance formula, uniform 7×7 windows,
   population variances, C1=(0.01R)^2 and C2=(0.03R)^2. Only fully valid windows
   wholly inside the image contribute. No per-image normalization is used;
   using metric depths with metric R is equivalent to scaling both by R.
   No eligible windows returns null. This convention must not be mixed with
   Gaussian-window SSIM or sample-covariance variants without disclosure.
5. Boundary precision/recall/F1: mark both pixels of valid horizontal/vertical
   pairs whose absolute depth jump is >=0.05 m. Match predicted/reference edges
   within Euclidean distance <=1 target pixel. Precision and recall use their
   respective edge-map counts. Both empty yields 1; one empty yields F1=0.
   All thresholds are returned in each metrics record.
6. Edge preservation: average over valid horizontal/vertical neighbour pairs of
   `1 - abs(g_pred-g_gt)/(abs(g_pred)+abs(g_gt)+1e-12)`, with signed first
   differences. Equal flat gradients contribute 1, opposing gradients near 0.
   This is the project's explicit gradient similarity measure, not a claim of a
   standardized benchmark metric. No valid neighbour pair yields null.
7. Latency: one untimed warmup followed by three timed, batch-size-one CPU
   reconstructions, using perf_counter. Store all values and report median ms.
   Loading, degradation, metrics and serialization are excluded. Python traced
   allocations are enabled during timing, so this harness has tracing overhead.
8. Memory: tracemalloc allocation peak during timed calls, plus OS lifetime peak
   RSS when supported. Neither is an isolated per-inference native/GPU memory
   peak. The lifetime RSS includes earlier work in this benchmark process;
   use separate fresh processes for controlled native-memory comparisons.

The benchmark runs checkpoint models on CPU by default, so there is no CUDA
asynchrony in its measured path. A future GPU benchmark must synchronize before
and after timing and record peak allocated/reserved GPU memory separately.

Training/validation logs contain mean **batch losses**, not final masked test
metrics. Choose checkpoints on validation data; hold out test data until the
protocol is fixed. Comparison is per-sample JSON, avoiding an undocumented
mixture of per-image averaging and global pixel weighting.

`--checkpoint` enables deep/hybrid results only with a checkpoint bearing trained
provenance. `--ablations` adds six hybrid variants, each disabling one inference
component using identical inputs. These are inference ablations. Training losses
and uncertainty heads require separately retrained variants for a complete
training ablation study; inference toggles alone do not establish their training
contribution. The checkpoint's scale/protocol provenance must match the intended
comparison; do not claim a 4×-only checkpoint was trained at 2× and 8×.

Ranked diagnostic panels are candidates for human failure analysis. Repeated
scales of one scene are not independent failure cases. Final submission requires
at least three actual distinct examples with explanations and all model outputs.
