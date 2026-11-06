# TRIDENT corrective experimental design

Status: proposal only, awaiting explicit approval before implementation. No revised model, loss code, runnable experiment configuration, training notebook or optimizer update is introduced by this document. Every numerical setting below is a candidate for approval, not an approved configuration or a demonstrated improvement.

## Evidence and purpose

The existing diagnostic report and current ResidualCNN, feature packing, loss composition, dense measurement operator and validation boundary wrapper were inspected. Evidence is in [the diagnostic report](../custom_diagnostics/report.md) and [analysis.json](../custom_diagnostics/analysis.json).

The selected custom model adds about +3.00 mm on average and worsens 65.16% of validation pixels. Its reconstructed initialization adds approximately 171–172 mm on eight samples. The likelihood formula, masking and checkpoint selection passed verification. Uncertainty never hit its clamp bounds; its calibration was uneven, and its depth-output gradient norm was 1.65–3.01 times that of the other weighted terms in eight probes. The boundary head has no supervision.

These findings support controlled tests of initialization, correction parameterization and objectives. They do not establish a single cause, nor show that the proposed corrections work.

## 1. Common experimental contract

All initial proposed pilots retain:

- Dense 4x SR, original area-mean degradation and bicubic initializer; full 480x640 RGB/targets and 120x160 depth observations.
- The identical 672 training frames / 219 scenes and 123 validation frames / 30 scenes. The 654 official test frames remain locked. No test pixels, evaluation or tuning.
- Existing masks, metric implementations and threshold definitions. Global valid-pixel RMSE/MAE; frame-mean depth-boundary F1 at 0.05 m / 1 pixel tolerance.
- Adam, lr 1e-4, betas (0.9,0.999), eps 1e-8, weight decay 0, batch 1, FP32, seed 69, constant LR and gradient clipping norm 5.
- A proposed two-epoch screening budget per explicitly approved row, validation every epoch, and earliest minimum-validation-RMSE checkpoint. No automatic extension, seed sweep or matrix execution.
- Fresh initialization from the same seed for each row; no fine-tuning of existing checkpoints. Matching source/runtime, hardware where practical, shuffled order and initial backbone/head tensors across paired rows.
- Complete model/Adam/scheduler/Python/NumPy/CPU/CUDA/DataLoader RNG state and experiment provenance. Separate immutable run directories and new comparison records.

Existing bicubic, baseline CNN and original custom pilot remain historical anchors. They are not retrained, overwritten or silently treated as fully controlled replicates if the environment changes. If source/runtime numerical parity cannot be demonstrated, an explicitly approved matched replication is needed before attributing effects to one factor.

## 2. Revised architecture proposal

Retain the existing 473,059 trainable parameters: 8 input channels, encoder 32/64, bottleneck 128, decoder 64/32, skip concatenations and nearest decoder upsampling. Keep residual, log-scale and boundary heads. No normalization layers, added blocks, channel changes or new trainable gate are proposed for the initial controls.

Let D0 be bicubic depth, F the final decoder features, r_raw = W_r F + c_r, and alpha a fixed positive scalar:

    correction = alpha * r_raw
    predicted_depth = max(D0 + correction, 1e-6)

Expose raw and applied corrections separately for monitoring. Preserve the current log-scale clipping [-6,6] because saturation was not observed. In the later boundary-supervised variant, expose boundary_logits and sigmoid(boundary_logits) from the existing 1x1 boundary head. The extra output tensor adds no parameters.

This is a proposal for a separately versioned model and checkpoint identity. Legacy ResidualCNN and LightweightCNN loading must retain their current behavior.

### Exact-zero versus near-zero final head

Two alternatives isolate the initialization choice:

- Identity start: W_r = 0, c_r = 0. The starting prediction equals the bicubic estimate for valid D0 >= 1e-6.
- Near-zero start: W_r = 1e-3 * W_default, c_r = 0. The candidate epsilon is fixed before training and compared directly with identity start; it is not selected by scanning validation results.

Construct all modules in the original order and consume the original random draws before applying either final-head initialization. Other head/backbone tensors and shuffle state must remain matched. Initialize only fresh runs; never reinitialize after checkpoint restoration.

With W_r = 0, reconstruction gradients initially reach the final head while its feature-path gradient is zero on the first step. This is expected, not proof of disconnection. An active uncertainty branch can still update the shared backbone. The near-zero alternative tests whether preserving a small initial reconstruction gradient changes this transient.

Zero-start residual branches have prior precedent in [Fixup initialization](https://arxiv.org/pdf/1901.09321). This proposal changes only the final depth-refinement head; it does not implement the full Fixup recipe or claim its published results apply to TRIDENT.

### Controlled scaling

The first scaling control uses alpha = 0.1 versus alpha = 1, with all other settings matched. Alpha stays fixed, positive and non-trainable. It is stored in checkpoint metadata and restored exactly.

Do not combine alpha = 0 with a zero-initialized head: a learnable gate and head can then both have zero reconstruction gradients. Fixed scaling also changes gradients and optimization; Adam may partly compensate for it. It is not a guarantee that learned corrections remain small, and cannot by itself guarantee removal of a signed bias.

An optional later safety control could replace the linear correction by:

    correction = R * tanh(r_raw / R), R = 0.10 m, alpha = 1

This isolates a bounded output mapping with derivative one at zero. R is a proposed fixed control, not a validation-tuned value. A 100 mm bound can exclude legitimate corrections at difficult edges, and the observed final residual was already small. Therefore this is deferred, not the primary proposed fix for the +3 mm bias. Do not combine scaling and bounding in an initial row.

No residual-magnitude penalty, scheduled gate or learned scale is added to the initial experiment family.

## 3. Loss formulation and precise ablation meanings

Use the existing definitions, including their current normalization:

    L_d = mean_M |prediction - GT|
    L_g = mean_valid_horizontal_pairs |delta_x prediction - delta_x GT|
        + mean_valid_vertical_pairs   |delta_y prediction - delta_y GT|
    L_e = mean_M w_edge(GT) * |prediction - GT|
    L_m = mean_M_LR |area_pool_4(prediction) - observed_LR|
    L_u = mean_M [exp(-s) * |prediction - GT| + s]

M is the GT validity mask; M_LR is the observed LR mask. L_e retains its existing pixel-count denominator and both-endpoint depth-jump weights.

Define:

    L_R = L_d + 0.2 L_g + 0.1 L_e + 0.2 L_m
    L_J = L_R + 0.05 L_u

The current effective custom objective is L_J. The main uncertainty ablation compares L_J with L_R, changing only uncertainty supervision. This is reconstruction-only, not literally pixel-depth-only.

A literal depth-only control uses L_D = L_d. It is reached by removing gradient, edge and measurement terms in separate matched steps. A second pair compares L_D with L_D + 0.05 L_u. Thus removing several auxiliary terms is never reported as the isolated effect of uncertainty.

When uncertainty is disabled, the future loss implementation should exclude its training graph explicitly. Multiplying a computed term by zero can create zero gradients and misleading Adam state. Disabled heads must be marked untrained/invalid in metadata and must not supply calibrated confidence to inference or hybrid fusion.

Total loss values across different objectives are not model-quality rankings. Reconstruction metrics, measurement drift and boundary quality remain the comparison criteria.

## 4. Should uncertainty get a calibration objective?

Not in the first corrective comparison. The negative likelihood is mathematically valid, no scale collapse occurred, and adding regularization immediately would confound the initialization/objective tests. First establish whether removing the uncertainty term changes reconstruction quality.

If retained, evaluate b as a Laplace scale: conditional mean absolute error should approximately match b, not Gaussian standard deviation. Report 50/90/95% coverage, scale-decile calibration, edge/non-edge calibration and error ranking. Global coverage alone can conceal uninformative or conditionally incorrect scales; this motivates reporting both coverage and conditional calibration, consistent with [Levi et al.](https://arxiv.org/abs/1905.11659).

### Optional regional calibration regularizer

One later, separately isolated candidate is:

    E_edge = eligible GT pixels incident to a valid 0.05 m depth jump
    E_flat = other eligible GT pixels
    C_G = [log(mean_G exp(s) + epsilon_b)
         - log(mean_G stop_gradient(|prediction-GT|) + epsilon_b)]^2
    L_cal = mean over nonempty groups G in {E_edge,E_flat} of C_G
    L_JC = L_J + lambda_cal * L_cal

Proposed initial controls: epsilon_b = 1e-6 m and lambda_cal = 0.01. They are candidates for approval; no claim is made that these weights are effective. Groups receive equal weight when both exist; empty groups are excluded and the remaining group weights renormalized. GT group labels are used only in the training objective and diagnostic stratification, never as inference inputs.

The stopped residual target prevents a direct incentive to increase the predicted depth error to match uncertainty. Scale gradients still pass through shared features, so this is not reconstruction-neutral. Matching group means also does not guarantee calibrated tails or useful ranking. These limitations require the metrics above and a matched L_J control.

Do not add an arbitrary penalty toward s = 0: that corresponds to a 1 m scale. Do not force NLL to be nonnegative, change inactive clamp bounds, or add uncertainty spatial smoothing without separate evidence; smoothing could erase the edge-dependent scale variation we need.

### Calibration-only alternatives and leakage

A later head-only control would need to stop gradients through both the depth-error target and the features supplied to the uncertainty head. Detaching the depth error alone still allows shared-backbone changes. Global gradient clipping can also couple otherwise separated branches; reconstruction and calibration parameter sets would need independent clipping/optimization treatment and explicit trajectory-parity tests before claiming reconstruction is unaffected.

Post-hoc positive scale calibration, such as b' = T*b, is another separate option. It cannot fix depth predictions and a global T may not resolve opposing edge/non-edge errors. Calibration must not be fitted on the 123 validation frames used to compare outcomes. Honest independent calibration would require a scene-safe subset of training scenes excluded from weight fitting, or scene-wise cross-fitting. That changes data allocation or compute and needs separate approval; the initial matrix preserves all 672 fitting frames. Do not claim independent calibration from in-sample training residuals. [Calibrated regression](https://proceedings.mlr.press/v80/kuleshov18a.html) provides relevant background, not a guarantee for these depth data.

## 5. Supervised boundary-confidence head

Target meaning: probability of a GT depth discontinuity at a pixel. It is not the probability that a depth estimate is accurate and must not automatically replace the fusion reliability signal.

For pixel i:

    Y_i = 1 if any in-image cardinal neighbour j has
          valid GT at i and j, and |GT_i - GT_j| >= 0.05 m
          else 0

Mark both endpoints of each qualifying pair, matching the current depth-edge evaluation convention. Use metres and GT depth, not RGB texture edges, predicted depth edges or sensor-invalid transitions. There is no label dilation; the one-pixel tolerance belongs to the evaluation metric only.

Define the training eligibility mask B_i as valid GT at i and all its available in-image cardinal neighbours. Ignore ambiguous neighborhoods around invalid GT. Image borders use only available neighbours. In the current fully valid dense protocol all target pixels remain eligible.

Use existing boundary-head logits z:

    L_b = mean_B BCEWithLogits(z, Y)
    L_RB = L_R + lambda_b * L_b, proposed lambda_b = 0.01

Compute unreduced BCE, then apply B and divide by its valid count; return a connected zero when no eligible pixels exist. Use logits directly for stable computation as described in [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.11/generated/torch.nn.BCEWithLogitsLoss.html). Keep sigmoid probabilities for outputs.

Start with unweighted BCE so the head has an ordinary Bernoulli-probability interpretation. The approximately 9.7% GT edge prevalence makes the all-negative solution a concern; report precision/recall, PR-AUC, probability calibration and head F1 at a fixed 0.5 threshold. Do not silently add positive-class weighting, focal loss, Dice or threshold tuning. Those would be separate factors and positive weighting changes probability calibration.

Depth-boundary F1 and auxiliary-head F1 must have distinct names. Initially the supervised head influences depth only through shared-backbone gradients; do not also gate the residual with its probability. That would add another architectural factor. The old unsupervised head is not reinterpreted as supervised.

## 6. Dense measurement consistency stays unchanged

Retain the current operator, mask, normalization and 0.2 weight in all primary candidates. Area pooling matches the actual synthetic dense LR input, and the audit found only float32 accumulation roundoff. No extra factor of 16 or hard projection is required.

The explicit removal of L_m in one literal-depth-only diagnostic row below is the only planned exception. It isolates that removal, not a recommendation for the revised default. Sparse/noisy completion is outside this experiment family; do not apply dense area consistency to that task.

## 7. Ablation matrix

All rows below are proposals with no results. Each training row is a fresh, separately approved two-epoch pilot with the common contract. Parent identifies the exact matched contrast. Every unspecified setting is inherited from that parent; no row silently combines initial fixes.

| ID | Parent | Single factor changed | Objective / architecture setting | Stage |
| --- | --- | --- | --- | --- |
| H0 | existing | Historical anchor, no new run | Current initialization, alpha=1, L_J | reference |
| A1 | H0 | Final residual initialization | Exact zero W/b; alpha=1; L_J | first |
| A2 | A1 | Residual weight initialization | epsilon=1e-3 default W; bias=0; L_J | optional initialization control |
| A3 | A1 | Fixed residual scale | alpha=0.1; exact-zero head; L_J | scaling control |
| B0 | A1 | Remove uncertainty objective | L_R; other settings identical | first |
| B1 | B0 | Remove gradient objective | L_d + 0.1 L_e + 0.2 L_m | literal-depth path |
| B2 | B1 | Remove edge objective | L_d + 0.2 L_m | literal-depth path |
| B3 | B2 | Remove measurement objective | L_D = L_d | literal-depth path only |
| B4 | B3 | Add uncertainty objective | L_d + 0.05 L_u | matched pure-depth uncertainty pair |
| C1 | B0 | Add boundary supervision | L_R + 0.01 L_b; no uncertainty training | later auxiliary control |
| C2 | A1 | Add regional calibration term | L_J + 0.01 L_cal; boundary loss off | later calibration control |
| C3 | A1 | Add boundary supervision | L_J + 0.01 L_b; calibration term off | optional interaction control |
| S1 | A1 | Bound correction mapping | R*tanh(r_raw/R), R=0.10 m; alpha=1; L_J | deferred safety control |

A1 versus H0 tests initialization. B0 versus A1 isolates uncertainty removal with reconstruction terms fixed. A3 versus A1 isolates scaling. B4 versus B3 tests uncertainty on a literal depth-only objective. The B0-to-B3 chain isolates one removal at each step; effects are conditional on that parent and cannot be summed as independent global effects.

A1/B0/C1/C3 form an optional 2x2 uncertainty-by-boundary-supervision interaction study. Do not execute C3 or a combined calibration/scaling/boundary variant before the separate controls are understood.

Recommended smallest implementation scope for discussion is A1 and B0, retaining H0 and the frozen baseline/bicubic anchors. A2/A3 and the later rows remain separately scoped options. No choice of a parent row authorizes its children. A matched legacy replication is conditional on a source/runtime parity failure and also requires explicit approval.

A CSV copy is in [ablation_matrix.csv](ablation_matrix.csv); it is a design table, not a runnable experiment configuration. Every candidate remains pending approval. No effect size or winner is predicted.

## 8. Monitoring and decision rules

Record actual applied correction statistics, not only the raw head: signed mean/bias, MAE/RMS magnitude, p50/p90/p99/max, positive-depth and optional tanh saturation fractions, pixel/frame improvement rates, edge/non-edge effects, and the exact MSE cross-term decomposition already used in diagnostics. Report raw corrections alongside these to detect compensation for alpha.

Keep all loss components both raw and weighted, including measurement drift. Log uncertainty scales, clamp fractions, conditional calibration and coverage; mark uncertainty unavailable in disabled-head variants. In boundary variants, record label prevalence, auxiliary metrics and nonzero finite head gradients.

Measure per-term depth-output and shared-backbone gradient norms/cosines on a fixed training-only probe. Log pre-clip norms, clipping frequency and actual LR. Diagnostic probes must not consume training shuffle/RNG state, mutate weights or become extra optimizer steps. Report runtime and CUDA allocated/reserved peak memory; extra diagnostics add overhead, so epoch time alone is not inference latency.

Use identical validation IDs/files/masks for all paired comparisons. Selection remains global validation RMSE, with the earliest epoch on ties. Do not select by signed total loss or by whichever metric happens to improve.

For review, report paired frame/scene outcomes and the RMSE/MAE/boundary trade-off against both anchors. An initialization change is not a success merely because epoch-zero error improves. A candidate with improved RMSE and worse MAE remains a trade-off, not an overall win. Require differences to exceed measured replay tolerance; no universal significance claim from correlated pixels. Any formal practical non-inferiority margins or later multi-seed confirmation must be specified before those runs.

## 9. Implementation plan, gated on approval

No items in this section have been implemented.

1. Freeze an approved manifest of run IDs, budgets, seeds and candidate constants. Preserve official configs and all old checkpoints. Create separately versioned proposed configuration files only after approval.
2. Add a distinct versioned residual model, reusing the current channel topology and module construction order. Preserve the legacy model registry entries, output behavior and checkpoint loading. Record alpha, initialization policy, output mapping and boundary-head semantics in new checkpoint metadata; all initial candidate variants retain 473,059 trainable parameters.
3. Add a separate objective composer that reuses the verified masking/gradient/edge/measurement functions, supports explicit disabled components and raw-logit BCE, and records active losses. Do not alter old loss defaults. Disabled uncertainty cannot silently drive hybrid inference.
4. Add deterministic boundary-target/mask generation and correction/calibration monitoring modules. Dataset bytes, splits, classical implementations and metric definitions remain unchanged.
5. Test identity and epsilon initialization, matching non-head RNG/tensors, positive alpha and the double-zero trap; first-step gradient expectations; masked BCE/empty labels; uncertainty detachment semantics; unchanged dense area consistency; legacy/V2 serialization; and exact resume without resetting learned heads. Test frozen baseline/checkpoint inference parity.
6. Only after implementation approval, run synthetic and real training-sample forward/backward preflights with zero optimizer steps on Windows CPU and Colab GPU. Confirm counts, finite gradients, masked targets, initial correction magnitudes and checkpoint/RNG roundtrip.
7. Prepare a distinct Colab notebook for only the approved run IDs. Preserve the working existing-runtime setup and CUDA PyTorch. Keep execution disabled until the relevant launch approval; save full optimizer/scheduler/RNG/source/configuration state and new outputs separately.
8. Training, later runs, official test evaluation and inference export each stay outside the present authorization. Stop at the approved two-epoch budget and review the actual evidence before any expansion.

Near-zero preflight should report an initial mean absolute correction below the proposed 0.1 mm diagnostic threshold on a fixed training-only probe; otherwise report the discrepancy rather than automatically retune epsilon. Exact-zero identity is a deterministic invariant. Do not demand nonzero reconstruction-backbone gradients at the first step of an exact-zero head.

A truly detached calibration-head control would additionally need independent clipping/optimization accounting and an invariance test; simply detaching the residual target is insufficient. New checkpoint loaders must reconstruct the proposed output transform while leaving all old checkpoints compatible.

## 10. Current deliverables and validation boundary

This work adds only design artifacts under docs/corrective_design: this document, a non-runnable CSV matrix and a preservation/verification record. No model, loss, test, training script, notebook or official configuration is changed, and no real dataset pixels are loaded for this design task.

The existing 144-test suite is run as a regression check and a preservation inventory checks the existing source/configuration/diagnostic records, supplied custom ZIP, baseline checkpoints and classical result files. Final verification is recorded in verification.json. No proposed correction has experimental efficacy evidence yet.

Implementation and training both await explicit approval.

