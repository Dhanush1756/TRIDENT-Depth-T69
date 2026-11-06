# Custom TRIDENT residual pilot: diagnostic report

The selected checkpoint reproduces the reported regression. No numerical bug was found in the investigated uncertainty formula, masks, loss normalization, dense-SR measurement operator or checkpoint selection. The clearest failure is a small, systematically biased correction that makes most pixels worse. Uncertainty does not collapse or saturate, but its calibration is uneven and its objective substantially reweights regression gradients. Initialization and coupled optimization are plausible contributors; their causal effects remain unproven.

Both existing checkpoints were evaluated on the same 123 development validation frames: 30 scenes and 37,785,600 valid pixels per checkpoint, with full 480x640 targets and 120x160 dense 4x observations. No optimizer was created or stepped by the diagnostic module. No training, fine-tuning, hyperparameter changes, checkpoint writes, inference exports, official test inference or benchmark overwrites were performed.

## 1. Provenance and checkpoint selection

The run ZIP, supplied environment record/freeze, source archive, dataset archive, configuration hashes, frame IDs, RGB/depth/mask file hashes and target-mask hashes were inspected. The environment JSON hash matches the hash embedded in the checkpoints. Original training source modules match the recorded source archive byte-for-byte.

Training used Python 3.13.16 / PyTorch 2.11.0+cu130 on Tesla T4. Diagnostics used the local Windows CPU environment recorded in analysis.json. Both checkpoints strictly load the 473,059-parameter nearest-decoder architecture. Their histories, metadata, optimizer step counts and selection records agree.

Epoch 2 has lower validation RMSE than epoch 1. The best-checkpoint pointer, saved best epoch, comparison metadata and exact comparison checkpoint SHA256 agree. There is no evidence of loading a wrong epoch, architecture or inference export.

| Method, original Colab comparison | RMSE (m) | MAE (m) | Boundary F1 |
| --- | ---: | ---: | ---: |
| Bicubic | 0.0585272133 | 0.0162030076 | 0.67796594 |
| Baseline CNN | 0.0583331947 | 0.0162285307 | 0.69136013 |
| Custom residual | 0.0587230350 | 0.0171603756 | 0.67801509 |

CPU replay of both custom epochs differs from the recorded RMSE by less than 1e-9 m, MAE by less than 6.4e-9 m and boundary F1 by less than 2.3e-7. These small backend differences are recorded separately; original results were not replaced.

## 2. Uncertainty formula is correct; negative values are legitimate

The head emits s = log(b), where b is a positive Laplace scale measured in metres. It is not Gaussian log variance. The implementation is:

    L_uncertainty = mean_valid(exp(-s) * abs(prediction-target) + s)

This equals Laplace negative log-likelihood with the constant log(2) omitted. The expression matches [PyTorch's Laplace distribution](https://docs.pytorch.org/docs/2.11/distributions.html#laplace), and independent tests verify its value and derivatives. Laplace standard deviation would be sqrt(2)*b.

For an unclamped pixel, the derivatives before mean reduction are:

    dL/dprediction = sign(prediction-target) / b
    dL/ds = 1 - abs(prediction-target) / b

A continuous density can exceed one, so its negative log-density can be negative. At the pointwise optimum b = absolute error, the expression is 1 + log(error_in_metres), negative for errors below approximately 0.368 m. Negative s is not negative uncertainty: exp(s) remains positive. Adding log(2) changes a constant offset and does not fix reconstruction behavior.

The clamp bounds b between exp(-6) and exp(6) metres. Its lower limit is approximately 2.479 mm, preventing a zero-scale likelihood singularity in this implementation. Neither bound is active in these validation predictions.

Across both full validation passes, the maximum difference from torch.distributions.Laplace is 7.153e-7 in the unweighted uncertainty loss. The float64 NumPy reference differs by at most 5.769e-7. Losses and diagnostic gradients are finite.

### Why total loss is dominated in value

| Weighted validation contribution | Epoch 1 | Epoch 2 |
| --- | ---: | ---: |
| Depth | +0.01918261 | +0.01716038 |
| Gradient | +0.00450269 | +0.00448270 |
| Edge | +0.00382126 | +0.00362071 |
| Measurement | +0.00192880 | +0.00142759 |
| Uncertainty | -0.14400352 | -0.16049803 |

At epoch 2, positive contributions total approximately +0.02669139; uncertainty contributes -0.16049803, reproducing the negative total near -0.13380664. Approximately 86% of the total-loss reduction between epochs comes from the uncertainty contribution.

Scalar dominance does not itself prove gradient dominance. On eight fixed validation frames, the weighted uncertainty gradient with respect to predicted depth has 1.65–3.01 times the norm of all other weighted depth-output gradients combined. Their cosines are 0.885–0.950: largely aligned, not directly opposing. This confirms substantial precision-based reweighting, not a sign conflict. Shared-backbone gradient competition remains a hypothesis because these probes do not reconstruct every training-step gradient.

## 3. Uncertainty is not collapsed; calibration is uneven

| Full-pixel statistic | Epoch 1 | Epoch 2 |
| --- | ---: | ---: |
| Mean Laplace b (mm) | 13.238 | 17.983 |
| Mean log(b) | -4.4860 | -4.1131 |
| Mean absolute error / b | 1.6059 | 0.9031 |
| Fraction at/below lower clamp | 0 | 0 |
| Fraction at/above upper clamp | 0 | 0 |
| Positive-depth clamp fraction | 0 | 0 |

Mean scale increases between epochs. The strongly negative uncertainty term therefore does not result from a shrinking or zero-scale collapse. At epoch 2, the full predicted scale range is approximately 8.54–585.14 mm, far from the bounds.

The decomposition E[abs(error)/b] + E[log(b)] changes from 1.6059 - 4.4860 to 0.9031 - 4.1131. The log-scale becomes less negative while the likelihood improves. The error/scale component is responsible for the net reduction.

For a centered Laplace interval, nominal coverage p has half-width -b*log(1-p):

| Nominal coverage | Epoch 1 actual | Epoch 2 actual |
| --- | ---: | ---: |
| 50% | 44.951% | 77.296% |
| 90% | 88.332% | 92.610% |
| 95% | 91.052% | 94.267% |

The central distribution is over-wide at epoch 2, while the 95% tail is slightly under-covered. Global mean scale near global MAE conceals conditional miscalibration. Sampled scale/error Spearman correlation is 0.332: some ranking ability, with substantial limitations.

GT edge regions use depth jumps >=0.05 m, marking both adjacent pixels:

| Region | Pixels | Bicubic MAE (mm) | Custom MAE (mm) | Predicted b (mm) |
| --- | ---: | ---: | ---: | ---: |
| Edge | 3,673,432 | 95.458 | 95.711 | 21.909 |
| Non-edge | 34,112,168 | 7.668 | 8.702 | 17.560 |

The predicted scale is too large relative to central non-edge errors and too small relative to edge errors. This is a confirmed calibration limitation, rather than numerical collapse or saturation. It also cautions against interpreting these scales as calibrated hybrid-routing confidence.

## 4. Residual corrections are small but biased and globally harmful

| Diagnostic | Epoch 1 | Epoch 2 |
| --- | ---: | ---: |
| Mean signed correction (mm) | -5.549 | +2.997 |
| Mean absolute correction (mm) | 6.223 | 3.120 |
| Pixels improved over bicubic | 25.632% | 34.836% |
| Pixels worsened over bicubic | 74.367% | 65.162% |

The final residual is not exploding. A systematic +3 mm correction is nevertheless harmful when bicubic is already accurate on most non-edge pixels. Both edge and non-edge MAE worsen. The sampled correlation between the correction and the ideal correction, GT minus bicubic, is -0.0261.

An exact energy identity establishes the direction of the error:

    e0 = bicubic - GT
    r = custom - bicubic
    MSE_custom = E[e0^2] + 2 E[e0*r] + E[r^2]
               = 0.0034254347 + 0.0000082187 + 0.0000147415
               = 0.0034483949 m^2

Both added terms are positive at epoch 2: the correction adds variance and, on average, aligns with existing error rather than cancelling it. This confirms unhelpful corrections globally without changing predictions or testing alternative residual multipliers.

The reduction in correction magnitude and improved validation metrics from epoch 1 to 2 show learning activity. They do not establish an improvement over the initializer or prove that extra epochs would fix the regression.

## 5. Initialization is not identity-preserving

The seeded custom residual head uses standard random Conv2d initialization and starts with a nonzero correction. Its initial bias is +0.162036 m. On the eight fixed validation examples, the initial custom correction is approximately +0.171–0.172 m, much larger than the desired millimetre-scale refinements. The reconstructed baseline initialization produces roughly 49–79 mm corrections on those examples.

At epoch 2 the custom head bias is still approximately +0.156443 m. Its learned feature contribution averages about -0.153446 m, leaving the observed +3 mm correction. The model has learned substantial cancellation before making useful fine corrections. This is an optimization handicap, not evidence of floating-point cancellation failure.

The non-identity initialization is confirmed. Its causal contribution to the final regression is a hypothesis requiring a controlled experiment. The baseline also starts with nonzero corrections, so initialization alone cannot be declared the explanation.

Initialization was reconstructed using local PyTorch. Untouched boundary-head weights differ from the stored checkpoint by at most 1.49e-8, and the bias is identical. The boundary weights are exactly identical between epochs. Therefore initial-reference comparisons require a numerical tolerance; bitwise identity across all parameters and PyTorch versions is not claimed.

## 6. Depth, gradient and edge normalization/masks are correct

Independent float64 NumPy references were evaluated on every frame for both checkpoints:

| Loss term | Maximum absolute implementation/reference difference |
| --- | ---: |
| Depth | 5.884e-9 |
| Gradient | 6.313e-9 |
| Edge | 1.739e-8 |
| Uncertainty | 5.769e-7 |
| Measurement | 7.849e-9 |

Depth and uncertainty use the valid-GT pixel count. Gradient loss excludes invalid neighbour pairs and sums horizontal and vertical means; it intentionally does not divide that sum by two. Edge loss averages absolute error times edge weight over valid pixels, not over the sum of weights. Edge weights start at one and increase at both endpoints of GT jumps, so edge density changes effective emphasis under this documented convention.

The actual dense reference masks are fully valid. Additional synthetic partial-mask/NaN tests verify exclusion of invalid targets and neighbour pairs. No missing normalization divisor, invalid-pixel leakage or masked NaN bug was found.

## 7. Dense measurement consistency uses the matching operator

The low-resolution input is the mean of each nonoverlapping 4x4 GT block. Comparing avg_pool2d(prediction,4,4) against these observations is the correct matching operator. It is not point sampling or the same-resolution sparse-completion constraint.

The loss uses the LR-valid mask and a mean over observed LR pixels. No extra factor of 16 is missing: the low-resolution mean and the pooling derivative account for the resolution change. The constraint remains soft, with its approved 0.2 weight.

GT re-pooling differs from stored observations by at most 1.907e-6 m of float32 accumulation roundoff. Dense observations receive no synthetic sparse noise, despite sparse-only settings also appearing in the shared pipeline configuration.

| Unweighted validation term | Bicubic | Custom epoch 2 |
| --- | ---: | ---: |
| Depth | 0.016203008 | 0.017160382 |
| Gradient | 0.022387479 | 0.022413503 |
| Edge | 0.035205806 | 0.036207100 |
| Measurement | 0.005490954 | 0.007137969 |

All four reconstruction terms are worse than bicubic. Measurement MAE increases by about 30%. Their weighted positive sum is approximately 0.02529927 for bicubic and 0.02669139 for custom. A lower total likelihood-based loss therefore does not imply better reconstruction. Total losses across baseline/custom models also differ in composition because only the custom model has an uncertainty head.

## 8. Boundary head and unused/disconnected components

The boundary-head weight and bias, 33 parameters total, have no loss gradients or Adam state and are exactly unchanged between checkpoints. This was documented before the pilot. The head is computed but not used for depth reconstruction, comparison metrics or the current fusion confidence input. Depth boundaries still receive gradient/edge supervision through the depth output.

The other 24 parameter tensors have expected Adam states and steps: 672 after epoch 1 and 1,344 after epoch 2. Full-frame module probes show finite, nonzero gradients through the encoder, bottleneck, decoder, residual head and uncertainty head. No whole core block is disconnected.

The dense/sparse protocol input is zero in this dense-only experiment. The corresponding 288 first-convolution weights have zero first/second Adam moments. This is expected conditional-feature inactivity, not disconnection of the full layer. Tiny cross-version differences from reconstructed initialization must not be mistaken for actual learning.

Removing the unused 33-parameter head alone is not established as a reconstruction fix. Its absence of supervision is a confirmed design limitation, not a newly discovered likelihood bug.

## 9. Confirmed findings versus hypotheses

| Classification | Finding |
| --- | --- |
| Confirmed numerical bug | None found in investigated loss formulas, masking, normalization, dense consistency or checkpoint selection. |
| Confirmed wiring limitation | Boundary-confidence head is unsupervised and unused by reconstruction/comparison, as previously documented. |
| Confirmed outcome | Biased corrections worsen most pixels and all four direct reconstruction terms. |
| Confirmed calibration limitation | Over-wide center, insufficient edge uncertainty, moderate error ranking; no clamp collapse. |
| Confirmed optimization setup | Large initial non-identity correction and substantial precision-based gradient reweighting. |
| Hypothesis | Much of the short pilot is spent learning to cancel the initial correction. |
| Hypothesis | Shared-feature uncertainty optimization and precision weighting hinder the desired reconstruction metrics. Direct output-gradient conflict is not established. |
| Hypothesis | Two epochs are insufficient for this coupled model. Improvement across epochs alone does not prove longer training will fix it. |
| Unsupported explanation | Wrong checkpoint, numerical divergence, entire-network disconnection, hard-clamp saturation or zero-scale collapse. |

## 10. Corrective experiment ideas for later review

These are conceptual controls, not newly authorized or scheduled runs. No new training configuration, hyperparameter change or launch notebook was created.

1. An identity-initialization control would isolate the measured initial residual offset while preserving parameter count. Improvement must be demonstrated, not assumed.
2. An uncertainty-objective ablation with other reconstruction settings matched would separate likelihood reweighting from architecture capacity. The present comparison changes both architecture and effective objective.
3. If uncertainty remains useful, investigate calibration by region and coverage on scene-safe development data and isolate branch-optimization choices. Clamp changes are not supported as a primary fix by this run, because neither bound is active.
4. Decide explicitly whether boundary confidence should receive masked GT-derived supervision or remain an acknowledged unused output. Any architecture/loss change needs approval; removing the tiny unused head alone is not a demonstrated fix.

Extra epochs or multiple seeds could later test transient/convergence explanations under controlled objectives. No longer training, fine-tuning, test-set tuning or export is initiated by this report.

## 11. Evidence scope, artifacts and reproducibility

Full-pixel means, coverage, clamp fractions, residual energy and regional statistics use all 123 validation frames. Quantiles, scale deciles and rank correlations use 563,965 deterministic, phase-mixed samples per checkpoint: every 67th flattened valid pixel with frame-dependent offset. Gradient weighting uses eight fixed validation frames; module connectivity uses one full frame per checkpoint. Neither establishes what every training step did.

Pixels are spatially correlated. No pixel-independence significance claim or causal conclusion from a single two-epoch seed is made. CPU/backend differences are quantified by the replay checks. Exact CUDA resumption was not attempted locally.

Evidence files:

- [analysis.json](analysis.json): full configurations, original history/comparison, environment, hashes, calibration, residual identities, regional sums and gradient probes.
- [per_frame.json](per_frame.json): paired results for every validation ID and both checkpoints.
- artifacts/custom_diagnostics/epoch_001_sample.npz and epoch_002_sample.npz: sampled calibration arrays with explicit column labels.
- trident_depth/evaluation/custom_diagnostics.py: diagnostic core and independent reference operators.
- tests/test_custom_diagnostics.py: analytic NLL/gradient, clamp, partial-mask normalization and degenerate-calibration checks.

Run the core diagnostics again only into fresh destinations:

~~~powershell
.\.venv\Scripts\python.exe -m trident_depth.evaluation.custom_diagnostics --archive "${USER_HOME}\Downloads\pilot_dense4_residual_seed69-20261106T121048Z-1-001.zip" --environment "${USER_HOME}\Downloads\pilot_dense4_residual_seed69_environment.json" --output artifacts/custom_diagnostics_repeat --report-dir docs/custom_diagnostics_repeat
.\.venv\Scripts\python.exe -m pytest -q
~~~

The complete suite after diagnostic code changes passed: 144 passed, zero failures, one existing upstream Starlette/httpx deprecation warning, 28.99 seconds. Preservation hashes are checked again after testing. Training remains paused pending review.


## Appendix: epoch-2 calibration bins and frame outcomes

Deciles use the documented deterministic sample; coverage and region tables above use all valid pixels.

| Predicted-scale decile | Mean b (mm) | Actual MAE (mm) | Mean absolute error / b |
| --- | ---: | ---: | ---: |
| 1 | 10.841 | 4.744 | 0.436 |
| 2 | 11.912 | 6.846 | 0.574 |
| 3 | 12.735 | 8.053 | 0.632 |
| 4 | 13.540 | 9.534 | 0.703 |
| 5 | 14.519 | 12.097 | 0.833 |
| 6 | 15.807 | 14.205 | 0.898 |
| 7 | 17.464 | 18.138 | 1.038 |
| 8 | 19.606 | 21.536 | 1.098 |
| 9 | 22.468 | 29.364 | 1.298 |
| 10 | 40.940 | 47.183 | 1.522 |

| Epoch | Frames with lower RMSE than bicubic | Frames with lower MAE than bicubic |
| --- | ---: | ---: |
| 1 | 0/123 | 0/123 |
| 2 | 0/123 | 0/123 |

Original source/configuration, existing benchmark records and supplied checkpoint ZIP/environment hashes were rechecked after the full test suite and remained unchanged.
