# Measured NYU development-validation baselines

These are actual classical measurements on a frozen, training-derived validation subset, not final test results.
No neural network training or neural/hybrid evaluation was run.

Official partition: {'train': 795, 'test': 654}. Development partition: {'test': 654, 'train': 672, 'validation': 123}.
Subset: 8 distinct scenes, selected before method scoring.
Scene types: {'bedroom': 1, 'living_room': 1, 'furniture_store': 1, 'kitchen': 1, 'bookstore': 1, 'dining_room': 1, 'reception_room': 1, 'bathroom': 1}.

Per-image arithmetic macro means are reported below. Latency is the median of per-image medians, CPU batch size 1, one warmup and three repetitions.
Tracemalloc is enabled during timing. Process RSS is the lifetime peak, not an isolated per-method native/GPU peak.
PSNR range: 10 m; SSIM: uniform 7x7 valid windows. Depth-edge threshold: 0.05 m; boundary tolerance: 1 pixel.

Evaluation uses full 480x640 images, no Eigen crop and no reference resize. These values must not be directly compared with incompatible published protocols.

## dense_sr

| Scale | Method | RMSE m | MAE m | PSNR dB | SSIM | Edge preservation | Boundary F1 | Latency ms | Traced peak MiB | Lifetime RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | bicubic | 0.03502 | 0.00944 | 51.03407 | 0.99645 | 0.57561 | 0.88391 | 13.93 | 8.28 | 158.32 |
| 2 | bilinear | 0.04000 | 0.01077 | 49.80763 | 0.99548 | 0.58787 | 0.85067 | 13.19 | 8.28 | 158.32 |
| 2 | classical | 0.04393 | 0.01388 | 48.71129 | 0.99480 | 0.52001 | 0.83007 | 588.92 | 56.26 | 158.54 |
| 2 | guided | 0.04945 | 0.01630 | 47.67415 | 0.99358 | 0.49058 | 0.82272 | 467.67 | 47.47 | 158.32 |
| 2 | joint_bilateral | 0.04357 | 0.01379 | 48.77502 | 0.99485 | 0.52030 | 0.83079 | 484.40 | 47.47 | 158.32 |
| 2 | nearest | 0.04837 | 0.01151 | 48.13189 | 0.99365 | 0.27014 | 0.93871 | 11.99 | 8.28 | 158.32 |
| 4 | bicubic | 0.06214 | 0.01994 | 45.82152 | 0.98864 | 0.47193 | 0.70223 | 12.78 | 8.00 | 160.35 |
| 4 | bilinear | 0.06900 | 0.02246 | 44.89091 | 0.98693 | 0.47543 | 0.65687 | 11.86 | 8.00 | 160.35 |
| 4 | classical | 0.06690 | 0.02358 | 45.06923 | 0.98844 | 0.46909 | 0.72615 | 655.87 | 56.26 | 160.35 |
| 4 | guided | 0.06972 | 0.02545 | 44.71231 | 0.98793 | 0.45767 | 0.73275 | 463.15 | 47.47 | 160.35 |
| 4 | joint_bilateral | 0.06700 | 0.02359 | 45.05230 | 0.98833 | 0.46888 | 0.72576 | 504.04 | 47.47 | 160.35 |
| 4 | nearest | 0.08637 | 0.02514 | 42.87253 | 0.97934 | 0.08340 | 0.75177 | 11.08 | 8.00 | 160.35 |
| 8 | bicubic | 0.10203 | 0.03900 | 41.45659 | 0.97429 | 0.39921 | 0.50743 | 17.55 | 7.94 | 160.48 |
| 8 | bilinear | 0.11144 | 0.04326 | 40.69146 | 0.97288 | 0.40451 | 0.46396 | 16.33 | 7.94 | 160.48 |
| 8 | classical | 0.10918 | 0.04340 | 40.83676 | 0.97539 | 0.41052 | 0.56292 | 791.29 | 56.26 | 160.52 |
| 8 | guided | 0.10967 | 0.04430 | 40.80454 | 0.97594 | 0.40940 | 0.58929 | 646.33 | 47.47 | 160.48 |
| 8 | joint_bilateral | 0.10936 | 0.04345 | 40.82250 | 0.97521 | 0.41028 | 0.56103 | 651.24 | 47.47 | 160.48 |
| 8 | nearest | 0.13557 | 0.04737 | 38.91604 | 0.95393 | 0.02826 | 0.48721 | 13.54 | 7.94 | 160.48 |

## sparse_completion

| Scale | Method | RMSE m | MAE m | PSNR dB | SSIM | Edge preservation | Boundary F1 | Latency ms | Traced peak MiB | Lifetime RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | bicubic | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 23.50 | 9.38 | 162.18 |
| 2 | bilinear | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 23.28 | 9.38 | 162.18 |
| 2 | classical | 0.21627 | 0.06050 | 35.38107 | 0.97384 | 0.32483 | 0.70948 | 916.76 | 56.26 | 162.38 |
| 2 | guided | 0.21305 | 0.06007 | 35.52211 | 0.97573 | 0.35808 | 0.70830 | 742.99 | 47.47 | 162.18 |
| 2 | joint_bilateral | 0.21696 | 0.06073 | 35.35523 | 0.97333 | 0.32291 | 0.70881 | 743.30 | 47.47 | 162.18 |
| 2 | nearest | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 23.43 | 9.38 | 162.18 |
| 4 | bicubic | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 23.35 | 9.38 | 163.45 |
| 4 | bilinear | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 22.39 | 9.38 | 163.45 |
| 4 | classical | 0.21627 | 0.06050 | 35.38107 | 0.97384 | 0.32483 | 0.70948 | 806.53 | 56.26 | 163.47 |
| 4 | guided | 0.21305 | 0.06007 | 35.52211 | 0.97573 | 0.35808 | 0.70830 | 654.11 | 47.47 | 163.45 |
| 4 | joint_bilateral | 0.21696 | 0.06073 | 35.35523 | 0.97333 | 0.32291 | 0.70881 | 656.37 | 47.47 | 163.45 |
| 4 | nearest | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 23.37 | 9.38 | 163.45 |
| 8 | bicubic | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 24.22 | 9.38 | 164.73 |
| 8 | bilinear | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 23.65 | 9.38 | 164.73 |
| 8 | classical | 0.21627 | 0.06050 | 35.38107 | 0.97384 | 0.32483 | 0.70948 | 932.38 | 56.26 | 164.73 |
| 8 | guided | 0.21305 | 0.06007 | 35.52211 | 0.97573 | 0.35808 | 0.70830 | 765.25 | 47.47 | 164.73 |
| 8 | joint_bilateral | 0.21696 | 0.06073 | 35.35523 | 0.97333 | 0.32291 | 0.70881 | 751.38 | 47.47 | 164.73 |
| 8 | nearest | 3.31160 | 3.01231 | 10.33850 | 0.02203 | 0.00268 | 0.23433 | 25.23 | 9.38 | 164.73 |

Sparse completion retains 480x640 spatial resolution; its scale parameter is conditioning metadata. Classical outputs can be identical across those scales by design.

Full per-image precision/recall, provenance, source/manifest hashes, configurations, timings and metrics are in the accompanying JSON/CSV files.
Raw NPZ predictions and intermediate arrays remain under `artifacts/nyu/validation_baselines/` and are excluded from Git.

Official filled depth is an inpainted reference, not an independently measured dense sensor ground truth.
Do not use these eight development scenes to claim full NYU test performance or cross-dataset generalization.

## Observed dense-SR trade-off

Bicubic interpolation has lower macro-average RMSE than the current classical refinement at all three scales on this frozen subset. The classical pipeline therefore does not uniformly improve depth accuracy.
Boundary F1 and runtime must be read alongside RMSE; no parameters were changed to hide this result.
