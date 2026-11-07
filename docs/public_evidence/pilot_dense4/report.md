# Dense 4x baseline CNN pilot audit

No new training. Official test pixels were not loaded. Existing classical benchmark results were preserved.

| Method | Validation RMSE (m) | Validation MAE (m) |
| --- | ---: | ---: |
| bicubic (CPU replay) | 0.0585272133 | 0.0162030076 |
| cnn_epoch_1 (CPU replay) | 0.0585125132 | 0.0165221998 |
| cnn_epoch_2 (CPU replay) | 0.0583331948 | 0.0162285306 |

Recorded Colab best: epoch 2, RMSE 0.0583331947 m, MAE 0.0162285307 m.

- rmse_m: CNN minus bicubic -0.0001940185 m; reduction 0.3315%.
- mae_m: CNN minus bicubic 0.0000255230 m; reduction -0.1575%.

Both checkpoints strictly load 11,873 model parameters, six Adam moment pairs and a constant scheduler. Adam steps are 672/1,344; scheduler epochs are 1/2. Saved history exactly matches the external log and best-epoch selection. Python, NumPy and CPU RNG states load; DataLoader RNG matches a seeded replay of each epoch's shuffle. CUDA state has the expected 16-byte layout and seed 69, but was not restored on a GPU.

Uncertainty loss is expected to be zero: the baseline has no log_scale head. The loss explicitly supplies a connected zero for that optional term. A nonzero configured uncertainty weight does not create an uncertainty head or imply learned uncertainty.

Evaluation uses exactly 123 scene-safe development validation frames (30 scenes), full 480x640, area-mean 4x degradation, identical target_valid masks, and the training runner's global valid-pixel double-precision RMSE/MAE. Bicubic is the unchanged initializer computation, without CNN refinement. The earlier eight-frame classical summary is not used as the comparator.

CPU replay agrees with both recorded validation epochs within 1e-6 m and validation loss within 1e-6. CPU software versions differ from Colab, so this is numerical verification rather than bitwise reproduction. CUDA resume remains unverified locally.

`report.json` contains full training/evaluation configurations, history, environment, source/dataset/config/checkpoint hashes, audit details and protected-file hashes. `per_frame.json` contains exact IDs, scenes, file/mask hashes and paired error sums. Imported checkpoints and original logs are preserved under `artifacts/pilot_dense4_audit`.

All 369 evaluated RGB/depth/mask files were also verified byte-for-byte against the checksummed training dataset archive. No test entries were decoded for this verification.

The RMSE reduction is small (0.3315%) and MAE regresses (0.1575%); this pilot does not demonstrate a consistent improvement across the two metrics. Longer training has not been authorized.

To reproduce into fresh directories from the repository root:

```powershell
.\.venv\Scripts\python.exe -m trident_depth.evaluation.pilot_audit --archive "${USER_HOME}\Downloads\runs-20261106T111552Z-1-001.zip" --output artifacts/pilot_dense4_audit_repeat --report-dir docs/pilot_dense4_repeat
.\.venv\Scripts\python.exe -m pytest -q
```

After the audit, the complete test suite passed: **135 passed**, zero failures, one existing upstream Starlette/httpx deprecation warning. `pip check` found no broken requirements. Protected configuration/model/classical-result hashes and imported checkpoint hashes were rechecked after tests and remained unchanged.
