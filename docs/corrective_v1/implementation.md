# A1 and B0 implementation and preflight report

Implementation and preparation are complete. **Neither experiment has training approval.** No optimizer steps, trained checkpoint writes, official test evaluation or inference export were performed. These checks establish compatibility and initialization properties, not improved reconstruction quality.

Only A1 and B0 are implemented. No other proposed corrective row is executable through this workflow.

## Frozen experiment contract

| Setting | A1 | B0 |
| --- | --- | --- |
| Model/checkpoint identity | residual_cnn_v2 | residual_cnn_v2 |
| Contract identity | A1_B0_v1 / A1 | A1_B0_v1 / B0 |
| Final residual weights/bias | exactly zero | exactly zero |
| Fixed alpha | 1 | 1 |
| Objective | L_J | L_R |
| Uncertainty forward/graph | active | never executed |
| Boundary-confidence head | retained, unsupervised | retained, unsupervised |
| Total parameters | 473,059 | 473,059 |

Topology remains encoder 32/64, bottleneck 128, decoder 64/32, skip concatenations, nearest decoder upsampling and the original three heads. All original constructor RNG draws occur before zeroing the final residual head. Zeroing consumes no additional RNG draws. Raw/applied residuals are exposed separately; with alpha=1 they coincide. Positive-depth output retains the 1e-6 m floor. Exact identity applies to bicubic values at/above that floor.

Both rows retain full-frame dense 4x area degradation, bicubic initialization, FP32, RGB in [0,1], depths in metres, seed 69, Adam lr=1e-4, betas=(0.9,0.999), eps=1e-8, weight decay=0, batch size=1, clipping norm=5 and constant LR. Frozen budgets are two epochs with validation after each epoch and earliest minimum global validation RMSE selection. Config and pipeline edits are rejected by the separate runner. The pipeline is checked against independently frozen values, including the existing 10 m validity ceiling.

The existing loss implementations/defaults are reused unchanged:

```text
L_R = L_depth + 0.2 L_gradient + 0.1 L_edge + 0.2 L_measurement
L_J = L_R + 0.05 L_uncertainty
L_measurement = masked mean |avg_pool_4(prediction) - observed_LR|
```

B0 removes only the uncertainty objective. Its log-scale head is not called, its parameters receive grad=None and it supplies no uncertainty for hybrid inference. A B0-only inference guard prevents silently treating unavailable uncertainty as calibrated confidence. The parameters remain present for exact topology/count and paired initialization. Neither row supervises boundary confidence; there is no new boundary target or auxiliary objective.

## Data and provenance verification

The approved manifest SHA256 remains `d1cdbcf0ff3ee96f2e0c173c3c4701ed3f5c526ab517ce3b988268c3c6a391f5`. Counts remain 672 training frames / 219 scenes, 123 validation frames / 30 scenes and 654 locked official test frames / 215 scenes. The manifest audit verifies scene separation and official split labels. It checks file existence for all records without loading test arrays.

The original 123-frame reference SHA256 is also fixed. Validation IDs/order, scenes and RGB/depth/mask file SHA256 values were checked against it, without evaluating validation predictions in this task. Future validation additionally checks the exact tensor mask SHA256. Per-frame/scene identifiers, squared/absolute error sums, valid counts and boundary F1 are saved in each future epoch history for a paired audit. The unchanged original metric implementations calculate global valid-pixel RMSE/MAE and frame-mean boundary F1 (0.05 m threshold, 1 pixel tolerance).

Checkpoints serialize the model identity/configuration, model tensors, Adam moments/step counters when present, scheduler state, Python/NumPy/CPU/CUDA/DataLoader RNG state, history, epoch/best epoch, objective/head availability, split hash, source/environment record and config/pipeline provenance. Fresh construction zeros the head; loading then restores saved tensors strictly, without resetting a learned head. Resume requires matching experiment identity, pipeline, manifest and recorded PyTorch version. New workflow resumes remain bounded at epoch two.

## Verified CPU preflight

The machine ran Python 3.10.8 with PyTorch 2.14.1+cpu. See [the measured preflight JSON](cpu_preflight_verified.json), including source-file SHA256 values, configurations, file provenance, losses, individual tensor gradient norms and temporary checkpoint round-trips.

Each row passed one synthetic and two real 480x640 training-frame forward/backward probes: `nyu_00003` (office_0003) and `nyu_00005` (office_0004), with 120x160 dense observations. They were the same frames for both rows.

- Exact bicubic identity and zero residual magnitude on every probe.
- Identical complete initial state tensors across A1/B0; all non-residual-head tensors and post-constructor RNG match the legacy constructor with seed 69.
- Finite losses and active gradients; parameters unchanged after every forward/backward probe; zero optimizer steps.
- Both residual heads receive nonzero reconstruction gradients.
- Reconstruction gradients to the backbone are zero initially in both rows because the residual weights are zero. A1's uncertainty branch supplies nonzero backbone gradients. B0's total backbone gradients are zero, and its uncertainty-head gradients are None. The boundary-head gradients are None in both.
- Model/optimizer/scheduler and Python/NumPy/CPU/shuffle RNG round-trips pass. Preflight Adam moments are honestly empty because no optimizer step occurred. Unit fixtures separately verify nonempty synthetic moments and completed-epoch scheduler/history state without stepping an optimizer.
- A fixed training-only probe records weighted per-term depth-output/shared-backbone gradient norms and cosines, plus A1 scale-decile and edge/flat coverage summaries. These are training diagnostics, not independent uncertainty calibration.

For `nyu_00003`, the enc1 first-weight gradient norm is 0.00196414 for A1 and exactly 0 for B0. Residual-head weight norms are 0.000750544 and 0.000739011 respectively. These derivatives verify expected connectivity; they are not evidence of efficacy.

CPU checks cannot establish CUDA numerical behavior, actual GPU memory or throughput. The notebook runs the same paired preflight on the selected Colab GPU, explicitly requires CUDA, checks CUDA RNG round-trip, and records allocated/reserved memory before any training launch. GPU execution is prepared but has not been performed here.

## Tests and preservation

The complete suite passes: **171 tests**, comprising all existing 144 plus 27 new tests. One existing Starlette/httpx deprecation warning remains. Tests cover paired RNG/tensors, identity, first-step gradients, subsequent backbone connectivity with a synthetic nonzero-head fixture, explicit uncertainty exclusion, dense measurement parity, invalid/empty masks, serialization, strict loading, resume state, legacy prediction parity, frozen scope/pipeline, unavailable hybrid uncertainty and disabled notebook execution.

The preservation record is [verification.json](verification.json). Only two existing files were edited: additive model registration/configuration serialization in `models/checkpoint.py`, and the B0-specific unavailable-uncertainty guard in `inference.py`. Legacy model classes, losses, training runner, approved phase5 configurations, notebooks, diagnostic/design reports, stored pilot checkpoints and classical benchmark JSONs retain their original bytes. Temporary serialization fixtures are untrained and automatically removed. No historical run directory is changed.

## Files introduced

| File | Purpose |
| --- | --- |
| trident_depth/models/residual_v2.py | Separate model identity, original construction order, exact-zero head, alpha=1, optional graph exclusion |
| trident_depth/training/corrective.py | Frozen A1/B0 contracts and isolated objective composition using existing losses |
| trident_depth/training/corrective_data.py | Exact reference validation IDs/scenes/file hashes |
| trident_depth/training/corrective_monitoring.py | Fixed training-only gradient and calibration probes, with no parameter/RNG updates |
| trident_depth/training/corrective_preflight.py | Paired synthetic/real CPU or GPU zero-step compatibility checks and temporary resume audit |
| trident_depth/training/corrective_train.py | Separately gated, two-epoch-only runner, validation, monitoring and complete checkpoint/history recovery |
| configs/corrective_v1/A1.json and B0.json | Separate immutable experiment identities/output directories; reuse the original pipeline file |
| tests/test_corrective_v1.py | New invariant and compatibility tests |
| notebooks/colab_corrective_A1_B0.ipynb | CUDA-preserving setup, full tests, verified data staging, paired GPU preflight, two disabled launch flags |
| docs/corrective_v1 | This report, measured CPU preflight and preservation/test verification |

The new runner logs raw/weighted loss components; correction mean/absolute/RMS/p50/p90/p99/max and improvement fraction; MSE correction/cross terms; A1 uncertainty scale/clamp/coverage; clipping frequency/norm; runtime; learning rate; and CUDA peak allocation/reservation. The fixed training probe adds regional/decile calibration and gradient diagnostics each epoch. Per-frame statistics are available in validation history; aggregated component and quantile entries are averages of batch statistics, not pooled distribution quantiles. The report must retain that distinction during later comparison. Raw and applied correction statistics coincide because alpha is fixed to one.

## Exact local preparation commands

Run from the repository root in PowerShell; these commands do not train:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m trident_depth.training.corrective_preflight --manifest data/nyu/converted/manifest_development.json --device cpu --output artifacts/A1_B0_cpu_recheck_1.json
Get-FileHash -Algorithm SHA256 artifacts/colab/trident_A1_B0_source.zip
```

The preflight refuses to overwrite a report; choose another new report path for repeat runs. A reviewed source archive is supplied at `artifacts/colab/trident_A1_B0_source.zip`. To package a later reviewed revision, run `python scripts/pack_colab.py --kind source --output artifacts/colab/trident_A1_B0_source_reviewed_v2.zip` and use that archive's name/SHA256. The packer refuses existing archive paths. Do not replace historical source archives.

## Exact Colab preparation and later launch instructions

1. Upload `notebooks/colab_corrective_A1_B0.ipynb` to Colab and select a GPU runtime. Both `TRAIN_A1_APPROVED=False` and `TRAIN_B0_APPROVED=False`, plus `EXPORT_APPROVED=False`, must remain false for preparation.
2. Upload the new source ZIP to `MyDrive/TRIDENT-T69/trident_A1_B0_source.zip`. Keep the existing verified `nyu_converted.zip` beside it. No baseline/custom checkpoint upload is required; these rows start fresh.
3. Leave `SOURCE_MODE="archive"` and paste the local `Get-FileHash` value into `SOURCE_SHA256`. Keep the existing dataset hash unchanged. The ZIP contains repository files directly, without an enclosing folder.
4. Alternatively publish the reviewed local repository yourself, choose `SOURCE_MODE="github"` and set the full 40-character reviewed commit. A moving branch or local unpublished changes are insufficient. Both rows must use the same source/runtime/GPU for a controlled pair.
5. Run the settings, mount, source, dependency/test, dataset and paired GPU-preflight sections. Setup uses `sys.executable` in Colab's existing Python, preserves CUDA PyTorch, installs only the additional approved requirements and installs the repository with `--no-deps --no-build-isolation`. Source/dataset SHA256 and split checks remain enforced.
6. Review the GPU JSON saved under `MyDrive/TRIDENT-T69/preparation/A1_B0/`. It must show `cuda_verified=true`, `optimizer_steps=0`, exact identity, finite expected gradients, matching constructor tensors/RNG and successful checkpoint round-trips. Setup also runs the complete project test suite on CPU in Colab. Any failure blocks launch.
7. You can run the remaining notebook cells with both flags false: both launch helpers return before creating a training run or environment record. **Stop and obtain separate training approval.**
8. Only after explicit approval for A1, set `TRAIN_A1_APPROVED=True`, retain `TRAIN_B0_APPROVED=False`, then execute the A1 launch cell. Only after explicit approval for B0, enable its flag and execute its cell, with A1 disabled. A notebook flag is a technical guard, not an authorization grant.
9. Each approved launch creates its own environment JSON/package freeze and invokes `python -m trident_depth.training.corrective_train` with its frozen config, local-cache manifest, Drive output directory, `--device cuda --record-boundary-f1 --provenance <row environment JSON> --execute-training`. The paths are `configs/corrective_v1/A1.json` / `B0.json` and `MyDrive/TRIDENT-T69/runs/corrective_v1_A1_dense4_seed69` / `corrective_v1_B0_dense4_seed69`. Nothing continues beyond epoch two.
10. After a disconnection, restore the exact source/environment/device, rerun preparation, and set only the approved row's `RESUME` entry to `"epoch_001.pt"`. Its launch adds `--resume <same Drive run>/epoch_001.pt`, restores all states and runs epoch two only. A partial epoch is replayed from the last complete checkpoint. Without any complete checkpoint, preserve the interrupted directory and review a fresh output path instead of overwriting it. Epoch-two checkpoints cannot extend the budget. Source/environment mismatches deliberately block resume.

Download the GPU preparation JSON for review now. Training A1, training B0, later paired efficacy analysis and any additional experiment/export/test evaluation remain pending their appropriate approval. Neither A1 nor B0 has an observed trained result.
