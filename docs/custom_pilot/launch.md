# Approved custom residual dense 4x pilot: exactly two epochs

This approval covers the custom residual component of TRIDENT, not the complete
hybrid fusion system or a longer experiment. No official test evaluation or
inference-weight export is included. Training has not been launched locally.

## Verified configuration

`configs/phase5/pilot_dense4_residual.json` copies the approved
`dense_sr_4x_residual_cnn.json` and changes only `epochs` (20 to 2) and `output`
(a separate pilot directory). The original approved configuration is unchanged.

| Setting | Pilot |
| --- | --- |
| Architecture | Existing ResidualCNN, 473,059 parameters |
| Encoder / bottleneck / decoder channels | 32, 64 / 128 / 64, 32, with skip concatenation |
| Decoder upsampling | nearest, already in the approved custom config |
| Outputs | positive metric depth residual refinement, clipped log-scale uncertainty, boundary confidence |
| Input / output | 120x160 depth / 480x640 depth with aligned 480x640 RGB |
| Protocol | Dense 4x, nonoverlapping area-mean degradation, bicubic initializer |
| Optimizer | Adam, lr 1e-4, betas (0.9, 0.999), eps 1e-8, weight decay 0 |
| Batch / precision / seed | 1 / FP32, no AMP / 69 |
| Loss weights | depth 1, gradient 0.2, edge 0.1, uncertainty 0.05, measurement 0.2 |
| Scheduler / gradient clip | Constant learning rate / norm 5 |
| Epochs | Exactly 2, validation after each |
| Training / validation / locked test | 672 / 123 / 654 frames; 219 / 30 / 215 scenes |
| Model selection | Lowest global valid-pixel validation RMSE, earliest epoch on ties |

No architecture-specific hyperparameter changes are needed. Nearest decoder
upsampling was already approved for deterministic CUDA backward; see
[PyTorch's deterministic algorithm notes](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html).
The notebook additionally checks actual CUDA forward/backward compatibility
before the first optimizer step.

The local full-frame CPU diagnostic in `preflight_cpu.json` verified the exact
parameter count, finite losses and active gradients, unchanged model parameters,
and model/optimizer/scheduler/RNG serialization and restoration. It performed
zero optimizer steps and saved no trained checkpoint. CUDA execution cannot be
verified from this CPU host; the notebook repeats this preflight on its GPU.
The complete local test suite passed: 140 tests, with one existing upstream
Starlette/httpx deprecation warning. Original architecture/configuration files,
the baseline notebook/audit, and existing classical results retained their hashes.

The approved uncertainty head receives Laplace-loss gradients; unlike the baseline,
its uncertainty loss can be nonzero and can legitimately be negative. The
boundary-confidence head has no direct supervision in the approved loss. This is
preserved explicitly. Reported boundary F1 measures boundaries in reconstructed
depth, not the unsupervised confidence head.

## Upload the new source, reuse the existing dataset

The previous source ZIP lacks the custom notebook and per-epoch boundary metric
recording. Build a new source archive; keep the baseline source archive intact:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/pack_colab.py --kind source --output artifacts/colab/trident_custom_pilot_source.zip
```

Upload that archive to `MyDrive/TRIDENT-T69/trident_custom_pilot_source.zip` and
copy the command's printed SHA256 into the notebook's `SOURCE_SHA256` setting.
If the destination archive already exists, choose a new archive filename and
update the notebook path rather than overwriting a reviewed artifact. The source
archive includes the baseline audit's fixed per-frame identities and masks under
`docs/pilot_dense4/per_frame.json`.

Reuse the existing `nyu_converted.zip` on Drive. Its SHA256 is already filled in:
`0e97b6d1a2ae7a83c4383566eee782dc04dc68048f9aff9590ebe0c7d9df1cd6`.
The original development manifest hash remains
`d1cdbcf0ff3ee96f2e0c173c3c4701ed3f5c526ab517ce3b988268c3c6a391f5`.
The existing audited baseline checkpoint must remain at
`MyDrive/TRIDENT-T69/runs/pilot_dense4_lightweight_seed69/epoch_002.pt`, SHA256
`a1fb5080360d9e693d43e67b3aa105da05be93af6baca6d1c84bee979ce3fb43`.
Restore that full checkpoint from the reviewed run archive if needed. Do not
create an inference export for this comparison.

## Exact Colab launch steps

1. Upload `notebooks/colab_dense4_custom_pilot.ipynb` to Colab. Use a fresh GPU
   runtime to avoid an old source checkout. Python 3.10-3.13 is supported.
2. Keep `SOURCE_MODE="archive"`, set `SOURCE_SHA256`, and leave the verified
   dataset hash intact. Adjust DRIVE_ROOT only if your Drive folder differs.
3. Run sections 1-6. They mount Drive, verify/extract the new source, retain
   Colab's existing CUDA-enabled PyTorch, install only additional dependencies,
   run all tests on CPU, verify the scene-safe split and run the GPU preflight.
   No virtual environment is created; no CPU Torch wheel is installed.
4. Set `TRAINING_APPROVED=True` in section 1 and execute that settings cell again.
   Approval is already granted for this pilot; no additional confirmation is
   needed. Set `RUN_COMPARISON=True` too if you want section 8 to run immediately
   after training. Keep `EXPORT_APPROVED=False`.
5. Execute section 7. The exact launch command is:

```python
subprocess.run([
    sys.executable, "-m", "trident_depth.training.train",
    "--experiment", str(EXPERIMENT),
    "--manifest", str(MANIFEST),
    "--output", str(RUN_DIR),
    "--device", "cuda",
    "--record-boundary-f1",
    "--provenance", str(ENV_FILE),
    "--execute-training",
], check=True)
```

The notebook writes ENV_FILE before launching. EXPERIMENT is exclusively
`configs/phase5/pilot_dense4_residual.json`; RUN_DIR is exclusively
`MyDrive/TRIDENT-T69/runs/pilot_dense4_residual_seed69`. Do not substitute the
20-epoch parent configuration. Training stops after epoch two.

6. Run section 8 with `RUN_COMPARISON=True`. It executes:

```python
subprocess.run([
    sys.executable, "-m", "trident_depth.evaluation.custom_pilot",
    "--custom-run", str(RUN_DIR),
    "--baseline-checkpoint", str(BASELINE),
    "--manifest", str(MANIFEST),
    "--reference-frames", str(REFERENCE_FRAMES),
    "--output", str(RUN_DIR / "validation_comparison"),
    "--device", "cuda",
], check=True)
```

The comparison refuses incomplete pilots and existing output folders. It
re-evaluates bicubic, baseline epoch two and the best custom checkpoint together
on precisely the same 123 validation IDs, RGB/depth files and target masks from
the baseline audit. All methods use the existing global valid-pixel RMSE/MAE
implementation, and the existing boundary metric with a 0.05 m edge threshold
and 1-pixel tolerance. Boundary F1 is the arithmetic mean of per-frame F1 values.
It verifies replay against both the original baseline audit and the custom logs.

7. Set TRAINING_APPROVED back to False after completion. Download the custom run
   folder for review, then stop. No extra experiment or inference export follows.

## Saved outputs, resume and limits

Each epoch checkpoint retains model state, Adam state, constant scheduler state,
Python/NumPy/CPU/CUDA/DataLoader RNG state, completed history, architecture options,
full experiment and pipeline config, scene/split audit, source/environment record
and its SHA256, and boundary-evaluation settings. The environment record includes
the reviewed source identity, dataset/config hashes and baseline/reference hashes.
Original model code and existing benchmark records are not overwritten.

Each history row records global validation RMSE/MAE, frame-mean boundary F1,
training/validation losses and components, epoch wall time, and peak CUDA
allocated/reserved memory. Runtime includes boundary evaluation overhead; it is
not directly comparable to the original baseline's epoch time, which omitted it.
GPU memory/time are measured in Colab, not invented from this CPU preflight.

The Drive run contains `epoch_001.pt`, `epoch_002.pt`, `history.jsonl`, `run.json`,
`best_checkpoint.json`, and after comparison a `validation_comparison/` folder
with `comparison.json`, `per_frame.json`, and `comparison.md`. Both custom epochs
remain available; selection uses validation RMSE only. The existing baseline and
classical records remain untouched.

After a disconnection, rerun setup with the same source, environment and RUN_ID.
Set `RESUME="epoch_001.pt"` to finish epoch two. The notebook rejects other resume
filenames, and the runner verifies configuration, source/environment record and
boundary settings before restoring state. Interrupted epochs are repeated from
the last completed epoch. With no completed checkpoint, use a fresh RUN_ID/path
only after preserving the interrupted folder; the two-epoch budget stays fixed.
After epoch two, run only the comparison cell; do not relaunch or extend training.

No custom pilot results exist yet. The baseline reference remains RMSE
0.0583331948 m / MAE 0.0162285306 m; bicubic remains 0.0585272133 m /
0.0162030076 m. A nonzero uncertainty objective changes the total-loss composition,
so compare reconstruction metrics rather than total losses across architectures.
