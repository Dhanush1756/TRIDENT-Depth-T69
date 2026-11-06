# Colab pilot workflow (prepared; no training launched)

Windows remains the CPU development and FastAPI environment. All real neural
training runs on a Colab CUDA GPU after explicit approval. Open
`notebooks/colab_dense4_baseline_pilot.ipynb` using Colab's Upload notebook action.
The notebook has eight sections: settings/runtime, Drive, reviewed source,
dependencies/GPU, dataset staging/audit, pilot review/provenance, guarded
launch/resume, and guarded CPU export. Both execution flags default to false.

## Code, dataset and runtime

GitHub contains code, configurations and notebooks. The local working tree currently
contains unpublished files: cloning the existing remote alone will not retrieve
these additions. Publish reviewed changes yourself, then set the notebook's
`GIT_COMMIT` to the exact full commit. The notebook rejects a different or dirty
checkout. Alternatively, upload a source archive of the reviewed local files;
its SHA256 becomes the source identity. This task does not push to GitHub.

Drive stores the uploaded dataset archive, environment record, package snapshot
and persistent training checkpoints under `MyDrive/TRIDENT-T69/`. Colab holds
disposable code, the existing Colab Python runtime and an extracted local dataset
cache under `/content`. Dataset reads use the local cache; checkpoints are written
directly to Drive. Recreate the cache after a runtime reset. This follows Colab's
[recommendation to extract dataset archives locally](https://research.google.com/colaboratory/faq.html),
reducing small Drive reads. GPU allocation and runtime lifetime are not guaranteed.

## Windows preparation commands

Run from the repository root. These commands package existing files; they do not
download data, train, or change benchmark records. Archives are never overwritten.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/pack_colab.py --kind dataset --output artifacts/colab/nyu_converted.zip
.\.venv\Scripts\python.exe scripts/pack_colab.py --kind source --output artifacts/colab/trident_source.zip
```

Upload `nyu_converted.zip` to `MyDrive/TRIDENT-T69/`; upload `trident_source.zip`
only for archive mode. Copy printed SHA256 values into notebook settings. The ZIP
contains converted files directly, including original manifests, provenance,
RGB, metric depth, masks and raw sensor arrays. No MAT download is needed in Colab.
Allow several GB for the archive and extracted dataset; archive size depends on
compression. Packaging prints exact bytes; extraction checks free space using
the ZIP's uncompressed sizes. Keep the original local conversion intact.

## Environment and configuration

Use a Python 3.10–3.13 Colab GPU runtime, including Python 3.13.16. Setup uses
`PY = sys.executable` directly; it does not create a virtual environment or invoke
ensurepip. Colab's CUDA-enabled PyTorch (>=2.6,<3) must already be installed and
`torch.cuda.is_available()` must return true before dependency installation.

Setup resolves the additional `requirements-colab.txt` packages with a pip dry-run
report and rejects any plan replacing torch, torchvision, torchaudio, torchtext,
Triton or NVIDIA packages. It installs the resolved wheel URLs with `--no-deps`,
so pip cannot introduce an unreviewed PyTorch replacement through another
dependency. Already satisfied packages are retained. See the official
[pip install options](https://pip.pypa.io/en/stable/cli/pip_install/) for dry-run
reports and dependency suppression. The repository installs editable with
`--no-deps --no-build-isolation`, using the existing build tools. No CPU Torch
requirements are installed. The torch version, CUDA version, module location and
CUDA availability are compared before and after installation.

Pinned NumPy 2.2.6 and Pillow 11.3 support newer Colab Python versions; Windows
packages and all approved experiment JSON files remain unchanged. All notebook
commands use the same Colab executable. Fresh subprocesses import the installed
dependencies, avoiding stale imports from the notebook kernel. The complete test
suite runs on CPU after installation, before allowing a training launch.

The notebook records exact Python, torch/CUDA, numerical package versions and GPU,
source identity, dataset archive SHA256, experiment SHA256 and pipeline SHA256.
At the first approved launch it saves this record next to the run and also saves
`pip freeze`. Reconnection requires an identical record; if Colab changes the
base runtime, restore a matching environment rather than silently continuing.
The full freeze includes inherited Colab packages for inspection; do not blindly
install it on Windows. Colab installation and GPU execution have not been verified
from this local CPU machine. Package installation errors or test failures stop
the notebook before training.

The unmodified pilot remains dense 4x, lightweight CNN, bicubic initializer,
two epochs, batch 1, Adam learning rate 1e-4, seed 69, FP32 and full 480x640
frames. A constant scheduler preserves the approved learning rate. CLI runtime
overrides `--manifest`, `--output`, and `--device auto|cpu|cuda` relocate storage
and select hardware without changing the stored experiment identity. The runner
defaults to automatic CUDA/CPU selection; the Colab notebook additionally refuses
pilot training without CUDA. Local CPU testing remains supported.

The approved development manifest must retain SHA256
`d1cdbcf0ff3ee96f2e0c173c3c4701ed3f5c526ab517ce3b988268c3c6a391f5`.
Checks cover scene separation, official split labels, referenced RGB/depth/mask
files and counts: 672 train, 123 validation, 654 locked test. Manifest audit reads
test metadata only. Training and model selection never load official test pixels.
Validation uses all 123 development validation frames and global valid-pixel RMSE
in metres. Existing classical benchmark results remain untouched.

## Approval and disconnection recovery

Leave `TRAINING_APPROVED=False` until explicitly approved. After approval, change
the flag and execute the launch cell. It invokes the existing runner with its
additional `--execute-training` opt-in. Setup cells alone never train.

Every completed epoch saves model weights, Adam state, constant scheduler state,
Python/NumPy/CPU/CUDA/DataLoader RNG states, completed history and dataset/config
provenance in `runs/pilot_dense4_lightweight_seed69/epoch_XXX.pt`. Checkpoints use
temporary files and rename; `.part` files are incomplete and must not be resumed.
Drive still has service/flush failure risks; verify checkpoint availability after
each epoch. An interruption before a complete checkpoint loses that epoch's work.

Reconnect with identical source/environment/settings, mount Drive, rebuild the
dataset cache, keep the same RUN_ID and set `RESUME="epoch_001.pt"` for a completed
first epoch. The total budget remains two epochs, so this runs epoch two only.
Use the latest complete checkpoint. The runner restores scheduler/optimizer/RNG
and repairs missing or partial trailing history from the checkpoint, rejecting
conflicting or newer history. It does not resume partway through an epoch.
If disconnected before the first completed epoch, there is no resumable
checkpoint: choose a new RUN_ID and repeat the pilot from epoch one. Preserve
the interrupted directory for inspection.
PyTorch/CUDA version or GPU changes can prevent exact replay; the notebook refuses
a changed environment. Legacy checkpoints without scheduler state are rejected
by the updated training runner rather than silently losing state. Inference
loading remains backward compatible.

After epoch two there are no remaining training epochs: export/review rather than
requesting a resume. The export cell can find the best epoch from the latest
checkpoint even if a disconnection interrupted writing the best-checkpoint pointer.

## CPU application export

After reviewing real pilot results, enable `EXPORT_APPROVED`. The exporter accepts
only an existing checkpoint marked trained, loads it on CPU, and writes the same
version-1 model/config/provenance format used by FastAPI. It omits optimizer and RNG
to make an inference-only artifact and records the source checkpoint SHA256.
It refuses to overwrite exports and verifies loading with the application's
checkpoint loader. Export does not evaluate the test set or create benchmark results.

Download `baseline_dense4_inference.pt` into local `checkpoints/`, then run:

```powershell
$env:TRIDENT_CHECKPOINT = (Resolve-Path checkpoints/baseline_dense4_inference.pt).Path
$env:TRIDENT_DEVICE = 'cpu'
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The artifact and end-to-end CPU application inference can be verified only after
the first real training checkpoint exists. Current tests exercise serialization,
resume mechanics and export rejection of untrained weights without optimizer steps.
