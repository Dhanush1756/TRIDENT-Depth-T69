# TRIDENT-Depth 2.0 · Team T69

Google Colab GPU training preparation: [pilot notebook](notebooks/colab_dense4_baseline_pilot.ipynb)
and [Windows/Drive/Colab setup and resume guide](docs/colab_training.md).
Training is disabled by default and awaits explicit approval.

The approved two-epoch custom pilot has a [separate Colab notebook](notebooks/colab_dense4_custom_pilot.ipynb)
and [exact launch instructions](docs/custom_pilot/launch.md). It preserves the
473,059-parameter architecture and does not authorize longer training or test evaluation.

Trust-Routed Intelligent Depth Enhancement Network for Hackathon 2026 P4.

**Status:** validated data pipeline, classical reconstruction and metrics;
untrained model definitions and gradient/checkpoint tests; training/evaluation
tooling; working local classical API and frontend. No final trained-model
results or hosted URL exist. See [phase status](docs/phase_status.md) and the
[official PDF audit](docs/pdf_requirements_audit.md) for incomplete gates.

Official NYU dataset preparation is documented in [NYU integration](docs/nyu_preparation.md).
The source MAT format, official split and scene-level development holdout are
kept distinct. Neural training remains disabled until explicit approval.
All 1,449 NYU pairs are converted and audited. Read the
[dataset integrity report](docs/nyu/integrity_report.md) and
[measured classical validation results](docs/nyu/validation_results.md).
There are 288 actual validation records; final test-set and trained-model
performance are still unmeasured.

Phase 5 review materials are in the [training preparation plan](docs/training_preparation/plan.md):
audited baselines, separate protocol/scale/model configs, checkpoint-resume
checks and explicit GPU budget assumptions. The two-epoch dense 4x baseline CNN
pilot completed in Colab; see its [checkpoint audit and paired validation
comparison](docs/pilot_dense4/report.md). Longer and residual-network training
remain paused pending approval. Official test performance remains unmeasured.

## Windows setup and run

From PowerShell in the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000**. FastAPI serves the frontend on the same port;
no separate frontend server or Node build is required. API docs are at `/docs`.
These commands do not require PowerShell environment activation or the `py`
launcher. Existing `.venv` users can skip creation. For the complete model test
suite, first install CPU PyTorch:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-learning-cpu.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

Model tests skip explicitly if Torch is absent. Backend tests skip explicitly
if app dependencies are absent. A Windows CPU environment snapshot is in
`docs/environment_windows_cpu.txt`; it includes the optional PDF audit reader.
Do not install the CPU Torch requirements over a Colab/Kaggle GPU runtime.
OpenCV 4.11 is selected to work with NumPy 1.26; the global OpenCV 4.12 install
requires NumPy 2 and must not leak into this virtual environment.

## Real-file smoke check

This optional command fetches only a small Open3D/Redwood sample archive, capped
at 10 MiB and checked against its upstream hash. It is not a NYU/KITTI training
dataset. It refuses to overwrite an existing smoke directory.

```powershell
.\.venv\Scripts\python.exe scripts/fetch_smoke_sample.py
.\.venv\Scripts\python.exe -m trident_depth.cli --manifest data/smoke_redwood/manifest.json --config configs/dense_sr.json --split test --output artifacts/my_dense4
.\.venv\Scripts\python.exe -m trident_depth.cli --manifest data/smoke_redwood/manifest.json --config configs/sparse_completion.json --split test --output artifacts/my_sparse
```

Each new output directory contains `sample.npz` (metric depth, separate bool
masks, optional RGB) and `metadata.json` (shape, protocol, seed and provenance).
Use a new output directory when rerunning. Synthetic arrays in tests are
clearly labelled test fixtures, not experimental evidence.

With the server running in another terminal, the real-file HTTP verification
can be repeated with `python scripts/verify_local_app.py --sample
artifacts/my_dense4/sample.npz --output artifacts/my_http_smoke.json` using the
virtual environment's Python executable.

## Dataset preparation

Read [data contracts](docs/data_pipeline.md). Export registered RGB, numeric
depth and optional masks into PNG/TIFF/NPY files and provide a JSON manifest.
All depth conversion factors and scene/split labels must be explicit. Preserve
official splits and audit scene overlap. Data is not downloaded automatically.
The official labelled NYU MAT/HDF5 has a dedicated converter and integrity audit;
see [NYU preparation](docs/nyu_preparation.md). KITTI native integration remains unverified.
Middlebury disparity is not metric depth without calibration.

Dense SR uses area downsampling and requires fully valid reference depth and
dimensions divisible by 2, 4 or 8. Sparse/noisy completion uses a different
same-resolution sampling protocol. See configs for density, hole and noise
controls; its scale parameter does not reduce spatial resolution.

## Classical evaluation

```powershell
.\.venv\Scripts\python.exe -m trident_depth.evaluation.benchmark --manifest data/smoke_redwood/manifest.json --config configs/dense_sr.json --output artifacts/my_benchmark_dense --label smoke_only_redwood
.\.venv\Scripts\python.exe -m trident_depth.evaluation.benchmark --manifest data/smoke_redwood/manifest.json --config configs/sparse_completion.json --output artifacts/my_benchmark_sparse --label smoke_only_redwood
.\.venv\Scripts\python.exe -m trident_depth.evaluation.failure_analysis --report artifacts/my_benchmark_dense/results.json --output artifacts/my_diagnostics
```

Each benchmark covers all three scales and six classical methods, saves raw
arrays/intermediates and records actual measurements. Set `--split` for a
different partition. See [metric definitions](docs/evaluation_protocol.md).
Diagnostics do not substitute for three independent real failure cases.

## Models and training — not executed

`LightweightCNN` is the explicit baseline, and `ResidualCNN` is the custom
multiscale model. Neither is DKN/FDKN. Architecture and tradeoffs are documented
in [architecture](docs/architecture.md) and [design](docs/design_document.md).

The training runner is implemented but has **not** been executed. It requires
nonempty audited train and validation data and an explicit opt-in flag. Before
using it, inspect dataset units/splits, choose a verified GPU runtime and edit
`configs/train_nyu.json` paths/device/run name. Its default CUDA setting is
deliberate: this machine's installed CPU-only Torch cannot run that setting.
Training remains subject to the project's approval gate.

```powershell
# Run only after dataset preparation and training authorization.
.\.venv\Scripts\python.exe -m trident_depth.training.train --experiment configs/train_nyu.json --execute-training
```

The runner records seeds, configuration, manifest SHA-256, batch-loss histories
and epoch checkpoints. Use validation data to select a checkpoint. Configure
separate runs for each scale/protocol and baseline/custom model, with distinct
output directories. Default batch size 1 accommodates varying image sizes and
RGB availability. Save checkpoints outside ephemeral GPU notebook storage.

On Colab/Kaggle, clone this repo, install `requirements-app.txt` using `python
-m pip install -r requirements-app.txt`, retain the runtime's GPU-enabled Torch,
verify `torch.cuda.is_available()`, mount data/persistent output and run the same
module command with notebook-appropriate paths. No notebook session is required
for production serving. Hosted training has not been tested here.

With an actual trained checkpoint, compare and ablate it:

```powershell
# Replace this example path with an actual validated checkpoint.
.\.venv\Scripts\python.exe -m trident_depth.evaluation.benchmark --manifest data/nyu/converted/manifest_development.json --config configs/dense_sr.json --split validation --checkpoint artifacts/training/nyu_residual_scale4/epoch_020.pt --ablations --output artifacts/nyu_comparison --label nyu_development_validation
```

The harness records each scale separately. A checkpoint trained only at 4× is
not automatically a trained multiscale model. Report such other-scale results
as generalization, or evaluate matching checkpoints in separate reports.
Inference ablations do not replace separately trained loss/head ablations.

## Application inputs and downloads

- Low-resolution mode outputs H×scale by W×scale; optional RGB/reference must
  already match that target resolution and camera frame.
- Completion mode keeps H×W, with optional same-size RGB/reference.
- Numeric depth accepts PNG/TIFF/NPY; masks accept 0/1/255. RGB must be RGB-mode
  PNG/JPEG. Set metres-per-stored-unit from source documentation.
- Inputs and target have a 1-megapixel cap, uploads an 8 MiB cap per file. The
  server rejects empty support, bad shapes, object arrays and invalid settings.
- Accuracy metrics appear only with valid ground truth. Runtime is a single
  reconstruction measurement; use the benchmark for repeated measurements.
- NPZ contains float32 metric depth, validity and intermediate arrays. PNG is
  uint16 millimetres and is omitted for ranges above 65.535 m.
- Deep/hybrid choices stay disabled without a trained checkpoint. To enable:

```powershell
$env:TRIDENT_CHECKPOINT = 'checkpoints/your_trained_checkpoint.pt'
$env:TRIDENT_DEVICE = 'cpu'
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

## Docker and deployment

```powershell
docker build -t trident-depth .
docker run --rm -p 8000:8000 trident-depth
```

The default image serves classical reconstruction. For neural serving, build
with `--build-arg INSTALL_LEARNING=1` and mount a trained checkpoint read-only
under `/checkpoints`, setting `TRIDENT_CHECKPOINT` to its container path.
Docker Desktop must be running; build/run is currently unverified because the
local daemon is unavailable. Do not commit checkpoints, datasets or secrets.

A public Python/Docker-capable host is still needed; Sites/Cloudflare Worker
hosting cannot run this FastAPI/PyTorch backend. Run one worker under the
current memory/concurrency design, set reverse-proxy body/rate limits, test
phone/laptop uploads and measure actual production latency before submission.
No public deployment or Git push was performed.

## Files and responsibilities

| Path | Purpose |
| --- | --- |
| `trident_depth/config.py`, `logging_utils.py` | Validated JSON configuration and explicit logging setup |
| `trident_depth/data/sample.py`, `io.py` | Array contracts, units and RGB/depth/mask loading |
| `trident_depth/data/preprocessing.py`, `degradation.py` | Aligned mask-aware resizing; separate dense/sparse degradations |
| `trident_depth/data/splits.py`, `dataset.py` | Scene/file leakage checks and lazy split-specific access |
| `trident_depth/data/nyu.py`, `nyu_splits.py`, `nyu_integrity.py` | Native NYU conversion, official partitions, development holdout and full integrity report |
| `trident_depth/cli.py` | Export one prepared sample plus provenance |
| `trident_depth/classical/` | Nine identifiable classical methods and reconstruction pipeline |
| `trident_depth/evaluation/` | Metrics, benchmarks, inference ablations and diagnostic panels |
| `trident_depth/evaluation/nyu_preparation.py` | Frozen NYU validation subset, protocol checks, registration panels and real classical comparisons |
| `trident_depth/models/` | Baseline/custom networks, feature conditioning, fusion and checkpoint I/O |
| `trident_depth/training/` | Torch dataset bridge, masked losses and opt-in training runner |
| `trident_depth/inference.py` | Classical/deep/hybrid inference with trained-checkpoint guard |
| `backend/main.py` | Upload API, validation, previews and downloads |
| `frontend/` | Responsive vanilla HTML/CSS/JS upload/compare interface |
| `configs/` | Dense, sparse and training configurations |
| `tests/` | Numeric, file I/O, leakage, model, loss and API tests |
| `scripts/fetch_smoke_sample.py` | Small checksummed real RGB-D smoke fixture acquisition |
| `scripts/download_nyu.py`, `inspect_nyu.py`, `export_nyu_reports.py` | Approved official downloads, source-format inspection and auditable report export |
| `requirements*.txt`, `pyproject.toml` | Pinned runtime/test choices and package metadata |
| `Dockerfile`, `.dockerignore`, `.gitignore` | Deployment recipe and artifact exclusions |
| `docs/` | Data/evaluation protocols, design, diagram, PDF audit, phase status and demo |
| `eval_P4_T69.md` | Submission report with correct USNs and honest incomplete results |

No final accuracy improvements, texture rejection, cross-dataset generalization
or novel research contribution are claimed. See limitations/ethics in the design
document. The organizer's referenced design template is still required to audit
its exact sections.
