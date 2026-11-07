# TRIDENT-Depth 2.0 - Team T69

Computer Vision Hackathon 2026, **P4: Edge-Preserving Depth Map Upsampling and Refinement**.
Python FastAPI serves a responsive HTML/CSS/JavaScript application for metric depth reconstruction.
TRIDENT means **Trust-Routed Intelligent Depth Enhancement Network**. Assigned problem:
**PS-04**. Repository: https://github.com/Dhanush1756/TRIDENT-Depth-T69.

Classical baselines, A1/B0 pilots and extended_v1 training are completed. The user reports the final
T4 benchmark failed during output saving. **854/861 saved task/frame records are verified;
six tasks have 123/123 frames and sparse_noisy_holes has 116/123.** See the
[partial independent benchmark audit](docs/partial_independent_recovery_20261107/report.md).
The official test set remains locked.

Live application: **[TRIDENT-Depth on Modal](https://dhanush1756--trident-depth-t69-web.us-east.modal.run)**.
Public HTTP and Chrome browser checks passed with the exact epoch-12 checkpoint on CPU.
These synthetic functionality checks are separate from scientific evaluation.

## Architecture and training overview

Registered RGB, depth and validity masks feed classical interpolation/filtering or the
473,059-parameter residual encoder-decoder. Extended inference reuses bicubic/nearest-valid
initialization and supervised depth-boundary logits. The hybrid combines learned depth with
depth-supported classical/geometry refinement and protocol-specific measurement consistency.
See [the architecture diagram](docs/architecture.md) and [design](DESIGN.md).

The extended model starts from verified B0 tensors and was trained for twelve epochs on
672 NYU training frames, with 123 validation frames from disjoint scenes. Adam, batch size 1,
FP32 and seed 69 follow the frozen [training contract](docs/extended_v1.md). The unused
uncertainty head stays frozen; the approved depth/gradient/edge/measurement/boundary/false-edge
objective remains unchanged. Checkpoints contain optimizer, scheduler and RNG state.
Training-time validation and the independent 854-record audit are distinct evidence.

## Independent saved-record results

These are recovered partial independent results, not training-time validation or live-app smoke metrics.

| Task | Frames | Extended RMSE m | Hybrid RMSE m |
| --- | --- | ---: | ---: |
| dense2 | 123/123 | 0.030476 | 0.025449 |
| dense4 | 123/123 | 0.044345 | 0.040598 |
| dense8 | 123/123 | 0.073959 | 0.072035 |
| sparse | 123/123 | 0.099851 | 0.096253 |
| noisy | 123/123 | 0.010945 | 0.011270 |
| incomplete | 123/123 | 0.408790 | 0.408922 |
| sparse_noisy_holes | 116/123 | 0.425872 | 0.425327 |

Hybrid does not win every task: classical has lower sparse RMSE, bicubic has lower noisy RMSE,
and raw extended slightly improves incomplete RMSE over hybrid. Hole-heavy errors remain large.
See eval_P4_T69.md for all baseline rows, pooled/per-frame definitions, memory and timing scope.

## Run locally on Windows 11 CPU

From the repository root, use Python 3.10 or 3.11 for these pinned CPU requirements
(local verification used 3.10.8; Docker targets 3.11). Colab uses its separate requirements
and Python runtime; NumPy 1.26.4 in the CPU file is not a Python 3.13 installation target:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-app.txt
.venv\Scripts\python.exe -m backend.serve
```

Open http://127.0.0.1:8000; API documentation is at `/docs`, health and exact capabilities at
`/api/health`. No frontend build or Node installation is required. Existing `.venv` users can
skip creation. Set `$env:TRIDENT_HOST='127.0.0.1'` for local-only binding.

Classical operation does not need Torch or a checkpoint. To enable the extended model:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-learning-cpu.txt
$env:TRIDENT_MODEL_FORMAT='extended_v1'
$env:TRIDENT_CHECKPOINT='checkpoints/epoch_012.pt'
$env:TRIDENT_DEVICE='cpu'
.venv\Scripts\python.exe -m backend.serve
```

Copy the approved checkpoint into that ignored local directory yourself. The application requires
SHA256 `09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd` before loading;
it fails closed on a mismatch. It does not fetch, train or modify weights. The supplied epoch-12
checkpoint passed its state audit and local/public synthetic app checks; this is
not a new scientific validation result. Do not install CPU Torch into Colab.
Legacy checkpoints remain supported with `TRIDENT_MODEL_FORMAT=legacy`.

## Supported inputs and methods

- Numeric depth: PNG/TIFF/NPY; explicitly select metres per stored unit (millimetres = 0.001,
  metres = 1). Validity masks accept 0, 1 or 255. RGB guidance must be RGB-mode PNG/JPEG and
  registered to the target grid; resizing alone does not establish registration.
- Dense SR produces H x scale by W x scale at 2x/4x/8x. Extended learned dense SR requires clean,
  fully valid LR measurements, registered RGB and a fixed 10 m range.
- Sparse/noisy/incomplete completion keeps input H x W. Extended completion uses conditioning
  value 4; this is **not 4x spatial upsampling**. Mark missing depth through zeros or a validity
  mask. Noisy measurements are not hard-copied by the extended hybrid.
- Nearest, bilinear, bicubic, guided, joint bilateral and complete classical refinement work
  without learned weights. Deep/extended hybrid appear only with an appropriate checkpoint.
  B0 supports deep dense4 only; legacy uncertainty-based B0 hybrid is not advertised.
- Optional ground truth must match the output grid and use explicit units. Accuracy metrics are
  omitted when it is absent. Upload latency is a single application measurement, not benchmark latency.
- At most 8 MiB per file and 1 megapixel per input/target. One reconstruction at a time (HTTP 429
  when busy). Downloads contain float32 metres in NPZ and optional uint16 millimetres in PNG.
  Previews share a fixed user-selected range; invalid pixels are black.

## Verification

```powershell
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m pip check
```

The full model suite requires CPU Torch. Synthetic test fixtures are not experimental results.
App integration tests cover classical and synthetic extended inference, input validation,
checkpoint rejection and unchanged parameters. See [submission audit](docs/submission_audit.md)
for actual checks and limitations. No current task changes any scientific/training source or data.

## Reproduce the scientific evaluation

Use [the epoch-12 validation notebook](notebooks/colab_extended_v1_epoch012_validation_T69.ipynb)
and [its frozen protocol](docs/extended_v1_epoch012_evaluation.md). It verifies the exact checkpoint,
training source/configuration, original scene split, masks and environment. All seven tasks use
123 validation frames; B0 is evaluated only on dense4. Outputs must use fresh directories.
The original benchmark failed during output saving. Preserve its log and JSONL. Do not rerun
the complete benchmark or merge unverified replacement records. The documented directory fix
is proposed only; original provenance and the matching execution environment must be verified
before any separate seven-frame continuation. No training or test evaluation is authorized here.

NYU has 672 training and 123 development-validation frames from disjoint scenes. Official test
samples remain excluded. Completion degradations are synthetic on real NYU filled references;
raw-sensor and KITTI generalization are unverified. Existing [288-record classical results](docs/nyu/validation_results.md)
and earlier pilot evidence are preserved. See [submission evaluation](eval_P4_T69.md) for
all 57 recovered task/method aggregates and the incomplete combined-task limitation.

## Deployment

```powershell
docker build -t trident-depth .
docker run --rm -p 8000:8000 trident-depth
```

The default image serves classical methods as a non-root user. For CPU learned inference build
with `--build-arg INSTALL_LEARNING=1`, mount the approved checkpoint read-only under `/checkpoints`,
and set `TRIDENT_MODEL_FORMAT=extended_v1`, `TRIDENT_CHECKPOINT=/checkpoints/epoch_012.pt`.
Set `PORT` for host integration. See [deployment instructions](docs/deployment.md).
Docker's CLI is installed, but its engine was unavailable during final review. No successful
Docker build/run is claimed. The existing Modal public endpoint is independently verified.

## Organization and design

`backend/` handles uploads and version-aware application dispatch; `frontend/` is the served UI.
`trident_depth/` contains preserved data, classical, learned, hybrid and evaluation implementations.
`configs/` and `notebooks/` identify reproducible experiments. `tests/` contains regression checks.
Data, weights and run artifacts are ignored by Git and excluded from the Docker build context.

Read [DESIGN.md](DESIGN.md), [architecture](docs/architecture.md),
[official requirements audit](docs/pdf_requirements_audit.md) and [evaluation protocol](docs/evaluation_protocol.md).
Earlier phase documents are historical evidence, not the current project status.

| Member | USN |
| --- | --- |
| Dhanush S | 1RUA24CSE0125 |
| Dhruthi R Kulkarni | 1RUA24CSE0129 |
| E Sasi Vaishnav | 1RUA24CSE0140 |
| Jathin Kirani Nagaraj | 1RUA24CSE0178 |

Limitations: seven combined-task records are missing; original run provenance, replay/isolation
certification and real failure-case panels remain unverified. No universal texture-rejection,
accuracy improvement, or cross-dataset generalization claim is made.

Public release: review [the checklist](docs/PUBLIC_RELEASE_CHECKLIST.md) and [the complete proposed file list](docs/PUBLIC_RELEASE_FILE_LIST.md) before staging or publishing. Original path-bearing evidence stays private; sanitized copies are separate.

Modal hosting: [resources, costs and exact deployment steps](docs/MODAL_DEPLOYMENT.md).
The public endpoint is verified; cold starts and ongoing usage remain subject to hosting costs.

Automatic GitHub-to-Modal deployment is [prepared but not yet run](docs/GITHUB_ACTIONS_SETUP.md).
Publication, historical-path privacy resolution, owner-managed secrets and first-deployment
approval remain required. See [the final submission checklist](docs/FINAL_SUBMISSION_CHECKLIST.md).

## Privacy cleanup release

The public branch history has been sanitized for machine-specific home paths. Four dataset-derived images are withheld pending licensing/privacy approval. Original evidence and model weights remain in private storage. See [the cleanup verification](docs/LOCAL_HISTORY_VERIFICATION.md).

Deployment automation is not enabled by this release: the prepared workflow is manual-only and still requires the owner opt-in variable and production approval. Automatic deployment integration is a separate next phase.
