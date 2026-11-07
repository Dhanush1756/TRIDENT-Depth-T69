# Modal deployment - public endpoint verified

Live application: https://dhanush1756--trident-depth-t69-web.us-east.modal.run

Deployed to the owner's authenticated account on 2026-11-07. Public HTTP checks and
isolated Chrome interaction checks passed with the exact epoch-12 model. Checks used
synthetic inputs only. No datasets or Colab runtime were accessed. See the separate
[deployment verification report](modal_deployment_20261107/report.md) for evidence;
the earlier preparation report is preserved.

The local exact epoch-12 checkpoint has passed its SHA256 audit:
`09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd`.
Local synthetic browser checks exercised the real weights, not accuracy evaluation.
No running Colab runtime is accessed. The final independent benchmark remains separate.

## Resources and cost gate

Proposed: one CPU Function, 2 physical cores, 4096 MiB RAM, no GPU, maximum one
container and one concurrent input, minimum zero containers, 30-second idle scale-down.
Python 3.11 / CPU Torch uses the existing pinned requirements. Existing source and
all model/protocol restrictions remain unchanged. Maximum request execution time is 300 s.
The Linux CPU image built successfully on Modal. Production latency under load remains unqualified.

At [Modal's listed prices](https://modal.com/pricing), checked 2026-11-07:
CPU $0.0000131/core/s and RAM $0.00000222/GiB/s yield approximately $0.1263
per active hour, or $0.0210 per 10 active minutes, at the requested resource usage.
Load time and the idle window are billable; actual usage exceeding requests costs more.
Image builds, egress and usage outside this estimate can add charges. Credits depend
on the authenticated account's remaining allowance; do not assume this is free.
One continuously active container would cost about $3.03/day at that resource usage.
This is a concurrency limit, not a dollar budget. Review account billing controls.
No GPU is allocated: the exact model already works on local CPU. Reconsider GPU only
after measuring hosted latency and obtaining approval for its additional costs.

The private Volume stores only the approximately 5.75 MB approved checkpoint, not
datasets or training archives. Published storage pricing is $0.09/GiB/month after
included allowances; confirm account eligibility. App HTTP endpoints are public.
The Volume is workspace-private and mounted read-only; it is never served as static files.

**Show these resources and costs to the account owner before cloud deployment or upload.**
These resources and costs were shown before deployment; the account owner completed authentication.

## Local tools and account login

Run from repository root in PowerShell. Preparation used a separate ignored tools
environment; application/training requirements were not changed.

```powershell
.venv\Scripts\python.exe -m venv release_review/modal_env
release_review\modal_env\Scripts\python.exe -m pip install -r requirements-modal.txt
release_review\modal_env\Scripts\python.exe -m modal token new
```

The last command opens Modal's authentication flow in your browser. Confirm your
own intended account/workspace. Never paste tokens into chat, source files or reports.
The initial inspection found no local Modal credential file or token environment
variables; a browser login alone does not configure this terminal.
Account authentication is required before the commands below. Do not create a new
paid subscription or upgrade plans as part of this workflow.

## After cost review and authentication

```powershell
# Immutable, single-file upload. Hash checked BEFORE any Modal lookup/network call.
release_review\modal_env\Scripts\python.exe scripts/modal_checkpoint.py release_review/checkpoint_audit_input/epoch_012.pt
# Deploy the CURRENT working source, not an old source ZIP.
$env:PYTHONUTF8='1'
$env:PYTHONIOENCODING='utf-8'
release_review\modal_env\Scripts\python.exe -m modal deploy scripts/modal_deploy.py
```

Upload refuses overwriting an existing checkpoint. If it already exists, stop and
verify its remote hash rather than forcing replacement. A classical-only deployment
may omit the upload step. Without the expected remote weight, only the six classical
methods are advertised; a present checkpoint with wrong SHA causes startup failure.
After uploading weights to an already-running classical-only deployment, redeploy so
new containers reload the checkpoint. No hot-reload or runtime mutation is assumed.

The image explicitly includes backend/, frontend/, trident_depth/, three dependency
files and this deployment module. Automatic source mounting is disabled. Repository
root, datasets, docs, configs, notebooks, experiment outputs, ZIPs and credentials are
not included. Model storage is not part of the image. Original ApplicationEngine checks
the hash before checkpoint loading and uses the original strict loader.

## Public verification gate

Use the actual HTTPS URL returned by Modal. The live URL above passed these checks.

```powershell
$appUrl = Read-Host 'Paste the actual deployed HTTPS application URL'
.venv\Scripts\python.exe scripts/check_deployment.py $appUrl --require-learned --output release_review/modal_public_smoke_01.json
```

For deliberate classical-only deployment omit --require-learned. The script checks
frontend assets, health, exact hash when learned is required, all available methods,
bad/missing uploads and invalid parameters, PNG previews and finite metric NPZ numeric
downloads. It sends synthetic inputs only, never training/validation/test samples.
It refuses to overwrite a report. Timings are smoke observations, not benchmark results.

Then open the actual URL in a browser: upload a registered RGB/depth pair, check the
loading state, all available method choices, results, error messages, NPZ/PNG downloads,
keyboard navigation and mobile layout. Repeat after a cold start. Inspect host logs
for startup failures without publishing credentials or upload contents. Record the real
URL in submission documentation only after public HTTP AND browser checks pass.

Modal SDK import and local tests do not prove the cloud image builds, the account is
authenticated or the public deployment works. Docker is separate and has only been
reviewed statically; no successful Docker build is claimed.

## Stop hosting

Scale-to-zero stops idle compute charges after the configured idle window; an active
public endpoint can still incur usage. To stop the deployed application deliberately:

```powershell
release_review\modal_env\Scripts\python.exe -m modal app stop trident-depth-t69
```

Stopping the app terminates its containers and availability. It does not delete the private
checkpoint Volume. Do not delete or overwrite that Volume as part of routine redeployment.

Official references: [ASGI hosting](https://modal.com/docs/guide/webhooks),
[read-only Volumes](https://modal.com/docs/sdk/py/latest/Volume),
[browser authentication](https://modal.com/docs/cli/latest/token).
