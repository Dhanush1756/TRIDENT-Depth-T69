# Deployment instructions

The [Modal CPU deployment](MODAL_DEPLOYMENT.md) passed public HTTP and browser checks.
Live URL: https://dhanush1756--trident-depth-t69-web.us-east.modal.run
Review resource costs before redeployment. Docker remains a separate, unbuilt option.

Run from repository root using the README commands. `python -m backend.serve` uses one worker,
`PORT` (default 8000), and `TRIDENT_HOST` (default 0.0.0.0). Bind 127.0.0.1 for local-only use.
`GET /api/health` lists actual loaded capabilities; a startup failure is not a healthy deployment.

## Docker

```sh
docker build -t trident-depth .
docker run --rm -p 8000:8000 trident-depth
# Optional learned image; first obtain the exact approved checkpoint separately.
docker build --build-arg INSTALL_LEARNING=1 -t trident-depth-learned .
docker run --rm -p 8000:8000 --mount type=bind,source="$(pwd)/checkpoints",target=/checkpoints,readonly -e TRIDENT_MODEL_FORMAT=extended_v1 -e TRIDENT_CHECKPOINT=/checkpoints/epoch_012.pt -e TRIDENT_DEVICE=cpu trident-depth-learned
```

On Windows, use the absolute checkpoint-folder path in the mount's `source`; do not commit that
machine-specific path. The default image serves classical reconstruction without Torch. Images
run non-root. Checkpoints/data/ZIP archives/secrets are excluded from the build context. Docker
build/run has not been verified in this review; perform both before submission.

## Hosting checklist

Use a Python/Docker host with sufficient RAM for the chosen CPU model and one request worker.
Configure a read-only checkpoint mount, exact format/hash, PORT, HTTPS, request body limits and
rate limits. Each multipart file is bounded to 8 MiB; set the reverse-proxy total-request limit to
accommodate several files. Keep upload processing ephemeral. Frontend and API use the same origin.
The Modal application uses the owner's existing authenticated account; its checkpoint Volume is private.

After deployment verify `/`, `/static/app.js`, `/api/health`, `/docs`, every classical option,
known aligned RGB/depth and learned deep/hybrid if configured. Verify invalid input errors and
numeric downloads, not only HTTP 200. Test a phone and laptop, cold start and actual latency.
Do not use a temporary Colab runtime as the production backend or interrupt the active benchmark.
Record the real public URL only after these checks succeed.

## Publication review

No credentials should be committed. Existing historical experiment reports include machine paths;
they are provenance, not application defaults. Review a separate publication copy for privacy
rather than altering preserved evidence. Existing notebooks use configurable Colab/Drive paths.
Run `git status --short` and inspect staged files; never stage data, weights, artifacts or ZIPs.
