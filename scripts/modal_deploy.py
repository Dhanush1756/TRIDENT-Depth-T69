"""CPU-only Modal hosting; import is local and does not deploy or upload weights."""
import os
from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
SHA256 = "09bb8fc5edfff40c7fe27a39d44eb571d02726e0bb22a78c28afcd7f0bd844bd"
VOLUME_NAME = "trident-t69-private-models"
CHECKPOINT = f"/models/extended_v1/{SHA256}/epoch_012.pt"

# Explicit source allowlist: never mount the repository root, data or artifacts.
image = modal.Image.debian_slim(python_version="3.11").workdir("/app")
for name in ("requirements.txt", "requirements-app.txt", "requirements-learning-cpu.txt"):
    image = image.add_local_file(ROOT / name, f"/app/{name}", copy=True)
image = image.run_commands("python -m pip install -r /app/requirements-learning-cpu.txt")
for name in ("backend", "frontend", "trident_depth"):
    image = image.add_local_dir(ROOT / name, f"/app/{name}",
                                ignore=["**/__pycache__/**", "**/*.pyc", "**/.env*",
                                        "**/*.pt", "**/*.pth", "**/*.zip", "**/*.pem",
                                        "**/*.key", "**/*.npy", "**/*.npz"], copy=True)
image = image.add_local_file(Path(__file__).resolve(), "/app/scripts/modal_deploy.py", copy=True)
image = image.env({"PYTHONPATH": "/app:/app/scripts", "TRIDENT_DEVICE": "cpu",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
app = modal.App("trident-depth-t69", include_source=False)


def configure_checkpoint(path=CHECKPOINT):
    """Missing weight permits classical service; corrupt weight fails closed."""
    if Path(path).is_file():
        os.environ["TRIDENT_CHECKPOINT"] = str(path)
        os.environ["TRIDENT_MODEL_FORMAT"] = "extended_v1"
        # ApplicationEngine verifies the SHA256 before deserializing.
    else:
        os.environ.pop("TRIDENT_CHECKPOINT", None)
        os.environ["TRIDENT_MODEL_FORMAT"] = "legacy"


@app.function(image=image, cpu=2.0, memory=4096, min_containers=0,
              max_containers=1, scaledown_window=30, timeout=300,
              volumes={"/models": volume.with_mount_options(read_only=True)})
@modal.concurrent(max_inputs=1)
@modal.asgi_app()
def web():
    configure_checkpoint()
    from backend.main import app as fastapi_app
    return fastapi_app
