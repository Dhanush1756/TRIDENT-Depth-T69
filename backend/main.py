"""Bounded multipart API; uploads and outputs remain in memory per request."""
from contextlib import asynccontextmanager
import base64
import io
import math
import os
from pathlib import Path
import threading
import time
import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from trident_depth.data.sample import clean_depth
from trident_depth.evaluation.metrics import evaluate
from backend.engine import ApplicationEngine

ROOT = Path(__file__).resolve().parent.parent
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 1024 * 1024
CLASSICAL_METHODS = ["nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "classical"]
gate = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    checkpoint = os.environ.get("TRIDENT_CHECKPOINT")
    app.state.engine = ApplicationEngine(checkpoint, os.environ.get("TRIDENT_DEVICE", "cpu"),
                                         os.environ.get("TRIDENT_MODEL_FORMAT", "legacy"))
    yield


app = FastAPI(title="TRIDENT-Depth T69", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


def read_upload(upload, *, rgb=False):
    payload = upload.file.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("each upload must be at most 8 MiB")
    suffix = Path(upload.filename or "").suffix.lower()
    stream = io.BytesIO(payload)
    if suffix == ".npy" and not rgb:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("only NumPy format 1.0/2.0 is supported")
        if len(shape) != 2 or math.prod(shape) > MAX_PIXELS or dtype.kind not in "buif":
            raise ValueError("NumPy depth/mask must be numeric HxW with at most 1 megapixel")
        stream.seek(0)
        return np.load(stream, allow_pickle=False)
    allowed = {".png", ".jpg", ".jpeg"} if rgb else {".png", ".tif", ".tiff"}
    if suffix not in allowed:
        raise ValueError("RGB supports PNG/JPEG; metric depth/masks support PNG/TIFF/NPY")
    with Image.open(stream) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("input image exceeds 1 megapixel")
        if rgb and image.mode != "RGB":
            raise ValueError("guidance must be an RGB image")
        return np.array(image)


def encode_png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def preview(depth, valid, depth_range):
    normalized = np.clip(depth / depth_range, 0, 1)
    # Shared metric range for all previews; invalid pixels are black.
    colour = np.stack([255 * normalized, 255 * np.sqrt(normalized), 255 * (1 - normalized)], axis=-1)
    return encode_png(np.where(valid[..., None], colour, 0).astype(np.uint8))


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/health")
def health():
    engine = getattr(app.state, "engine", None)
    trained = engine is not None and engine.model is not None
    return dict(status="ok", trained_checkpoint_loaded=trained,
                methods=list(engine.capabilities()) if engine else CLASSICAL_METHODS,
                capabilities=engine.capabilities() if engine else {},
                max_upload_bytes=MAX_BYTES, max_target_pixels=MAX_PIXELS,
                checkpoint=engine.metadata if trained and engine.extended else
                    ({"model": "legacy", "uncertainty_available": getattr(engine.model,"uncertainty_enabled",True)} if trained else None))


@app.post("/api/reconstruct")
def reconstruct_api(depth: UploadFile = File(...), rgb: UploadFile | None = File(None),
                    mask: UploadFile | None = File(None), ground_truth: UploadFile | None = File(None),
                    scale: int = Form(4), protocol: str = Form("dense_sr"), method: str = Form("classical"),
                    depth_scale_to_m: float = Form(0.001), gt_scale_to_m: float = Form(0.001),
                    max_depth_m: float = Form(10.0), noisy: bool = Form(False)):
    if not gate.acquire(blocking=False):
        raise HTTPException(429, "Another reconstruction is running; retry shortly")
    try:
        if scale not in {2, 4, 8} or protocol not in {"dense_sr", "sparse_completion"}:
            raise ValueError("select scale 2/4/8 and a supported input protocol")
        if method not in CLASSICAL_METHODS + ["deep", "hybrid"]:
            raise ValueError("unknown reconstruction method")
        if not np.isfinite([depth_scale_to_m, gt_scale_to_m, max_depth_m]).all() or min(depth_scale_to_m, gt_scale_to_m, max_depth_m) <= 0:
            raise ValueError("depth conversion factors and maximum depth must be finite positive")
        raw = read_upload(depth)
        if raw.ndim != 2 or raw.dtype.kind not in "uif":
            raise ValueError("depth must be a numeric HxW metric map, not a colour image")
        observed, valid = clean_depth(raw.astype(np.float64) * depth_scale_to_m,
                                      read_upload(mask) if mask is not None else None, max_depth=max_depth_m)
        if not valid.any():
            raise ValueError("depth contains no valid positive measurements in the selected range")
        size = tuple(v * scale for v in observed.shape) if protocol == "dense_sr" else observed.shape
        if math.prod(size) > MAX_PIXELS:
            raise ValueError("target exceeds 1 megapixel; use smaller input or scale")
        guidance = read_upload(rgb, rgb=True) if rgb is not None else None
        if guidance is not None and guidance.shape != (*size, 3):
            raise ValueError(f"RGB must be registered and match target HxWx3 {(*size, 3)}")
        target = target_valid = None
        if ground_truth is not None:
            target, target_valid = clean_depth(read_upload(ground_truth).astype(np.float64) * gt_scale_to_m, max_depth=max_depth_m)
            if target.shape != size or not target_valid.any():
                raise ValueError("ground truth must match target shape and contain valid measurements")
        engine = app.state.engine
        if method in {"deep","hybrid"} and engine.extended and max_depth_m != 10:
            raise ValueError("extended model uses a fixed 10 metre depth range")
        start = time.perf_counter()
        result = engine.run(observed, valid, size, guidance, method=method, scale=scale, protocol=protocol, noisy=noisy)
        elapsed_ms = 1000 * (time.perf_counter() - start)
        metrics = evaluate(result["depth"], target, target_valid, data_range_m=max_depth_m) if target is not None else None
        stream = io.BytesIO()
        arrays = dict(depth_m=result["depth"], valid=result["valid"])
        arrays.update({f"map_{key}": value for key, value in result["maps"].items()})
        np.savez_compressed(stream, **arrays)
        # 16-bit millimetre PNG is optional; never silently clip range or wrap.
        png16 = None
        if float(result["depth"].min()) >= 0 and float(result["depth"].max()) <= 65.535:
            png16 = encode_png(np.rint(result["depth"] * 1000).astype(np.uint16))
        return dict(method=method, protocol=protocol, scale=scale, input_shape=list(observed.shape),
                    output_shape=list(size), units="metres", latency_ms=elapsed_ms, metrics=metrics,
                    metric_note="Ground-truth reference supplied" if metrics is not None else "No ground truth supplied; accuracy metrics unavailable",
                    input_valid_fraction=float(valid.mean()), input_preview=preview(observed, valid, max_depth_m),
                    output_preview=preview(result["depth"], result["valid"], max_depth_m),
                    rgb_preview=encode_png(guidance) if guidance is not None else None,
                    output_npz=base64.b64encode(stream.getvalue()).decode("ascii"), output_png16=png16,
                    checkpoint=health()["checkpoint"] if method in {"deep","hybrid"} else None,
                    intermediate_previews={k: preview(v, np.isfinite(v) & (v > 0), max_depth_m)
                                           for k, v in result["maps"].items() if k in {"initial", "classical", "geometry", "denoised", "filled", "guided", "joint_bilateral", "neural", "edge_aware"}})
    except (ValueError, TypeError, OSError, EOFError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise HTTPException(400, str(exc)) from exc
    finally:
        gate.release()
