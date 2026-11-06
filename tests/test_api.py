import base64
import io
import numpy as np
import pytest
from PIL import Image
pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from backend.main import app, gate


def png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("TRIDENT_CHECKPOINT", raising=False)
    with TestClient(app) as c:
        yield c


def test_health_and_frontend(client):
    result = client.get("/api/health").json()
    assert result["status"] == "ok" and not result["trained_checkpoint_loaded"]
    assert "hybrid" not in result["methods"]
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200


@pytest.mark.parametrize("method", ["nearest", "bilinear", "bicubic", "guided", "joint_bilateral", "classical"])
def test_upload_and_numeric_download(client, method):
    depth = png(np.full((4, 4), 2000, np.uint16))
    rgb = png(np.full((16, 16, 3), 90, np.uint8))
    result = client.post("/api/reconstruct", files={"depth": ("depth.png", depth), "rgb": ("rgb.png", rgb)},
                         data={"method": method, "scale": 4}).json()
    assert result["output_shape"] == [16, 16] and result["metrics"] is None
    with np.load(io.BytesIO(base64.b64decode(result["output_npz"]))) as out:
        np.testing.assert_allclose(out["depth_m"], 2)
    image = np.array(Image.open(io.BytesIO(base64.b64decode(result["output_png16"]))))
    np.testing.assert_array_equal(image, 2000)


def test_supplied_gt_and_completion(client):
    d = png(np.full((16, 16), 2000, np.uint16))
    result = client.post("/api/reconstruct", files={"depth": ("d.png", d), "ground_truth": ("g.png", d)},
                         data={"protocol": "sparse_completion", "method": "nearest"})
    assert result.status_code == 200
    data = result.json()
    assert data["output_shape"] == [16, 16]
    assert data["metrics"]["rmse_m"] == 0 and data["metrics"]["psnr_infinite"]


@pytest.mark.parametrize("data", [{"scale": 3}, {"method": "fake"}, {"method": "hybrid"},
    {"depth_scale_to_m": 0}, {"max_depth_m": -1}, {"protocol": "unknown"}])
def test_invalid_settings(client, data):
    d = png(np.full((4, 4), 2000, np.uint16))
    response = client.post("/api/reconstruct", files={"depth": ("d.png", d)}, data=data)
    assert response.status_code in (400, 422)


def test_wrong_guidance_empty_depth_and_malformed_upload(client):
    d = png(np.full((4, 4), 2000, np.uint16))
    rgb = png(np.zeros((4, 4, 3), np.uint8))
    assert client.post("/api/reconstruct", files={"depth": ("d.png", d), "rgb": ("r.png", rgb)}).status_code == 400
    assert client.post("/api/reconstruct", files={"depth": ("d.png", b"invalid")}).status_code == 400
    empty = png(np.zeros((4, 4), np.uint16))
    assert client.post("/api/reconstruct", files={"depth": ("d.png", empty)}).status_code == 400


def test_oversized_numpy_header_rejected_without_allocation(client):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, dict(descr="<f4", fortran_order=False, shape=(1000000, 1000000)))
    response = client.post("/api/reconstruct", files={"depth": ("d.npy", stream.getvalue())})
    assert response.status_code == 400 and "megapixel" in response.json()["detail"]


def test_busy_request_returns_retryable_error(client):
    gate.acquire()
    try:
        d = png(np.ones((4, 4), np.uint16))
        assert client.post("/api/reconstruct", files={"depth": ("d.png", d)}).status_code == 429
    finally:
        gate.release()
