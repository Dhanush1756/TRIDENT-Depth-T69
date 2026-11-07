"""Deployment safety tests; no Modal network calls, datasets or optimizer steps."""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from scripts import modal_checkpoint, check_deployment
from backend.main import app


def test_wrong_checkpoint_rejected_before_modal_import(tmp_path, monkeypatch):
    path = tmp_path / "epoch_012.pt"
    path.write_bytes(b"not approved weights")
    monkeypatch.setitem(sys.modules, "modal", None)
    with pytest.raises(ValueError, match="SHA256"):
        modal_checkpoint.upload(path)
    assert path.read_bytes() == b"not approved weights"


def test_upload_is_one_file_and_never_overwrites(tmp_path, monkeypatch):
    calls = []
    class Batch:
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def put_file(self, local, remote): calls.append((local, remote))
    class Volume:
        def batch_upload(self, *, force):
            assert force is False
            return Batch()
    def from_name(name, *, create_if_missing):
        assert name == "trident-t69-private-models" and create_if_missing
        return Volume()
    path = tmp_path / "verified.pt"
    monkeypatch.setattr(modal_checkpoint, "verify", lambda p: path)
    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Volume=SimpleNamespace(from_name=from_name)))
    modal_checkpoint.upload(path)
    assert calls == [(str(path), modal_checkpoint.REMOTE)]


def test_modal_source_allowlist_and_cpu_limits():
    tree = ast.parse(Path("scripts/modal_deploy.py").read_text())
    source = ast.unparse(tree)
    assert 'include_source=False' in source and 'read_only=True' in source
    assert 'max_containers=1' in source and 'min_containers=0' in source
    assert 'scaledown_window=30' in source and 'max_inputs=1' in source
    assert 'gpu=' not in source and "'backend', 'frontend', 'trident_depth'" in source
    assert '/app:/app/scripts' in source  # Modal script-mode imports modal_deploy by basename.
    assert 'add_local_dir(ROOT,' not in source


def test_missing_checkpoint_clears_stale_environment(tmp_path, monkeypatch):
    tree = ast.parse(Path("scripts/modal_deploy.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_checkpoint")
    import os
    namespace = dict(Path=Path, os=os, CHECKPOINT="unused")
    exec(compile(ast.Module(body=[node], type_ignores=[]), "modal-config", "exec"), namespace)
    monkeypatch.setenv("TRIDENT_CHECKPOINT", "stale")
    monkeypatch.setenv("TRIDENT_MODEL_FORMAT", "extended_v1")
    namespace["configure_checkpoint"](tmp_path / "missing.pt")
    assert "TRIDENT_CHECKPOINT" not in os.environ
    assert os.environ["TRIDENT_MODEL_FORMAT"] == "legacy"
    # Present invalid bytes must reach the original hash check, never fall back.
    path = tmp_path / "invalid.pt"
    path.write_bytes(b"invalid")
    namespace["configure_checkpoint"](path)
    assert os.environ["TRIDENT_CHECKPOINT"] == str(path)
    assert os.environ["TRIDENT_MODEL_FORMAT"] == "extended_v1"


def test_deployment_http_checker_against_actual_classical_app(monkeypatch):
    monkeypatch.delenv("TRIDENT_CHECKPOINT", raising=False)
    monkeypatch.setenv("TRIDENT_MODEL_FORMAT", "legacy")
    with TestClient(app) as client:
        result = check_deployment.check(client)
    assert result["status"] == "PASS" and len(result["checks"]) == 6
    assert result["datasets_accessed"] is False
