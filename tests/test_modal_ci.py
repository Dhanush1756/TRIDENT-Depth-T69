"""CI preflight refuses mismatched resources without mutating cloud state."""
import hashlib
import sys
from types import SimpleNamespace
from pathlib import Path
import pytest
from scripts import modal_preflight as preflight


def install_fake(monkeypatch, *, url=None, content=b'checkpoint'):
    calls=[]
    class Function:
        @staticmethod
        def from_name(app,name):
            assert (app,name)==('trident-depth-t69','web')
            return SimpleNamespace(get_web_url=lambda:url or preflight.PUBLIC_URL)
    class Volume:
        @staticmethod
        def from_name(name, *, create_if_missing):
            assert name=='trident-t69-private-models' and create_if_missing is False
            calls.append('read existing volume')
            return SimpleNamespace(read_file=lambda path:iter([content]))
    monkeypatch.setitem(sys.modules,'modal',SimpleNamespace(Function=Function,Volume=Volume))
    return calls


def test_preflight_matches_without_writing_or_creating(monkeypatch):
    calls=install_fake(monkeypatch)
    monkeypatch.setattr(preflight,'SHA256',hashlib.sha256(b'checkpoint').hexdigest())
    result=preflight.verify_existing_resources()
    assert result['status']=='PASS' and not result['resources_created']
    assert not result['private_checkpoint_written_to_disk'] and calls==['read existing volume']


def test_preflight_rejects_wrong_app_before_volume_access(monkeypatch):
    calls=install_fake(monkeypatch,url='https://wrong.invalid')
    with pytest.raises(ValueError,match='URL differs'):preflight.verify_existing_resources()
    assert not calls


def test_preflight_rejects_wrong_checkpoint(monkeypatch):
    install_fake(monkeypatch)
    with pytest.raises(ValueError,match='SHA256 differs'):preflight.verify_existing_resources()


def test_deployment_uses_existing_volume_only():
    source=Path('scripts/modal_deploy.py').read_text()
    assert 'create_if_missing=False' in source and 'create_if_missing=True' not in source
