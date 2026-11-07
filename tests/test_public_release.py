"""Publication-copy tests: preserve evidence and numeric results, never mutate Git."""
import json
import pytest
from scripts import prepare_public_release as release

SYNTHETIC_HOME = "C:" + "\\Users\\" + "Example Person"
SYNTHETIC_UNIX_HOME = "/home" + "/example"


def test_json_redaction_preserves_metrics_hashes_and_original_bytes():
    original=dict(path=SYNTHETIC_HOME+r"\Downloads\run\epoch.pt",rmse_m=.04434492164352799,
                  count=123,hash="a"*64,nested=[True,None,dict(value=1.25)])
    raw=json.dumps(original).encode()
    safe=release.sanitized(raw,'.json')
    result=json.loads(safe)
    assert result['path']==r"${USER_HOME}\Downloads\run\epoch.pt"
    assert {k:v for k,v in result.items() if k!='path'}=={k:v for k,v in original.items() if k!='path'}
    assert json.loads(raw)==original and release.sanitized(safe,'.json')==safe


def test_text_redaction_and_credentials_require_manual_review():
    text=('From '+SYNTHETIC_HOME+r'\Downloads\project; '+SYNTHETIC_UNIX_HOME+'/project').encode()
    assert b'Example Person' not in release.sanitized(text,'.md')
    assert SYNTHETIC_UNIX_HOME.encode() not in release.sanitized(text,'.md')
    with pytest.raises(ValueError,match='credential'):
        release.sanitized(('ghp_'+'X'*36).encode(),'.txt')
    assert release.sanitized(b'normal code with no private path','.py')==b'normal code with no private path'


def test_public_copies_are_exclusive_and_original_remains_exact(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'ROOT',tmp_path)
    path=tmp_path/'docs/private.md';path.parent.mkdir();raw=(SYNTHETIC_HOME+r'\data').encode()
    path.write_bytes(raw)
    monkeypatch.setattr(release,'candidates',lambda:['docs/private.md'])
    rows=release.make_sanitized_copies()
    assert path.read_bytes()==raw and len(rows)==1
    assert (tmp_path/rows[0]['public_copy']).is_file()
    with pytest.raises(FileExistsError):release.make_sanitized_copies()


def test_export_preserves_source_and_git_index_without_commit(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'ROOT',tmp_path)
    p=tmp_path/'trident_depth/model.py';p.parent.mkdir();p.write_bytes(b'# preserved scientific code\r\n')
    monkeypatch.setattr(release,'candidates',lambda:['trident_depth/model.py'])
    commands=[]
    def git(*args):commands.append(args);return b''
    monkeypatch.setattr(release,'git',git)
    out=tmp_path/'release_review/export'
    rows=release.export(out)
    assert (out/'trident_depth/model.py').read_bytes()==p.read_bytes()
    assert not rows[0]['path_redacted']
    assert all(command==('diff','--cached','--raw') for command in commands)
    with pytest.raises(ValueError):release.export(out)
    with pytest.raises(ValueError):release.export(tmp_path/'wrong-location')
