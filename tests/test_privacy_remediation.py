"""Privacy preparation must not write Git history or mutate private evidence."""
import hashlib
import json
from pathlib import Path
import pytest
from scripts import privacy_history_plan as plan
from scripts import prepare_public_release as release


def test_git_object_serialization_roundtrips_empty_tree():
    assert plan.tree_id({})=='4b825dc642cb6eb9a060e54bf8d69288fbee4904'
    assert plan.object_id('blob',b'hello\n')=='ce013625030ba8dba906f756967f9e9ca394464a'


def test_commit_projection_preserves_identity_metadata_and_message():
    tree='a'*40;parent='b'*40
    raw=(f'tree {tree}\nparent {parent}\nauthor Team <team@example.invalid> 1 +0000\n'
         'committer Team <team@example.invalid> 2 +0000\n\nScientific provenance stays unchanged\n').encode()
    assert plan.project_commit(raw,tree,{parent:parent})==raw
    projected=plan.project_commit(raw,'c'*40,{parent:'d'*40})
    assert projected.split(b'\nauthor ',1)[1]==raw.split(b'\nauthor ',1)[1]
    assert b'tree '+b'c'*40 in projected and b'parent '+b'd'*40 in projected


def test_sensitive_json_keys_and_values_preserve_scientific_numbers():
    home='C:'+chr(92)+'Users'+chr(92)+'Example Person'
    source={home+chr(92)+'archive.zip':dict(rmse=.04550963874424752,seed=69,sha256='a'*64),
            'source':home+chr(92)+'manifest.json'}
    raw=json.dumps(source).encode();clean=release.sanitized(raw,'.json')
    assert len(plan.locations(raw,'.json'))==2 and not plan.locations(clean,'.json')
    assert json.loads(clean)==release.redact(source)
    assert json.loads(raw)==source


def test_withheld_image_excluded_even_if_candidate_list_includes_it(tmp_path,monkeypatch):
    monkeypatch.setattr(release,'ROOT',tmp_path)
    image=next(iter(release.HELD_IMAGES));p=tmp_path/image;p.parent.mkdir(parents=True);p.write_bytes(b'private-image')
    source=tmp_path/'backend/test.py';source.parent.mkdir();source.write_bytes(b'# unchanged source\n')
    monkeypatch.setattr(release,'candidates',lambda:[image,'backend/test.py'])
    monkeypatch.setattr(release,'git',lambda *a:b'')
    out=tmp_path/'release_review/public'
    result=release.export(out)
    assert not (out/image).exists() and p.read_bytes()==b'private-image'
    assert [r['path'] for r in result]==['backend/test.py']
    assert set(json.loads((out/'PUBLIC_RELEASE_MANIFEST.json').read_text())['excluded_images'])==release.HELD_IMAGES


def test_history_tool_rejects_git_mutations_before_execution(monkeypatch):
    monkeypatch.setattr(plan.subprocess,'check_output',lambda *a,**k:pytest.fail('must not execute mutation'))
    for command in ('update-ref','hash-object','commit-tree','filter-repo','push','reset'):
        with pytest.raises(ValueError,match='read-only'):plan.git(command)


def test_generic_readme_example_is_relative_without_touching_other_files():
    raw=b"checkpoint = '"+plan.CHECKPOINT_PLACEHOLDER.encode()+b"'\n"
    clean=plan.relative_checkpoint_example(raw,'README.md')
    assert clean==b"checkpoint = 'checkpoints/your_trained_checkpoint.pt'\n"
    assert plan.relative_checkpoint_example(raw,'other.md')==raw
