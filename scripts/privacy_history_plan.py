"""Read-only, in-memory Git history projection. Never writes Git objects or refs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from scripts.prepare_public_release import ROOT, HELD_IMAGES, HOME, UNIX_HOME, SECRET, decode, redact, sanitized

READ_ONLY_COMMANDS={"rev-parse","rev-list","for-each-ref","cat-file","ls-tree"}
MODAL_CREDENTIAL=re.compile(r"\b(?:ak|as)-[A-Za-z0-9]{20,}\b")
CHECKPOINT_PLACEHOLDER="C:"+chr(92)+"path"+chr(92)+"to"+chr(92)+"actual_trained_checkpoint.pt"


def relative_checkpoint_example(raw,name):
    return raw.replace(CHECKPOINT_PLACEHOLDER.encode(),b"checkpoints/your_trained_checkpoint.pt") if name=="README.md" else raw


def git(*args,input_data=None):
    if not args or args[0] not in READ_ONLY_COMMANDS:
        raise ValueError("only explicitly read-only Git commands are permitted")
    return subprocess.check_output(["git","-C",str(ROOT),*args],input=input_data)


def object_id(kind,data):
    return hashlib.sha1(kind.encode()+b" "+str(len(data)).encode()+b"\0"+data).hexdigest()


def tree_id(leaves):
    root={}
    for path,(mode,oid) in leaves.items():
        node=root;parts=path.split(b"/")
        for part in parts[:-1]:node=node.setdefault(part,{})
        node[parts[-1]]=(mode,oid)
    def build(node):
        raw=b""
        for name,value in sorted(node.items(),key=lambda x:x[0]+(b"/" if isinstance(x[1],dict) else b"")):
            mode,oid=(b"40000",build(value)) if isinstance(value,dict) else value
            raw+=mode+b" "+name+b"\0"+bytes.fromhex(oid)
        return object_id("tree",raw)
    return build(root)


def project_commit(raw,new_tree,parent_map):
    header,message=raw.split(b"\n\n",1);lines=[]
    for line in header.splitlines():
        if line.startswith(b"tree "):line=b"tree "+new_tree.encode()
        elif line.startswith(b"parent "):line=b"parent "+parent_map[line[7:].decode()].encode()
        lines.append(line)
    projected=b"\n".join(lines)+b"\n\n"+message
    if projected!=raw and b"gpgsig " in header:
        raise ValueError("changed signed commit requires separate signature-impact review")
    return projected


def locations(raw,suffix):
    text=decode(raw)
    if text is None:return []
    found=[]
    def active(value):return bool(HOME.search(value) or UNIX_HOME.search(value))
    def walk(value,pointer="$"):
        if isinstance(value,str) and active(value):found.append(dict(location=pointer,kind="value",exact_private_value=value))
        elif isinstance(value,list):
            for i,v in enumerate(value):walk(v,pointer+f"[{i}]")
        elif isinstance(value,dict):
            for i,(key,v) in enumerate(value.items()):
                if active(key):found.append(dict(location=pointer+f"[key#{i}]",kind="dictionary key",exact_private_value=key))
                walk(v,pointer+"."+key)
    if suffix==".json":walk(json.loads(text))
    else:
        found=[dict(location=f"line {i}",kind="text",exact_private_value=line)
               for i,line in enumerate(text.splitlines(),1) if active(line)]
    return found


def project():
    refs_before=git("for-each-ref","--format=%(refname) %(objectname)")
    if git("rev-parse","--show-object-format").strip()!=b"sha1":raise ValueError("unsupported object format")
    commits=git("rev-list","--reverse","--topo-order","--all").decode().splitlines()
    main=set(git("rev-list","main").decode().splitlines())
    ids=[line.split()[0] for line in git("rev-list","--objects","--all").decode().splitlines()]
    stream=git("cat-file","--batch",input_data=("\n".join(ids)+"\n").encode());objects={};pos=0
    for oid in ids:
        end=stream.index(b"\n",pos);header=stream[pos:end].split();size=int(header[2]);raw=stream[end+1:end+1+size];pos=end+2+size
        kind=header[1].decode();assert object_id(kind,raw)==oid
        objects[oid]=(kind,raw)
        text=decode(raw)
        if text and (SECRET.search(text) or MODAL_CREDENTIAL.search(text)):
            raise ValueError("credential signature found in object "+oid+"; stop for owner review")
    refs_parsed=dict(line.split() for line in refs_before.decode().splitlines())
    extra_trees=sorted({oid for oid in refs_parsed.values() if objects[oid][0]=='tree'})
    roots=commits+extra_trees
    trees={};image_objects=set()
    for commit in roots:
        leaves={}
        for entry in git("ls-tree","-r","-z","--full-tree",commit).split(b"\0"):
            if not entry:continue
            metadata,path=entry.split(b"\t",1);mode,kind,oid=metadata.split()
            if kind!=b"blob":raise ValueError("submodule requires separate review")
            leaves[path]=(mode,oid.decode())
            if path.decode() in HELD_IMAGES:image_objects.add(oid.decode())
        original_tree=commit if objects[commit][0]=='tree' else objects[commit][1].splitlines()[0][5:].decode()
        assert tree_id(leaves)==original_tree,"original tree roundtrip differs"
        trees[commit]=leaves
    replacements={};exposure={};mapping={};removed=[];scanned=set();unchanged=0;hygiene={}
    for commit in roots:
        projected={}
        for path,(mode,oid) in trees[commit].items():
            name=path.decode();raw=objects[oid][1];suffix=Path(name).suffix
            if name in HELD_IMAGES or oid in image_objects:
                removed.append(dict(root_object=commit,root_type=objects[commit][0],path=name,blob=oid));continue
            findings=locations(raw,suffix)
            if findings:
                if suffix not in {".json",".md"}:raise ValueError("non-documentation private path requires review: "+name)
                clean=sanitized(raw,suffix)
                if suffix==".json":assert json.loads(decode(clean))==redact(json.loads(decode(raw))),"semantic preservation failed"
                assert not locations(clean,suffix)
                if oid in replacements:assert replacements[oid]==clean
                replacements[oid]=clean
                row=exposure.setdefault((name,oid),dict(path=name,blob=oid,original_sha256=hashlib.sha256(raw).hexdigest(),sanitized_sha256=hashlib.sha256(clean).hexdigest(),locations=findings,commits=[],tree_roots=[],main_commits=[]))
                row['commits' if objects[commit][0]=='commit' else 'tree_roots'].append(commit)
                if commit in main:row['main_commits'].append(commit)
            else:clean=raw
            relative=relative_checkpoint_example(clean,name)
            if relative!=clean:
                hygiene[oid]=dict(path=name,original_blob=oid,reason="replace a non-sensitive absolute placeholder example with a relative checkpoint example")
                clean=relative
            projected[path]=(mode,object_id("blob",clean));scanned.add(projected[path][1])
            if name.startswith(("trident_depth/","configs/","notebooks/")):
                assert clean==raw,"protected scientific bytes changed";unchanged+=1
        if objects[commit][0]=='tree':mapping[commit]=tree_id(projected)
        else:
            raw_commit=project_commit(objects[commit][1],tree_id(projected),mapping)
            assert not locations(raw_commit,".commit")
            mapping[commit]=object_id("commit",raw_commit)
    assert refs_before==git("for-each-ref","--format=%(refname) %(objectname)"),"refs changed during read-only projection"
    refmap={}
    for line in refs_before.decode().splitlines():
        name,oid=line.split()
        if oid not in mapping:raise ValueError("non-commit ref needs separate tag review")
        refmap[name]=dict(original=oid,projected=mapping[oid])
    return dict(status="IN_MEMORY_PROJECTION_ONLY",actual_history_clean=False,git_history_rewritten=False,
        git_objects_written=0,git_refs_changed=False,commits=len(commits),main_commits=len(main),
        original_exposures=list(exposure.values()),projected_path_findings=0,credential_signature_findings=0,
        projected_held_images=0,removed_image_entries=removed,image_object_ids=sorted(image_objects),
        commit_mapping={c:mapping[c] for c in commits},tree_ref_mapping={c:mapping[c] for c in extra_trees},
        documentation_hygiene=list(hygiene.values()),
        signed_commits_kept_byte_identical=[c for c in commits if b'gpgsig ' in objects[c][1].split(b'\n\n',1)[0] and mapping[c]==c],
        refs=refmap,protected_file_versions_unchanged=unchanged,
        unique_projected_blobs_scanned=len(scanned),json_numeric_values_and_other_fields_preserved=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,required=True);args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/"release_review") or output.exists():parser.error("use a NEW ignored release_review directory")
    report=project();output.mkdir(parents=True)
    with (output/"private_exact_exposure_and_projection.json").open("x",encoding="utf-8") as f:json.dump(report,f,indent=2)
    public={k:v for k,v in report.items() if k not in {"original_exposures","refs","removed_image_entries"}}
    public['exposure_summary']=[{k:v for k,v in row.items() if k!='locations'}|dict(locations_count=len(row['locations'])) for row in report['original_exposures']]
    public['main_projection']=report['refs']['refs/heads/main']
    with (output/"public_summary.json").open("x") as f:json.dump(public,f,indent=2)
    print(json.dumps(dict(status=report['status'],commits=report['commits'],main_commits=report['main_commits'],exposed_records=len(report['original_exposures']),main_exposed_records=sum(bool(r['main_commits']) for r in report['original_exposures']),projected_path_findings=0,main_projection=public['main_projection'],git_history_rewritten=False),indent=2))


if __name__=="__main__":main()
