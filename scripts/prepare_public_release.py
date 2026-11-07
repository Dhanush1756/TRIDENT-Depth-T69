"""Prepare reviewed public copies without touching originals, Git index or history."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[1]
HOME=re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\r\n\"<>`]+",re.I)
UNIX_HOME=re.compile(r"/(?:home|Users)/[^/\s\"<>`]+")
SECRET=re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|sk-[A-Za-z0-9]{32,}")
FOLDERS={".github","backend","frontend","trident_depth","configs","tests","docs","notebooks","scripts"}
ROOT_FILES={".gitignore",".dockerignore",".gitattributes","Dockerfile","pyproject.toml","README.md","DESIGN.md","eval_P4_T69.md"}
HELD_IMAGES={
    "docs/nyu/registration_1.png", "docs/nyu/registration_2.png",
    "docs/training_preparation/prediction_review_dense_sr.png",
    "docs/training_preparation/prediction_review_sparse_completion.png",
}


def digest(raw):return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(["git","-C",str(ROOT),*args])


def decode(raw):
    try:return raw.decode("utf-16") if raw[:2] in (b"\xff\xfe",b"\xfe\xff") else raw.decode("utf-8")
    except UnicodeError:return None


def redact(value):
    if isinstance(value,str):return UNIX_HOME.sub("${USER_HOME}",HOME.sub("${USER_HOME}",value))
    if isinstance(value,list):return [redact(v) for v in value]
    if isinstance(value,dict):
        pairs=[(redact(k),redact(v)) for k,v in value.items()]
        if len({k for k,_ in pairs})!=len(value):raise ValueError("redaction would collide dictionary keys")
        return dict(pairs)
    return value


def sanitized(raw,suffix):
    text=decode(raw)
    if text is None:return raw
    if SECRET.search(text):raise ValueError("credential signature found; manual review required")
    # JSON escaping is decoded first, so doubled Windows backslashes are handled.
    if suffix==".json":
        obj=json.loads(text);safe=redact(obj)
        if safe==obj:return raw
        return (json.dumps(safe,indent=2,ensure_ascii=True)+"\n").encode()
    safe=redact(text)
    return raw if safe==text else safe.encode("utf-8")


def candidates():
    paths=sorted(set(git("ls-files","--cached","--others","--exclude-standard").decode().splitlines()))
    return [p for p in paths if p not in HELD_IMAGES and (ROOT/p).is_file() and (
        Path(p).parts[0] in FOLDERS or p in ROOT_FILES or Path(p).name.startswith("requirements"))]


def make_sanitized_copies():
    directory=ROOT/"docs/public_evidence"
    if directory.exists():raise FileExistsError("public evidence copies already exist; preserve/review rather than overwrite")
    rows=[]
    for name in candidates():
        if not name.startswith("docs/"):continue
        source=ROOT/name;raw=source.read_bytes();safe=sanitized(raw,source.suffix)
        if safe==raw:continue
        dest=directory/Path(name).relative_to("docs")
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("xb") as stream:stream.write(safe)
        rows.append(dict(original=name,public_copy=dest.relative_to(ROOT).as_posix(),
                         original_sha256=digest(raw),public_copy_sha256=digest(safe)))
    directory.mkdir(exist_ok=True,parents=True)
    with (directory/"manifest.json").open("x") as f:
        json.dump(dict(scope="path-redacted publication copies, not original hash-verification artifacts",
                       rule="only machine home prefixes replaced by ${USER_HOME}; JSON numeric/boolean/null values and content hashes retained",
                       copies=rows),f,indent=2)
    return rows


def export(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/"release_review") or output.exists():
        raise ValueError("use a NEW directory inside ignored release_review")
    index_before=git("diff","--cached","--raw")
    output.mkdir(parents=True)
    files=[]
    for name in candidates():
        if name in HELD_IMAGES:continue
        source=ROOT/name;raw=source.read_bytes()
        if source.is_symlink():raise ValueError("public export refuses symlinks")
        safe=sanitized(raw,source.suffix)
        if safe!=raw and not (name.startswith("docs/") or source.suffix==".md"):
            raise ValueError("non-documentation source needs manual privacy review: "+name)
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open("xb") as f:f.write(safe)
        files.append(dict(path=name,sha256=digest(safe),original_sha256=digest(raw),path_redacted=safe!=raw))
    with (output/"PUBLIC_RELEASE_NOTICE.md").open("x") as f:
        f.write("# Review export - not committed or published\n\nOriginal scientific evidence remains untouched in the private workspace. Some historical documentation has machine-home paths redacted; see docs/public_evidence/manifest.json for original/copy hashes. Metrics and source code were not changed. This export contains no Git history, datasets, weights, run archives or private audit profiles. It requires human approval and dataset-image licence review before publishing. Final validation results remain pending; see deployment documentation for current hosting status.\n")
    with (output/"PUBLIC_RELEASE_MANIFEST.json").open("x") as f:json.dump(dict(approval_required=True,excluded_images=sorted(HELD_IMAGES),files=files),f,indent=2)
    if git("diff","--cached","--raw")!=index_before:raise RuntimeError("Git index changed externally during export")
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--make-sanitized-copies",action="store_true")
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    if args.make_sanitized_copies:print(json.dumps(make_sanitized_copies(),indent=2))
    if args.output:
        files=export(args.output);print(f"Review export: {len(files)} files; nothing staged, committed or pushed")
    if not args.output and not args.make_sanitized_copies:print("\n".join(candidates()))


if __name__=="__main__":main()
