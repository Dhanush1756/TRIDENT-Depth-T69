"""Audit saved metrics only; never infer, train, or open dataset sample files."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
from trident_depth.extended_v1.evaluation import aggregate, methods_for


def read_records(path):
    rows=[]; seen=set()
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            row=json.loads(line); key=(row['task'],row['id'])
            if key in seen: raise ValueError('duplicate task/frame')
            seen.add(key); rows.append(row)
    return rows


def recover(jsonl, reference, supplied_csv):
    rows=read_records(jsonl)
    ref=json.loads(Path(reference).read_text()); ids=[r['id'] for r in ref]
    expected={r['id']:r for r in ref}
    if len(ids)!=123 or len(set(ids))!=123: raise ValueError('validation reference differs')
    tasks=['dense2','dense4','dense8','sparse','noisy','incomplete','sparse_noisy_holes']
    groups={t:[] for t in tasks}
    for r in rows:
        if r['task'] not in groups or r['id'] not in expected: raise ValueError('unexpected task/frame')
        e=expected[r['id']]
        for k in ('scene_id','mask_sha256','source_files_sha256'):
            if r[k]!=e[k]: raise ValueError('reference mismatch: '+k)
        if set(r['methods'])!=set(methods_for({'name':r['task']})): raise ValueError('method set mismatch')
        for v in r['methods'].values():
            if v['errors']['pixels']<=0 or any(not math.isfinite(v['errors'][k]) or v['errors'][k]<0 for k in ('squared_error_m2','absolute_error_m')):
                raise ValueError('invalid error sums')
        groups[r['task']].append(r)
    results={}; flat=[]; missing={}
    for task,frames in groups.items():
        missing[task]=[i for i in ids if i not in {f['id'] for f in frames}]
        results[task]={}
        for method in methods_for({'name':task}):
            if not frames: continue
            values=[f['methods'][method] for f in frames]
            a=aggregate(values);results[task][method]=a
            flat.append(dict(task=task,frames=len(frames),method=method,
                rmse_m_pooled=a['rmse_m'],mae_m_pooled=a['mae_m'],
                psnr_db_frame_mean=a['mean_frame_psnr_db'],global_psnr_db=a['global_psnr_db'],
                ssim_frame_mean=a['mean_frame_ssim'],
                boundary_precision_frame_mean=a['mean_frame_boundary_precision'],
                boundary_recall_frame_mean=a['mean_frame_boundary_recall'],
                boundary_f1_frame_mean=a['mean_frame_boundary_f1'],
                median_inference_seconds_frame_mean=statistics.mean(v['timing']['median_seconds'] for v in values),
                median_inference_seconds_frame_median=a['latency_median_frame_seconds'],
                peak_allocated_bytes=a['peak_allocated_bytes'],peak_reserved_bytes=a['peak_reserved_bytes']))
    lookup={(r['task'],r['method']):r for r in flat};checked=0
    with Path(supplied_csv).open(newline='',encoding='utf-8-sig') as stream:
        supplied=list(csv.DictReader(stream))
    seen=set()
    for r in supplied:
        key=(r['task'],r['method'])
        if key in seen: raise ValueError('duplicate CSV row')
        seen.add(key); actual=lookup[key]
        for k,v in r.items():
            if k in ('task','method'):continue
            if not math.isclose(float(v),actual[k],rel_tol=1e-12,abs_tol=1e-12): raise ValueError('CSV mismatch: '+str(key)+' '+k)
            checked+=1
    return dict(status='PARTIAL',task_frame_pairs=len(rows),expected_pairs=861,
                counts={k:len(v) for k,v in groups.items()},missing_ids=missing,
                supplied_csv_rows_verified=len(supplied),supplied_csv_numeric_values_verified=checked,
                reference_metadata_checks='PASS; metadata only, no dataset sample opened',
                aggregation=results,table=flat,scientific_completion=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jsonl',type=Path,required=True);p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--reference',type=Path,default=Path('docs/pilot_dense4/per_frame.json'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():p.error('output must be NEW')
    report=recover(args.jsonl,args.reference,args.csv)
    report['input_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.jsonl,args.csv,args.reference)}
    args.output.mkdir(parents=True)
    with (args.output/'aggregation.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    with (args.output/'aggregates.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(report['table'][0]));w.writeheader();w.writerows(report['table'])
    print(json.dumps({k:report[k] for k in ('task_frame_pairs','counts','missing_ids','supplied_csv_rows_verified','supplied_csv_numeric_values_verified')},indent=2))


if __name__=='__main__':main()
