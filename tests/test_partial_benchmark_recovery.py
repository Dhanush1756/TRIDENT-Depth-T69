"""Aggregation audits without dataset or checkpoint access."""
import csv
import json
import pytest
from scripts.recover_partial_benchmark import recover,read_records,methods_for


def fixture_files(tmp_path):
    reference=[dict(id=f'frame_{i}',scene_id='scene',mask_sha256='mask',source_files_sha256={}) for i in range(123)]
    errors=dict(pixels=4,squared_error_m2=16.,absolute_error_m=8.)
    value=dict(errors=errors,metrics={k:.5 for k in ('psnr_db','ssim','boundary_precision','boundary_recall','boundary_f1','edge_preservation')},regions={'all':errors},
               timing=dict(median_seconds=.01,memory=dict(peak_allocated_bytes=10,peak_reserved_bytes=20)))
    row=dict(reference[0],task='dense4',methods={m:value for m in methods_for({'name':'dense4'})})
    paths=[tmp_path/n for n in ('frames.jsonl','reference.json','provided.csv')]
    paths[0].write_text(json.dumps(row)+'\n');paths[1].write_text(json.dumps(reference))
    paths[2].write_text('task,frames,method,rmse_m_pooled,mae_m_pooled\ndense4,1,extended,2,2\n')
    return paths


def test_pooled_metrics_and_incomplete_status(tmp_path):
    paths=fixture_files(tmp_path);r=recover(*paths)
    assert r['status']=='PARTIAL' and not r['scientific_completion']
    assert r['aggregation']['dense4']['extended']['rmse_m']==2
    assert len(r['missing_ids']['dense4'])==122
    assert r['supplied_csv_numeric_values_verified']==3


def test_csv_mismatch_is_error(tmp_path):
    paths=fixture_files(tmp_path);paths[2].write_text(paths[2].read_text().replace(',2,2',',3,2'))
    with pytest.raises(ValueError,match='CSV mismatch'):recover(*paths)


def test_duplicate_records_and_reference_mismatch_rejected(tmp_path):
    paths=fixture_files(tmp_path);raw=paths[0].read_text();paths[0].write_text(raw+raw)
    with pytest.raises(ValueError,match='duplicate'):read_records(paths[0])
    row=json.loads(raw);row['mask_sha256']='wrong';paths[0].write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError,match='reference mismatch'):recover(*paths)
