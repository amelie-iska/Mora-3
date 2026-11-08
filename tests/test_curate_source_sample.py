"""Real source data must stay immutable; synthetic rows here are test fixtures only."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def sampler():
    path=Path(__file__).resolve().parents[1]/'scripts/curate_source_sample.py'
    assert path.exists(), 'Real-data sample curator is missing'
    spec=importlib.util.spec_from_file_location('sample_curator',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def rows(n=10):
    return [dict(id=str(i),source='test-fixture-only',split='train',graph_json=json.dumps(dict(id=str(i),nodes=[dict(id='n',type='text')],edges=[]))) for i in range(n)]


def fixture(tmp_path, values):
    path=tmp_path/'source.parquet';pq.write_table(pa.Table.from_pylist(values),path);return path


def test_bounded_deterministic_curation_preserves_source_and_original_split(tmp_path):
    source=fixture(tmp_path,rows());before=hashlib.sha256(source.read_bytes()).hexdigest()
    m=sampler();a=m.curate([source],tmp_path/'a',rows_per_source=3,scan_rows=6,seed=19)
    b=m.curate([source],tmp_path/'b',rows_per_source=3,scan_rows=6,seed=19)
    first=pq.read_table(tmp_path/'a/sample-000.parquet').to_pylist()
    assert first==pq.read_table(tmp_path/'b/sample-000.parquet').to_pylist()
    assert len(first)==3 and all(int(row['id'])<6 and row['split']=='train' for row in first)
    assert a['sources'][0]['scanned_rows']==6
    assert a['sources'][0]['source_unchanged'] is True
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    assert a['training_ready'] is False
    assert 'license' in ' '.join(a['outstanding_gates']).lower()
    assert (tmp_path/'a/manifest.json').is_file()


def test_invalid_graphs_and_repeated_ids_are_rejected(tmp_path):
    values=rows(3);values[1]['graph_json']='not-json';values.append(values[0].copy())
    result=sampler().curate([fixture(tmp_path,values)],tmp_path/'sample',rows_per_source=8,scan_rows=10)
    source=result['sources'][0]
    assert source['accepted_rows']==2
    assert source['rejected_rows']==2
    assert source['selected_rows']==2


def test_existing_output_is_never_overwritten(tmp_path):
    source=fixture(tmp_path,rows(1));out=tmp_path/'existing';out.mkdir();sentinel=out/'user.txt';sentinel.write_text('protected')
    with pytest.raises(FileExistsError):sampler().curate([source],out)
    assert sentinel.read_text()=='protected'


def test_sampling_requires_positive_consistent_limits(tmp_path):
    with pytest.raises(ValueError):sampler().curate([],tmp_path/'sample',rows_per_source=4,scan_rows=2)


def test_nonprotein_record_id_schema_is_accepted_without_rewriting_rows(tmp_path):
    values=rows(3)
    for row in values:row['record_id']=row.pop('id')
    source=fixture(tmp_path,values)
    out=tmp_path/'sample'
    result=sampler().curate([source],out,rows_per_source=3,scan_rows=3)
    assert result['sources'][0]['selected_rows']==3
    assert pq.read_table(out/'sample-000.parquet').to_pylist()==values
