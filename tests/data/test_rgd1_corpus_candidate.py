"""Owner-review candidate only: live corpus file is deliberately unchanged."""
import importlib.util
import hashlib
import json
from pathlib import Path
import pytest
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
P=ROOT/'outputs/implementation/i201-rgd1-corpus-guard/data_intake_corpus_candidate.py'
spec=importlib.util.spec_from_file_location('rgd1_corpus_candidate',P)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_input_projection_refuses_but_audit_codec_preserves_bytes():
    t=m.CorpusTask('RGD1_energy',{'reactant':'C'},{'DE_F':1},{},())
    raw=t.canonical_bytes()
    assert m.CorpusTask.from_bytes(raw).canonical_bytes()==raw
    with pytest.raises(ValueError,match='RGD1.*quarantine'):t.input_bytes()

@pytest.mark.parametrize('split',['train','validation','test'])
@pytest.mark.parametrize('supported_only',[True,False])
def test_reader_refuses_before_yield_regardless_split_or_adapter_flags(tmp_path,split,supported_only):
    raw=b'{"task":"RGD1_energy"}'
    row=dict(source_family='RGD1-computed-energy',split=split,model_adapter_supported=True,
        native_schema='mora3.CorpusTask/v1',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest())
    p=tmp_path/'s.parquet';pq.write_table(pa.Table.from_pylist([row]),p)
    f=tmp_path/'manifest.json';f.write_text(json.dumps({'shards':[dict(path='s.parquet',rows=1,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())]}))
    with pytest.raises(ValueError,match='RGD1.*quarantine'):
        next(m.read_corpus_tasks(f,root=tmp_path,split=split,supported_only=supported_only))
