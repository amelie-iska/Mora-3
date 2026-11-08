"""Synthetic fixtures test identity, integrity and split behavior, not coverage."""
import hashlib
import json
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

class Adapter:
    def convert(self, *args):
        import importlib.util
        assert importlib.util.find_spec("mora3.data.rnacentral"), "RNA adapter is missing"
        from mora3.data.rnacentral import convert
        return convert(*args)

rnacentral = Adapter()
from mora3.data.schema import GraphRecord
from mora3.data.splits import leakage_groups, audit_splits


def cluster(name):
    return json.dumps(dict(schema='toricblm.nonprotein_similarity_split.v1', cluster_id=name))


def fixture():
    row = dict(record_id='local-1', dataset='toricblm_rnacentral_8192',
               entry_id='URS0000000001', sequence='ACTG', split='train',
               split_cluster=cluster('cluster-a'))
    api = dict(rnacentral_id='URS0000000001', sequence='ACUG', length=4,
               md5=hashlib.md5(b'ACTG').hexdigest(), is_active=True,
               description='RNA description', rna_type='lncRNA')
    return row, json.dumps(api).encode()


def test_verified_projection_preserves_local_sequence_and_missing_structure():
    row, raw = fixture()
    r = rnacentral.convert(row, raw)
    assert r.nodes[0].kind == 'rna'
    values = {m.name:m for m in r.nodes[0].measurements}
    assert values['sequence'].value == 'ACTG'
    assert values['structure'].available is False
    assert r.nodes[1].measurements[0].value == 'RNA description'
    assert (r.edges[0].source,r.edges[0].target,r.edges[0].directed)==(0,1,True)
    assert r.provenance[0].sha256 == hashlib.sha256(raw).hexdigest()
    assert r.provenance[0].license == 'CC0-1.0'
    assert GraphRecord.from_bytes(r.canonical_bytes()) == r
    assert 'proxy' not in r.canonical_bytes().decode()


@pytest.mark.parametrize('field,value', [('rnacentral_id','URS0000000002'),
    ('sequence','AAAA'),('length',5),('md5','0'*32),('is_active',False)])
def test_rejects_unverified_or_inactive_sequence(field,value):
    row, raw = fixture(); api=json.loads(raw);api[field]=value
    with pytest.raises(ValueError):rnacentral.convert(row,json.dumps(api).encode())


@pytest.mark.parametrize('field,value', [('dataset','another-source'),('sequence',''),
    ('sequence','AC?G'),('split','unknown'),('split_cluster','')])
def test_rejects_wrong_source_and_missing_split_evidence(field,value):
    row, raw=fixture();row[field]=value
    with pytest.raises(ValueError):rnacentral.convert(row,raw)


def test_duplicate_sequences_transitively_join_upstream_clusters():
    row,raw=fixture();a=rnacentral.convert(row,raw)
    row.update(record_id='local-2',split_cluster=cluster('cluster-b'));b=rnacentral.convert(row,raw)
    row.update(record_id='local-3',entry_id='URS0000000003',sequence='AAAA')
    api=json.loads(raw);api.update(rnacentral_id=row['entry_id'],sequence='AAAA',md5=hashlib.md5(b'AAAA').hexdigest())
    c=rnacentral.convert(row,json.dumps(api).encode())
    assert len(set(leakage_groups([a,b,c]).values()))==1
    assert len(audit_splits([a,b,c],{'local-1':'train','local-2':'train','local-3':'test'}))==1


def test_cluster_identity_excludes_split_and_record_specific_metadata():
    row,raw=fixture()
    row['split_cluster']=json.dumps(dict(schema='toricblm.nonprotein_similarity_split.v1',
        cluster_id='shared',split='train',cluster_signature={'accession_prefix':'first'}))
    a=rnacentral.convert(row,raw)
    row.update(record_id='other',entry_id='URS0000000004',sequence='AAAA',split='test')
    row['split_cluster']=json.dumps(dict(schema='toricblm.nonprotein_similarity_split.v1',
        cluster_id='shared',split='test',cluster_signature={'accession_prefix':'second'}))
    api=json.loads(raw);api.update(rnacentral_id=row['entry_id'],sequence='AAAA',md5=hashlib.md5(b'AAAA').hexdigest())
    b=rnacentral.convert(row,json.dumps(api).encode())
    assert len(audit_splits([a,b],{'local-1':'train','other':'test'})) == 1
