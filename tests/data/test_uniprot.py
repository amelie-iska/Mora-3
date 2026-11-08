"""Synthetic contract fixtures, not real protein coverage."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from mora3.data.schema import GraphRecord
from mora3.data.splits import leakage_groups


def api():
    assert importlib.util.find_spec('mora3.data.uniprot'), 'UniProt adapter missing'
    from mora3.data import uniprot
    return uniprot


def entry():
    return dict(primaryAccession='P12345', entryType='UniProtKB reviewed (Swiss-Prot)',
        annotationScore=4.0, sequence=dict(value='MACDEFGHIK', length=10),
        entryAudit=dict(entryVersion=3, sequenceVersion=1),
        proteinDescription=dict(recommendedName=dict(fullName=dict(value='Example protein'))),
        comments=[dict(commentType='FUNCTION',texts=[dict(value='Database function')])])


def raw(e=None):
    return json.dumps({'results':[entry() if e is None else e]}).encode()


def test_projection_provenance_missing_structure_and_roundtrip():
    b=raw();r=api().convert(b,'P12345','2026_03')
    assert r.nodes[0].measurements[0].value=='MACDEFGHIK'
    assert r.nodes[0].measurements[1].status=='missing'
    assert {m.name:m.value for m in r.nodes[1].measurements}==dict(name='Example protein',annotation_score=4,function_0='Database function')
    assert (r.edges[0].source,r.edges[0].target,r.edges[0].directed)==(0,1,True)
    assert r.provenance[0].sha256==hashlib.sha256(b).hexdigest()
    assert r.provenance[0].license=='CC-BY-4.0'
    assert 'entry:3' in r.provenance[0].revision
    assert GraphRecord.from_bytes(r.canonical_bytes())==r


@pytest.mark.parametrize('score',[None,3,0,6,4.5,True,'5',float('nan')])
def test_score_filter_fails_closed(score):
    e=entry();e['annotationScore']=score
    with pytest.raises(ValueError): api().convert(raw(e),'P12345','2026_03')


@pytest.mark.parametrize('field,value', [('entryType','UniProtKB unreviewed (TrEMBL)'),('sequence',dict(value='MA*',length=3)),('sequence',dict(value='macd',length=4)),('sequence',dict(value='MACD',length=5)),('entryAudit',dict(entryVersion=0,sequenceVersion=1)),('proteinDescription',{}),('primaryAccession','bad')])
def test_invalid_entry_rejected(field,value):
    e=entry();e[field]=value
    with pytest.raises(ValueError): api().convert(raw(e),e['primaryAccession'],'2026_03')


def test_missing_identity_duplicate_identity_and_release_rejected():
    for b,acc,rel in [(raw(),'Q12345','2026_03'),(raw(),'P12345',''),(json.dumps({'results':[entry(),entry()]}).encode(),'P12345','2026_03')]:
        with pytest.raises(ValueError): api().convert(b,acc,rel)


def test_exact_sequences_group_across_accessions():
    e=entry();e['primaryAccession']='Q12345'
    a=api().convert(raw(),'P12345','2026_03');b=api().convert(raw(e),'Q12345','2026_03')
    assert len(set(leakage_groups([a,b]).values()))==1


def test_curation_checksum_rejections_and_immutable_output(tmp_path):
    e=entry();e['primaryAccession']='Q12345';e['annotationScore']=3
    b=json.dumps({'results':[entry(),e]}).encode();source=tmp_path/'response.json';source.write_bytes(b)
    out=tmp_path/'curated';sha=hashlib.sha256(b).hexdigest()
    m=api().curate(source,out,expected_sha256=sha,release='2026_03')
    assert m['accepted']==1 and len(m['rejected'])==1
    assert set(m['split_assignments'].values())=={'train'}
    assert m['split_violations']==[] and m['training_ready'] is False
    assert hashlib.sha256((out/'records.jsonl').read_bytes()).hexdigest()==m['checksum_files']['records.jsonl']
    assert source.read_bytes()==b
    with pytest.raises(FileExistsError): api().curate(source,out,expected_sha256=sha,release='2026_03')
    with pytest.raises(ValueError,match='checksum'): api().curate(source,tmp_path/'bad',expected_sha256='0'*64,release='2026_03')
    assert not (tmp_path/'bad').exists()


def test_no_function_is_absent_not_imputed():
    e=entry();e.pop('comments')
    r=api().convert(raw(e),'P12345','2026_03')
    assert [m.name for m in r.nodes[1].measurements]==['name','annotation_score']


def test_cli_produces_curated_artifact(tmp_path):
    import subprocess
    source=tmp_path/'response.json';source.write_bytes(raw())
    result=subprocess.run([sys.executable,'scripts/curate_uniprot.py','--source',str(source),
        '--sha256',hashlib.sha256(raw()).hexdigest(),'--release','2026_03',
        '--output',str(tmp_path/'out')],capture_output=True,text=True)
    assert result.returncode==0, result.stderr
    assert json.loads((tmp_path/'out'/'manifest.json').read_text())['accepted']==1
