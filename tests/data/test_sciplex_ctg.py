"""Synthetic CSV fixtures test contracts; actual source stays in outputs."""
import hashlib
import importlib.util
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from mora3.data.schema import GraphRecord
from mora3.data.splits import leakage_groups, audit_splits

HEADER = 'CTG,replicate,cell_type,treatment,dose,real_dose\n'
REV = '079639c50811dd43a206a779ab2f0199a147c98f'

def api():
    assert importlib.util.find_spec('mora3.data.sciplex_ctg'), 'sci-Plex CTG adapter missing'
    from mora3.data import sciplex_ctg
    return sciplex_ctg

def raw(rows='0.7,1,A549,Dex,100,50\n1.05,1,A549,Dex,0,0\n'):
    return (HEADER + rows).encode()

def test_preserves_source_pair_context_missing_and_relative_readout():
    records = api().convert(raw(), revision=REV)
    a, b = records
    assert a != b and a.record_id != b.record_id
    assert GraphRecord.from_bytes(a.canonical_bytes()) == a
    assert a.provenance[0].sha256 == hashlib.sha256(raw()).hexdigest()
    assert a.provenance[0].license == 'UNRESOLVED-NO-TRAINING'
    fields = {m.name:m for n in a.nodes for m in n.measurements}
    assert fields['treatment'].value == 'Dex'
    assert fields['dose'].value == 100 and fields['real_dose'].value == 50
    assert fields['real_dose'].units == 'source-unspecified'
    assert fields['CTG'].value == .7 and fields['CTG'].units == 'relative_viability'
    for name in ['exposure_time','baseline_CT G'.replace(' ',''),'structure','expression']:
        assert fields[name].status == 'missing'
    assert fields['response_state'].value == 'treated'
    assert {m.name:m.value for m in b.nodes[2].measurements}['response_state'] == 'vehicle_control'
    assert [(e.source,e.target,e.kind,e.directed) for e in a.edges] == [
        (0,1,'applied_in_assay',True),(1,2,'has_assay_readout',True)]

def test_two_masks_hide_readout_and_keep_context():
    r = api().convert(raw(), revision=REV)[0]
    forward, inverse = api().tasks(r)
    assert forward.available_targets(r) == (('node',2,'CTG'),)
    assert ('node',2,'CTG') not in forward.input_fields(r)
    assert ('node',0,'treatment') in forward.input_fields(r)
    assert inverse.available_targets(r) == (('node',0,'treatment'),)
    assert ('node',2,'CTG') in inverse.input_fields(r)
    assert ('node',0,'treatment') not in inverse.input_fields(r)
    # Unknown time remains unused, never a fabricated conditioning value.
    assert ('node',1,'exposure_time') not in forward.input_fields(r)
    with pytest.raises(ValueError, match='context'):
        forward.require_same_context(r,api().convert(raw(),revision=REV)[1])

@pytest.mark.parametrize('row',[
    'nan,1,A549,Dex,100,50\n','-0.1,1,A549,Dex,100,50\n',
    '0.7,0,A549,Dex,100,50\n','0.7,1,A549,unknown,100,50\n',
    '0.7,1,K562,Dex,100,50\n','0.7,1,A549,Dex,-1,50\n',
    '0.7,1,A549,Dex,0,50\n','0.7,1,A549,Dex,100,\n',
    '0.7,1,A549,Dex,100,50,extra\n'])
def test_invalid_rows_fail_closed(row):
    with pytest.raises(ValueError): api().convert(raw(row), revision=REV)

def test_duplicates_header_and_revision_fail_closed():
    for b, rev in [(raw('0.7,1,A549,Dex,100,50\n'*2),REV),
                   (raw().replace(b'CTG,replicate',b'CTG,CTG'),REV),
                   (raw(),'main'),(HEADER.encode(),REV)]:
        with pytest.raises(ValueError): api().convert(b,revision=rev)

def test_study_groups_prevent_cross_row_heldout_claims():
    records=api().convert(raw(),revision=REV)
    assert len(set(leakage_groups(records).values()))==1
    assert audit_splits(records,{records[0].record_id:'train',records[1].record_id:'test'})

def test_quarantine_manifest_checksums_immutable_and_not_train(tmp_path):
    source=tmp_path/'ctg.csv';source.write_bytes(raw());out=tmp_path/'out'
    sha=hashlib.sha256(raw()).hexdigest()
    m=api().curate(source,out,expected_sha256=sha,revision=REV)
    assert m['accepted']==2 and m['training_ready'] is False
    assert m['license_status']=='unresolved' and m['quarantined'] is True
    assert set(m['split_assignments'].values())=={'quarantine'}
    assert m['source_unchanged'] and source.read_bytes()==raw()
    assert m['pairing_audit']['source_rows']==2 and m['pairing_audit']['cross_source_joins']==0
    assert hashlib.sha256((out/'records.jsonl').read_bytes()).hexdigest()==m['checksum_files']['records.jsonl']
    assert (out/'tasks.jsonl').stat().st_size>0
    with pytest.raises(FileExistsError): api().curate(source,out,expected_sha256=sha,revision=REV)
    with pytest.raises(ValueError,match='checksum'):
        api().curate(source,tmp_path/'bad',expected_sha256='0'*64,revision=REV)
    assert not (tmp_path/'bad').exists()
    with pytest.raises(ValueError,match='license'):
        api().require_training_eligible(m)


def test_existing_vocabulary_cli_refuses_quarantine(tmp_path):
    import subprocess
    source=tmp_path/'ctg.csv';source.write_bytes(raw());out=tmp_path/'quarantine'
    api().curate(source,out,expected_sha256=hashlib.sha256(raw()).hexdigest(),revision=REV)
    result=subprocess.run([sys.executable,'scripts/build_vocabulary.py',
        '--manifest',str(out/'manifest.json'),'--output',str(tmp_path/'vocab'),
        '--field','treatment','--max-length','2','--vocab-size','270'],
        capture_output=True,text=True)
    assert result.returncode != 0
    assert 'explicit train assignment' in result.stderr
    assert not (tmp_path/'vocab').exists()
