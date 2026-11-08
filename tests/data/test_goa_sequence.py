import copy
import hashlib
import json
import pytest
from test_goa_source_packet import fixture
from mora3.data import goa_packets as g


def setup(tmp_path):
    row,_=fixture(tmp_path)
    packet=g.verify_goa_packet(row,root=tmp_path)
    source={'results':[{'primaryAccession':'P12345','sequence':{'value':'ACDE','length':4}}]}
    raw=json.dumps(source).encode();(tmp_path/'kb.json').write_bytes(raw)
    receipt=dict(path='kb.json',sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),compression='none',source_release='2026_03',entries=1)
    return packet,receipt


def test_exact_canonical_source_preserves_inputs_targets(tmp_path):
    packet,r=setup(tmp_path)
    v=g.bind_canonical_sequence(packet,root=tmp_path,page_receipt=r,results_index=0)
    assert v.inputs()==packet.inputs() and v.targets()==packet.targets()
    assert v.provenance()['sequence_source_verified'] is True
    assert v.provenance()['training_eligibility_granted'] is False
    assert packet.provenance()['sequence_source_verified'] is False


@pytest.mark.parametrize('bad',['sequence','accession','length','hash','release','index','escape','form'])
def test_reject_inconsistent_or_unsupported_source(tmp_path,bad):
    packet,r=setup(tmp_path)
    d=json.loads((tmp_path/'kb.json').read_bytes());index=0
    if bad=='sequence':d['results'][0]['sequence']['value']='AAAA'
    if bad=='accession':d['results'][0]['primaryAccession']='OTHER'
    if bad=='length':d['results'][0]['sequence']['length']=5
    if bad=='form':
        p=packet.provenance();p['subject']='UniProtKB:P12345-2'
        packet=g.VerifiedGOAPacket(packet.condition_bytes,packet.target_bytes,json.dumps(p).encode())
    raw=json.dumps(d).encode();(tmp_path/'kb.json').write_bytes(raw)
    r.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    if bad=='hash':r['sha256']='0'*64
    if bad=='release':r['source_release']='other'
    if bad=='index':index=True
    if bad=='escape':r['path']='../kb.json'
    with pytest.raises(ValueError):g.bind_canonical_sequence(packet,root=tmp_path,page_receipt=r,results_index=index)
