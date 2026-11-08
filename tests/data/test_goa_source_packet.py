import copy
import hashlib
import json
import pytest
from mora3.data import goa_packets


def fixture(tmp_path):
    assertion={'geneProductId':'UniProtKB:P12345','goId':'GO:0000001','qualifier':'enables','goEvidence':'IDA'}
    raw=json.dumps({'results':[assertion]}).encode();path=tmp_path/'raw.json';path.write_bytes(raw)
    seq='ACDE';sha=hashlib.sha256(seq.encode()).hexdigest()
    packet={'schema':'mora3.CorpusTask/v1','task':'source_GOA_assertion_packet_prediction',
        'conditions':{'protein_sequence':seq},'targets':{'source_GO_annotations':[assertion],'typed_GO_annotations':[]},
        'audit':{'binding':{'subject':'UniProtKB:P12345','parent_accession':'P12345','canonical_labels_inherited':False},
                 'source_refs':[{'raw_file':'raw.json','raw_sha256':hashlib.sha256(raw).hexdigest(),'results_index':0}]}}
    payload=json.dumps(packet).encode()
    row={'native_payload':payload,'native_payload_sha256':hashlib.sha256(payload).hexdigest(),
         'record_id':'UniProtKB:P12345','task_name':packet['task'],'native_schema':packet['schema'],
         'split':'train','admitted_for_task':True,'full_axis_values':4,
         'admission_receipt_json':json.dumps({'license':'CC-BY-4.0','canonical_labels_inherited':False,'qualified_parent_entry':True,'semantic_not_residue_or_experimental_response_task':True}),
         'group_ids_json':json.dumps(['accession:P12345','sequence:'+sha])}
    return row,packet


def test_raw_source_and_predictor_exclusion(tmp_path):
    row,p=fixture(tmp_path);before=copy.deepcopy(row)
    v=goa_packets.verify_goa_packet(row,root=tmp_path)
    assert v.inputs()=={'protein_sequence':'ACDE'}
    assert v.targets()==p['targets']
    v.targets()['source_GO_annotations'].clear()
    assert v.targets()==p['targets'] and row==before
    assert v.provenance()['sequence_source_verified'] is False


@pytest.mark.parametrize('mutation',['source','sequence','subject','rights','escape','extra_input'])
def test_fail_closed(tmp_path,mutation):
    row,p=fixture(tmp_path)
    if mutation=='source':p['targets']['source_GO_annotations'][0]['goEvidence']='IEA'
    if mutation=='sequence':p['conditions']['protein_sequence']='AAAA'
    if mutation=='subject':p['audit']['binding']['subject']='UniProtKB:OTHER'
    if mutation=='rights':row['admission_receipt_json']='{}'
    if mutation=='escape':p['audit']['source_refs'][0]['raw_file']='../outside.json'
    if mutation=='extra_input':p['conditions']['GO']='GO:0000001'
    row['native_payload']=json.dumps(p).encode();row['native_payload_sha256']=hashlib.sha256(row['native_payload']).hexdigest()
    with pytest.raises(ValueError):goa_packets.verify_goa_packet(row,root=tmp_path)
