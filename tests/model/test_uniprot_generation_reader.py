import copy
import hashlib
import json
import pytest
from test_uniprot_native_reader import fixture, canonical
from mora3.data import uniprot_annotations as u


def generation_fixture():
    e,r=fixture()
    e['comments'].append({'commentType':'FUNCTION','texts':[{'value':'Catalyzes substrate conversion.','evidences':[{'evidenceCode':'ECO:0000269','id':'1'}]}]})
    t=json.loads(r['native_payload']);t['task']='source_UniProt_annotation_conditioned_canonical_sequence'
    t['conditions']={'source_experimentally_supported_function_texts':['Catalyzes substrate conversion.'],'source_experimental_associated_GO_ids':[],'source_explicit_experimental_NOT_GO_ids':[],'source_experimentally_supported_EC_ids':[],'source_kinetic_conditions':[{'parameter_type':'maximumVelocity','source_value':3,'source_unit':'umol/min/mg','substrate_label':None,'assay_pH':None,'assay_temperature':None}],'organism_taxon_id':None}
    t['targets']={'canonical_sequence':'MAGK'}
    r['task_name']=t['task'];bind(r,t)
    return e,r,t


def bind(r,t):
    r['native_payload']=canonical(t);r['native_payload_sha256']=hashlib.sha256(r['native_payload']).hexdigest()


def test_generation_detached_conditions_and_no_sequence_input():
    e,r,t=generation_fixture();v=u.generation_view(r,e)
    assert v.inputs()==t['conditions'] and v.targets()=={'canonical_sequence':'MAGK'}
    x=v.inputs();x['source_kinetic_conditions'].clear()
    assert v.inputs()==t['conditions']
    assert not v.heldout_eligible and v.family_group is None


@pytest.mark.parametrize('bad',['inferred','parent_evidence','product','target','extra','heldout','hash','units'])
def test_generation_fail_closed(bad):
    e,r,t=generation_fixture()
    if bad=='inferred': e['comments'][-1]['texts'][0]['evidences'][0]['evidenceCode']='ECO:0000250'
    if bad=='parent_evidence':
        e['comments'][-1]['evidences']=e['comments'][-1]['texts'][0].pop('evidences')
    if bad=='product':e['comments'][-1]['molecule']='Processed product'
    if bad=='target':t['targets']['canonical_sequence']='MASK'
    if bad=='extra':t['conditions']['canonical_sequence']='MAGK'
    if bad=='heldout':r['split']='test'
    if bad=='units':t['conditions']['source_kinetic_conditions'][0]['source_unit']='kcat'
    bind(r,t)
    if bad=='hash':r['native_payload_sha256']='0'*64
    with pytest.raises(ValueError):u.generation_view(r,e)


def test_target_sequence_and_hidden_names_never_enter_conditions():
    e,r,t=generation_fixture();a=u.generation_view(r,e)
    e['sequence']['value']='MAVG';e['genes']=[{'geneName':{'value':'hidden'}}]
    t['audit']['identity']['sequence_sha256']=hashlib.sha256(b'MAVG').hexdigest()
    r['group_ids_json']=json.dumps(['accession:P12345','sequence:'+t['audit']['identity']['sequence_sha256']])
    t['targets']['canonical_sequence']='MAVG';bind(r,t)
    b=u.generation_view(r,e)
    assert a.inputs()==b.inputs() and a.targets()!=b.targets()


def test_go_negation_isoform_and_ec_scope():
    e,_,_=generation_fixture()
    def go(id,code,**kw):
        return dict(database='GO',id=id,properties=[{'key':'GoEvidenceType','value':code},{'key':'Qualifier','value':'NOT|enables'}],**kw)
    e['uniProtKBCrossReferences']=[go('GO:1','IDA:source'),go('GO:2','IEA:source'),go('GO:3','EXP:source',isoformId='P12345-2')]
    ec={'value':'1.2.-.-','evidences':[{'evidenceCode':'ECO:0000269'}]}
    e['proteinDescription']={'recommendedName':{'ecNumbers':[ec]},'contains':[{'recommendedName':{'ecNumbers':[dict(ec,value='9.9.9.9')]}}]}
    c=u.generation_conditions(e)
    assert c['source_experimental_associated_GO_ids']==[]
    assert c['source_explicit_experimental_NOT_GO_ids']==['GO:1']
    assert c['source_experimentally_supported_EC_ids']==['1.2.-.-']
