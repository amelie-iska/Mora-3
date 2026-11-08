import importlib.util,copy,json
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());spec=importlib.util.spec_from_file_location('goap',root/'scripts/data_intake_goa_projection.py');m=importlib.util.module_from_spec(spec)
try:spec.loader.exec_module(m)
except FileNotFoundError:m=None

def module():assert m is not None,'GOA form/evidence projection missing';return m

def annotation():return dict(id='a1',geneProductId='UniProtKB:P00001',qualifier='NOT|enables',goId='GO:0000001',goName=None,goAspect='molecular_function',goEvidence='IDA',evidenceCode='ECO:0000314',reference='PMID:1',assignedBy='UniProt',date='20200101',withFrom=[{'connectedXrefs':[{'db':'UniProtKB','id':'A'},{'db':'UniProtKB','id':'B'}]},{'connectedXrefs':[{'db':'UniProtKB','id':'C'}]}],extensions=[{'connectedXrefs':[{'relation':'occurs_in','db':'CL','id':'1'}]}])

def test_goa_not_relation_and_and_or_are_preserved_without_closedworld_negatives():
 a=annotation();r=module().annotation_projection(a)
 assert r['negated'] is True and r['relations']==['enables'] and r['term_name'] is None and r['missing_is_negative'] is False
 assert r['with_from_or_of_and']==[['UniProtKB:A','UniProtKB:B'],['UniProtKB:C']]
 assert r['extension_or_of_and']==[[{'relation':'occurs_in','identifier':'CL:1'}]]
 assert json.loads(r['full_source_json'])==a

def test_unknown_qualifier_and_future_nested_fields_do_not_become_false_negative():
 a=annotation();a['qualifier']=None;a['future']={'nested':[None,{'x':3}]};r=module().annotation_projection(a)
 assert r['negated'] is None and r['relations']==[] and json.loads(r['full_source_json'])['future']=={'nested':[None,{'x':3}]}

def test_form_scope_uses_exact_form_sequence_and_never_copies_canonical_labels():
 baseline={'P00001':'ACDEFGHIK'};forms={('P00001','PRO_1'):{'verified_sequence':'CDE'}};isoforms={'P00001-2':'ACDFGHIK'}
 assert module().bind_subject('UniProtKB:P00001-PRO_1',baseline,forms,isoforms)['sequence']=='CDE'
 assert module().bind_subject('UniProtKB:P00001-2',baseline,forms,isoforms)['sequence']=='ACDFGHIK'
 r=module().bind_subject('UniProtKB:P00001-3',baseline,forms,isoforms);assert r['sequence'] is None and r['canonical_labels_inherited'] is False

def test_annotation_packet_target_changes_cannot_change_sequence_predictors():
 a=annotation();binding=dict(parent_accession='P00001',subject='UniProtKB:P00001',sequence='ACDEFGHIK',sequence_binding='exact_current_canonical_source',canonical_labels_inherited=False)
 t=module().prediction_task(binding,[a],source_snapshot='2026-07-28 10:08')
 b=copy.deepcopy(a);b.update(qualifier='involved_in',goId='GO:9999999',goName='changedname',reference='PMID:999',symbol='changed');assert t.input_bytes()==module().prediction_task(binding,[b],source_snapshot='2026-07-28 10:08').input_bytes()
 assert json.loads(t.canonical_bytes())['conditions']=={'protein_sequence':'ACDEFGHIK'}
