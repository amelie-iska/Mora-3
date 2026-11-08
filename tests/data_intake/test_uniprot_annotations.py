import importlib.util,copy,json,hashlib
from pathlib import Path
import pytest,pyarrow.parquet as pq
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
spec=importlib.util.spec_from_file_location('ann',root/'scripts/data_intake_uniprot_annotations.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def ev(code='ECO:0000269'):return dict(evidenceCode=code,source='PubMed',id='1')
def entry():return dict(primaryAccession='P00001',entryType='UniProtKB reviewed (Swiss-Prot)',annotationScore=5.0,entryAudit=dict(entryVersion=2,sequenceVersion=1),sequence=dict(value='ACDEFGHIK',length=9,molWeight=1018),organism=dict(taxonId=9606,scientificName='fixture'),proteinDescription=dict(recommendedName=dict(fullName=dict(value='hiddenname'),ecNumbers=[dict(value='2.7.-.-',evidences=[ev()])])),comments=[dict(commentType='FUNCTION',texts=[dict(value='sourcefunction',evidences=[ev()])])],uniProtKBCrossReferences=[dict(database='GO',id='GO:0000001',properties=[dict(key='GoTerm',value='F:source function'),dict(key='GoEvidenceType',value='IDA:source')])],features=[],references=[dict(citation=dict(id='123'))])
def test_exact_identity_reviewed_and_score_gates():
 e=entry();assert m.validate_entry(e,dict(Entry='P00001',Sequence='ACDEFGHIK',Annotation='5'))['annotation_score']==5
 for change in [dict(entryType='UniProtKB unreviewed (TrEMBL)'),dict(annotationScore=3.0)]:
  q=copy.deepcopy(e);q.update(change)
  with pytest.raises(ValueError):m.validate_entry(q)
 with pytest.raises(ValueError):m.validate_entry(e,dict(Entry='P00001',Sequence='WRONG',Annotation='5'))
def test_partial_ec_is_hierarchy_not_filled_negative():
 e=m.normalize_ec('2.7.-.-');assert e['identifier']=='2.7.-.-' and e['known_prefixes']==['2','2.7'] and e['complete'] is False
 q=m.normalize_ec('2.7.11.1');assert q['complete'] is True
 q=m.normalize_ec('unexpected-source-EC');assert q['identifier']=='unexpected-source-EC' and q['format_status']=='unparsed_source_identifier'
def test_go_aspect_evidence_and_unknown_negation_are_separate():
 r=m.normalize_go(entry()['uniProtKBCrossReferences'][0]);assert r['aspect']=='molecular_function' and r['term_name']=='source function' and r['source_evidence_code']=='IDA';assert r['negated'] is None and r['relation'] is None and r['closed_world_labels'] is False
 x=entry()['uniProtKBCrossReferences'][0];x['properties'].append(dict(key='Qualifier',value='NOT|enables'));r=m.normalize_go(x);assert r['negated'] is True and r['qualifier_source']=='NOT|enables'
def test_exact_fuzzy_and_isoform_coordinates_not_conflated():
 f=dict(type='Binding site',location=dict(start=dict(value=3,modifier='EXACT'),end=dict(value=4,modifier='EXACT')),alternativeSequence=dict(originalSequence='DE',alternativeSequences=['AA']))
 p=m.normalize_feature(f,'ACDEFGHIK','P00001');assert p['interval0']==[2,4] and p['coordinate_supervision'] is True
 g=copy.deepcopy(f);g['location']['end']['modifier']='LESS_THAN';q=m.normalize_feature(g,'ACDEFGHIK','P00001');assert q['interval0'] is None and q['coordinate_supervision'] is False
 g=copy.deepcopy(f);g['location']['sequence']='P00001-2';q=m.normalize_feature(g,'ACDEFGHIK','P00001');assert q['coordinate_supervision'] is False and q['scope']=='unverified_isoform_reference'
 g=copy.deepcopy(f);g['alternativeSequence']['originalSequence']='XX';q=m.normalize_feature(g,'ACDEFGHIK','P00001');assert q['coordinate_supervision'] is False and q['coordinate_status']=='reference_sequence_mismatch'
def test_disulfide_annotation_is_endpoint_pair_not_all_residues_in_interval():
 f=dict(type='Disulfide bond',location=dict(start=dict(value=1,modifier='EXACT'),end=dict(value=9,modifier='EXACT')))
 p=m.normalize_feature(f,'ACDEFGHIK','P00001');assert p['coordinate_semantics']=='endpoint_pair' and p['endpoint_indices0']==[0,8] and p['range_supervision'] is False

def test_rare_nested_source_fields_survive_arrow_after_first_record(tmp_path):
 a=entry();b=entry();b['primaryAccession']='P00002';b['extraAttributes']=dict(rareNested=dict(values=[dict(unexpected='retained',evidences=[ev('ECO:0000250')])]))
 t=m.full_source_table([a,b]);p=tmp_path/'rows.parquet';pq.write_table(t,p);back=pq.read_table(p).to_pylist();assert [json.loads(r['full_source_json']) for r in back]==[a,b]
def test_evidence_binds_to_exact_nested_text_not_entire_container():
 obj=dict(texts=[dict(value='experimental',evidences=[ev()]),dict(value='inferred',evidences=[ev('ECO:0000250')])]);bindings=m.evidence_bindings(obj,'/comments/0');assert {b['source_pointer'] for b in bindings}=={'/comments/0/texts/0','/comments/0/texts/1'};assert bindings[0]['evidences']!=bindings[1]['evidences']
def test_calculated_sequence_properties_and_vmax_never_become_measured_kcat():
 e=entry();e['comments'].append(dict(commentType='BIOPHYSICOCHEMICAL PROPERTIES',kineticParameters=dict(maximumVelocities=[dict(velocity=2.,unit='umol/min/mg',evidences=[ev()])]),phDependence=dict(texts=[dict(value='Optimum pH7.0')])));facts=m.physical_facts(e)
 assert any(f['parameter_type']=='sequence_mass' and f['evidence_role']=='provider_computed_from_canonical_sequence' for f in facts)
 assert any(f['parameter_type']=='maximumVelocity' and f['unit']=='umol/min/mg' for f in facts)
 assert not any(f['parameter_type'] in ('kcat','pI') for f in facts)
def test_annotation_prediction_names_citations_and_labels_cannot_reach_inputs():
 e=entry();a=m.annotation_prediction_task(e);e['proteinDescription']['recommendedName']['fullName']['value']='changed';e['comments'][0]['texts'][0]['value']='changed';e['references'][0]['citation']['id']='changed';e['uniProtKBCrossReferences'][0]['id']='GO:0000002';b=m.annotation_prediction_task(e);assert a.input_bytes()==b.input_bytes() and a.canonical_bytes()!=b.canonical_bytes()
def test_conditioned_generation_excludes_sequence_identity_computed_features_and_citations():
 e=entry();a=m.annotation_conditioned_sequence_task(e);q=copy.deepcopy(e);q['sequence']=dict(value='CCCCCCCCC',length=9,molWeight=9999);q['primaryAccession']='P00002';q['proteinDescription']['recommendedName']['fullName']['value']='differentname';q['references'][0]['citation']['id']='changed';q['comments'][0]['texts'][0]['evidences'][0]['id']='differentcitation';b=m.annotation_conditioned_sequence_task(q);assert a.input_bytes()==b.input_bytes() and a.canonical_bytes()!=b.canonical_bytes();d=json.loads(a.input_bytes());assert 'ACDEFGHIK' not in json.dumps(d) and '1018' not in json.dumps(d) and 'hiddenname' not in json.dumps(d)
def test_missing_or_only_inferred_conditions_do_not_invent_generation_supervision():
 e=entry();e['comments'][0]['texts'][0]['evidences']=[ev('ECO:0000250')];e['proteinDescription']['recommendedName']['ecNumbers'][0]['evidences']=[ev('ECO:0000250')];e['uniProtKBCrossReferences'][0]['properties'][1]['value']='IEA:source';assert m.annotation_conditioned_sequence_task(e) is None;assert m.annotation_prediction_task(e)

def test_string_ec_source_pointer_preserves_literal_source_value():
 e=entry();e['comments'].append(dict(commentType='CATALYTIC ACTIVITY',reaction=dict(ecNumber='1.2.3.4')));r=next(r for r in m.typed_assertions(e) if r['category']=='EC_identifier' and r['source_pointer'].endswith('/ecNumber'));assert json.loads(r['source_value_json'])=='1.2.3.4'
def test_binding_ligand_relationships_are_retained_with_feature_coordinates():
 f=dict(type='Binding site',location=dict(start=dict(value=3,modifier='EXACT'),end=dict(value=3,modifier='EXACT')),ligand=dict(name='ATP',id='ChEBI:CHEBI:15422'),ligandPart=dict(name='adenosine'))
 n=m.normalize_feature(f,'ACDEFGHIK','P00001');assert n['ligand']==f['ligand'] and n['ligand_part']==f['ligandPart']
def test_sequence_like_annotation_text_filter_is_not_target_dependent():
 e=entry();e['comments'][0]['texts'][0]['value']='Peptide ACDEFGHIK binds ligand';a=m.annotation_conditioned_sequence_task(e);e['sequence']=dict(value='CCCCCCCCC',length=9,molWeight=9999);b=m.annotation_conditioned_sequence_task(e);assert a.input_bytes()==b.input_bytes() and 'ACDEFGHIK' not in a.input_bytes().decode()
