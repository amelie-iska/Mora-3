import importlib.util,json
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());s=importlib.util.spec_from_file_location('formbinding',root/'scripts/data_intake_uniprot_form_binding.py');m=importlib.util.module_from_spec(s)
try:s.loader.exec_module(m)
except FileNotFoundError:m=None

def module():assert m is not None,'explicit same-descriptor form binding missing';return m

def test_old_form_alias_uses_same_descriptor_not_prefix_heuristic():
 iso={'isoformIds':['P00001-2','Q99999-7'],'isoformSequenceStatus':'Described'};b=module().bind_isoform_descriptor(iso,'ACDEFGHIK',{'P00001-2':'ACDFGHIK'})
 assert b['verified_sequence']=='ACDFGHIK' and b['source_verified_ids']==['P00001-2'] and b['declared_ids']==['P00001-2','Q99999-7']
 other={'isoformIds':['Q99999-2'],'isoformSequenceStatus':'Described'};assert module().bind_isoform_descriptor(other,'ACDEFGHIK',{'P00001-2':'ACDFGHIK'})['verified_sequence'] is None

def test_same_descriptor_conflicting_native_sequences_are_quarantined():
 iso={'isoformIds':['P00001-2','Q99999-7'],'isoformSequenceStatus':'Described'};b=module().bind_isoform_descriptor(iso,'ACDEFGHIK',{'P00001-2':'AAAA','Q99999-7':'CCCC'})
 assert b['verified_sequence'] is None and b['status']=='conflicting_source_sequences'

def test_feature_on_verified_isoform_requires_that_exact_predictor_sequence():
 f=dict(type='Binding site',location=dict(sequence='P00001-2',start=dict(value=3,modifier='EXACT'),end=dict(value=3,modifier='EXACT')))
 e=dict(primaryAccession='P00001',sequence=dict(value='ACDEFGHIK'));r=module().bind_feature(f,e,{'P00001-2':{'verified_sequence':'ACDFGHIK','status':'verified_source_form'}})
 assert r['coordinate_supervision'] is True and r['interval0']==[2,3] and r['reference_id']=='P00001-2'
 assert r['canonical_entry_input_coordinate_supervision'] is False and r['requires_exact_reference_sequence_input'] is True
 f['location']['start']['modifier']='LESS_THAN';assert module().bind_feature(f,e,{'P00001-2':{'verified_sequence':'ACDFGHIK','status':'verified_source_form'}})['coordinate_supervision'] is False
