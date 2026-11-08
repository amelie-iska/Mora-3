import importlib.util,json
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
spec=importlib.util.spec_from_file_location('genv2',root/'scripts/data_intake_uniprot_generation_v2.py');m=importlib.util.module_from_spec(spec)
try:spec.loader.exec_module(m)
except FileNotFoundError:m=None

def ev():return dict(evidenceCode='ECO:0000269',source='PubMed',id='123')
def fixture():return dict(primaryAccession='P00001',entryType='UniProtKB reviewed (Swiss-Prot)',annotationScore=5,sequence=dict(value='ACDEFGHIK',length=9,molWeight=1018),organism=dict(taxonId=9606),comments=[],features=[],uniProtKBCrossReferences=[])
def module():assert m is not None,'scope-safe generation projection missing';return m

def test_isoform_specific_kinetics_do_not_condition_canonical_sequence():
 e=fixture();e['comments']=[dict(commentType='BIOPHYSICOCHEMICAL PROPERTIES',molecule='Isoform 2',kineticParameters=dict(michaelisConstants=[dict(constant=2,unit='mM',substrate='ATP',evidences=[ev()])]))]
 assert module().generation_task(e) is None

def test_scoped_ec_does_not_condition_canonical_but_canonical_ec_remains():
 e=fixture();e['comments']=[dict(commentType='CATALYTIC ACTIVITY',molecule='Isoform 2',reaction=dict(ecNumber='1.2.3.4',evidences=[ev()]))]
 assert module().generation_task(e) is None
 e['proteinDescription']=dict(recommendedName=dict(ecNumbers=[dict(value='2.7.-.-',evidences=[ev()])]))
 t=module().generation_task(e);c=json.loads(t.canonical_bytes())['conditions'];assert c['source_experimentally_supported_EC_ids']==['2.7.-.-']

def test_canonical_kinetics_keep_units_local_scope_and_exclude_generic_ph():
 e=fixture();e['comments']=[dict(commentType='BIOPHYSICOCHEMICAL PROPERTIES',kineticParameters=dict(maximumVelocities=[dict(velocity=4,unit='nmol/min/mg',enzyme='enzyme',evidences=[ev()])]),phDependence={'texts':[{'value':'pH 7'}]})]
 t=module().generation_task(e);c=json.loads(t.canonical_bytes())['conditions'];assert c['source_kinetic_conditions']==[dict(parameter_type='maximumVelocity',source_value=4,source_unit='nmol/min/mg',substrate_label=None,assay_pH=None,assay_temperature=None)]
 e['sequence']={'value':'YYYYYYYYY','length':9,'molWeight':9000};e['primaryAccession']='P99999';e['comments'][0]['kineticParameters']['maximumVelocities'][0]['evidences'][0]['id']='999';assert t.input_bytes()==module().generation_task(e).input_bytes()
