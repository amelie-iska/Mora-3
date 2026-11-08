import importlib.util,json
from pathlib import Path
root=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
spec=importlib.util.spec_from_file_location('sup',root/'scripts/data_intake_uniprot_supplement.py')
m=importlib.util.module_from_spec(spec) if spec else None
if spec and spec.loader:
 try:spec.loader.exec_module(m)
 except FileNotFoundError:m=None

def fixture():return dict(primaryAccession='P00001',sequence=dict(value='ACDEFGHIK',length=9,molWeight=1018),comments=[],features=[])
def module():assert m is not None,'physical/form projection is not implemented';return m

def test_actual_mass_spectrometry_keys_and_zero_error_are_not_lost():
 e=fixture();e['comments']=[dict(commentType='MASS SPECTROMETRY',method='MALDI',molWeight=55211.89,molWeightError=0.0,evidences=[dict(evidenceCode='ECO:0000269')])]
 r=[r for r in module().physical_records(e) if r['parameter_type']=='mass_spectrometry_source_mass'][0]
 assert r['value']==55211.89 and r['error']==0.0 and r['method']=='MALDI'
 assert r['unit'] is None and r['numeric_objective_eligible'] is False
 assert json.loads(r['source_value_json'])['molWeight']==55211.89

def test_vmax_is_not_kcat_and_generic_ph_is_not_assay_context():
 e=fixture();e['comments']=[dict(commentType='BIOPHYSICOCHEMICAL PROPERTIES',kineticParameters=dict(maximumVelocities=[dict(velocity=2.5,unit='umol/min/mg',enzyme='enzyme')]),phDependence=dict(texts=[dict(value='Optimum pH is 7.')]))]
 rows=module().physical_records(e);v=next(r for r in rows if r['parameter_type']=='maximumVelocity')
 assert v['value']==2.5 and v['unit']=='umol/min/mg' and v['assay_pH'] is None and v['assay_temperature'] is None
 assert v['converted_to_kcat'] is False
 p=next(r for r in rows if r['parameter_type']=='phDependence');assert p['value'] is None and p['unit'] is None and '7.' in p['source_value_json']

def test_unknown_biophysical_keys_and_molecule_scope_remain_lossless():
 e=fixture();e['comments']=[dict(commentType='BIOPHYSICOCHEMICAL PROPERTIES',molecule='Isoform 2',futureProperty=dict(units='new',nested=[1,None,{'rare':True}]))]
 r=next(r for r in module().physical_records(e) if r['parameter_type']=='futureProperty')
 assert r['molecule_scope']=='Isoform 2' and r['canonical_property_supervision'] is False
 assert json.loads(r['source_value_json'])=={'units':'new','nested':[1,None,{'rare':True}]}

def test_displayed_described_and_external_isoforms_do_not_share_sequence_truth():
 e=fixture();e['comments']=[dict(commentType='ALTERNATIVE PRODUCTS',events=['Alternative splicing'],isoforms=[dict(name={'value':'1'},isoformIds=['P00001-1'],isoformSequenceStatus='Displayed'),dict(isoformIds=['P00001-2'],isoformSequenceStatus='Described',sequenceIds=['VSP_1']),dict(isoformIds=['P00001-3'],isoformSequenceStatus='External')])]
 rows=module().form_records(e);by={r['form_id']:r for r in rows}
 assert by['P00001-1']['verified_sequence']=='ACDEFGHIK'
 assert by['P00001-2']['verified_sequence'] is None and by['P00001-3']['verified_sequence'] is None
 assert by['P00001-2']['sequence_feature_ids']==['VSP_1']
 assert all(r['canonical_labels_inherited'] is False for r in rows)

def test_processed_chain_exact_bounds_and_isoform_scope_stay_distinct():
 e=fixture();e['features']=[dict(type='Chain',featureId='PRO_1',location=dict(start=dict(value=2,modifier='EXACT'),end=dict(value=4,modifier='EXACT'))),dict(type='Peptide',featureId='PRO_2',location=dict(sequence='P00001-2',start=dict(value=2,modifier='EXACT'),end=dict(value=4,modifier='EXACT')))]
 by={r['form_id']:r for r in module().form_records(e)}
 assert by['PRO_1']['canonical_interval0']==[1,4] and by['PRO_1']['verified_sequence']=='CDE'
 assert by['PRO_2']['canonical_interval0'] is None and by['PRO_2']['verified_sequence'] is None
