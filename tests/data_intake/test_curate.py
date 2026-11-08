import importlib.util
from pathlib import Path
import numpy as np
import pytest
SPEC=importlib.util.spec_from_file_location('data_intake_curate',Path(__file__).parents[2]/'scripts/data_intake_curate.py')
m=importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(m)

def test_stereo_identity_preserved_but_leakage_group_shared():
 a=m.molecular_identity('C[C@H](O)Cl');b=m.molecular_identity('C[C@@H](O)Cl')
 assert a['canonical_isomeric_smiles']!=b['canonical_isomeric_smiles']
 assert a['molecule_group']==b['molecule_group']

@pytest.mark.parametrize('smiles',['garbage','*',''])
def test_invalid_or_wildcard_molecule_rejected(smiles):
 with pytest.raises(ValueError):m.molecular_identity(smiles)

def test_count_positions_and_integer_values():
 out=m.validate_count_chunk(np.array([0,1]),np.array([[0.,2.],[3.,0.]]),start=0,n_genes=2)
 assert out.dtype==np.int32 and out.sum()==5

@pytest.mark.parametrize('idx,val',[(np.array([1,2]),np.zeros((2,2))), (np.array([0,1]),np.array([[0.,-.1],[0.,1.]])),(np.array([0,1]),np.array([[0.,.5],[0.,1.]])),(np.array([0,1]),np.array([[0.,np.nan],[0.,1.]]))])
def test_invalid_count_blocks_rejected(idx,val):
 with pytest.raises(ValueError):m.validate_count_chunk(idx,val,start=0,n_genes=2)

def test_finite_labels_do_not_impute_missing():
 assert m.label_value('') is None and m.label_value('nan') is None
 assert m.label_value('0')==0.
 with pytest.raises(ValueError):m.label_value('inf')

def test_fresh_output_refuses_existing(tmp_path):
 target=tmp_path/'existing';target.mkdir();(target/'protected').write_text('keep')
 with pytest.raises(FileExistsError):m.fresh_output(target)
 assert (target/'protected').read_text()=='keep'

def test_source_stereo_survives_nonstereo_scaffold():
 s='C/C1=C/C=C(/C(C)C)CC/C(C)=C/CC/C(C)=C/CC1.O=P([O-])([O-])OP(=O)([O-])O'
 identity=m.molecular_identity(s)
 assert '/' in identity['canonical_isomeric_smiles']
 assert identity['scaffold_group'].startswith('scaffold:')
def test_decimal_annotation_score_and_exact_sequence_required():
 score=dict(Annotation='4.0',Reviewed='reviewed',Sequence='ACDE')
 assert m.annotation_quality(score,'ACDE')==(4,True)
 assert m.annotation_quality(score,'ACDF')==(4,False)
 assert m.annotation_quality(dict(score,Annotation='3.0'),'ACDE')==(3,False)

def test_freesolv_computed_label_never_default_experimental_target():
 x=m.property_label_contract('FreeSolv-original',{'expt':-5.7,'d_expt':0.6,'calc':-8.6,'d_calc':0.05})
 assert x['default_target_fields']==['expt']
 assert x['computed_target_opt_in_fields']==['calc']
 assert x['audit_uncertainty_fields']==['d_expt','d_calc']
 assert x['label_evidence']['expt']=='experimental'
 assert x['label_evidence']['calc']=='computed'
