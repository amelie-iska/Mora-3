from pathlib import Path
import sys,json,hashlib
import pytest
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(R/'scripts'))
from data_intake_PubChem_tasks import validate_native,smiles_prior,codec_task
from data_intake_corpus import CorpusTask
import selfies as sf
from rdkit import Chem

def row(smiles='[13CH3][C@H](O)C(=O)[O-]',supported=True):
 iso=Chem.MolToSmiles(Chem.MolFromSmiles(smiles),canonical=True,isomericSmiles=True)
 return dict(native_smiles=smiles,native_current_RDKit_parse_pass=True,source_canonical_isomeric_smiles=iso,source_current_canonical_graph_sha256=hashlib.sha256(iso.encode()).hexdigest(),standard_selfies=sf.encoder(smiles) if supported else None,graph_roundtrip_pass=supported,standard_selfies_target_present=supported,status='standard_SELFIES_graph_roundtrip_pass' if supported else 'standard_SELFIES_encoding_or_decoding_quarantine',coordinate_geometry_present=False,source_record_id='audit-only',source_native_split='train')

def test_valid_smiles_remains_eligible_when_standard_SELFIES_is_unavailable():
 r=row(supported=False);v=validate_native(r);assert v['smiles_eligible'] and not v['selfies_eligible'];assert json.loads(smiles_prior(r,{}).canonical_bytes())['targets']=={'native_smiles':r['native_smiles']}
 with pytest.raises(ValueError,match='SELFIES'):codec_task(r,'smiles_to_selfies',{})

def test_source_graph_hash_and_stereo_are_independently_checked():
 r=row();r['source_current_canonical_graph_sha256']='0'*64
 with pytest.raises(ValueError,match='canonical'):validate_native(r)
 r=row();r['standard_selfies']=sf.encoder('[13CH3][C@@H](O)C(=O)[O-]')
 with pytest.raises(ValueError,match='graph'):validate_native(r)

def test_computed_codec_preserves_charge_isotope_and_directional_target_roles():
 r=row();a=json.loads(codec_task(r,'smiles_to_selfies',{}).canonical_bytes());b=json.loads(codec_task(r,'selfies_to_smiles',{}).canonical_bytes())
 assert a['conditions']=={'native_smiles':r['native_smiles'],'representation_task':'standard_SELFIES_codec'} and a['targets']=={'standard_selfies':r['standard_selfies']}
 assert b['conditions']=={'standard_selfies':r['standard_selfies'],'representation_task':'standard_SELFIES_codec'} and b['targets']=={'source_canonical_isomeric_smiles':r['source_canonical_isomeric_smiles']}
 assert a['audit']['evidence_class']=='computed_representation_of_source_reference_graph_not_experiment'

def test_hidden_targets_and_source_ids_lengths_labels_cannot_change_inputs():
 r=row();task=codec_task(r,'smiles_to_selfies',dict(source_record_id='one',target_length=10));raw=json.loads(task.canonical_bytes());raw['targets']={'hidden':'changed'};raw['audit']={'source_record_id':'changed','target_length':999999,'GO':'fake','assay':'fake'};raw['missing']=['changed']
 assert CorpusTask(raw['task'],raw['conditions'],raw['targets'],raw['audit'],raw['missing']).input_bytes()==task.input_bytes()
 assert 'audit-only' not in task.input_bytes().decode() and 'source_native_split' not in task.input_bytes().decode()

def test_invalid_native_graph_is_not_admitted_and_mask_claims_cannot_override_it():
 r=row();r['native_smiles']='C1bad'
 with pytest.raises(ValueError,match='native'):validate_native(r)
 r=row();r['native_current_RDKit_parse_pass']=False
 with pytest.raises(ValueError,match='native'):smiles_prior(r,{})

def test_incoherent_SELFIES_flags_and_fabricated_geometry_rejected():
 r=row();r['graph_roundtrip_pass']=False
 with pytest.raises(ValueError,match='SELFIES'):validate_native(r)
 r=row();r['coordinate_geometry_present']=True
 with pytest.raises(ValueError,match='geometry'):validate_native(r)

def test_full_long_native_axis_is_preserved_and_audit_only_length_not_conditioned():
 r=row('C'*400,supported=False);t=smiles_prior(r,{'length':400});d=json.loads(t.canonical_bytes());assert len(d['targets']['native_smiles'])==400;assert d['conditions']=={'prior_domain':'molecular_reference_native_SMILES'};assert '400' not in t.input_bytes().decode()

def test_codec_direction_must_be_explicit():
 with pytest.raises(ValueError,match='direction'):codec_task(row(),'guess',{})
