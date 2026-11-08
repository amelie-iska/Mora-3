from pathlib import Path
import sys,hashlib,copy
import pytest,selfies as sf
from rdkit import Chem
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(R/'scripts'))
from data_intake_full_pubchem_census import validate_rows

def source(smiles='[13CH3][C@H](O)C(=O)[O-]',codec=True):
 iso=Chem.MolToSmiles(Chem.MolFromSmiles(smiles),isomericSmiles=True);return dict(native_smiles=smiles,source_record_id='test_fixture_only',source_native_file_row_index=17,native_current_RDKit_parse_pass=True,source_canonical_isomeric_smiles=iso,source_current_canonical_graph_sha256=hashlib.sha256(iso.encode()).hexdigest(),standard_selfies=sf.encoder(smiles) if codec else None,standard_selfies_target_present=codec,graph_roundtrip_pass=codec,coordinate_geometry_present=False,status='fixture',rare_unprojected_feature={'immutable':[1,2,3]})

def test_validation_is_audit_only_and_does_not_receive_train_codec_eligibility():
 r=source();a,g,e,c=validate_rows([r],'validation');assert a[0]['derived_native_graph_pass'];assert not a[0]['derived_SELFIES_graph_pass'];assert a[0]['source_SELFIES_target_present'];assert not e

def test_unsupported_codec_retains_valid_full_native_graph_with_source_bytes_unchanged():
 r=source(codec=False);before=copy.deepcopy(r);a,g,e,c=validate_rows([r],'train');assert a[0]['derived_native_graph_pass'] and not a[0]['derived_SELFIES_graph_pass'];assert a[0]['admission_gate_code']==0;assert r==before;assert a[0]['source_native_SMILES_sha256']==hashlib.sha256(r['native_smiles'].encode()).digest();assert g[0] and not e

def test_native_corruption_is_quarantined_with_original_reference_not_repaired():
 r=source();r['source_current_canonical_graph_sha256']='0'*64;a,g,e,c=validate_rows([r],'train');assert a[0]['admission_gate_code']&1 and not a[0]['derived_native_graph_pass'];assert e[0]['source_record_id']=='test_fixture_only';assert r['source_current_canonical_graph_sha256']=='0'*64

def test_whole_and_component_leakage_guards_keep_salt_roles_without_claiming_same_material():
 a,g,e,c=validate_rows([source('CC[NH3+].[Cl-]')],'train');assert a[0]['derived_native_graph_pass'];assert len(g[0])>=4;assert not e

def test_mode_must_not_default_to_validation_or_training():
 with pytest.raises(ValueError,match='mode'):validate_rows([source()],'unspecified')
