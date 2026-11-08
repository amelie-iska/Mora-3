from pathlib import Path
import sys,json,time,hashlib
import pytest
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists());sys.path.insert(0,str(R/'scripts'))
from data_intake_full_pubchem_guards import BuildGuard,immutable_json,task_identifier

def test_prewrite_guard_rejects_reservation_before_any_protected_bytes_change(tmp_path):
 p=tmp_path/'old-source.bin';p.write_bytes(b'protected');g=BuildGuard(tmp_path,census_cap=100,total_cap=200,deadline_epoch=time.time()+100,min_free=0)
 before=p.read_bytes()
 with pytest.raises(RuntimeError,match='census'):g.check('census',reserve=100)
 assert p.read_bytes()==before and sorted(x.name for x in tmp_path.iterdir())==['old-source.bin']

def test_task_phase_does_not_borrow_unused_census_allowance(tmp_path):
 (tmp_path/'materialized').mkdir();(tmp_path/'materialized/a').write_bytes(b'x'*30);g=BuildGuard(tmp_path,census_cap=100,total_cap=120,deadline_epoch=time.time()+100,min_free=0)
 with pytest.raises(RuntimeError,match='materialized'):g.check('materialized',reserve=1)

def test_deadline_enforced_and_does_not_delete_checkpoint(tmp_path):
 p=tmp_path/'checkpoint.json';immutable_json(p,{'completed':5000});g=BuildGuard(tmp_path,census_cap=1000,total_cap=2000,deadline_epoch=time.time()-1,min_free=0)
 with pytest.raises(RuntimeError,match='deadline'):g.check('census')
 assert json.loads(p.read_text())=={'completed':5000}

def test_source_chunk_checkpoints_are_exclusive_not_rewritten(tmp_path):
 p=tmp_path/'chunk-0001.json';immutable_json(p,{'rows':5000});before=p.read_bytes()
 with pytest.raises(FileExistsError):immutable_json(p,{'rows':0})
 assert p.read_bytes()==before

def test_task_identity_preserves_existing_pilot_namespace_recipe():
 from data_intake_corpus import canonical
 source='HF:sagawa/pubchem-10m-canonicalized@f83219601635a0a80fc99c13a9ca37f99ef34f0a:train:17'
 assert task_identifier('PubChem-standardSELFIES-codec',source,'smiles_to_selfies')==hashlib.sha256(canonical(['PubChem-standardSELFIES-codec',source,'smiles_to_selfies'])).hexdigest()
 assert task_identifier('PubChem-standardSELFIES-codec',source,'smiles_to_selfies')!=task_identifier('PubChem-standardSELFIES-codec',source,'selfies_to_smiles')

def test_total_and_phase_caps_are_checked_including_preserved_partial_files(tmp_path):
 (tmp_path/'partial').write_bytes(b'x'*60);g=BuildGuard(tmp_path,census_cap=100,total_cap=120,deadline_epoch=time.time()+100,min_free=0)
 assert g.check('census',reserve=30)['all_bytes']==60
 with pytest.raises(RuntimeError,match='total'):g.check('census',reserve=61)
 assert (tmp_path/'partial').stat().st_size==60

def test_free_space_reservation_is_checked_before_materialization(tmp_path):
 g=BuildGuard(tmp_path,census_cap=100,total_cap=120,deadline_epoch=time.time()+100,min_free=2**80)
 with pytest.raises(RuntimeError,match='free'):g.check('census',reserve=1)
