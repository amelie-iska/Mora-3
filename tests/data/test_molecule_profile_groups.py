"""Read-only complete source groups; source fixtures do not authorize training."""
import hashlib,json,sqlite3
from pathlib import Path
import pytest


def encoded(x):return json.dumps(x,sort_keys=True,separators=(',',':')).encode()

def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


@pytest.fixture
def group_files(tmp_path):
    condition=b'{"assay_modality":"L1000","source_profile_array":"synthetic-unit"}'
    obs=[dict(canonical_isomeric_smiles=s,split='train',source_family='Rosetta-CDRP-molecule-profile-L1000',condition_sha256=hashlib.sha256(condition).hexdigest(),task_id=str(i))for i,s in enumerate(('CCO','CCO','N'))]
    db=tmp_path/'source-groups.sqlite';c=sqlite3.connect(db);c.execute('CREATE TABLE source_groups (condition BLOB PRIMARY KEY, condition_sha TEXT NOT NULL, observations TEXT NOT NULL)');c.execute('INSERT INTO source_groups VALUES(?,?,?)',(condition,hashlib.sha256(condition).hexdigest(),json.dumps(obs)));c.commit();c.close()
    m=tmp_path/'manifest.json';m.write_bytes(encoded(dict(schema='Mora3.CompleteNativeRosettaConditionGroups/v1',train_condition_groups=1,complete_retained_inverse_rows=[dict(split='train',rows=3)],encoding_failures=[],artifacts=[ref(db)])))
    proof=tmp_path/'independent.json';proof.write_bytes(encoded(dict(schema='Mora3.IndependentCompleteNativeRosettaConditionGroups/v1',producer_manifest=ref(m),complete_native_source_observations=3,complete_exact_full_native_condition_groups=1,every_artifact_bytes_SHA_and_currentidentity_stable=True,all_source_condition_bytes_exact_SQLite_BLOB_primary_keys=True,all_target_source_values_preserved=True,grouped_JSONL_and_immutable_database_complete_both_directions=True,native_order_and_duplicate_observation_multisets_exact=True,uniform_observation_1_over_n_weights_exact=True,source_reward_or_global_training_admission=False)))
    return m,proof,db,condition


def make(files):
    from mora3.data.molecule_profile_groups import ImmutableRosettaSourceGroups
    m,p,db,c=files
    return ImmutableRosettaSourceGroups(m,ref(m)['sha256'],independent_manifest=p,independent_sha256=ref(p)['sha256'],root=m.parent)


def test_all_duplicate_observations_source_order_and_read_only_lookup(group_files):
    r=make(group_files);m,p,db,c=group_files;before=db.read_bytes()
    assert r.molecules_for_condition(c)==('CCO','CCO','N')
    assert r.binding()['source_observations']==3 and r.binding()['condition_groups']==1
    assert r.binding()['database_hash_bytes']==db.stat().st_size
    assert r.binding()['training_authorization']is False
    with pytest.raises(ValueError,match='not.*source'):r.molecules_for_condition(b'{"different":1}')
    assert db.read_bytes()==before;r.close()


def test_modified_database_and_wrong_independent_manifest_refuse(group_files):
    m,p,db,c=group_files
    r=make(group_files)
    with db.open('ab')as f:f.write(b'corruption')
    with pytest.raises(ValueError,match='identity'):r.molecules_for_condition(c)
    r.close()
    with pytest.raises(ValueError,match='bytes|SHA'):make(group_files)


def test_small_review_proof_hash_and_group_count_authentication(group_files):
    m,p,db,c=group_files;d=json.loads(p.read_bytes());d['complete_exact_full_native_condition_groups']=2;p.write_bytes(encoded(d))
    with pytest.raises(ValueError,match='complete'):make(group_files)
