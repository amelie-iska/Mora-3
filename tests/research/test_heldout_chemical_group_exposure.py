"""Target-inaccessible, actual original FreeSolv heldout source-group audit."""
import pytest
from pathlib import Path
from mora3.research.heldout_chemical_group_exposure import (
    group_overlap, audit_original_freesolv_heldout_group_exposure,
)

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')

def _row(source,role,id,mol,scaffold):
    return dict(source=source,role=role,task_id=id*64,
                molecule_group='molecule:'+mol*64,scaffold_group='scaffold:'+scaffold*64)

def test_hash_overlap_preserves_role_and_group_kind():
    train={'AZ':(_row('AZ','train','a','1','2'),),
           'FreeSolv':(_row('FreeSolv','train','b','3','4'),)}
    held={'validation':(_row('FreeSolv','validation','c','1','5'),),
          'test':(_row('FreeSolv','test','d','6','4'),)}
    result=group_overlap(train,held)
    assert result['heldout_comparisons']['validation_vs_AZ_train']['molecule_hash_collision_count']==1
    assert result['heldout_comparisons']['validation_vs_AZ_train']['scaffold_hash_collision_count']==0
    assert result['heldout_comparisons']['test_vs_FreeSolv_train']['molecule_hash_collision_count']==0
    assert result['heldout_comparisons']['test_vs_FreeSolv_train']['scaffold_hash_collision_count']==1
    assert result['global_split_admitted'] is False


def test_no_role_substitution_or_duplicate_task():
    train={'AZ':(_row('AZ','train','a','1','2'),),
           'FreeSolv':(_row('FreeSolv','train','b','3','4'),)}
    held={'validation':(_row('FreeSolv','validation','c','5','6'),),
          'test':(_row('FreeSolv','test','d','7','8'),)}
    assert group_overlap(train,held)['all_scoped_group_hashes_disjoint'] is True
    with pytest.raises(ValueError):
        group_overlap(train,{'validation':held['validation']+held['validation'],'test':held['test']})
    with pytest.raises(ValueError):
        group_overlap(train,{'validation':(_row('FreeSolv','train','c','5','6'),),'test':held['test']})
    with pytest.raises(ValueError):
        group_overlap({'AZ':train['AZ'],'FreeSolv':()},held)


def test_original_heldout_only_source_sidecars_and_no_target_columns():
    r=audit_original_freesolv_heldout_group_exposure(ROOT)
    assert r['train_rows']=={'AZ':4184,'FreeSolv':556}
    assert r['heldout_rows']=={'validation':42,'test':44}
    assert r['group_sidecars_authenticated']==22
    assert r['read_columns_heldout']==['task_id','source_file','source_group_ids_json','effective_groups_json']
    assert r['heldout_native_shards_opened'] is False
    assert r['heldout_target_payload_decoded'] is False
    assert r['all_scoped_group_hashes_disjoint'] is True
    assert len(r['heldout_comparisons'])==5
    assert r['global_split_admitted'] is False
    assert r['training_admission'] is False
    assert r['optimizer_updates']==r['actual_proxy_runs']==r['actual_target_runs']==0
