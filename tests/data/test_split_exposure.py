"""Synthetic streaming group-role checks, not whole-corpus qualification."""
from copy import deepcopy
import pytest
from mora3.data.split_exposure import HeldoutGroupExposure


def test_cross_role_transitive_bridge_and_duplicate_identity():
    audit=HeldoutGroupExposure(max_heldout_keys=20,max_heldout_rows=20,max_groups_per_row=8,max_examples=10)
    audit.add_heldout('v',1,('group:a',),'v:0')
    audit.add_heldout('t',2,('group:b',),'t:0')
    audit.freeze_heldout()
    audit.add_train('train1',('group:a','group:x'),'s:0')
    audit.add_train('train2',('group:x','group:b'),'s:1')
    audit.add_train('v',('different',),'s:2')
    out=audit.finish()
    assert out['cross_role_group_keys']==2 and out['train_rows_with_group_exposure']==2
    assert out['cross_role_task_ids']==1 and out['train_rows_with_task_id_exposure']==1
    assert not out['declared_cross_role_disjoint']
    assert out['role_rows']=={'train':3,'validation':1,'test':1}


def test_validation_test_edges_and_missing_groups_are_explicit():
    audit=HeldoutGroupExposure(max_heldout_keys=20,max_heldout_rows=20,max_groups_per_row=8,max_examples=10)
    audit.add_heldout('v',1,('shared',),'v')
    audit.add_heldout('t',2,('shared',),'t')
    audit.add_heldout('u',2,(),'u')
    audit.freeze_heldout();audit.add_train('a',(),'a')
    out=audit.finish()
    assert out['cross_role_group_keys']==1
    assert out['missing_group_rows']=={'train':1,'validation':0,'test':1}
    assert out['all_rows_have_declared_groups']is False


def test_disjoint_and_same_train_repeated_key_never_claims_deduplication():
    audit=HeldoutGroupExposure(max_heldout_keys=20,max_heldout_rows=20,max_groups_per_row=8,max_examples=2)
    audit.add_heldout('v',1,('V',),'v');audit.add_heldout('t',2,('T',),'t')
    audit.freeze_heldout()
    for i in range(20):audit.add_train('same-train',('A',),'s:'+str(i))
    out=audit.finish();assert out['declared_cross_role_disjoint']is True
    assert out['within_train_duplicate_audited']is False and out['unknown_biological_dependency_excluded']is False
    assert out['role_rows']['train']==20 and out['training_authorized']is False
    with pytest.raises(ValueError):audit.add_train('later',('A',),'later')


def test_example_budget_never_truncates_counts():
    audit=HeldoutGroupExposure(max_heldout_keys=20,max_heldout_rows=20,max_groups_per_row=8,max_examples=1)
    audit.add_heldout('v',1,('A','B'),'v');audit.freeze_heldout()
    for i in range(10):audit.add_train('x'+str(i),('A','B'),'s:'+str(i))
    out=audit.finish()
    assert len(out['examples'])==1 and out['cross_role_group_keys']==2
    assert out['train_rows_with_group_exposure']==10


@pytest.mark.parametrize('fault',('keys','rows','groups','role','duplicate-group','malformed-id','unfinished','order','bool-budget'))
def test_invalid_or_over_budget_refuses(fault):
    if fault=='bool-budget':
        with pytest.raises(ValueError):HeldoutGroupExposure(max_heldout_keys=True,max_heldout_rows=1,max_groups_per_row=1,max_examples=1)
        return
    audit=HeldoutGroupExposure(max_heldout_keys=1,max_heldout_rows=1,max_groups_per_row=2,max_examples=1)
    if fault=='keys':args=('x',1,('A','B'),'x')
    elif fault=='groups':args=('x',1,('A','B','C'),'x')
    elif fault=='role':args=('x',0,('A',),'x')
    elif fault=='duplicate-group':args=('x',1,('A','A'),'x')
    elif fault=='malformed-id':args=('',1,('A',),'x')
    elif fault=='unfinished':
        with pytest.raises(ValueError):audit.finish()
        return
    elif fault=='order':
        with pytest.raises(ValueError):audit.add_train('x',('A',),'x')
        return
    else:
        audit.add_heldout('x',1,('A',),'x');args=('y',2,('A',),'y')
    with pytest.raises(ValueError):audit.add_heldout(*args)
