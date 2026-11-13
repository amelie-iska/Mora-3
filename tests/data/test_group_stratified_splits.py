"""Synthetic allocation tests; no evidence of whole-corpus biological independence."""
from collections import Counter
from copy import deepcopy
import pytest
from mora3.data.splits import allocate_stratified_groups,assign_stratified_splits,audit_stratified_splits
from mora3.data.schema import GraphRecord,Node
from mora3.data.provenance import Provenance,LeakageKey


def records():
    return tuple(GraphRecord(str(i),(Node('test'),),(),
        (Provenance('test-source','v1','record:'+str(i),'a'*64,'test-only'),),
        (LeakageKey('declared-group-v1','group:'+str(i//2)),))for i in range(40))


def counts(n=20):
    return {str(i):dict(rows=2,strata={'modality:A':2,'source:fixture':2},fixed_split=None)for i in range(n)}


def test_exact_balanced_feasible_equal_groups_and_input_order_invariance():
    data=counts();before=deepcopy(data)
    a=allocate_stratified_groups(data,seed=19,fractions=(.6,.2,.2))
    b=allocate_stratified_groups(dict(reversed(tuple(data.items()))),seed=19,fractions=(.6,.2,.2))
    assert a==b and data==before
    assert a['rows_by_split']=={'train':24,'validation':8,'test':8}
    assert a['strata_by_split']['validation']['modality:A']==8
    assert a['optimality_proven']is False and a['unknown_leakage_excluded']is False


def test_component_counts_support_large_row_populations_without_row_materialization():
    data={str(i):dict(rows=10_000_000,strata={'native:RNA':10_000_000},fixed_split=None)for i in range(10)}
    out=allocate_stratified_groups(data,seed=2,fractions=(.8,.1,.1))
    assert out['total_rows']==100_000_000
    assert out['rows_by_split']=={'train':80_000_000,'validation':10_000_000,'test':10_000_000}


def test_transitive_links_and_official_roles_are_not_split_to_balance():
    rs=list(records());rs[2]=GraphRecord(rs[2].record_id,rs[2].nodes,(),rs[2].provenance,
        rs[2].leakage_keys+(LeakageKey('declared-group-v1','group:0'),))
    strata={x.record_id:('modality:A',)for x in rs}
    fixed={'0':'test','10':'validation'}
    out=assign_stratified_splits(rs,strata=strata,fixed_assignments=fixed,seed=29,fractions=(.8,.1,.1))
    assert all(out['assignments'][str(i)]=='test'for i in range(4))
    assert out['assignments']['10']==out['assignments']['11']=='validation'
    audit=audit_stratified_splits(rs,out['assignments'],strata=strata,fixed_assignments=fixed)
    assert audit['declared_group_conflicts']==[]and audit['fixed_role_conflicts']==[]
    assert audit['rows']==40 and audit['known_constraints_satisfied']is True
    bad=dict(out['assignments']);bad['1']='train'
    assert not audit_stratified_splits(rs,bad,strata=strata,fixed_assignments=fixed)['known_constraints_satisfied']
    with pytest.raises(ValueError,match='fixed'):
        assign_stratified_splits(rs,strata=strata,fixed_assignments={'0':'train','3':'test'},seed=29)


def test_repeated_original_provenance_across_revisions_stays_together():
    a=GraphRecord('a',(Node('test'),),(),(Provenance('src','v1','same','a'*64,'test'),))
    b=GraphRecord('b',(Node('test'),),(),(Provenance('src','v2','same','b'*64,'test'),))
    with pytest.raises(ValueError,match='fixed'):
        assign_stratified_splits((a,b),strata={'a':('A',),'b':('B',)},fixed_assignments={'a':'train','b':'test'},seed=1)


def test_one_giant_component_reports_empty_partitions_without_fabrication():
    out=allocate_stratified_groups({'giant':dict(rows=101,strata={'A':101},fixed_split='train')},seed=2)
    assert out['rows_by_split']=={'train':101,'validation':0,'test':0}
    assert out['empty_requested_splits']==['validation','test']
    assert out['ratios_matched_exactly']is False


@pytest.mark.parametrize('change',('bool-count','negative','stratum-overcount','unknown-role','extra-field','empty','zero-fraction-fixed','group-cap','stratum-cap'))
def test_invalid_metadata_or_impossible_policy_refuses(change):
    data=counts(2);kw={}
    if change=='bool-count':data['0']['rows']=True
    elif change=='negative':data['0']['rows']=-1
    elif change=='stratum-overcount':data['0']['strata']['modality:A']=3
    elif change=='unknown-role':data['0']['fixed_split']='heldout-maybe'
    elif change=='extra-field':data['0']['hidden_target']=[1,2]
    elif change=='empty':data={}
    elif change=='zero-fraction-fixed':data['0']['fixed_split']='test';kw['fractions']=(1.,0.,0.)
    elif change=='group-cap':kw['max_groups']=1
    else:kw['max_strata']=1
    with pytest.raises(ValueError):allocate_stratified_groups(data,seed=1,**kw)


@pytest.mark.parametrize('bad',('missing-stratum','unknown-fixed','duplicate-tag','empty-tag','record-budget'))
def test_native_metadata_exact_coverage_and_bounds(bad):
    rs=records();strata={x.record_id:('modality:A',)for x in rs};fixed={};kw={}
    if bad=='missing-stratum':strata.pop('0')
    elif bad=='unknown-fixed':fixed={'missing':'test'}
    elif bad=='duplicate-tag':strata['0']=('A','A')
    elif bad=='empty-tag':strata['0']=('',)
    else:kw['max_records']=2
    with pytest.raises(ValueError):assign_stratified_splits(iter(rs),strata=strata,fixed_assignments=fixed,seed=7,**kw)


def test_stratum_counts_and_fixed_roles_are_independently_recomputed():
    rs=records();strata={x.record_id:('A'if int(x.record_id)<18 else'B',)for x in rs}
    out=assign_stratified_splits(rs,strata=strata,fixed_assignments={'0':'test'},seed=7)
    assert Counter(out['assignments'].values())==out['rows_by_split']
    for split in ('train','validation','test'):
        expected=Counter(tag for x in rs if out['assignments'][x.record_id]==split for tag in strata[x.record_id])
        assert all(out['strata_by_split'][split][tag]==expected[tag]for tag in ('A','B'))
    audit=audit_stratified_splits(rs,out['assignments'],strata=strata,fixed_assignments={'0':'train'})
    assert audit['fixed_role_conflicts']and not audit['known_constraints_satisfied']


@pytest.mark.parametrize('which',('group','stratum'))
def test_mixed_type_metadata_keys_fail_explicitly_before_sorting(which):
    data=counts(2)
    if which=='group':data[1]=data.pop('1')
    else:data['0']['strata'][1]=1
    with pytest.raises(ValueError):allocate_stratified_groups(data,seed=1)
