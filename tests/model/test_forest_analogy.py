"""Learned whole-forest retrieval on explicit synthetic, non-admitted examples."""
from dataclasses import replace
import copy
import pytest
import torch
from test_forest_flow import make,problem,skeleton,state,SOURCE
from mora3.model.forest_analogy import (ForestAnalogySpec,ForestAnalogyItem,NativeForestAnalogyHead,
    ForestAnalogyIndex,forest_analogy_loss,forest_analogy_parameter_roles)
from mora3.model.parameterization import parameter_ledger,width_family

GROUP='d'*64


def head(width=8):
    return NativeForestAnalogyHead(make(width),ForestAnalogySpec(4,GROUP,max_memory=32,max_centers=8)).double()


def item(i,*,split='train',group=None):
    r,t=problem(dose=float(i+1));f=skeleton(r,t);f=replace(f,states=state(1)[0]+i*.2)
    return ForestAnalogyItem(str(i),r,t,f,(group or 'group:'+str(i),),split,GROUP,'e'*64)


def test_actual_learned_queries_do_not_include_hidden_answers():
    h=head();r,t=problem();r2,t2=problem(target=10000.);r3,t3=problem(dose=2.)
    assert torch.equal(h.query(r,t),h.query(r2,t2))
    assert not torch.equal(h.query(r,t),h.query(r3,t3))
    a=item(0);b=replace(a,forest=replace(a.forest,states=a.forest.states.clone()))
    b.forest.states[1,1]+=3
    assert torch.equal(a.forest.states[-1],b.forest.states[-1])
    assert not torch.equal(h.memory_key(a),h.memory_key(b))


def test_explicit_pair_loss_learns_shared_context_and_both_readouts():
    h=head();queries=(item(0,group='query:0'),item(1,group='query:1'));memories=(item(3),item(4),item(5))
    positive=torch.tensor([[True,False,False],[False,True,False]])
    known=torch.tensor([[True,True,False],[True,True,True]])
    result=forest_analogy_loss(h,queries,memories,positive,known)
    q=torch.cat([h.query(x.record,x.task)for x in queries]);m=torch.cat([h.memory_key(x)for x in memories])
    logits=q@m.T/h.spec.temperature
    expected=(torch.logsumexp(logits.masked_fill(~known,float('-inf')),1)
        -torch.logsumexp(logits.masked_fill(~positive,float('-inf')),1)).mean()
    assert torch.equal(result['loss'],expected)and result['known_pair_count']==5
    result['loss'].backward()
    for p in(h.query_readout.weight,h.memory_readout.weight,h.field.geometry.payload.embedding.weight,h.field.state_projection.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()and p.grad.abs().sum()>0
    assert h.field.readout.weight.grad is None


@pytest.mark.parametrize('fault',['query-heldout','memory-heldout','same-group','unknown-positive','no-negative','group-policy'])
def test_loss_refuses_wrong_split_groups_and_fabricated_negatives(fault):
    h=head();q=(item(0,group='query'),);m=(item(1),item(2))
    positive=torch.tensor([[True,False]]);known=torch.tensor([[True,True]])
    if fault=='query-heldout':q=(replace(q[0],split='test'),)
    if fault=='memory-heldout':m=(replace(m[0],split='validation'),m[1])
    if fault=='same-group':m=(replace(m[0],groups=q[0].groups),m[1])
    if fault=='unknown-positive':known[0,0]=False
    if fault=='no-negative':known[0,1]=False
    if fault=='group-policy':q=(replace(q[0],group_policy_sha256='f'*64),)
    with pytest.raises(ValueError):forest_analogy_loss(h,q,m,positive,known)


def test_index_is_train_only_and_actual_l2_with_exact_reference():
    h=head();memories=tuple(item(i)for i in range(10));index=ForestAnalogyIndex(h,memories,centers=3)
    query=item(13,split='test',group='group:0')
    found=index.search(query,k=4,probes=3,exact=False)
    oracle=index.search(query,k=4,exact=True)
    assert found['identities']==oracle['identities']and found['distances']==oracle['distances']
    assert '0'not in found['identities']and found['examined_candidates']==9
    narrow=index.search(query,k=4,probes=1)
    assert narrow['examined_candidates']<=9 and narrow['exact_global_retrieval']is False
    keys=torch.cat([h.memory_key(x)for x in memories])
    distances=torch.linalg.vector_norm(keys-h.query(query.record,query.task),dim=1)
    eligible=[i for i,x in enumerate(memories)if not set(x.groups)&set(query.groups)]
    expected=sorted(eligible,key=lambda i:(float(distances[i].detach()),str(i)))[:4]
    assert oracle['identities']==tuple(str(i)for i in expected)
    assert oracle['directed_semantic_mapping_verified']is False
    with pytest.raises(ValueError):ForestAnalogyIndex(h,(replace(memories[0],split='test'),))


def test_index_stale_weights_or_mutable_memory_cannot_silently_change_results():
    h=head();items=(item(1),item(2));index=ForestAnalogyIndex(h,items)
    q=item(8,split='validation');original=index.search(q,k=2,exact=True)
    items[0].forest.states.add_(999)
    assert index.search(q,k=2,exact=True)==original
    with torch.no_grad():h.memory_readout.weight.add_(.1)
    with pytest.raises(ValueError,match='stale|changed'):index.search(q,k=2,exact=True)


def test_index_empty_after_group_exclusion_reports_shortfall_not_duplicates():
    h=head();index=ForestAnalogyIndex(h,(item(1,group='same'),item(2,group='same')))
    out=index.search(item(9,group='same'),k=3,exact=True)
    assert out['identities']==()and out['shortfall']==3


def test_analogy_axes_are_complete_across_the_native_width_family():
    hs=[head(w)for w in(8,12,16)];ledgers=[]
    for h in hs:
        roles=forest_analogy_parameter_roles(h,identifier_axis=tuple('geometry:'+str(i)for i in range(8)))
        assert set(roles)==set(dict(h.named_parameters(remove_duplicate=False)))
        ledgers.append(parameter_ledger(h,roles))
    assert len(set(width_family(*ledgers).total_numel))==3


def test_strict_readback_refuses_coordinate_or_temperature_change_before_copy():
    h=head();saved=copy.deepcopy(h.state_dict());copy.deepcopy(h).load_state_dict(saved)
    before={n:p.clone()for n,p in h.named_parameters()};saved['query_readout.weight'].add_(10)
    saved['_extra_state']['spec']['temperature']=.9
    with pytest.raises(ValueError):h.load_state_dict(saved)
    assert all(torch.equal(before[n],p)for n,p in h.named_parameters())
