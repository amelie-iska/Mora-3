"""Native canonical GOA shares the twelve-objective encoder without target input."""
from dataclasses import replace
import copy
import pytest
import torch
from test_rfam_cooperative import build as base_build
from test_goa_objective import view
from mora3.model import goa_cooperative as module


def build(tmp_path, device):
    v=view(tmp_path)
    base=base_build(device)
    assert hasattr(module,'GOACooperativeModel'), 'native GOA cooperative interface missing'
    m=module.GOACooperativeModel(base.native,base.annotation,base.residue.head.feature_axis,
        base.profile.axes,codec=base.decoder.codec,max_tokens=64,
        annotation_registry=__import__('annotation_fixture').fixture_registry(),
        family_axis=base.rfam.family_axis,rfam_manifest_sha256='a'*64,
        goa_packets=(v,)).to(device).double()
    return m,v


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_goa_shared_encoder_every_gradient_and_target_exclusion(tmp_path,device):
    m,v=build(tmp_path,device)
    from mora3.model.uniprot_go import goa_graph_task
    r,t=goa_graph_task(v)
    graph=m.annotation.payload.forward_task(r,task=t,identifiers=m.annotation.sequence_identifiers())
    z=m.goa(m.annotation.backbone([graph])[0][:,0])
    oracle=torch.nn.functional.softplus(-z[0,0])
    actual=m.goa_objective((v,))
    ps=tuple(m.parameters())
    a=torch.autograd.grad(actual,ps,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(oracle,ps,allow_unused=True)
    for x,y in zip(a,b):
        assert (x is None)==(y is None)
        if x is not None:torch.testing.assert_close(x,y,atol=1e-11,rtol=1e-10)
    actual.backward()
    assert m.annotation.payload.embedding.weight.grad.abs().sum()>0
    assert m.native.response.backbone.feature.weight.grad.abs().sum()>0
    assert m.goa.output.weight.grad[1].count_nonzero()==0
    changed=replace(v,packet=replace(v.packet,target_bytes=b'{"hidden":"different"}'))
    assert torch.equal(m.goa_logits((v,)),m.goa_logits((changed,)))
    with pytest.raises(ValueError):m.goa_objective((changed,))
    assert len(list(m.parameters()))==len(set(map(id,m.parameters())))
    state=copy.deepcopy(m.get_extra_state());state['goa']['policy']='stale'
    with pytest.raises(ValueError):m.set_extra_state(state)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_goa_quarantine_connected_zero_and_pool_rejection(tmp_path,device):
    from test_goa_applicability import setup,bind
    import json
    args=list(setup(tmp_path));p=args[0];t=p.targets()
    negative=copy.deepcopy(t['source_GO_annotations'][0]);negative['qualifier']='NOT|enables'
    t['source_GO_annotations'].append(negative)
    args[0]=replace(p,target_bytes=json.dumps(t).encode())
    q=bind(args,tmp_path)
    m,v=build(tmp_path,device)
    # Reader-issued quarantine is explicitly part of this finite pool.
    from mora3.model.uniprot_go import NativeGOAHead
    m.goa=NativeGOAHead(4,(v,q)).to(device).double()
    loss=m.goa_objective((q,));loss.backward()
    assert loss.item()==0
    assert all(p.grad is None or p.grad.count_nonzero()==0 for p in m.parameters())
    with pytest.raises(ValueError):m.goa_logits(())
