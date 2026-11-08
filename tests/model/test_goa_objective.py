"""Native policy binding must reject stale axes/masks and isolate targets."""
import copy
from dataclasses import replace
import pytest
import torch
from test_goa_applicability import setup, bind
from mora3.model import uniprot_go as go


def view(tmp_path):
    return bind(setup(tmp_path),tmp_path)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_native_head_reference_and_latent_gradient(tmp_path,device):
    v=view(tmp_path)
    torch.manual_seed(10901)
    head=go.NativeGOAHead(3,[v]).to(device=device,dtype=torch.float64)
    x=torch.randn(1,3,device=device,dtype=torch.float64,requires_grad=True)
    z=head(x);loss=head.objective(z,[v])
    reference=torch.nn.functional.softplus(-z[0,0])
    params=(x,*head.parameters())
    actual=torch.autograd.grad(loss,params,retain_graph=True)
    expected=torch.autograd.grad(reference,params)
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=1e-13,atol=1e-15)
    assert torch.count_nonzero(actual[1][1])==0
    state=head.get_extra_state()
    assert state['policy']=='exact-subject-coarse-GOA-observed-BCE-v1'
    assert state['packets'][0]['packet_sha256']
    changed=copy.deepcopy(state);changed['policy']='old'
    with pytest.raises(ValueError):head.set_extra_state(changed)


@pytest.mark.parametrize('bad',['axis','mask','label','status','provenance','packet'])
def test_native_objective_rejects_changed_packets(tmp_path,bad):
    v=view(tmp_path);head=go.NativeGOAHead(3,[v]);other=v
    if bad=='axis':other=replace(v,axis_ids=v.axis_ids[::-1])
    if bad=='mask':other=replace(v,observed=(False,False))
    if bad=='label':other=replace(v,labels=(0.,0.))
    if bad=='status':other=replace(v,status='applicable_future')
    if bad=='provenance':
        p=v.packet.provenance();p['goa_objective']['policy']='old'
        import json
        other=replace(v,packet=replace(v.packet,provenance_bytes=json.dumps(p).encode()))
    if bad=='packet':other=replace(v,packet=replace(v.packet,target_bytes=b'{}'))
    with pytest.raises(ValueError):head.objective(torch.zeros(1,2),[other])


def test_quarantine_has_connected_zero_gradient(tmp_path):
    v=view(tmp_path)
    # Use actual source conflict so quarantine is reader-issued.
    import json
    args=list(setup(tmp_path));p=args[0];t=p.targets()
    negative=copy.deepcopy(t['source_GO_annotations'][0]);negative['qualifier']='NOT|enables'
    t['source_GO_annotations'].append(negative)
    args[0]=replace(p,target_bytes=json.dumps(t).encode())
    q=bind(args,tmp_path);head=go.NativeGOAHead(3,[v,q])
    z=head(torch.ones(1,3));loss=head.objective(z,[q]);loss.backward()
    assert loss.item()==0 and all(torch.count_nonzero(p.grad)==0 for p in head.parameters())


def test_graph_bridge_contains_sequence_only_conditions(tmp_path):
    v=view(tmp_path)
    record,task=go.goa_graph_task(v)
    from mora3.model.payload import PayloadEncoder
    from mora3.tokenization.shortest_path import ByteTokenizer
    encoder=PayloadEncoder(ByteTokenizer((),controls=('pad','node','edge')),4)
    x=encoder.forward_task(record,task=task,identifiers=torch.eye(1))
    changed=replace(v,packet=replace(v.packet,target_bytes=b'{"changed": true}'))
    r,t=go.goa_graph_task(changed)
    y=encoder.forward_task(r,task=t,identifiers=torch.eye(1))
    assert torch.equal(x['features'],y['features'])
