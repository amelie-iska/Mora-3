"""Physical symmetries, native masks and actual shared TokenGT gradients."""
from dataclasses import replace
import copy
import math
import numpy as np
import pytest
import torch
from mora3.data.schema import GraphRecord,Node,Measurement
from mora3.data.provenance import Provenance
from mora3.data.tasks import ConditionalTask
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.geometry_conditioning import GeometryConditioner
from mora3.reasoning.geometry_analogy import (GeometryPolicy,physical_distances,signed_volume,
    torus_distances,geometry_objective,relation_transport_energy,relational_distortion,
    geometry_regularized_native_loss)

H='a'*64

def fixture():
    xyz=((0.,0.,0.),(1.,.2,0.),(1.3,1.,.1),(.1,1.2,.9))
    nodes=tuple(Node('atom',tuple(Measurement(k,v,'angstrom','observed',True)for k,v in zip(('x','y','z'),row))+
        (Measurement('frame','native-1',None,'observed',True),Measurement('outcome',3.,'dimensionless','observed',True)))for row in xyz)
    r=GraphRecord('fixture',nodes,(),(Provenance('fixture','v1','n1',H,'CC0'),))
    t=ConditionalTask(tuple((('node',i,m.name),'target'if m.name=='outcome'else'condition')for i,n in enumerate(nodes)for m in n.measurements),topology_known=True)
    p=GeometryPolicy(H,'molecular_structure',('x','y','z'),'angstrom','frame','native-1',1.,'b'*64)
    return r,t,p,torch.tensor(xyz,dtype=torch.float64)


def test_physical_distances_are_rigid_invariant_but_chirality_is_not_reflection_invariant():
    _,_,_,x=fixture();theta=.6
    rot=torch.tensor([[math.cos(theta),-math.sin(theta),0.],[math.sin(theta),math.cos(theta),0.],[0.,0.,1.]],dtype=x.dtype)
    y=x@rot.T+torch.tensor([2.,-3.,5.])
    torch.testing.assert_close(physical_distances(x,scale=1.),physical_distances(y,scale=1.),rtol=1e-12,atol=1e-12)
    assert signed_volume(x,(0,1,2,3)).item()==pytest.approx(signed_volume(y,(0,1,2,3)).item())
    mirrored=x*torch.tensor([-1.,1.,1.])
    torch.testing.assert_close(physical_distances(x,scale=1.),physical_distances(mirrored,scale=1.))
    assert signed_volume(mirrored,(0,1,2,3)).item()==pytest.approx(-signed_volume(x,(0,1,2,3)).item())


def test_periodic_torsion_does_not_use_a_linear_angle_metric():
    x=torch.tensor([[math.pi-.1],[-math.pi+.1]],dtype=torch.float64)
    assert torus_distances(x,periods=(2*math.pi,))[0,1].item()==pytest.approx(.2)
    with pytest.raises(ValueError):torus_distances(x,periods=(0.,))


@pytest.mark.parametrize('width',(8,12,16))
def test_geometry_objective_reuses_existing_parameters_and_is_width_normalized(width):
    r,t,p,_=fixture();torch.manual_seed(20261110)
    core=CooperativeModel(('A','B'),width=width,parameterization_policy='mup-cooperative-v1',base_width=8).double()
    bridge=GeometryConditioner(core.response.backbone,id_width=4,parameterization_policy='mup-geometry-v1',base_width=8).double()
    params={n:id(v)for n,v in bridge.named_parameters()};before=torch.get_rng_state().clone()
    result=geometry_objective(bridge,r,t,p)
    assert result['node_embeddings'].shape==(4,width)
    assert result['new_learned_parameters']==0 and result['latent_metric']=='RMS-coordinate-distance'
    result['loss'].backward()
    assert core.response.backbone.layers[0].attention.qkv.weight.grad.abs().sum()>0
    assert bridge.payload.embedding.weight.grad.abs().sum()>0
    assert params=={n:id(v)for n,v in bridge.named_parameters()}
    assert torch.equal(before,torch.get_rng_state())
    assert torch.isfinite(result['loss'])


def test_hidden_target_and_audit_change_neither_embeddings_nor_geometric_loss():
    r,t,p,_=fixture();torch.manual_seed(20261110)
    core=CooperativeModel(('A','B'),width=8,parameterization_policy='mup-cooperative-v1',base_width=8).double();bridge=GeometryConditioner(core.response.backbone,id_width=4,parameterization_policy='mup-geometry-v1',base_width=8).double()
    a=geometry_objective(bridge,r,t,p)
    changed=replace(r,record_id='different-audit',nodes=tuple(replace(n,measurements=tuple(replace(m,value=9000.)if m.name=='outcome'else m for m in n.measurements))for n in r.nodes))
    b=geometry_objective(bridge,changed,t,p)
    assert torch.equal(a['node_embeddings'],b['node_embeddings'])and torch.equal(a['loss'],b['loss'])
    assert a['condition_sha256']==b['condition_sha256']


def test_unobserved_coordinates_units_frames_or_hidden_geometry_refuse_before_neural_work():
    r,t,p,_=fixture();core=CooperativeModel(('A','B'),width=8,parameterization_policy='mup-cooperative-v1',base_width=8);bridge=GeometryConditioner(core.response.backbone,id_width=4,parameterization_policy='mup-geometry-v1',base_width=8)
    for name,change in [('x',{'status':'predicted'}),('x',{'units':'nm'}),('frame',{'value':'other'})]:
        node=r.nodes[0];node=replace(node,measurements=tuple(replace(m,**change)if m.name==name else m for m in node.measurements))
        with pytest.raises(ValueError):geometry_objective(bridge,replace(r,nodes=(node,*r.nodes[1:])),t,p)
    hidden=replace(t,roles=tuple((ref,'evaluation'if ref[2]=='x'else role)for ref,role in t.roles))
    with pytest.raises(ValueError):geometry_objective(bridge,r,hidden,p)
    with pytest.raises(ValueError):replace(p,domain='gene_regulatory_network')
    with pytest.raises(ValueError):geometry_objective(GeometryConditioner(core.response.backbone,id_width=2,parameterization_policy='mup-geometry-v1',base_width=8),r,t,p)


def test_relational_distortion_uses_correspondence_not_embedding_offsets():
    x=torch.tensor([[0.,0.],[1.,0.],[1.,2.]],dtype=torch.float64)
    y=x@torch.tensor([[0.,-1.],[1.,0.]],dtype=x.dtype)+2
    d=physical_distances(x,scale=1.);e=physical_distances(y,scale=1.)
    assert relational_distortion(d,e,((0,0),(1,1),(2,2))).item()==pytest.approx(0.,abs=1e-20)
    assert not torch.allclose(y-x,(y-x)[0].expand_as(x))
    with pytest.raises(ValueError):relational_distortion(d,e,((0,0),(1,0)))


def test_relation_specific_transport_avoids_forcing_inhibition_to_equal_states():
    z=torch.tensor([[1.,2.],[-1.,-2.]],dtype=torch.float64,requires_grad=True)
    identity=torch.eye(2,dtype=torch.float64)[None]
    assert relation_transport_energy(z,((0,1),),identity,-identity).item()==0
    assert relation_transport_energy(z,((0,1),),identity,identity).item()>0
    assert torch.autograd.gradcheck(lambda x:relation_transport_energy(x,((0,1),),identity,identity),(z,))


def test_auxiliary_loss_never_grants_native_training_admission():
    class NotAnOwner:pass
    with pytest.raises(ValueError):geometry_regularized_native_loss(NotAnOwner(),lambda:None,None,None,None,geometry_module_path='x')
