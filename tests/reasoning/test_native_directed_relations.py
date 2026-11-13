"""Source-conditioned directed adapters and categorical naturality witnesses."""
from dataclasses import replace
import pytest
from mora3.data.schema import GraphRecord,Node,Edge,Measurement
from mora3.data.tasks import ConditionalTask
from mora3.data.provenance import Provenance
from mora3.reasoning.directed_topology import Arrow,DirectedComplex,DirectedFunctor
from mora3.reasoning.native_relations import relation_complex,verify_natural_transformation
from test_directed_topology import diamond,H


def fixture():
    nodes=tuple(Node('gene',(Measurement('response',float(i),'dimensionless','observed',True),))for i in range(3))
    edges=(Edge(0,1,'inhibits',True,(Measurement('cost',1.,'action-count','observed',True),)),
           Edge(1,2,'activates',True,(Measurement('cost',2.,'action-count','observed',True),)))
    record=GraphRecord('audit',nodes,edges,(Provenance('fixture','v1','system',H,'CC0'),))
    roles=tuple((('node',i,'response'),'target')for i in range(3))+tuple((('edge',i,'cost'),'condition')for i in range(2))
    return record,ConditionalTask(roles,topology_known=True)


def test_native_grn_preserves_direction_sign_and_unit_cost_separately():
    r,t=fixture();x,receipt=relation_complex(r,t,source_sha256=H,cost_field='cost',cost_unit='action-count')
    assert x.endpoint('node:0',('edge:0','edge:1'))=='node:2'
    assert x.arrows[0].relation=='inhibits'and x.arrows[0].cost==1.
    assert x.squares==()and receipt['causal_authority_issued']is False
    assert receipt['coordinate_geometry_inferred']is False
    with pytest.raises(ValueError):x.endpoint('node:1',('edge:0',))


def test_targets_and_record_ids_cannot_influence_directed_presentation():
    r,t=fixture();a,ra=relation_complex(r,t,source_sha256=H,cost_field='cost',cost_unit='action-count')
    r=replace(r,record_id='other',nodes=tuple(replace(n,measurements=(replace(n.measurements[0],value=10000.),))for n in r.nodes))
    b,rb=relation_complex(r,t,source_sha256=H,cost_field='cost',cost_unit='action-count')
    assert a==b and ra['condition_sha256']==rb['condition_sha256']


@pytest.mark.parametrize('bad',('predicted','unit','role','undirected'))
def test_invalid_native_directed_cost_or_role_refuses(bad):
    r,t=fixture();edge=r.edges[0];measurement=edge.measurements[0]
    if bad=='predicted':edge=replace(edge,measurements=(replace(measurement,status='predicted'),))
    elif bad=='unit':edge=replace(edge,measurements=(replace(measurement,units='joule'),))
    elif bad=='undirected':edge=replace(edge,directed=False)
    else:t=replace(t,roles=tuple((ref,'evaluation'if ref==('edge',0,'cost')else role)for ref,role in t.roles))
    with pytest.raises(ValueError):relation_complex(replace(r,edges=(edge,r.edges[1])),t,source_sha256=H,cost_field='cost',cost_unit='action-count')


def test_natural_transformation_requires_checked_commuting_square():
    source=DirectedComplex(('x','y'),(Arrow('r','x','y','r',1.,H),),(),cost_unit='dimensionless')
    for filled,expected in ((True,'proved'),(False,'refused')):
        target=diamond(filled)
        f=DirectedFunctor(source,target,{'x':'0','y':'1'},{'r':('a',)},{'r':('r',)},H)
        g=DirectedFunctor(source,target,{'x':'2','y':'3'},{'r':('d',)},{'r':('r',)},H)
        result=verify_natural_transformation(f,g,{'x':('c',),'y':('b',)},H)
        assert result['status']==expected
        assert result['continuous_homotopy_realization_proved']is False
    assert verify_natural_transformation(f,g,{'x':('c',),'y':('b',)},H)['status']=='refused'


def test_functor_is_bound_to_its_complete_declared_maps():
    x=diamond(True);f=DirectedFunctor(x,x,{v:v for v in x.vertices},{a.name:(a.name,)for a in x.arrows},{'r':('r',),'s':('s',)},H)
    f.objects.pop('0')
    with pytest.raises(ValueError):f.verify()
