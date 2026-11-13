"""Grandis-inspired finite directed square presentations; no continuum claim."""
from dataclasses import replace
import math
import numpy as np
import pytest
from mora3.reasoning.directed_topology import Arrow,Square,DirectedComplex,DirectedFunctor,verify_positive_chain_map,verify_chain_homotopy

H='a'*64

def diamond(fill=False):
    arrows=(Arrow('a','0','1','r',1.,H),Arrow('b','1','3','s',2.,H),
            Arrow('c','0','2','s',2.,H),Arrow('d','2','3','r',1.,H))
    return DirectedComplex(('0','1','2','3'),arrows,(Square(('a','b'),('c','d'),H),)if fill else(),cost_unit='dimensionless')


def test_paths_direction_identity_composition_and_weight():
    x=diamond();assert x.endpoint('0',('a','b'))=='3'
    assert x.compose('0',(),('a','b'))==('a','b')
    assert x.compose('0',('a',),('b',))==('a','b')
    assert x.path_cost(('a','b'))==3.
    assert x.shortest_cost('0','3')==3.
    assert math.isinf(x.shortest_cost('3','0'))
    with pytest.raises(ValueError):x.endpoint('3',('b','a'))


def test_same_endpoints_do_not_imply_homotopy_without_square():
    assert diamond().equivalent('0',('a','b'),('c','d')).status=='disproved-in-presentation'
    proof=diamond(True).equivalent('0',('a','b'),('c','d'))
    assert proof.status=='proved'and proof.rewrite_count==1
    assert diamond(True).equivalent('0',('a','b'),('c','d'),max_states=1).status=='unknown-budget'


def test_integer_chain_boundaries_and_cellular_betti():
    x=diamond(True);b1,b2=x.boundaries()
    assert np.array_equal(b1@b2,np.zeros((4,1),dtype=np.int64))
    assert x.betti_rational()==(1,0,0)
    assert diamond(False).betti_rational()==(1,1,0)
    # An underlying H1 class of a directed diamond is not a positive directed cycle.
    z=np.array([1,1,-1,-1])
    assert diamond().positive_cycle(z) is False
    circle=DirectedComplex(('v',),(Arrow('loop','v','v','r',1.,H),),(),cost_unit='dimensionless')
    assert circle.betti_rational()==(1,1,0)
    assert circle.positive_cycle(np.array([1]))is True
    assert circle.positive_cycle(np.array([-1]))is False


def test_positive_homology_witness_preserves_cone_information():
    x=diamond(True);z=np.array([1,1,-1,-1])
    # This mixed-sign cycle is a boundary, and so has the nonnegative zero representative.
    assert x.verify_positive_class(z,np.zeros(4,dtype=int),np.array([1]))
    assert not x.verify_positive_class(z,np.zeros(4,dtype=int),np.array([0]))


def test_functor_maps_generators_paths_and_declared_relations():
    x=diamond(True);f=DirectedFunctor(x,x,{v:v for v in x.vertices},{a.name:(a.name,)for a in x.arrows},
        {'r':('r',),'s':('s',)},H)
    receipt=f.verify();assert receipt['relations_preserved']and receipt['composition_preserved_by_construction']
    assert f.map_path('0',('a','b'))==('a','b')
    assert receipt['faithful_or_full_proved']is False
    bad=DirectedFunctor(x,diamond(False),f.objects,f.arrows,f.relations,H)
    assert bad.verify()['status']=='refused'


def test_functor_noncommuting_action_order_and_wrong_relation_refuse():
    x=diamond(True)
    with pytest.raises(ValueError):DirectedFunctor(x,x,{v:v for v in x.vertices},
        {'a':('c',),'b':('d',),'c':('a',),'d':('b',)}, {'r':('r',),'s':('s',)},H)
    with pytest.raises(ValueError):DirectedFunctor(x,x,{v:v for v in x.vertices},
        {a.name:(a.name,)for a in x.arrows},{'r':('s',),'s':('r',)},H)


def test_verified_nonnegative_chain_map_and_homotopy_equations():
    x=diamond(True);b1,b2=x.boundaries();chain=(b1,b2)
    identity=(np.eye(4,dtype=int),np.eye(4,dtype=int),np.eye(1,dtype=int))
    assert verify_positive_chain_map(chain,chain,identity)
    bad=list(identity);bad[1]=-bad[1]
    with pytest.raises(ValueError):verify_positive_chain_map(chain,chain,tuple(bad))
    homotopy=(np.zeros((4,4),dtype=int),np.zeros((1,4),dtype=int))
    assert verify_chain_homotopy(chain,chain,identity,identity,homotopy,require_positive=True)
    wrong=list(identity);wrong[1]=np.zeros((4,4),dtype=int)
    with pytest.raises(ValueError):verify_chain_homotopy(chain,chain,identity,tuple(wrong),homotopy,require_positive=True)


def test_explicit_square_evidence_and_parallel_endpoints_required():
    x=diamond()
    with pytest.raises(ValueError):replace(x,squares=(Square(('a','b'),('c','a'),H),))
    with pytest.raises(ValueError):Square(('a','b'),('c','d'),'')
    with pytest.raises(ValueError):replace(x,arrows=x.arrows+(x.arrows[0],))
    with pytest.raises(ValueError):Arrow('e','0','1','r',-1.,H)


def test_opposite_category_reverses_arrows_not_sign_or_cost():
    x=diamond(True);op=x.opposite()
    assert op.endpoint('3',('b','a'))=='0'
    assert op.equivalent('3',('b','a'),('d','c')).status=='proved'
    assert op.opposite()==x and op.path_cost(('b','a'))==3.
