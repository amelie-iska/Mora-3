"""Partitioned independent qualification checks; no training admission."""
from dataclasses import replace
import pytest
from mora3.extensions.scri.simulation_contracts import Intervention,SimulationBinding,simulation_group,compare_summary


def simulation():
    return SimulationBinding('a'*40,'b'*64,'c'*64,'d'*64,'engine-1',17,'generic-2D','s','control',
        (Intervention('gap_junction','junction-network','decrease',20.,'percent-blocked',0.,.01,'s'),),
        'simulation',None)


def test_simulation_lineage_does_not_split_near_duplicate_interventions_or_seed():
    a=simulation();b=replace(a,seed=18,interventions=(replace(a.interventions[0],value=21.),))
    assert simulation_group(a)==simulation_group(b)
    with pytest.raises(ValueError):replace(a,evidence_kind='experimental')
    with pytest.raises(ValueError):replace(a,seed=True)
    with pytest.raises(ValueError):replace(a,independent_experimental_support_sha256='not-a-digest')


def test_summary_comparison_requires_fixed_tolerance_and_native_keys():
    a={'case':'control','E_mem':0.,'E_gj':0.,'E_pump':0.,'E_total':0.,'d_mean_vmem_mV':0.,'d_wound_gj':0.}
    assert compare_summary((a,),(dict(a),),atol=0.,rtol=0.)['within_tolerance']
    b=dict(a,E_mem=1.)
    assert not compare_summary((a,),(b,),atol=0.,rtol=0.)['within_tolerance']
    with pytest.raises(ValueError):compare_summary((a,),(dict(a,extra=1),),atol=0.,rtol=0.)
