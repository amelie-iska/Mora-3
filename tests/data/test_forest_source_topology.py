"""Original complete source tree supplies optional topology labels; no updates."""
from dataclasses import replace
import pytest
from test_forest_source_exports import encoder,exported,heads_for,trace
from mora3.model.forest_topology import NativeForestTopologyHead,ForestTopologySpec
from mora3.reasoning.forest_topology import TopologyBounds
from mora3.model.forest_training import forest_joint_loss


def enabled(e):
    h,s=heads_for(e)
    h=replace(h,topology=NativeForestTopologyHead(h.flow,ForestTopologySpec(TopologyBounds(8,7,7),'5'*64)))
    return h,replace(s,heads_sha256=h.sha256,topology_weight=.25)


def test_actual_complete_source_parents_are_exported_as_optional_labels():
    e=encoder();x=exported(e);h,s=enabled(e);b=x.training_batch(h,s)
    assert b.topology.forests==(tuple(x.document()['parents']),)
    assert b.topology.root_count==1 and b.topology.evidence_sha256==x.sha256
    assert b.topology.construction_policy_sha256==h.topology.spec.construction_policy_sha256
    only=replace(b,flow=None,tokens=None,quality=None,analogy=None)
    value=forest_joint_loss(h,s,only)
    assert set(value['component_losses'])=={'topology'}and value['counts']['topology']==5
    value['loss'].backward();assert h.topology.action_readout.weight.grad.abs().sum()>0


def test_disabled_export_supervision_preserves_previous_objective_shape():
    e=encoder();h,s=heads_for(e);b=exported(e).training_batch(h,s)
    assert b.topology is None


@pytest.mark.parametrize('split',('validation','test'))
def test_new_topology_supervision_cannot_relabel_original_heldout(split):
    e=encoder();h,s=enabled(e)
    with pytest.raises(ValueError,match='train'):exported(e,trace(split)).training_batch(h,s)
