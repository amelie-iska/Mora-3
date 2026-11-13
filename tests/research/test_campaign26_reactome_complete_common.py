"""Native Reactome98 masked-role and genuine frozen candidate/common source checks."""
from pathlib import Path
from types import SimpleNamespace
import pytest

from mora3.research.campaign26_reactome_complete_common import (
    _frozen_prefix, verify_reactome_masked_packet,
)
from mora3.research.native_annotation_registry import load_original_annotation_assets
from mora3.research.native_annotation_campaign_routes import bind_reactome

ACTIVE=Path("/home/iska/Documents/amelie/bio/Mora-3")
CANDIDATE="1091e3f27d57cb70203ab9fce1136836d0cd96eef955b6682a3e6264abadc733"
COMMON="79d328f542d1158581875ef325e56f213b407992f95db6ee0c7352a3459669d4"

def test_masked_native_roles_cannot_become_observed_inputs():
    packet=SimpleNamespace(targets=(("reactant", "2"),))
    graph=SimpleNamespace(nodes=(1,),edges=(SimpleNamespace(measurements=()),))
    prep=SimpleNamespace(observed_positions=2,unavailable_reason=None,evaluate=lambda owner:None)
    assert verify_reactome_masked_packet(packet, (graph,(1,)),prep)==(1,1)
    with pytest.raises(ValueError,match="mask"):
        verify_reactome_masked_packet(packet,(SimpleNamespace(nodes=(1,),edges=(SimpleNamespace(measurements=("leak",)),)),(1,)),prep)
    with pytest.raises(ValueError,match="target"):
        verify_reactome_masked_packet(packet,(graph,(1,)),SimpleNamespace(observed_positions=1,unavailable_reason=None,evaluate=lambda owner:None))

def test_original_reactome_candidate_common_prefix_and_native_cursor():
    if not ACTIVE.is_dir():
        pytest.skip("Original source checkout not mounted")
    view,sources,availability,_,_=load_original_annotation_assets(ACTIVE)
    source=sources["reactome"]
    key="reactome_static_participant_roles"
    assert availability[key]["train_rows"]==1653
    assert availability[key]["validation_rows"]==availability[key]["test_rows"]==0
    candidate=bind_reactome(source,view,snapshot_sha256=CANDIDATE,batch_size=1)
    common=bind_reactome(source,view,snapshot_sha256=COMMON,batch_size=1)
    original,prev=_frozen_prefix(candidate,source)
    rebound,new=_frozen_prefix(common,source)
    assert prev==new
    assert all(x.batch.native_rows.equals(y.batch.native_rows,check_metadata=True)
               for x,y in zip(original,rebound,strict=True))
    assert sum(r[4][0] for r in new)>0
    assert all(r[0] and r[2] and r[3] for r in new)
