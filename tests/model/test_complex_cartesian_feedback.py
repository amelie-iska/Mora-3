from dataclasses import replace
import json,pytest,torch
from mora3.model.complex_cartesian_feedback import (
    ComplexCartesianFeedbackAdapter, bind_complex_states, feedback_head_parameter_roles,
)
from mora3.complex_affinity_feedback import ComplexPartition, MLIPInteractionFeedback
from mora3.model.complex_affinity_head import ComplexAffinityFeedbackHead
from mora3.model.cartesian_gfn_policy import box_coordinates_and_log_measure
from mora3.model.parameterization import parameter_ledger
from mora3.teacher_protocol import TeacherRefusal, coordinates_sha256
from mora3.teacher_protocol import SystemSpec
from test_cartesian_gfn_policy import target,policy

@pytest.fixture
def sample():
    torch.manual_seed(20261108)
    old=target(3,True);payload=json.loads(old.system.topology_json);payload['bonds']=[]
    whole=SystemSpec.from_export_mapping(payload)
    parts=[]
    for name,selected in [('left',(2,0)),('right',(1,))]:
        d=dict(payload,system_id=name,topology_id=name,atoms=[dict(payload['atoms'][i],ordinal=j) for j,i in enumerate(selected)],formal_charges=[0]*len(selected))
        parts.append(SystemSpec.from_export_mapping(d))
    partition=ComplexPartition(whole,tuple(parts));task=replace(old,system=whole,support=replace(old.support,system_sha256=whole.sha256))
    states=torch.zeros(1,task.support.dimension,dtype=torch.float64)
    frame=tuple(tuple(float(x) for x in row) for row in box_coordinates_and_log_measure(states,task.support)[0][0].tolist())
    f=MLIPInteractionFeedback(partition.sha256,coordinates_sha256(frame),task.teacher.sha256,'UMA',task.teacher.energy_reference_scheme,-10.,(-4.,-3.),('a'*64,'b'*64,'c'*64))
    return partition,task,states,f

def test_native_context_separate_heads_and_complete_roles(sample):
    p,t,z,f=sample
    heads=ComplexAffinityFeedbackHead(8,8,experimental_axes=(('Kd','nM'),),teacher_axes=(t.teacher.sha256,)).double()
    adapter=ComplexCartesianFeedbackAdapter(policy(),heads).eval()
    before=tuple(x.detach().clone() for x in adapter.parameters())
    with torch.no_grad():
        binding,loss=adapter.diagnostic_objectives(p,t,z,layer=1,observed=(None,),feedback=(f,),affinity_scales=(1.,),teacher_scales=(1.,),feedback_weight=.5,origin='synthetic_fixture')
    assert loss['affinity'] is None and loss['mlip_observations']==1 and loss['experimental_observations']==0
    assert not binding.training_authorized
    assert len(binding.participant_coordinate_sha256[0])==2
    ledger=parameter_ledger(adapter,adapter.parameter_roles())
    assert ledger.total_numel==sum(x.numel() for x in adapter.parameters())
    assert adapter.parameter_roles()['heads.affinity.weight'].axes[0]!=adapter.parameter_roles()['heads.interaction.weight'].axes[0]
    assert all(torch.equal(x,b) for x,b in zip(adapter.parameters(),before))
    with pytest.raises(TeacherRefusal,match='admission'):adapter.require_training_admission()
    with pytest.raises(TeacherRefusal,match='admission'):binding.require_training_admission()

@pytest.mark.parametrize('kind',['frame','teacher','partition','reference','participants'])
def test_preflight_rejects_inexact_feedback(sample,kind):
    p,t,z,f=sample
    if kind=='frame':f=replace(f,complex_xyz_sha256='0'*64)
    elif kind=='teacher':f=replace(f,teacher_sha256='0'*64)
    elif kind=='partition':f=replace(f,partition_sha256='0'*64)
    elif kind=='reference':f=replace(f,energy_reference_scheme='explicit-pinned-reference-assets')
    else:f=replace(f,participant_energies_ev=(-4.,-3.,-1.),evaluation_sha256=tuple(c*64 for c in 'abcd'))
    with pytest.raises(TeacherRefusal,match='exact teacher'):
        bind_complex_states(p,t,z,observed=(None,),feedback=(f,))

def test_heldout_origin_and_mismatched_frame_refuse_before_encoder(sample,monkeypatch):
    p,t,z,f=sample
    adapter=ComplexCartesianFeedbackAdapter(policy(),ComplexAffinityFeedbackHead(8,8,experimental_axes=(('Kd','nM'),),teacher_axes=(t.teacher.sha256,)).double())
    def forbidden(*a,**k):raise AssertionError('encoder reached')
    monkeypatch.setattr(adapter.policy.graph,'forward',forbidden)
    kwargs=dict(layer=1,observed=(None,),feedback=(f,),affinity_scales=(1.,),teacher_scales=(1.,),feedback_weight=.5)
    for role in ('train','validation','test','heldout','real_source'):
        with pytest.raises(TeacherRefusal,match='synthetic'):adapter.diagnostic_objectives(p,t,z,origin=role,**kwargs)
    with pytest.raises(TeacherRefusal,match='exact teacher'):
        adapter.diagnostic_objectives(p,t,z+.1,origin='synthetic_fixture',**kwargs)

def test_tied_experimental_teacher_head_refuses(sample):
    _,t,_,_=sample
    h=ComplexAffinityFeedbackHead(8,8,experimental_axes=(('Kd','nM'),),teacher_axes=(t.teacher.sha256,)).double()
    h.interaction.weight=h.affinity.weight
    with pytest.raises(ValueError,match='not be tied'):feedback_head_parameter_roles(h)
