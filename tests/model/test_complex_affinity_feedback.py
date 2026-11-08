"""Exact typed interaction subtraction and independent native objective gradients."""
from dataclasses import replace
import json

import pytest
import torch

from mora3.complex_affinity_feedback import ComplexPartition,evaluate_interaction_feedback
from mora3.model.complex_affinity_head import AffinityObservation,ComplexAffinityFeedbackHead
from mora3.teacher_protocol import (
    FAIRCHEM_REVISION,REPOSITORIES,FrozenTeacherClient,SocketEndpoint,
    SystemSpec,TeacherRefusal,TeacherSpec,canonical,
)


def system(name, ids, *, bonds=()):
    atoms=[dict(key=atom,entity='fixture',ordinal=i,element='He',atomic_number=2,
                name=None,component=None,residue=None,isotope=None) for i,atom in enumerate(ids)]
    return SystemSpec.from_export_mapping(dict(
        system_id=name,topology_id=name+'-topology',atoms=atoms,bonds=list(bonds),
        formal_charges=[0]*len(ids),total_charge=0,spin_multiplicity=1,
        pbc=[False]*3,cell_angstrom=None,protonation=None,cofactor_ids=[],chain_links=[],
        chemistry_context_json='{}',source_refs=[],sequence_chains=[],cofactor_atom_map=[],
    ))


def client(family):
    repo,rev=REPOSITORIES[family]
    spec=TeacherSpec('fixture-'+family,family,repo,rev,FAIRCHEM_REVISION,
        '/fixture/checkpoint.pt','a'*64,'omol','angstrom','eV','eV/angstrom',
        'raw-local-no-reference-restoration',None,None,'{}','b'*64,17)
    return FrozenTeacherClient(spec,SocketEndpoint('/tmp/mora3-unopened-affinity-fixture',
                                                   1.,8192,8192,1,1,1))


@pytest.fixture
def partition():
    return ComplexPartition(system('whole',('a','b','c')),
                            (system('left',('c','a')),system('right',('b',))))


def fake_exchange(calls):
    def exchange(_endpoint,raw):
        request=json.loads(raw)
        calls.append(request)
        frame,=request['frames']
        energy={'whole':-10.,'left':-4.,'right':-3.}[request['system']['system_id']]
        return canonical(dict(schema='Mora3.FrozenTeacherResponse/v1',
            request_id=request['request_id'],teacher_sha256=request['teacher_sha256'],
            system_sha256=request['system_sha256'],evaluations=[dict(
                frame_id=frame['frame_id'],xyz_sha256=frame['xyz_sha256'],status='ok',
                energy_ev=energy,forces_ev_per_angstrom=None,energy_unit='eV',
                force_unit='eV/angstrom',position_unit='angstrom',
                execution_fingerprint_sha256='b'*64,teacher_trainable_parameter_count=0,
                teacher_training_mode=False,force_relation='not_requested',
                conservative_evidence_sha256=None,refusal_reason=None)]))
    return exchange


@pytest.mark.parametrize('family',tuple(REPOSITORIES))
def test_teacher_receipts_match_exact_participant_geometry(partition,family,monkeypatch):
    calls=[]
    monkeypatch.setattr(SocketEndpoint,'exchange',fake_exchange(calls))
    frame=((0.,0.,0.),(1.,0.,0.),(0.,2.,0.))
    teacher=client(family)
    result=evaluate_interaction_feedback(partition,frame,teacher,request_id='unit')
    assert result.interaction_energy_ev==-3. and result.teacher_sha256==teacher.spec.sha256
    assert [r['frames'][0]['xyz_angstrom'] for r in calls]==[
        [list(x) for x in frame],[list(frame[2]),list(frame[0])],[list(frame[1])]]
    assert len({r['request_id'] for r in calls})==3


def test_no_missing_atoms_or_cross_participant_bonds(partition):
    with pytest.raises(TeacherRefusal,match='partition'):
        replace(partition,participants=(system('left',('a',)),system('right',('b',))))
    with pytest.raises(TeacherRefusal,match='Cross-participant'):
        replace(partition,complex_system=system('whole',('a','b','c'),
                                               bonds=({'first':'a','second':'b'},)))


def test_teacher_loss_does_not_create_experimental_labels(partition,monkeypatch):
    monkeypatch.setattr(SocketEndpoint,'exchange',fake_exchange([]))
    teacher=client('UMA')
    evidence=evaluate_interaction_feedback(partition,((0.,0.,0.),(1.,0.,0.),(0.,2.,0.)),
                                           teacher,request_id='gradient')
    head=ComplexAffinityFeedbackHead(8,4,experimental_axes=(('Kd','nM'),('Ki','nM')),
                                     teacher_axes=(teacher.spec.sha256,))
    inputs=torch.randn(2,8,requires_grad=True)
    loss=head.objectives(inputs,observed=(None,None),feedback=(evidence,None),
                         affinity_scales=(1.,1.),teacher_scales=(1.,),feedback_weight=.5)
    assert loss['affinity'] is None and loss['mlip_observations']==1
    loss['total'].backward()
    assert head.affinity.weight.grad is None
    assert torch.isfinite(head.interaction.weight.grad).all()
    assert inputs.grad[1].abs().sum()==0
    label=AffinityObservation(4.,'Kd','uM','c'*64)
    with pytest.raises(ValueError,match='endpoint and unit'):
        head.objectives(inputs,observed=(label,None),feedback=(None,None),
                        affinity_scales=(1.,1.),teacher_scales=(1.,),feedback_weight=.5)
    with pytest.raises(ValueError,match='finite-axis'):
        head.load_state_dict(dict(head.state_dict(),_extra_state=(8,4,(('Ki','nM'),),('a'*64,),1.)))
