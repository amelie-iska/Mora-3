import dataclasses
import json

import pytest
import torch
from torch.nn import functional as F

from mora3.model.cartesian_teacher_trajectory import cartesian_teacher_trajectory_loss
from mora3.teacher_protocol import FrozenTeacherClient, SocketEndpoint, TeacherRefusal, canonical
from test_cartesian_gfn_policy import policy, target


def _client(task):
    return FrozenTeacherClient(task.teacher, SocketEndpoint('/tmp/mora3-unopened-teacher-fixture',1.,8192,8192,1,1,1))


def _exchange(task, *, wrong_frame=False):
    calls = []

    def exchange(self, raw):
        request = json.loads(raw)
        calls.append(request)
        assert request['teacher_sha256'] == task.teacher.sha256
        assert request['system_sha256'] == task.system.sha256
        assert request['path_provenance'] == 'generative_dag_not_physical_time'
        assert request['return_forces'] is False
        evaluations = []
        for i, row in enumerate(request['frames']):
            evaluations.append(dict(frame_id='changed' if wrong_frame else row['frame_id'], xyz_sha256=row['xyz_sha256'],
                status='ok',energy_ev=float(i),forces_ev_per_angstrom=None,
                energy_unit='eV',force_unit='eV/angstrom',position_unit='angstrom',
                execution_fingerprint_sha256=task.teacher.execution_fingerprint_sha256,
                teacher_trainable_parameter_count=0,teacher_training_mode=False,force_relation='not_requested',
                conservative_evidence_sha256=None,refusal_reason=None))
        return canonical(dict(schema='Mora3.FrozenTeacherResponse/v1',request_id=request['request_id'],
                              teacher_sha256=task.teacher.sha256,system_sha256=task.system.sha256,evaluations=evaluations))

    return calls, exchange


def test_exact_terminal_teacher_rpc_and_gradient(monkeypatch):
    model = policy()
    task = dataclasses.replace(target(3,True),optional_stop=False)
    calls, exchange = _exchange(task)
    monkeypatch.setattr(SocketEndpoint,'exchange',exchange)
    generator = torch.Generator().manual_seed(19)
    rng_before = torch.get_rng_state().clone()
    result = cartesian_teacher_trajectory_loss(model,task,_client(task),batch_size=2,generator=generator,request_id='test')
    request, = calls
    positions = torch.tensor([x['xyz_angstrom'] for x in request['frames']],dtype=torch.float64)
    assert torch.equal(positions[:,0],torch.zeros((2,3),dtype=torch.float64))
    logits = torch.logit((positions[:,1:].reshape(2,-1)+2)/5)
    correction = (F.logsigmoid(logits)+F.logsigmoid(-logits)).sum(-1)
    expected = correction + positions.new_tensor([task.energy_log_reward(float(i)) for i in range(2)])
    torch.testing.assert_close(result['log_reward'],expected)
    assert result['target_sha256'] == task.sha256
    assert result['terminal_coordinate_sha256'] == tuple(x['xyz_sha256'] for x in request['frames'])
    assert not result['log_reward'].requires_grad
    result['loss'].backward()
    assert torch.isfinite(model.backbone.feature.weight.grad).all()
    assert model.backbone.feature.weight.grad.abs().sum()>0
    assert torch.equal(rng_before,torch.get_rng_state())


def test_teacher_refusal_preserves_private_generator(monkeypatch):
    model = policy()
    task = target()
    calls, exchange = _exchange(task,wrong_frame=True)
    monkeypatch.setattr(SocketEndpoint,'exchange',exchange)
    generator = torch.Generator().manual_seed(19)
    before = generator.get_state().clone()
    with pytest.raises(TeacherRefusal,match='reordered'):
        cartesian_teacher_trajectory_loss(model,task,_client(task),batch_size=2,generator=generator,request_id='test')
    assert len(calls)==1 and torch.equal(generator.get_state(),before)
    assert all(p.grad is None for p in model.parameters())
