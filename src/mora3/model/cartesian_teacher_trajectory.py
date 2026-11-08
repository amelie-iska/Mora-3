"""Score Cartesian trajectories against an exact frozen terminal energy batch."""

import torch

from mora3.boltzmann_target import BoltzmannTarget
from mora3.reasoning.gfn_sampling import sample_paths, score_paths
from mora3.teacher_protocol import FrozenTeacherClient, TeacherRefusal, text

from .cartesian_gfn_policy import SharedCartesianGFNPolicy, box_coordinates_and_log_measure, normalized_terminal_log_reward
from .profile_gfn import conditional_trajectory_balance


def cartesian_teacher_trajectory_loss(policy, target, teacher, *, batch_size, generator, request_id):
    """Return a differentiable score and authenticated terminal evidence.

    The caller owns corpus, target properness and optimizer admission. Energies
    stay detached and the teacher only sees the final complete physical frame
    of each path. A refused batch leaves the caller's generator retryable.
    """
    if (type(policy) is not SharedCartesianGFNPolicy or type(target) is not BoltzmannTarget
            or type(teacher) is not FrozenTeacherClient or teacher.spec.sha256 != target.teacher.sha256
            or type(batch_size) is not int or batch_size < 1
            or not isinstance(generator,torch.Generator) or not text(request_id)):
        raise TeacherRefusal('Exact Cartesian policy, target, frozen teacher and private generator required')
    bound = policy.bind(target)
    context = bound.context(batch_size)
    saved = generator.get_state()
    try:
        paths = sample_paths(bound,batch_size,generator=generator,optional_stop=target.optional_stop,
                             context=context.detach())
        path_indices = torch.arange(batch_size,device=context.device)
        terminals = paths['states'][path_indices,paths['lengths']-1]
        positions,_ = box_coordinates_and_log_measure(terminals,target.support)
        frames = tuple((str(i),tuple(tuple(row) for row in frame))
                       for i,frame in enumerate(positions.cpu().tolist()))
        evaluations = teacher.evaluate_batch(target.system,frames,request_id=request_id,return_forces=False,
                                             path_provenance='generative_dag_not_physical_time')
        energies = tuple(row.energy_ev for row in evaluations)
        _,reward = normalized_terminal_log_reward(terminals,energies,target,horizon=bound.horizon)
        scores = score_paths(bound,paths,context=context)
        result = conditional_trajectory_balance(bound.log_partition(context),**scores,log_reward=reward,
                                                mask=paths['mask'])
    except BaseException:
        generator.set_state(saved)
        raise
    return dict(**result,log_reward=reward,target_sha256=target.sha256,teacher_sha256=teacher.spec.sha256,
                terminal_coordinate_sha256=tuple(row.xyz_sha256 for row in evaluations),
                terminal_energy_ev=energies,path_lengths=tuple(paths['lengths'].cpu().tolist()))
