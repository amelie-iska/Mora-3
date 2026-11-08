"""Exact existing Cartesian context adapter; diagnostic only, no source admission.

Uses the complete fixed-system box representation already declared by native
Cartesian GFN. No invariance, new frame registration, experimental calibration,
rights, teacher availability, or biomolecular accuracy is inferred.
"""
from dataclasses import dataclass
import torch
from torch import nn
from mora3.teacher_protocol import TeacherRefusal,canonical,sha,coordinates_sha256
from mora3.complex_affinity_feedback import ComplexPartition,MLIPInteractionFeedback
from mora3.boltzmann_target import BoltzmannTarget
from .complex_affinity_head import ComplexAffinityFeedbackHead,AffinityObservation
from .cartesian_gfn_policy import SharedCartesianGFNPolicy,box_coordinates_and_log_measure
from .cartesian_mup import cartesian_mup_parameter_roles
from .parameterization import Axis,ParameterRole,parameter_ledger

@dataclass(frozen=True)
class ExactComplexStateBinding:
    partition_sha256:str
    target_sha256:str
    coordinate_sha256:tuple
    participant_coordinate_sha256:tuple
    feedback_sha256:tuple
    observed_affinity_source_sha256:tuple
    training_authorized:bool=False
    def require_training_admission(self):
        raise TeacherRefusal('Complex affinity source/heldout/rights/teacher/calibration admission is not implemented')


def bind_complex_states(partition,target,states,*,observed,feedback):
    """Pure complete-frame correspondence preflight; never calls a teacher or model."""
    if type(partition) is not ComplexPartition or type(target) is not BoltzmannTarget:
        raise TeacherRefusal('Exact complete partition and native Cartesian target required')
    partition.__post_init__();target.__post_init__()
    if target.system.sha256!=partition.complex_system.sha256:
        raise TeacherRefusal('Cartesian target must be the exact complete complex')
    if (type(observed) is not tuple or type(feedback) is not tuple
        or len(observed)!=len(feedback) or len(observed)!=len(states)
        or any(x is not None and type(x) is not AffinityObservation for x in observed)
        or any(x is not None and type(x) is not MLIPInteractionFeedback for x in feedback)):
        raise TeacherRefusal('Exact independent affinity/teacher observation masks required')
    xyz,_=box_coordinates_and_log_measure(states,target.support)
    frames=tuple(tuple(tuple(float(x) for x in row) for row in frame) for frame in xyz.detach().cpu().tolist())
    hashes=tuple(coordinates_sha256(frame) for frame in frames)
    indices={a:i for i,a in enumerate(target.system.atom_ids)}
    part_hashes=tuple(tuple(coordinates_sha256(tuple(frame[indices[a]] for a in p.atom_ids)) for p in partition.participants) for frame in frames)
    for h,f in zip(hashes,feedback,strict=True):
        if f is None:continue
        f.__post_init__()
        if (f.partition_sha256!=partition.sha256 or f.complex_xyz_sha256!=h
            or f.teacher_sha256!=target.teacher.sha256
            or f.teacher_family.upper()!=target.teacher.family.upper()
            or f.energy_reference_scheme!=target.teacher.energy_reference_scheme
            or len(f.participant_energies_ev)!=len(partition.participants)):
            raise TeacherRefusal('Feedback must use exact teacher, partition, complete frame and reference convention')
    return ExactComplexStateBinding(partition.sha256,target.sha256,hashes,part_hashes,
        tuple(None if f is None else f.sha256 for f in feedback),
        tuple(None if x is None else x.source_sha256 for x in observed))


def feedback_head_parameter_roles(head):
    if type(head) is not ComplexAffinityFeedbackHead or set(head._modules)!={'affinity','interaction'}:
        raise ValueError('Exact separate affinity/MLIP readout ownership required')
    width,base,experimental,teachers,mult=head.get_extra_state()
    roles={};hidden=Axis('width','hidden')
    for name,values,namespace in (('affinity',experimental,'experimental-endpoint-native-unit'),('interaction',teachers,'frozen-MLIP-electronic-interaction-eV')):
        layer=getattr(head,name)
        if (type(layer) is not nn.Linear or layer.in_features!=width or layer.out_features!=len(values)
            or layer.bias is None or tuple(layer.weight.shape)!=(len(values),width) or tuple(layer.bias.shape)!=(len(values),)):
            raise ValueError('Exact finite independent readout axes required')
        axis=Axis('finite',sha(canonical((namespace,values))))
        roles[name+'.weight']=ParameterRole('output',(axis,hidden));roles[name+'.bias']=ParameterRole('vector',(axis,))
    if head.affinity.weight is head.interaction.weight or head.affinity.bias is head.interaction.bias:
        raise ValueError('Experimental and teacher readouts must not be tied')
    parameter_ledger(head,roles)
    return roles


class ComplexCartesianFeedbackAdapter(nn.Module):
    def __init__(self,policy,heads):
        super().__init__()
        if type(policy) is not SharedCartesianGFNPolicy or type(heads) is not ComplexAffinityFeedbackHead:
            raise ValueError('Existing exact native Cartesian policy and separate feedback heads required')
        policy.get_extra_state();feedback_head_parameter_roles(heads)
        w,b,_,teachers,_=heads.get_extra_state()
        if (w,b)!=(policy.width,policy.base_width) or policy.teacher_sha256 not in teachers:
            raise ValueError('Common native width/base and frozen teacher axis required')
        self.policy=policy;self.heads=heads
        self._configuration=canonical((policy.get_extra_state(),heads.get_extra_state()))
    def get_extra_state(self):
        current=canonical((self.policy.get_extra_state(),self.heads.get_extra_state()))
        feedback_head_parameter_roles(self.heads)
        if current!=self._configuration:raise ValueError('Complex Cartesian source/operator configuration drift')
        return dict(schema='Mora3.DiagnosticComplexCartesianFeedback/v1',configuration_sha256=sha(current),training_authorized=False)
    def set_extra_state(self,state):
        if state!=self.get_extra_state():raise ValueError('Exact complex Cartesian checkpoint configuration required')
    def require_training_admission(self):
        raise TeacherRefusal('No complex affinity source/heldout/rights/teacher/calibration admission; diagnostic adapter only')
    def parameter_roles(self):
        self.get_extra_state()
        roles={'policy.'+n:r for n,r in cartesian_mup_parameter_roles(self.policy).items()}
        roles.update({'heads.'+n:r for n,r in feedback_head_parameter_roles(self.heads).items()})
        parameter_ledger(self,roles)
        return roles
    def diagnostic_objectives(self,partition,target,states,*,layer,observed,feedback,affinity_scales,teacher_scales,feedback_weight,origin):
        if origin!='synthetic_fixture':raise TeacherRefusal('Only explicit synthetic fixture diagnostics; no source admission')
        self.get_extra_state()
        binding=bind_complex_states(partition,target,states,observed=observed,feedback=feedback)
        bound=self.policy.bind(target)
        # All atoms and declared edges, current geometry, shared original TokenGT.
        context=self.policy.graph(bound.graphs(states,layer))[0][:,0]
        losses=self.heads.objectives(context,observed=observed,feedback=feedback,
            affinity_scales=affinity_scales,teacher_scales=teacher_scales,feedback_weight=feedback_weight)
        return binding,losses
