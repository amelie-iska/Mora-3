"""Separate observed affinity and computed MLIP feedback objectives."""
from dataclasses import dataclass
import math

import torch
from torch import nn

from mora3.complex_affinity_feedback import MLIPInteractionFeedback
from mora3.teacher_protocol import finite_float,hex_sha
from .mixed_payload_projection import exact_wire


@dataclass(frozen=True, slots=True)
class AffinityObservation:
    """An observed point in its declared native assay endpoint and unit."""

    value: float
    endpoint: str
    unit: str
    source_sha256: str

    def __post_init__(self):
        if (not finite_float(self.value)
                or type(self.endpoint) is not str or not self.endpoint.strip()
                or type(self.unit) is not str or not self.unit.strip()
                or not hex_sha(self.source_sha256)):
            raise ValueError('Finite observed affinity, native endpoint/unit and exact source identity required')


class ComplexAffinityFeedbackHead(nn.Module):
    """Width-scaled readouts for a shared complex geometry encoding.

    Every output axis remains finite. Affinity labels and MLIP teacher energies
    keep independent heads, units and observation masks.
    """

    def __init__(self,width,base_width,*,experimental_axes,teacher_axes,output_mult=1.):
        super().__init__()
        if (type(width) is not int or type(base_width) is not int or not 1<=base_width<=width
                or not finite_float(output_mult) or output_mult<=0
                or type(experimental_axes) is not tuple or not experimental_axes
                or any(type(a) is not tuple or len(a)!=2 or any(type(x) is not str or not x.strip() for x in a)
                       for a in experimental_axes)
                or len(set(experimental_axes))!=len(experimental_axes)
                or type(teacher_axes) is not tuple or not teacher_axes
                or any(not hex_sha(x) for x in teacher_axes)
                or len(set(teacher_axes))!=len(teacher_axes)):
            raise ValueError('Explicit ordered experimental endpoint/units and teacher identities required')
        self._binding = (width,base_width,experimental_axes,teacher_axes,output_mult)
        self.affinity = nn.Linear(width,len(experimental_axes))
        self.interaction = nn.Linear(width,len(teacher_axes))
        with torch.no_grad():
            factor = math.sqrt(width/base_width)
            for layer in (self.affinity,self.interaction):
                layer.weight.mul_(factor)
                layer.bias.mul_(factor)

    def get_extra_state(self):
        return self._binding

    def set_extra_state(self,state):
        if not exact_wire(state,self._binding):
            raise ValueError('Exact affinity/teacher finite-axis and width configuration required')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        self.set_extra_state(state_dict.get(prefix+'_extra_state'))
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def forward(self,context):
        width,base,_,_,mult = self._binding
        weight = self.affinity.weight
        if (not isinstance(context,torch.Tensor) or context.ndim!=2 or context.shape[1]!=width
                or not len(context) or context.device!=weight.device or context.dtype!=weight.dtype
                or not torch.isfinite(context).all()):
            raise ValueError('Exact finite shared complex geometry context required')
        scaled = context*(mult/(width/base))
        return self.affinity(scaled),self.interaction(scaled)

    def objectives(self,context,*,observed,feedback,affinity_scales,teacher_scales,feedback_weight):
        """Combine source-specific normalized residuals with an explicit auxiliary weight."""
        _,_,experimental_axes,teacher_axes,_ = self._binding
        if (type(observed) is not tuple or type(feedback) is not tuple
                or len(observed)!=len(feedback)
                or type(affinity_scales) is not tuple or len(affinity_scales)!=len(experimental_axes)
                or type(teacher_scales) is not tuple or len(teacher_scales)!=len(teacher_axes)
                or any(not finite_float(s) or s<=0 for s in (*affinity_scales,*teacher_scales))
                or not finite_float(feedback_weight) or feedback_weight<=0):
            raise ValueError('Explicit affinity/feedback masks, positive source scales and auxiliary weight required')
        affinity,interaction = self(context)
        if len(observed)!=len(affinity):
            raise ValueError('Exact one source disposition per encoded complex required')
        experimental_terms,teacher_terms = [],[]
        for i,(label,teacher) in enumerate(zip(observed,feedback,strict=True)):
            if label is not None:
                if type(label) is not AffinityObservation or (label.endpoint,label.unit) not in experimental_axes:
                    raise ValueError('Observed affinity endpoint and unit must match a frozen source axis')
                axis = experimental_axes.index((label.endpoint,label.unit))
                experimental_terms.append(((affinity[i,axis].double()-label.value)/affinity_scales[axis]).square())
            if teacher is not None:
                if type(teacher) is not MLIPInteractionFeedback or teacher.teacher_sha256 not in teacher_axes:
                    raise ValueError('Exact teacher feedback identity must match a frozen auxiliary axis')
                axis = teacher_axes.index(teacher.teacher_sha256)
                teacher_terms.append(((interaction[i,axis].double()-teacher.interaction_energy_ev)/teacher_scales[axis]).square())
        if not experimental_terms and not teacher_terms:
            raise ValueError('No observed experimental or computed feedback targets')
        experimental = torch.stack(experimental_terms).mean() if experimental_terms else None
        computed = torch.stack(teacher_terms).mean() if teacher_terms else None
        total = sum(term for term in (experimental,None if computed is None else feedback_weight*computed)
                    if term is not None)
        if not torch.isfinite(total):
            raise ValueError('Nonfinite affinity feedback loss')
        return dict(total=total,affinity=experimental,mlip_interaction=computed,
                    experimental_observations=len(experimental_terms),mlip_observations=len(teacher_terms))
