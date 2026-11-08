"""Opt-in complete overlapping nuclear instances on the cooperative backbone.

The native RGB encoder and task-selected context are shared by identity with the
semantic attachment. Independent slot projections preserve overlap; target masks
remain loss-only. This interface provides neither optimizer nor corpus admission,
and does not convert historical checkpoints or invent image/response joins.
"""
import copy
from torch import nn
import torch
from .mixed_payload_projection import exact_wire
from .color_cooperative import ColorCooperativeModel
from .color_semantic import ColorInstancePredictor
from .color_conditioning import ConditionedColorInstance
from .color_instance_task import ColorInstanceTask


class InstanceCooperativeModel(nn.Module):
    def __init__(self, parent, *, slots, max_output_bytes,
                 slot_policy='bounded-max64-v1',parameterization_policy='legacy',source_envelope=None):
        super().__init__()
        if type(parent) is not ColorCooperativeModel:
            raise ValueError('native color cooperative parent required')
        if (parameterization_policy not in ('legacy','mup-cooperative-v1')
                or getattr(parent,'parameterization_policy','legacy')!=parameterization_policy):
            raise ValueError('compatible explicit instance cooperative parameterization required')
        self.parameterization_policy=parameterization_policy
        new=parameterization_policy!='legacy'
        native = parent.color.color
        if slot_policy in('source-envelope-v3','source-layer-list-v4')and source_envelope is None:source_envelope=native.source_envelope
        instance = ColorInstancePredictor(native.width, slots=slots,
            max_output_bytes=max_output_bytes, alpha_policy=native.alpha_policy,
            input_policy=native.input_policy,slot_policy=slot_policy,source_envelope=source_envelope,**(
                dict(parameterization_policy='mup-color-v1',base_width=native.base_width,
                    output_mult=native.output_mult,native_color=native) if new else {}))
        instance.pixels = native.pixels
        # Image incidence and context incidence have different native ranks.
        # Share each complete existing branch, never replace graph attention.
        instance.backbone = native.backbone
        conditioned = ConditionedColorInstance(instance,
            id_width=parent.color.config['id_width'],**(
                dict(parameterization_policy='mup-color-context-v1',context=parent.color.context) if new else {}))
        conditioned.context = parent.color.context
        weight = native.head.weight
        conditioned.to(device=weight.device, dtype=weight.dtype)
        self.parent = parent
        self.instance = conditioned
        # Non-registering reference tuple: deepcopy preserves alias identity via
        # its memo; runtime ownership references are not checkpoint parameters.
        self._context_owners = (conditioned.context.graph.incidence,
                                conditioned.context.payload)
        self.configuration = dict(version=1,
            architecture='native-cooperative-overlapping-nuclear-instance-v1',
            parent=copy.deepcopy(parent.get_extra_state()),
            instance=copy.deepcopy(conditioned.get_extra_state()),
            instance_color=copy.deepcopy(instance.get_extra_state()),
            instance_head=copy.deepcopy(instance.head.get_extra_state()),
            pairing='independent-source-pools-no-image-response-join',
            training_admission=False)

        if new:self.configuration.update(version=3,parameterization_policy=parameterization_policy)

        self._mup_configuration=copy.deepcopy(self.configuration) if new else None
        self._parent_reference=(parent,)

    def get_extra_state(self):
        if self._mup_configuration is not None:
            if (not exact_wire(self.configuration,self._mup_configuration)
                    or self.parameterization_policy!='mup-cooperative-v1'
                    or self.parent is not self._parent_reference[0]):
                raise ValueError('cooperative image owner configuration drift')
        if self._mup_configuration is not None and self.parameterization_policy!='legacy':
            if (self.configuration.get('parameterization_policy')!=self.parameterization_policy
                    or self.parent.get_extra_state()!=self.configuration['parent']
                    or self.instance.get_extra_state()!=self.configuration['instance']
                    or self.instance.color.get_extra_state()!=self.configuration['instance_color']
                    or self.instance.color.head.get_extra_state()!=self.configuration['instance_head']):
                raise ValueError('cooperative instance parameterization configuration drift')
        return copy.deepcopy(self.configuration)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('cooperative image checkpoint configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def set_extra_state(self, state):
        if state != self.configuration:
            raise ValueError('cooperative instance configuration mismatch')

    def instance_objective(self, task, *, context_fields=None,
                           expected_image_sha256=None, pixel_block_size=None):
        if type(task) is not ColorInstanceTask:
            raise ValueError('source-bound native instance task required')
        if (self.parent.get_extra_state() != self.configuration['parent']
                or self.instance.get_extra_state() != self.configuration['instance']
                or self.instance.color.get_extra_state() != self.configuration['instance_color']
                or self.instance.color.head.get_extra_state() != self.configuration['instance_head']):
            raise ValueError('cooperative instance configuration changed')
        native = self.parent.color.color
        if (self.instance.context.graph.incidence is not self._context_owners[0]
                or self.instance.context.payload is not self._context_owners[1]):
            raise ValueError('cooperative instance shared ownership changed')
        if (self.instance.color.pixels is not native.pixels
                or self.instance.color.backbone is not native.backbone
                or self.instance.context is not self.parent.color.context):
            raise ValueError('cooperative instance shared ownership changed')
        # Also validate the parent's original cooperative transform ownership;
        # an alias replaced in place can otherwise affect both branches equally.
        backbone = self.parent.parent.annotation.backbone
        if self.parent.parent.configuration != self.parent.configuration['parent']:
            raise ValueError('cooperative parent configuration changed')
        for name in ('feature', 'role', 'graph_token', 'layers', 'final_norm'):
            shared = getattr(backbone, name)
            if (getattr(native.backbone, name) is not shared
                    or getattr(self.instance.context.graph, name) is not shared):
                raise ValueError('cooperative instance shared ownership changed')
        return task.evaluate_conditioned(self.instance,
            context_fields=context_fields, expected_image_sha256=expected_image_sha256,
            pixel_block_size=pixel_block_size)
