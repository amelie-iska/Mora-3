"""Opt-in nuclear semantic attachment to the native cooperative model.

Shares existing TokenGT transforms; image and context incidence remain explicit.
Independent source pools are not joined. No optimizer, training admission or
historical checkpoint migration is provided by this attachment.
"""
import copy
from torch import nn
import torch
from .mixed_payload_projection import exact_wire
from .lincs_cooperative import LincsCooperativeModel
from .color_semantic import ColorNuclearSemantic
from .color_conditioning import ConditionedColorSemantic
from .color_supervised_task import ColorSemanticTask


class ColorCooperativeModel(nn.Module):
    def __init__(self, parent, *, color_id_width, input_policy='native-max512-v1',parameterization_policy='legacy',source_envelope=None):
        super().__init__()
        if not isinstance(parent, LincsCooperativeModel):
            raise ValueError('native LINCS cooperative parent required')
        if type(color_id_width) is not int or color_id_width <= 0:
            raise ValueError('positive exact color identifier rank required')
        if parameterization_policy not in ('legacy','mup-cooperative-v1'):
            raise ValueError('known explicit color cooperative parameterization required')
        new=parameterization_policy!='legacy'
        if getattr(parent,'parameterization_policy','legacy')!=parameterization_policy:
            raise ValueError('cooperative parent/color parameterization policy mismatch')
        self.parameterization_policy=parameterization_policy
        backbone=parent.annotation.backbone
        width=backbone.feature.out_features
        if backbone.feature.in_features != width:
            raise ValueError('equal native feature and model width required')
        self.parent=parent
        spec=parent.annotation.get_extra_state().get('mup',{})
        color=ColorNuclearSemantic(width,alpha_policy='require-opaque',input_policy=input_policy,source_envelope=source_envelope,**(
            dict(parameterization_policy='mup-color-v1',base_width=spec['base_width'],
                output_mult=spec['output_mult'],backbone=backbone) if new else {}))
        if not new:
            for name in ('feature','role','graph_token','layers','final_norm'):
                setattr(color.backbone,name,getattr(backbone,name))
        self.color=ConditionedColorSemantic(color,id_width=color_id_width,**(
            dict(parameterization_policy='mup-color-context-v1') if new else {}))
        weight=backbone.feature.weight
        self.color.to(device=weight.device,dtype=weight.dtype)
        self.configuration=dict(version=1,architecture='native-cooperative-color-semantic-attachment-v1',
            parent=copy.deepcopy(parent.configuration),color=self.color.get_extra_state(),
            pairing='independent-source-pools-no-image-response-join',training_admission=False)

        if input_policy != 'native-max512-v1':
            self.configuration.update(version=2,
                native_color=copy.deepcopy(color.get_extra_state()))

        if new:self.configuration.update(version=3,parameterization_policy=parameterization_policy,
            native_color=copy.deepcopy(color.get_extra_state()))

        self._mup_configuration=copy.deepcopy(self.configuration) if new else None
        self._parent_reference=(parent,)

    def get_extra_state(self):
        if self._mup_configuration is not None:
            if (not exact_wire(self.configuration,self._mup_configuration)
                    or self.parameterization_policy!='mup-cooperative-v1'
                    or self.parent is not self._parent_reference[0]):
                raise ValueError('cooperative image owner configuration drift')
        if self._mup_configuration is not None and self.parameterization_policy!='legacy':
            self.color.get_extra_state()
            if (self.configuration.get('parameterization_policy')!=self.parameterization_policy
                    or self.parent.get_extra_state()!=self.configuration['parent']
                    or self.color.get_extra_state()!=self.configuration['color']
                    or self.color.color.get_extra_state()!=self.configuration['native_color']):
                raise ValueError('cooperative color parameterization configuration drift')
        return copy.deepcopy(self.configuration)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('cooperative image checkpoint configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def set_extra_state(self,state):
        if state != self.configuration:
            raise ValueError('cooperative color configuration mismatch')

    def color_objective(self,task,*,context_fields=None,expected_image_sha256=None):
        if type(task) is not ColorSemanticTask:
            raise ValueError('source-bound native color task required')
        if self.parent.configuration != self.configuration['parent']:
            raise ValueError('cooperative parent configuration changed')
        for name in ('feature','role','graph_token','layers','final_norm'):
            native=getattr(self.parent.annotation.backbone,name)
            if (getattr(self.color.color.backbone,name) is not native
                    or getattr(self.color.context.graph,name) is not native):
                raise ValueError('cooperative color shared ownership changed')
        return task.evaluate_conditioned(self.color,context_fields=context_fields,
            expected_image_sha256=expected_image_sha256)
