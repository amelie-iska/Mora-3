"""Opt-in task-selected scalar graph context for native color semantic logits.

Local adaptation: a learned spatially uniform logit offset from a shared TokenGT
context. Does not authorize joins, task admission, instance inference or calibrated
quality. Known graph topology remains a condition; missing selected fields refuse.
"""
from torch import nn
import torch
import math
import copy
from .mixed_payload_projection import exact_wire
from .color_semantic import ColorNuclearSemantic
from .geometry_conditioning import GeometryConditioner


class ConditionedColorSemantic(nn.Module):
    def __init__(self, color, *, id_width,parameterization_policy='legacy',context=None):
        super().__init__()
        if not isinstance(color, ColorNuclearSemantic):
            raise ValueError('native color nuclear semantic component required')
        if parameterization_policy not in ('legacy','mup-color-context-v1'):
            raise ValueError('known explicit color context parameterization required')
        new=parameterization_policy!='legacy'
        if new and getattr(color,'parameterization_policy','legacy')!='mup-color-v1':
            raise ValueError('explicit applied RGB color owner required')
        if context is not None and (not new or type(context) is not GeometryConditioner
                or context.config['id_width']!=id_width or context.parameterization_policy!='mup-geometry-v1'):
            raise ValueError('compatible explicit shared color context required')
        self.parameterization_policy=parameterization_policy
        self.color = color
        self.context = (context if context is not None else GeometryConditioner(color.backbone,id_width=id_width,
            **(dict(parameterization_policy='mup-geometry-v1',base_width=color.base_width) if new else {})))
        if new:
            self.context.get_extra_state()
            if any(getattr(self.context.graph,n) is not getattr(color.backbone,n)
                   for n in ('feature','role','graph_token','layers','final_norm')):
                raise ValueError('exact shared color context transforms required')
        self.offset = nn.Linear(color.width, 1)
        self.config = dict(version=1, architecture='task-graph-uniform-logit-offset-v1',
                           width=color.width, id_width=id_width,
                           pooling='existing-positional-payload',
                           training_admission=False)
        self._mup_configuration=None
        if new:
            with torch.no_grad():
                self.offset.weight.mul_(math.sqrt(color.width/color.base_width))
                self.offset.bias.mul_(math.sqrt(color.width/color.base_width))
            self.config.update(version=2,parameterization_policy=parameterization_policy,
                base_width=color.base_width,output_mult=color.output_mult,
                color=color.get_extra_state(),context=self.context.get_extra_state(),
                readout='finite-offset-native-rescale-sqrt-width-over-base-input-divided-by-width-over-base')
            self._mup_configuration=copy.deepcopy(self.config)

    def get_extra_state(self):
        if self._mup_configuration is not None:
            if (not exact_wire(self.config,self._mup_configuration)
                    or self.parameterization_policy!='mup-color-context-v1'
                    or not exact_wire(self.config['color'],self.color.get_extra_state())
                    or not exact_wire(self.config['context'],self.context.get_extra_state())
                    or any(getattr(self.context.graph,n) is not getattr(self.color.backbone,n)
                           for n in ('feature','role','graph_token','layers','final_norm'))):
                raise ValueError('conditioned color source/shared/readout configuration drift')
            return copy.deepcopy(self.config)
        return self.config.copy()

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('conditioned color checkpoint configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout_features(self,context):
        if self._mup_configuration is None:return context
        self.get_extra_state()
        return context*(self.color.output_mult/(self.color.width/self.color.base_width))

    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError('conditioned color configuration mismatch')

    def forward(self, pixels, record, task, *, recompute_pixels=False):
        # Select/scrub targets before pixel computation. No masks enter this API.
        context = self.context(record, task)
        return self.color(pixels,recompute_pixels=recompute_pixels) + self.offset(self._readout_features(context)).reshape(())


class ConditionedColorInstance(nn.Module):
    """Opt-in task-selected uniform per-slot offsets for native RGB masks.

    Reuses target-scrubbed graph selection and shared TokenGT transforms. This
    adaptation preserves independent overlapping slots; it does not admit source
    targets, increase capacity or establish segmentation quality.
    """
    def __init__(self, color, *, id_width,parameterization_policy='legacy',context=None):
        super().__init__()
        from .color_semantic import ColorInstancePredictor
        if not isinstance(color, ColorInstancePredictor):
            raise ValueError('native color instance predictor required')
        if parameterization_policy not in ('legacy','mup-color-context-v1'):
            raise ValueError('known explicit color context parameterization required')
        new=parameterization_policy!='legacy'
        if new and getattr(color,'parameterization_policy','legacy')!='mup-color-v1':
            raise ValueError('explicit applied RGB color owner required')
        if context is not None and (not new or type(context) is not GeometryConditioner
                or context.config['id_width']!=id_width or context.parameterization_policy!='mup-geometry-v1'):
            raise ValueError('compatible explicit shared color context required')
        self.parameterization_policy=parameterization_policy
        self.color = color
        self.context = (context if context is not None else GeometryConditioner(color.backbone,id_width=id_width,
            **(dict(parameterization_policy='mup-geometry-v1',base_width=color.base_width) if new else {})))
        if new:
            self.context.get_extra_state()
            if any(getattr(self.context.graph,n) is not getattr(color.backbone,n)
                   for n in ('feature','role','graph_token','layers','final_norm')):
                raise ValueError('exact shared color context transforms required')
        self.offset = nn.Linear(color.width, color.head.slots)
        self.config = dict(version=1, architecture='task-graph-per-slot-logit-offset-v1',
                           width=color.width, id_width=id_width, slots=color.head.slots,
                           pooling='existing-positional-payload', training_admission=False)
        self._mup_configuration=None
        if new:
            with torch.no_grad():
                self.offset.weight.mul_(math.sqrt(color.width/color.base_width))
                self.offset.bias.mul_(math.sqrt(color.width/color.base_width))
            self.config.update(version=2,parameterization_policy=parameterization_policy,
                base_width=color.base_width,output_mult=color.output_mult,
                color=color.get_extra_state(),context=self.context.get_extra_state(),
                readout='finite-offset-native-rescale-sqrt-width-over-base-input-divided-by-width-over-base')
            self._mup_configuration=copy.deepcopy(self.config)

    def get_extra_state(self):
        if self._mup_configuration is not None:
            if (not exact_wire(self.config,self._mup_configuration)
                    or self.parameterization_policy!='mup-color-context-v1'
                    or not exact_wire(self.config['color'],self.color.get_extra_state())
                    or not exact_wire(self.config['context'],self.context.get_extra_state())
                    or any(getattr(self.context.graph,n) is not getattr(self.color.backbone,n)
                           for n in ('feature','role','graph_token','layers','final_norm'))):
                raise ValueError('conditioned color source/shared/readout configuration drift')
            return copy.deepcopy(self.config)
        return self.config.copy()

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('conditioned color checkpoint configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _readout_features(self,context):
        if self._mup_configuration is None:return context
        self.get_extra_state()
        return context*(self.color.output_mult/(self.color.width/self.color.base_width))

    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError('conditioned color instance configuration mismatch')

    def forward(self, pixels, record, task):
        # Validate selected context before running the pixel encoder. Target masks
        # never enter the forward signature or the selected graph payload.
        context = self.context(record, task)
        offset=self.offset(self._readout_features(context))[0]
        if self.color.head.slot_policy=='source-layer-list-v4':
            return self.color.forward_layer_list(pixels,offset=offset,count_context_offset=context)
        return self.color(pixels)+offset.reshape(self.color.head.slots,1,1)


    def streamed_objective(self,pixels,record,task,instances,**options):
        """Complete source slots/pixels; targets remain loss-only."""
        if self.color.head.slot_policy!='source-envelope-v3':raise ValueError('explicit source-envelope instance policy required')
        self.get_extra_state()
        self.color._validate_source_targets(pixels,instances)
        context=self.context(record,task)
        offset=self.offset(self._readout_features(context))[0]
        return self.color.streamed_objective(pixels,instances,offset=offset,**options)

    def layer_list_objective(self,pixels,record,task,instances,**options):
        if self.color.head.slot_policy!='source-layer-list-v4':raise ValueError('explicit source ordered-layer-list policy required')
        self.get_extra_state();self.color._validate_source_targets(pixels,instances)
        context=self.context(record,task);offset=self.offset(self._readout_features(context))[0]
        return self.color.layer_list_objective(pixels,instances,offset=offset,count_context_offset=context,**options)

    @torch.no_grad()
    def generate_layer_list(self,pixels,record,task,**options):
        self.get_extra_state();context=self.context(record,task)
        offset=self.offset(self._readout_features(context))[0]
        return self.color.generate_layer_list(pixels,offset=offset,count_context_offset=context,**options)
