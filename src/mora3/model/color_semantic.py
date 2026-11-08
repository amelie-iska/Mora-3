"""Opt-in RGB nuclear semantic branch; no instance IDs or training admission.

Uses the existing padded multiscale pixel architecture and vanilla TokenGT
image-node context. This separate configuration never migrates brightfield
checkpoints. Forward accepts only pixels; source masks are loss-only targets.
"""
import torch
import copy
import math
from .mixed_payload_projection import exact_wire
from torch import nn
from torch.nn import functional as F
from .color_input import color_tensor
from .morphology import MultiscalePixels
from .tokengt import TokenGT


def nuclear_semantic_objective(logits, target):
    """Mean binary nuclear pixel NLL; never interprets targets as instances."""
    if (logits.ndim != 2 or not logits.numel() or not logits.is_floating_point()
            or target.shape != logits.shape or target.dtype != torch.bool
            or target.device != logits.device or not torch.isfinite(logits).all()):
        raise ValueError('finite aligned logits and native boolean semantic mask required')
    return F.binary_cross_entropy_with_logits(logits, target.to(logits.dtype))


class ColorNuclearSemantic(nn.Module):
    def __init__(self, width=16, *, alpha_policy, input_policy='native-max512-v1',
                 parameterization_policy='legacy',base_width=None,output_mult=1.0,backbone=None,attn_mult=1.0,encoder_var=1.0,source_envelope=None):
        super().__init__()
        if type(width) is not int or width < 2 or width % 2:
            raise ValueError('even positive width required')
        if alpha_policy != 'require-opaque':
            raise ValueError('explicit require-opaque alpha policy required')
        from .raster_source_envelope import RasterSourceEnvelope
        full=input_policy=='source-envelope-v2'
        if(input_policy not in('native-max512-v1','native-max1000-v1','source-envelope-v2')
                or(full and(type(source_envelope)is not RasterSourceEnvelope or parameterization_policy!='mup-color-v1'))
                or(not full and source_envelope is not None)):
            raise ValueError('explicit complete source native color policy required')
        if full:source_envelope.binding()
        self.source_envelope=source_envelope
        if (parameterization_policy not in ('legacy','mup-color-v1')
                or (parameterization_policy=='legacy' and (base_width is not None or output_mult!=1
                    or backbone is not None or attn_mult!=1 or encoder_var!=1))
                or (parameterization_policy!='legacy' and (type(base_width) is not int or base_width<2
                    or base_width%2 or width<base_width or type(output_mult) not in (int,float)
                    or not math.isfinite(output_mult) or output_mult<=0))):
            raise ValueError('explicit compatible color base/readout parameterization required')
        new=parameterization_policy!='legacy'
        source=None
        if backbone is not None:
            if type(backbone) is not TokenGT:raise ValueError('exact native shared color transforms required')
            source=backbone.configuration()
            if (source['schema']!='tokengt-native-configuration-v3' or source['feature_width']!=width
                    or source['width']!=width or source['parameterization']['base_width']!=base_width):
                raise ValueError('new native shared color configuration mismatch')
        self.parameterization_policy,self.base_width,self.output_mult=parameterization_policy,base_width,float(output_mult)
        self.input_policy=input_policy
        self.width=width
        self.alpha_policy=alpha_policy
        self.pixels=MultiscalePixels(width,input_channels=3,**(
            dict(parameterization_policy='mup-image-v1',base_width=base_width) if new else {}))
        options={} if not new else dict(scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',
            attn_mult=source['attn_mult'] if source else attn_mult,encoder_var=source['encoder_var'] if source else encoder_var,
            parameterization_policy='mup-tokengt-v1',base_width=base_width,
            base_ffn_width=source['parameterization']['base_ffn_width'] if source else base_width)
        self.backbone=TokenGT(width,1,width=width,heads=source['heads'] if source else 2,
            layers=source['layers'] if source else 1,ffn_width=source['ffn_width'] if source else None,**options)
        if source:
            for name in ('feature','role','graph_token','layers','final_norm'):
                setattr(self.backbone,name,getattr(backbone,name))
        self.head=nn.Conv2d(2*width,1,1)
        if new:
            with torch.no_grad():
                self.head.weight.mul_(math.sqrt(width/base_width));self.head.bias.mul_(math.sqrt(width/base_width))
        self._mup_configuration=None
        self._backbone_reference=(self.backbone,)
        self._source_transform_reference=(backbone,) if source else None
        if new:self._mup_configuration=self._current_parameterization()

    def _current_parameterization(self):
        state=dict(policy=self.parameterization_policy,base_width=self.base_width,output_mult=self.output_mult,
            width=self.width,alpha_policy=self.alpha_policy,input_policy=self.input_policy,
            native_backbone=self.backbone.configuration(),pixels=self.pixels.parameterization_configuration(),
            readout='finite-conv1x1-native-rescale-sqrt-width-over-base-input-divided-by-width-over-base')
        if self.source_envelope is not None:state['source_envelope']=self.source_envelope.binding()
        return state

    def _validate_mup_configuration(self):
        if (set(self._modules)!={'pixels','backbone','head'} or type(self.pixels) is not MultiscalePixels
                or type(self.backbone) is not TokenGT or type(self.head) is not nn.Conv2d
                or (self.head.in_channels,self.head.out_channels,self.head.kernel_size,self.head.stride,
                    self.head.padding,self.head.dilation,self.head.groups,self.head.padding_mode,
                    self.head.transposed,self.head.output_padding)
                    !=(2*self.width,1,(1,1),(1,1),(0,0),(1,1),1,'zeros',False,(0,0))):
            raise ValueError('RGB semantic operator configuration drift')
        if (self.backbone is not self._backbone_reference[0]
                or not exact_wire(self._current_parameterization(),self._mup_configuration)):
            raise ValueError('color native/readout/pixel configuration drift')
        if self._source_transform_reference is not None:
            source=self._source_transform_reference[0]
            if any(getattr(self.backbone,n) is not getattr(source,n)
                   for n in ('feature','role','graph_token','layers','final_norm')):
                raise ValueError('color shared transform ownership drift')

    def get_extra_state(self):
        state = dict(version=1,architecture='rgb-two-level-padded-bilinear-v1',
                    width=self.width,alpha_policy=self.alpha_policy,
                    normalization='uint8-div255',outputs=['nuclear-semantic'],attention='vanilla')
        if self.input_policy != 'native-max512-v1':
            state.update(version=2,input_policy=self.input_policy)
        if self.source_envelope is not None:state['source_envelope']=self.source_envelope.binding()
        if self._mup_configuration is not None:
            self._validate_mup_configuration()
            state.update(version=4 if self.source_envelope is not None else 3,input_policy=self.input_policy,mup=copy.deepcopy(self._mup_configuration))
        return state

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            current=self.state_dict()
            for key,value in current.items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('color checkpoint child configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def set_extra_state(self,state):
        if state != self.get_extra_state():
            raise ValueError('color nuclear semantic configuration mismatch')

    def forward(self,pixels,*,recompute_pixels=False):
        """Optionally trade backward compute for pixel activation storage.

        Per-call execution policy only: parameters, state and arithmetic are
        unchanged. Inference without autograd uses the ordinary pixel path.
        """
        if type(recompute_pixels) is not bool:
            raise ValueError('recompute_pixels must be boolean')
        if self._mup_configuration is not None:self.get_extra_state()
        p=self.head.weight
        image=color_tensor(pixels,alpha_policy=self.alpha_policy,device=p.device,dtype=p.dtype,input_policy=self.input_policy,source_envelope=self.source_envelope)
        if recompute_pixels and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            local=checkpoint(self.pixels,image,use_reentrant=False)
        else:
            local=self.pixels(image)
        tensor=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long,device=p.device),
                    roles=torch.zeros(1,dtype=torch.long,device=p.device),
                    identifiers=F.pad(torch.ones((1,1),dtype=p.dtype,device=p.device),
                                      (0,self.backbone.incidence.in_features//2-1)))
        latent,_=self.backbone([tensor])
        context=latent[:,0,:,None,None].expand(-1,-1,*image.shape[-2:])
        features=torch.cat((local,context),dim=1)
        if self._mup_configuration is not None:
            self.get_extra_state();features=features*(self.output_mult/(self.width/self.base_width))
        return self.head(features)[0,0]


class ColorInstancePredictor(nn.Module):
    """Opt-in native RGB/local plus image TokenGT-context mask slots.

    Independent configuration; no semantic checkpoint conversion or task-graph
    conditioning. Targets never enter forward. Slot capacity is not source
    admission or measured instance quality; output budget excludes activations.
    """
    def __init__(self, width=16, *, slots, max_output_bytes, alpha_policy,
                 input_policy='native-max512-v1', slot_policy='bounded-max64-v1',
                 parameterization_policy='legacy',base_width=None,output_mult=1.0,native_color=None,source_envelope=None):
        super().__init__()
        from .instance_affinity import OverlappingInstanceHead
        self.source_envelope=source_envelope
        if source_envelope is not None and slot_policy not in('source-envelope-v3','source-layer-list-v4'):
            raise ValueError('complete source instance slot policy required')
        # Reuse the exact native encoder/backbone construction and validation;
        # discard the semantic projection rather than register unused parameters.
        if native_color is not None:
            if (parameterization_policy!='mup-color-v1' or type(native_color) is not ColorNuclearSemantic
                    or native_color.width!=width or native_color.alpha_policy!=alpha_policy or native_color.input_policy!=input_policy
                    or native_color.parameterization_policy!=parameterization_policy or native_color.base_width!=base_width
                    or native_color.output_mult!=output_mult or native_color.source_envelope!=source_envelope):
                raise ValueError('exact compatible shared new semantic pixels/backbone required')
            native_color.get_extra_state();native=native_color
        else:
            native = ColorNuclearSemantic(width,alpha_policy=alpha_policy,input_policy=input_policy,
                parameterization_policy=parameterization_policy,base_width=base_width,output_mult=output_mult,source_envelope=source_envelope)
        self.parameterization_policy,self.base_width,self.output_mult=parameterization_policy,base_width,float(output_mult)
        self.width = width
        self.alpha_policy = alpha_policy
        self.input_policy = input_policy
        self.pixels = native.pixels
        self.backbone = native.backbone
        self.head = OverlappingInstanceHead(2*width, slots=slots,
                                           max_output_bytes=max_output_bytes,slot_policy=slot_policy,source_envelope=source_envelope,**(
            dict(parameterization_policy='mup-mask-slots-v1',base_channels=2*base_width,output_mult=output_mult)
            if parameterization_policy!='legacy' else {}))
        if slot_policy=='source-layer-list-v4':
            self.count=nn.Linear(width,slots+1)
            with torch.no_grad():
                self.count.weight.mul_(math.sqrt(width/base_width));self.count.bias.mul_(math.sqrt(width/base_width))
        self._mup_configuration=None
        self._shared_reference=(native.pixels,native.backbone)
        if parameterization_policy!='legacy':self._mup_configuration=self._current_parameterization()

    def _current_parameterization(self):
        state=dict(policy=self.parameterization_policy,base_width=self.base_width,output_mult=self.output_mult,
            width=self.width,alpha_policy=self.alpha_policy,input_policy=self.input_policy,
            native_backbone=self.backbone.configuration(),pixels=self.pixels.parameterization_configuration(),
            head=self.head.get_extra_state())
        if self.head.slot_policy=='source-layer-list-v4':
            state['source_layer_list']=dict(count_axis=tuple(range(self.head.slots+1)),
                count='finite-W-to-Splus1-MuReadout;supplied-source-layer-count-not-biological-nuclei',
                masks='ordered-source-plane-i-to-slot-i;observedemptyplanesincluded',
                mask_padding='not-observed;excluded-from-loss',objective='countCE+observed-layer-pixel-meanBCE',
                count_weight=1.0,mask_weight=1.0,distribution='categoricalKtimesindependentBernoulli-pixels-for-FIRSTKplanes')
        if self.source_envelope is not None:state['source_envelope']=self.source_envelope.binding_for_slot_policy(self.head.slot_policy)
        return state

    def get_extra_state(self):
        if self._mup_configuration is not None:
            if self.head.slot_policy=='source-layer-list-v4'and(type(self.count)is not nn.Linear
                    or(self.count.in_features,self.count.out_features)!=(self.width,self.head.slots+1)
                    or self.count._modules or self.count._buffers):raise ValueError('source layer count readout drift')
            if ((self.pixels,self.backbone)!=self._shared_reference
                    or not exact_wire(self._current_parameterization(),self._mup_configuration)):
                raise ValueError('color instance shared/readout configuration drift')
            return dict(version=5 if self.head.slot_policy=='source-layer-list-v4'else(4 if self.source_envelope is not None else 3),architecture='native-rgb-image-context-mask-slots-mup-v1',
                width=self.width,alpha_policy=self.alpha_policy,input_policy=self.input_policy,
                mup=copy.deepcopy(self._mup_configuration),training_admission=False)
        if self.head.slot_policy == 'extended-max512-v2':
            return dict(version=2, architecture='native-rgb-image-context-mask-slots-v2',
                        width=self.width, alpha_policy=self.alpha_policy,
                        input_policy=self.input_policy, slot_policy=self.head.slot_policy,
                        training_admission=False)
        return dict(version=1, architecture='native-rgb-image-context-mask-slots-v1',
                    width=self.width, alpha_policy=self.alpha_policy,
                    input_policy=self.input_policy, training_admission=False)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('color instance predictor configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            for key,value in self.state_dict().items():
                if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                    raise ValueError('color instance checkpoint configuration/policy mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def forward(self, pixels):
        if self.head.slot_policy=='source-layer-list-v4':return self.forward_layer_list(pixels)
        if self._mup_configuration is not None:self.get_extra_state()
        p = self.head.projection.weight
        image = color_tensor(pixels, alpha_policy=self.alpha_policy,
                             device=p.device, dtype=p.dtype, input_policy=self.input_policy,source_envelope=self.source_envelope)
        # Reject output overflow before allocating pixel encoder activations.
        if self.head.slots * image.shape[-2] * image.shape[-1] * p.element_size() > self.head.max_output_bytes:
            raise ValueError('instance output-byte policy exceeded')
        local = self.pixels(image)
        tensor = dict(features=local.mean((2,3)),
                      endpoints=torch.zeros((1,2),dtype=torch.long,device=p.device),
                      roles=torch.zeros(1,dtype=torch.long,device=p.device),
                      identifiers=F.pad(torch.ones((1,1),dtype=p.dtype,device=p.device),
                                        (0,self.backbone.incidence.in_features//2-1)))
        latent,_ = self.backbone([tensor])
        context = latent[:,0,:,None,None].expand(-1,-1,*image.shape[-2:])
        return self.head(torch.cat((local,context),dim=1))

    def _validate_source_targets(self,pixels,instances):
        import numpy as np
        if (not isinstance(pixels,np.ndarray)or pixels.ndim!=3 or not isinstance(instances,torch.Tensor)
                or instances.ndim!=3 or instances.dtype!=torch.bool or instances.shape[1:]!=pixels.shape[:2]
                or instances.shape[0]>self.head.slots):
            raise ValueError('complete aligned source planes required before encoder')

    def streamed_objective(self,pixels,instances,*,offset=None,recompute_pixels=True,**stream_options):
        """New complete source mode: all pixels/slots, no full-slot logits."""
        if self.head.slot_policy!='source-envelope-v3':raise ValueError('explicit complete source instance mode required')
        self.get_extra_state();self._validate_source_targets(pixels,instances);p=self.head.projection.weight
        image=color_tensor(pixels,alpha_policy=self.alpha_policy,device=p.device,dtype=p.dtype,
            input_policy=self.input_policy,source_envelope=self.source_envelope)
        if recompute_pixels and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            local=checkpoint(self.pixels,image,use_reentrant=False,preserve_rng_state=True)
        else:local=self.pixels(image)
        tensor=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long,device=p.device),
            roles=torch.zeros(1,dtype=torch.long,device=p.device),
            identifiers=F.pad(torch.ones((1,1),dtype=p.dtype,device=p.device),(0,self.backbone.incidence.in_features//2-1)))
        latent,_=self.backbone([tensor]);context=latent[:,0,:,None,None].expand(-1,-1,*image.shape[-2:])
        from .streamed_instance_objective import streamed_affine_mask_set_objective
        return streamed_affine_mask_set_objective(self.head,torch.cat((local,context),dim=1),instances,offset=offset,**stream_options)

    def _layer_list_features(self,pixels,*,recompute_pixels=False):
        if self.head.slot_policy!='source-layer-list-v4':raise ValueError('explicit ordered source-layer-list policy required')
        self.get_extra_state();p=self.head.projection.weight
        image=color_tensor(pixels,alpha_policy=self.alpha_policy,device=p.device,dtype=p.dtype,
            input_policy=self.input_policy,source_envelope=self.source_envelope)
        if recompute_pixels and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            local=checkpoint(self.pixels,image,use_reentrant=False,preserve_rng_state=True)
        else:local=self.pixels(image)
        tensor=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long,device=p.device),
            roles=torch.zeros(1,dtype=torch.long,device=p.device),
            identifiers=F.pad(torch.ones((1,1),dtype=p.dtype,device=p.device),(0,self.backbone.incidence.in_features//2-1)))
        latent=self.backbone([tensor])[0][:,0]
        context=latent[:,:,None,None].expand(-1,-1,*image.shape[-2:])
        return torch.cat((local,context),dim=1),latent

    def _layer_count_logits(self,latent,count_context_offset=None):
        if count_context_offset is not None:
            if(count_context_offset.shape!=latent.shape or count_context_offset.dtype!=latent.dtype
                    or count_context_offset.device!=latent.device or not torch.isfinite(count_context_offset).all()):
                raise ValueError('matching target-free known source context required')
            latent=latent+count_context_offset
        return self.count(latent*(self.output_mult/(self.width/self.base_width)))

    def forward_layer_list(self,pixels,*,offset=None,count_context_offset=None,recompute_pixels=False):
        features,latent=self._layer_list_features(pixels,recompute_pixels=recompute_pixels)
        masks=self.head(features)
        if offset is not None:masks=masks+offset[:,None,None]
        return dict(count_logits=self._layer_count_logits(latent,count_context_offset),mask_logits=masks)

    def layer_list_objective(self,pixels,instances,*,offset=None,count_context_offset=None,
            recompute_pixels=True,return_components=False,**stream_options):
        if type(return_components)is not bool:raise ValueError('explicit component reporting policy required')
        self._validate_source_targets(pixels,instances)
        features,latent=self._layer_list_features(pixels,recompute_pixels=recompute_pixels)
        count_logits=self._layer_count_logits(latent,count_context_offset)
        # Stack dimension is a target ONLY. It never enters the source encoder.
        target=count_logits.new_tensor([len(instances)],dtype=torch.long)
        count_loss=F.cross_entropy(count_logits.double(),target)
        from .streamed_instance_objective import streamed_ordered_affine_mask_list_objective
        mask_loss=streamed_ordered_affine_mask_list_objective(self.head,features,instances,offset=offset,**stream_options)
        loss=count_loss+mask_loss
        return dict(loss=loss,count_loss=count_loss,mask_loss=mask_loss)if return_components else loss

    @torch.no_grad()
    def generate_layer_list(self,pixels,*,offset=None,count_context_offset=None,count_rule='argmax',generator=None,
            slot_block_size=32,pixel_block_size=8192,max_workspace_bytes=256*1024*1024):
        if(count_rule not in('argmax','sample')or(count_rule=='sample'and not isinstance(generator,torch.Generator))
                or any(type(n)is not int or n<1 for n in(slot_block_size,pixel_block_size,max_workspace_bytes))):
            raise ValueError('explicit complete source-list generation/resource policy required')
        features,latent=self._layer_list_features(pixels);p=self.head.projection.weight
        count_logits=self._layer_count_logits(latent,count_context_offset)
        if not torch.isfinite(count_logits).all():raise ValueError('finite normalized source-count logits required')
        if count_rule=='sample':
            gd=generator.device
            if gd.type=='cuda'and gd.index is None:gd=torch.device('cuda',torch.cuda.current_device())
            if gd!=p.device:raise ValueError('isolated source-count generator must match actual model device')
        k=int(count_logits.argmax(-1))if count_rule=='argmax'else int(torch.multinomial(count_logits.softmax(-1),1,generator=generator))
        height,width=features.shape[2:];pixels_count=height*width
        if k*pixels_count>self.head.max_output_bytes:raise ValueError('complete ordered Boolean source-list output budget exceeded; no crop')
        qb=min(max(k,1),slot_block_size);pb=min(pixels_count,pixel_block_size)
        if(2*self.head.channels*pb+4*qb*pb)*p.element_size()>max_workspace_bytes:
            raise ValueError('complete ordered source generation workspace exceeded; no crop')
        if offset is None:offset=p.new_zeros(self.head.slots)
        if offset.shape!=(self.head.slots,)or offset.dtype!=p.dtype or offset.device!=p.device or not torch.isfinite(offset).all():
            raise ValueError('matching complete ordered source offsets required')
        output=torch.empty(k,pixels_count,dtype=torch.bool,device='cpu');x=features[0].reshape(self.head.channels,pixels_count).T
        factor=self.head.output_mult/(self.head.channels/self.head.base_channels)
        for left in range(0,k,slot_block_size):
            right=min(k,left+slot_block_size)
            for start in range(0,pixels_count,pixel_block_size):
                stop=min(pixels_count,start+pixel_block_size)
                logits=F.linear(x[start:stop]*factor,p[left:right,:,0,0],self.head.projection.bias[left:right]).T+offset[left:right,None]
                if not torch.isfinite(logits).all():raise ValueError('finite full ordered source mask logits required')
                output[left:right,start:stop]=(logits>=0).cpu()
        return dict(source_layer_count=k,ordered_masks=output.reshape(k,height,width),count_rule=count_rule,mask_rule='independent-logit-ge0',
            biological_object_count=False,empty_supplied_or_generated_layers_preserved=True)
