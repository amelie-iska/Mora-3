"""Full-resolution foreground diagnostic, not an instance segmentation model.

Local adaptation: CNN pixel features -> mean image node -> shared TokenGT ->
broadcast graph context -> pixel logits. No crop/resize or target topology.
Pooling loses spatial detail in the graph context; the pixel branch retains it.
Only image pixels and known image-node incidence are inputs in this version.
"""
import numpy as np
import copy
import math
from .mixed_payload_projection import exact_wire
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.livecell_graph import LiveCellGraphTask
from mora3.data.raster_morphology import RasterMorphologyTask
from mora3.data.segmentation_image import SegmentationImageInput
from mora3.data.full_source_raster_input import FullSourceRasterInput
from .raster_source_envelope import RasterSourceEnvelope
from .tokengt import TokenGT


def foreground_objective(logits, instances):
    """Mean Bernoulli NLL against union of all overlapping whole-cell masks."""
    if (logits.ndim != 2 or instances.ndim != 3 or instances.shape[1:] != logits.shape
            or instances.dtype != torch.bool or instances.device != logits.device
            or not torch.isfinite(logits).all()):
        raise ValueError('Aligned finite logits and boolean instance planes required')
    return F.binary_cross_entropy_with_logits(logits, instances.any(0).to(logits.dtype))


class MorphologyForeground(nn.Module):
    def __init__(self, width=16, *, normalization="unsigned-dtype-max"):
        super().__init__()
        if width < 2 or width % 2: raise ValueError('Even positive width required')
        if normalization not in ("unsigned-dtype-max", "image-minmax"):
            raise ValueError("Unknown image normalization")
        self.normalization = normalization
        self.pixels=nn.Sequential(nn.Conv2d(1,width,3,padding=1),nn.GELU(),
                                  nn.Conv2d(width,width,3,padding=1),nn.GELU())
        self.backbone=TokenGT(width,1,width=width,heads=2,layers=1)
        self.head=nn.Conv2d(2*width,1,1)

    def forward(self, task, *, for_training=False):
        return self._logits(task, for_training=for_training)[0]

    def _logits(self, task, *, for_training=False, external_context=None):
        if isinstance(task, FullSourceRasterInput):
            if type(task) is not FullSourceRasterInput or getattr(self, 'source_envelope', None) != task.source_envelope:
                raise ValueError('Matching explicit grayscale source envelope required')
            self.get_extra_state()
        if not isinstance(task,(LiveCellGraphTask,RasterMorphologyTask,SegmentationImageInput,FullSourceRasterInput)) or task.direction!='image_to_mask':
            raise ValueError('Only image_to_mask direction supported')
        param=self.head.weight
        if isinstance(task,(RasterMorphologyTask,SegmentationImageInput,FullSourceRasterInput)):
            if for_training: task.require_training()
            x=task
            image=task.input_image(device=param.device).to(param.dtype)
            image=image/np.iinfo(np.dtype(task.image_dtype)).max
        else:
            if for_training: task.example.require_training()
            # Decode pixels directly: do not construct targets or read polygons.
            x=task.example
            image=torch.from_numpy(np.frombuffer(bytes.fromhex(x.pixels_hex),dtype=x.dtype)
                                   .copy().reshape(x.height,x.width))
            image=image.to(device=param.device,dtype=param.dtype)/np.iinfo(np.dtype(x.dtype)).max
        if self.normalization == "image-minmax":
            # Input-only per-image affine map; constants map to zero. No fitted stats.
            low = image.amin()
            span = image.amax() - low
            image = (image - low) / torch.where(span > 0, span, torch.ones_like(span))
        local=self.pixels(image[None,None])
        # image_to_mask input_graph has exactly one image node, zero edges.
        graph=task.input_graph()
        if len(graph.nodes)!=1 or graph.edges: raise ValueError('Unknown input topology')
        tensor=dict(features=local.mean((2,3)),endpoints=torch.zeros((1,2),dtype=torch.long,device=param.device),
                    roles=torch.zeros(1,dtype=torch.long,device=param.device),
                    identifiers=F.pad(torch.ones((1,1),dtype=param.dtype,device=param.device),
                                      (0,self.backbone.incidence.in_features//2-1)))
        latent,_=self.backbone([tensor])
        if external_context is not None:
            if (external_context.shape != latent[:,0].shape
                    or external_context.device != latent.device
                    or external_context.dtype != latent.dtype
                    or not torch.isfinite(external_context).all()):
                raise ValueError('aligned finite external context required')
            # Opt-in additive graph context; retain the full native pixel path.
            latent = latent.clone()
            latent[:,0] = latent[:,0] + external_context
        context=latent[:,0,:,None,None].expand(-1,-1,x.height,x.width)
        features=torch.cat((local,context),dim=1)
        if getattr(self,'parameterization_policy','legacy')!='legacy':
            self.get_extra_state()
            features=features*(self.output_mult/(self.width/self.base_width))
        return self.head(features)[0]


class MultiscalePixels(nn.Module):
    """Two-level U-shaped adaptation; padded convolutions preserve all pixels.

    Bilinear interpolation to skip sizes (align_corners=False) supports odd and
    singleton dimensions. Not the original valid-convolution U-Net architecture.
    """
    def __init__(self, width, *, input_channels=1,parameterization_policy='legacy',base_width=None):
        super().__init__()
        if parameterization_policy not in ('legacy','mup-image-v1'):
            raise ValueError('explicit known image parameterization policy required')
        if parameterization_policy=='legacy' and base_width is not None:
            raise ValueError('legacy pixel encoder has no base width')
        if parameterization_policy!='legacy' and (type(width) is not int or type(base_width) is not int
                or base_width<2 or base_width%2 or width<base_width or width%2
                or type(input_channels) is not int or input_channels<1):
            raise ValueError('positive even pixel widths and finite input channels required')
        self.parameterization_policy,self.base_width=parameterization_policy,base_width
        self.width,self.input_channels=width,input_channels
        self._declaration=(width,input_channels,parameterization_policy,base_width)
        def block(a,b):
            return nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.GELU(),
                                 nn.Conv2d(b,b,3,padding=1),nn.GELU())
        self.enc0=block(input_channels,width)
        self.enc1=block(width,2*width)
        self.bottom=block(2*width,4*width)
        self.dec1=block(6*width,2*width)
        self.dec0=block(3*width,width)
        if parameterization_policy!='legacy':
            with torch.no_grad():
                for name,module in self.named_modules():
                    if type(module) is nn.Conv2d and name!='enc0.0':
                        module.bias.mul_(math.sqrt(width/base_width))

    def parameterization_configuration(self):
        if self._declaration!=(self.width,self.input_channels,self.parameterization_policy,self.base_width):
            raise ValueError('pixel parameterization declaration drift')
        if self.parameterization_policy=='legacy':return None
        from .parameterization import image_encoder_parameter_roles
        image_encoder_parameter_roles(self,channel_axis=tuple('input-channel:'+str(i) for i in range(self.input_channels)))
        return dict(schema='fixed-kernel-multiscale-pixels-v1',width=self.width,base_width=self.base_width,
            input_channels=self.input_channels,layout='skip2-before-bottom4-skip1-before-decoder2',
            interpolation='bilinear-align-corners-false',pool='max2-ceil-true',
            hidden_weights='native-fan-in-fixed-3x3-kernels',
            hidden_bias='fresh-native-times-sqrt-width-over-base',
            first_fixed_channel_projection='native-fan-in-invariant',hidden_adam_lr='base-over-width')

    def validate_checkpoint(self,state_dict,prefix):
        own=self.parameterization_configuration();key=prefix+'_pixel_parameterization'
        if own is not None or key in state_dict:
            if not exact_wire(own,state_dict.get(key)):
                raise ValueError('pixel encoder checkpoint configuration/policy mismatch')

    def _save_to_state_dict(self,destination,prefix,keep_vars):
        super()._save_to_state_dict(destination,prefix,keep_vars)
        cfg=self.parameterization_configuration()
        if cfg is not None:destination[prefix+'_pixel_parameterization']=copy.deepcopy(cfg)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.validate_checkpoint(state_dict,prefix)
        # A conditional sidecar is not a registered parameter/buffer.
        data=state_dict.copy();data.pop(prefix+'_pixel_parameterization',None)
        super()._load_from_state_dict(data,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def forward(self,image):
        self.parameterization_configuration()
        a=self.enc0(image)
        b=self.enc1(F.max_pool2d(a,2,ceil_mode=True))
        c=self.bottom(F.max_pool2d(b,2,ceil_mode=True))
        d=self.dec1(torch.cat((b,F.interpolate(c,size=b.shape[-2:],mode='bilinear',align_corners=False)),1))
        return self.dec0(torch.cat((a,F.interpolate(d,size=a.shape[-2:],mode='bilinear',align_corners=False)),1))


class MorphologyInstance(MorphologyForeground):
    """Shared image/TokenGT context; outputs foreground, right, down logits.

    Untrained optional architecture. Local affinity BCE is supplied separately;
    no instance decoding, biological pairing or generalization is implied.
    """
    def __init__(self,width=32,*,normalization='unsigned-dtype-max',parameterization_policy='legacy',
                 base_width=None,output_mult=1.0,backbone=None,attn_mult=1.0,encoder_var=1.0,
                 source_envelope=None):
        if parameterization_policy not in ('legacy','mup-image-v1'):
            raise ValueError('known explicit morphology parameterization required')
        if source_envelope is not None:
            if type(source_envelope) is not RasterSourceEnvelope or parameterization_policy != 'mup-image-v1':
                raise ValueError('Explicit opt-in grayscale source envelope required')
            source_envelope.binding()
        self.source_envelope = source_envelope
        if parameterization_policy=='legacy':
            if base_width is not None or output_mult!=1 or backbone is not None or attn_mult!=1 or encoder_var!=1:
                raise ValueError('legacy image owner has no muP declaration/shared constructor')
            super().__init__(width,normalization=normalization)
            self.width=width
            self.pixels=MultiscalePixels(width)
            self.head=nn.Conv2d(2*width,3,1)
            self.parameterization_policy='legacy';self._mup_configuration=None
            return
        nn.Module.__init__(self)
        if (type(width) is not int or width<2 or width%2 or type(base_width) is not int
                or base_width<2 or base_width%2 or width<base_width
                or type(output_mult) not in (int,float) or not math.isfinite(output_mult) or output_mult<=0
                or normalization not in ('unsigned-dtype-max','image-minmax')):
            raise ValueError('valid native image width/normalization/base/readout required')
        if backbone is not None:
            if type(backbone) is not TokenGT:raise ValueError('exact native shared image backbone required')
            cfg=backbone.configuration()
            if (cfg['schema']!='tokengt-native-configuration-v3' or cfg['width']!=width or cfg['feature_width']!=width
                    or cfg['parameterization']['base_width']!=base_width or cfg['attn_mult']!=attn_mult
                    or cfg['encoder_var']!=encoder_var):
                raise ValueError('native shared image backbone policy mismatch')
        self.width,self.normalization=width,normalization
        self.parameterization_policy,self.base_width,self.output_mult=parameterization_policy,base_width,float(output_mult)
        self.pixels=MultiscalePixels(width,parameterization_policy=parameterization_policy,base_width=base_width)
        self.backbone=(backbone if backbone is not None else TokenGT(width,1,width=width,heads=2,layers=1,
            scaling_policy='mup-v1',initialization_policy='mup-transformer-v1',attn_mult=attn_mult,encoder_var=encoder_var,
            parameterization_policy='mup-tokengt-v1',base_width=base_width,base_ffn_width=base_width))
        self.head=nn.Conv2d(2*width,3,1)
        with torch.no_grad():
            self.head.weight.mul_(math.sqrt(width/base_width));self.head.bias.mul_(math.sqrt(width/base_width))
        self._backbone_reference=(self.backbone,)
        self._mup_configuration=self._current_parameterization()

    def _current_parameterization(self):
        config = dict(schema='native-morphology-parameterization-v1',policy=self.parameterization_policy,
            width=self.width,normalization=self.normalization,
            base_width=self.base_width,output_mult=self.output_mult,
            native_backbone=self.backbone.configuration(),pixels=self.pixels.parameterization_configuration(),
            readout='finite-conv1x1-native-rescale-sqrt-width-over-base-input-divided-by-width-over-base')
        if self.source_envelope is not None:
            config['source_envelope'] = self.source_envelope.binding()
        return config

    def get_extra_state(self):
        from .instance_affinity import CONVENTION
        cfg=dict(version=1,architecture='two-level-padded-bilinear-v1',width=self.width,
                    normalization=self.normalization,affinity=CONVENTION,
                    outputs=['foreground','right','down'],attention='vanilla')
        if self._mup_configuration is not None:
            if (set(self._modules)!={'pixels','backbone','head'} or type(self.pixels) is not MultiscalePixels
                    or type(self.backbone) is not TokenGT or type(self.head) is not nn.Conv2d
                    or (self.head.in_channels,self.head.out_channels,self.head.kernel_size,self.head.stride,
                        self.head.padding,self.head.dilation,self.head.groups,self.head.padding_mode,
                        self.head.transposed,self.head.output_padding)
                        !=(2*self.width,3,(1,1),(1,1),(0,0),(1,1),1,'zeros',False,(0,0))):
                raise ValueError('native image readout/module configuration drift')
            if (self.backbone is not self._backbone_reference[0]
                    or not exact_wire(self._current_parameterization(),self._mup_configuration)):
                raise ValueError('native image parameterization/backbone configuration drift')
            cfg.update(version=3 if self.source_envelope is not None else 2,
                       mup=copy.deepcopy(self._mup_configuration))
        return cfg

    def set_extra_state(self,state):
        if state!=self.get_extra_state():
            raise ValueError('Morphology instance resume configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        if self._mup_configuration is not None:
            if not exact_wire(state_dict.get(prefix+'_extra_state'),self.get_extra_state()):
                raise ValueError('native image checkpoint configuration mismatch')
            self.pixels.validate_checkpoint(state_dict,prefix+'pixels.')
            self.backbone._validate_initialization_checkpoint(state_dict,prefix+'backbone.')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def forward(self,task,*,for_training=False):
        return self._logits(task,for_training=for_training)
