"""Bounded vanilla TokenGT reference (2207.02505v2, sections 2–3).

Inputs are precomputed entity features, NOT flattened ConvexTok sequences.
One graph token plus one token per node/edge. No conditional generation support.
Identifiers are explicit so callers can transport them under relabeling or
save/resume their RNG policy. Dense attention is quadratic; rank never truncated.
"""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F


def orthogonal_identifiers(nodes, width, *, seed=0, dtype=torch.float32, device='cpu'):
    """Square Gaussian QR rows padded to width; local RNG leaves global RNG alone."""
    if not isinstance(nodes, int) or not isinstance(width, int) or not 0 < nodes <= width:
        raise ValueError('Require 0 < nodes <= identifier width')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(nodes, nodes, generator=generator, dtype=torch.float64))
    return F.pad(q.T, (0, width - nodes)).to(device=device, dtype=dtype)


def parameter_count(module):
    """Count all unique parameters, including frozen parameters and heads supplied by caller."""
    return sum(p.numel() for p in module.parameters())


def _same_attention_wire(left,right):
    if type(left) is not type(right):return False
    if type(left) is dict:
        return left.keys()==right.keys() and all(_same_attention_wire(left[k],right[k]) for k in left)
    if type(left) is tuple:
        return len(left)==len(right) and all(_same_attention_wire(a,b) for a,b in zip(left,right))
    return left==right


def _attention_initialization(width,policy,encoder_var):
    """Pure explicit packed-slice recipe, not an inference from tensor shape."""
    if (type(width) is not int or width<=0
            or policy not in ('legacy','mup-transformer-v1')
            or type(encoder_var) not in (int,float) or not math.isfinite(encoder_var)
            or encoder_var<=0 or (policy=='legacy' and encoder_var!=1)):
        raise ValueError('explicit supported attention initialization and positive variance required')
    if policy=='legacy':return None
    return dict(schema='dense-attention-initialization-v1',policy=policy,
        encoder_var=float(encoder_var),width=width,
        slices=(('query',0,width),('key',width,2*width),('value',2*width,3*width)),
        distribution='normal-std-sqrt-encoder-var-over-width',
        query='zero-after-packed-normal-draw',bias='zero',
        reference='amelie-iska/mup@19814971934ef91dd546f88e913fc963e096d11c')


def _native_mup_declaration(width,heads,ffn_width,initialization,scaling,policy,base_width,base_ffn_width):
    """Opt-in fresh hidden-bias application; fixed ratios are declared explicitly."""
    if policy=='legacy':
        if base_width is not None or base_ffn_width is not None:
            raise ValueError('legacy native parameterization accepts no base widths')
        return None
    if (policy!='mup-tokengt-v1' or initialization!='mup-transformer-v1' or scaling!='mup-v1'
            or any(type(n) is not int or n<=0 for n in (width,heads,ffn_width))
            or type(base_width) is not int or type(base_ffn_width) is not int
            or base_width<=0 or base_ffn_width<=0 or base_width%heads
            or width<base_width or ffn_width<base_ffn_width
            or ffn_width*base_width!=base_ffn_width*width):
        raise ValueError('explicit compatible native muP base/hidden widths and initialization required')
    return dict(schema='tokengt-parameterization-v1',policy=policy,
        base_width=base_width,base_ffn_width=base_ffn_width,
        hidden_weights='retain-native-fan-in-kaiming-uniform',
        hidden_bias='fresh-sqrt-input-width-over-base-input-width',
        incidence='complete-fixed-ordered-endpoint-fan-in',
        attention='explicit-packed-qkv-query-zero-normal-v1')


class DenseAttention(nn.Module):
    """Ordinary softmax attention with True=valid keys.

    Legacy arithmetic is unchanged. Explicit ``mup-v1`` scales queries by
    sqrt(attn_mult)/d_head (pinned muP Transformer example). This policy alone
    does not qualify initialization, optimizer scaling or width transfer.
    Config and tensor state must both be retained for reconstruction; old
    tensor-only checkpoints must never be interpreted as the opt-in policy.
    Tropical attention is a separate operator and accepts no ordinary rule.
    """
    def __init__(self, width, heads, *, scaling_policy='legacy', attn_mult=1.0,
                 initialization_policy='legacy',encoder_var=1.0):
        super().__init__()
        if (type(width) is not int or type(heads) is not int
                or width <= 0 or heads <= 0 or width % heads):
            raise ValueError('Positive width must be divisible by heads')
        if scaling_policy not in ('legacy', 'mup-v1'):
            raise ValueError('Unknown ordinary attention scaling policy')
        if (type(attn_mult) not in (int, float) or not math.isfinite(attn_mult)
                or attn_mult <= 0 or (scaling_policy == 'legacy' and attn_mult != 1)):
            raise ValueError('Positive finite attn_mult required; legacy requires one')
        recipe=_attention_initialization(width,initialization_policy,encoder_var)
        if recipe is not None and scaling_policy!='mup-v1':
            raise ValueError('muP attention initialization requires explicit mup-v1 scaling')
        self.scaling_policy, self.attn_mult = scaling_policy, float(attn_mult)
        self.initialization_policy,self.encoder_var=initialization_policy,float(encoder_var)
        self._initialization_declaration=(initialization_policy,float(encoder_var))
        self.heads, self.head_width = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        if recipe is not None:
            # Fresh construction ONLY. Pinned Transformer/model.py291–309.
            # Draw all packed rows before zeroing Q, preserving reference RNG order.
            with torch.no_grad():
                nn.init.normal_(self.qkv.weight,mean=0.,std=math.sqrt(self.encoder_var/width))
                self.qkv.weight[:width].zero_()
                nn.init.normal_(self.output.weight,mean=0.,std=math.sqrt(self.encoder_var/width))
                self.qkv.bias.zero_();self.output.bias.zero_()

    def _initialization_configuration(self):
        """Bind the recipe, never assert that trained tensors still have initial values."""
        if ((self.initialization_policy,self.encoder_var)!=self._initialization_declaration
                or type(self.encoder_var) is not float):
            raise ValueError('attention initialization declaration drift')
        width=self.heads*self.head_width
        recipe=_attention_initialization(width,self.initialization_policy,self.encoder_var)
        if recipe is not None and self.scaling_policy!='mup-v1':
            raise ValueError('attention initialization/scaling drift')
        return recipe

    def scaling_config(self):
        """Explicit standalone configuration; tensor state has legacy keys."""
        config=dict(schema='dense-attention-scaling-v1', width=self.heads * self.head_width,
                    heads=self.heads, scaling_policy=self.scaling_policy,
                    attn_mult=self.attn_mult)
        recipe=self._initialization_configuration()
        if recipe is not None:config.update(schema='dense-attention-scaling-v2',initialization=recipe)
        return config

    @classmethod
    def from_scaling_config(cls, config):
        """Reconstruct only an exact versioned ordinary-attention declaration."""
        keys={'schema','width','heads','scaling_policy','attn_mult'}
        if type(config) is not dict or config.get('schema') not in (
                'dense-attention-scaling-v1','dense-attention-scaling-v2'):
            raise ValueError('Exact dense attention scaling configuration required')
        version2=config['schema']=='dense-attention-scaling-v2'
        if set(config)!=(keys|{'initialization'} if version2 else keys):
            raise ValueError('Exact dense attention scaling configuration required')
        policy,var='legacy',1.
        if version2:
            init=config['initialization']
            if type(init) is not dict or init.get('policy')!='mup-transformer-v1':
                raise ValueError('Exact opt-in initialization recipe required')
            policy,var=init['policy'],init.get('encoder_var')
            if not _same_attention_wire(init,_attention_initialization(config['width'],policy,var)):
                raise ValueError('attention initialization recipe mismatch')
        return cls(config['width'], config['heads'],
                   scaling_policy=config['scaling_policy'], attn_mult=config['attn_mult'],
                   initialization_policy=policy,encoder_var=var)

    def _save_to_state_dict(self,destination,prefix,keep_vars):
        super()._save_to_state_dict(destination,prefix,keep_vars)
        recipe=self._initialization_configuration()
        if recipe is not None:
            destination[prefix+'_attention_initialization']=copy.deepcopy(recipe)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        """Refuse recipe drift before this module's child tensors can be copied.

        No reinitialization/rescaling is available on load. Root model/optimizer/
        RNG transactionality still belongs to its separately staged trainer.
        Legacy states retain exactly their old tensor keys and interpretation.
        """
        recipe=self._initialization_configuration();key=prefix+'_attention_initialization'
        if (key in state_dict)!=(recipe is not None):
            raise ValueError('attention initialization checkpoint policy mismatch')
        if recipe is not None:
            if not _same_attention_wire(state_dict[key],recipe):
                raise ValueError('attention initialization checkpoint recipe mismatch')
            state_dict.pop(key)
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def forward(self, x, valid):
        self._initialization_configuration()
        if valid.dtype != torch.bool or valid.shape != x.shape[:2] or not valid.any(dim=1).all():
            raise ValueError('Each attention row needs at least one valid key')
        b, t, width = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.heads, self.head_width).unbind(2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        if self.scaling_policy == 'legacy':
            score = (q @ k.transpose(-1, -2)) / self.head_width ** 0.5
        else:
            score = (q * (math.sqrt(self.attn_mult) / self.head_width)) @ k.transpose(-1, -2)
        weight = score.masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
        return self.output((weight @ v).transpose(1, 2).reshape(b, t, width))


def _validate_native_policy(width, heads, ffn_width, attention, block, policy, mult,
                            initialization_policy='legacy',encoder_var=1.0):
    """Reject incompatible declarations before any random initialization."""
    if (any(type(v) is not int or v <= 0 for v in (width, heads, ffn_width, block))
            or width % heads or attention not in ('vanilla', 'tropical-code', 'tropical-blockwise')
            or policy not in ('legacy', 'mup-v1')
            or type(mult) not in (int, float) or not math.isfinite(mult) or mult <= 0
            or (policy == 'legacy' and mult != 1)
            or (attention != 'vanilla' and (policy != 'legacy' or mult != 1))):
        raise ValueError('Invalid native attention policy/dimensions')
    recipe=_attention_initialization(width,initialization_policy,encoder_var)
    if recipe is not None and (attention!='vanilla' or policy!='mup-v1'):
        raise ValueError('explicit ordinary mup-v1 attention required for new initialization')


class EncoderLayer(nn.Module):
    """Pre-norm residual attention and GELU FFN, dropout disabled for reference."""
    def __init__(self, width, heads, ffn_width, attention='vanilla', tropical_block_size=32, *,
                 scaling_policy='legacy',attn_mult=1.0,initialization_policy='legacy',encoder_var=1.0):
        super().__init__()
        _validate_native_policy(width,heads,ffn_width,attention,tropical_block_size,scaling_policy,
                                attn_mult,initialization_policy,encoder_var)
        self._attention_declaration = (width, heads, ffn_width, attention, tropical_block_size, scaling_policy, float(attn_mult))
        if initialization_policy!='legacy':
            self._attention_declaration+=(initialization_policy,float(encoder_var))
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        if attention == 'vanilla':
            self.attention = DenseAttention(width,heads,scaling_policy=scaling_policy,attn_mult=attn_mult,
                initialization_policy=initialization_policy,encoder_var=encoder_var)
        elif attention in ('tropical-code', 'tropical-blockwise'):
            from .tropical import TropicalCodeAttention
            self.attention = TropicalCodeAttention(width, heads, block_size=
                tropical_block_size if attention == 'tropical-blockwise' else None)
        else:
            raise ValueError('Unknown attention variant')
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(), nn.Linear(ffn_width, width))

    def validate_attention_configuration(self):
        """Bind current operator to the explicit native constructor declaration."""
        if type(self._attention_declaration) is not tuple or len(self._attention_declaration) not in (7,9):
            raise ValueError('exact native attention declaration required')
        w,h,ff,variant,block,policy,mult=self._attention_declaration[:7]
        init,var=(self._attention_declaration[7:] if len(self._attention_declaration)==9 else ('legacy',1.))
        _validate_native_policy(w,h,ff,variant,block,policy,mult,init,var)
        if len(self._attention_declaration)==9 and init=='legacy':
            raise ValueError('new declaration requires explicit new initialization')
        a = self.attention
        if variant == 'vanilla':
            expected = dict(schema='dense-attention-scaling-v1', width=w, heads=h,
                            scaling_policy=policy, attn_mult=mult)
            if init!='legacy':
                expected.update(schema='dense-attention-scaling-v2',
                                initialization=_attention_initialization(w,init,var))
            valid = (type(a) is DenseAttention and type(a.heads) is int
                     and type(a.head_width) is int and type(a.attn_mult) in (int, float)
                     and _same_attention_wire(a.scaling_config(),expected))
        else:
            from .tropical import TropicalCodeAttention
            valid = (type(a) is TropicalCodeAttention and a.heads == h
                     and a.head_width == w // h
                     and a.block_size == (block if variant == 'tropical-blockwise' else None))
        if (not valid or self.norm1.normalized_shape != (w,)
                or self.norm2.normalized_shape != (w,)
                or self.ffn[0].in_features != w or self.ffn[0].out_features != ff
                or self.ffn[2].in_features != ff or self.ffn[2].out_features != w):
            raise ValueError('Native attention configuration drift')
        return self._attention_declaration

    def forward(self, x, valid):
        x = x + self.attention(self.norm1(x), valid)
        return x + self.ffn(self.norm2(x))


class TokenGT(nn.Module):
    """Precomputed feature boundary, strict exact-ID and explicit-role contracts.

    Each graph dict contains floating features [entities,feature_width], long
    endpoints [entities,2], long roles [entities] (0 node,1 edge), and floating
    identifiers [nodes,id_width]. Node records may occur anywhere but must cover
    each node exactly once; edge records include self-loops. Ordered endpoint
    pairs encode directed incidence. Undirected adapters must supply both orders.
    Padding is excluded as keys at every layer and zeroed at the output. Graph
    readout is output[:,0]. Conditional masks fail closed until a leakage-safe
    objective/special-token policy is implemented.
    """
    def __init__(self, feature_width, id_width, *, width=128, heads=8, layers=2, ffn_width=None,
                 attention='vanilla',tropical_block_size=32,scaling_policy='legacy',attn_mult=1.0,
                 initialization_policy='legacy',encoder_var=1.0,
                 parameterization_policy='legacy',base_width=None,base_ffn_width=None):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in (feature_width, id_width, width, heads, layers)):
            raise ValueError('Positive integer native dimensions required')
        _validate_native_policy(width,heads,width if ffn_width is None else ffn_width,attention,
            tropical_block_size,scaling_policy,attn_mult,initialization_policy,encoder_var)
        ff=width if ffn_width is None else ffn_width
        mup=_native_mup_declaration(width,heads,ff,initialization_policy,scaling_policy,
                                   parameterization_policy,base_width,base_ffn_width)
        self._native_mup=(parameterization_policy,base_width,base_ffn_width)
        self._native_configuration = (feature_width, id_width, width, heads, layers, width if ffn_width is None else ffn_width, attention, tropical_block_size, scaling_policy, float(attn_mult))
        if initialization_policy!='legacy':
            self._native_configuration+=(initialization_policy,float(encoder_var))
        if min(feature_width, id_width, width, heads, layers) <= 0 or (ffn_width is not None and ffn_width <= 0):
            raise ValueError('Model dimensions must be positive')
        self.feature = nn.Linear(feature_width, width, bias=False)
        self.incidence = nn.Linear(2 * id_width, width, bias=False)
        self.role = nn.Embedding(2, width)
        self.graph_token = nn.Parameter(torch.zeros(width))
        nn.init.normal_(self.graph_token, std=0.02)
        self.layers = nn.ModuleList([EncoderLayer(width,heads,ffn_width or width,attention,
            tropical_block_size,scaling_policy=scaling_policy,attn_mult=attn_mult,
            initialization_policy=initialization_policy,encoder_var=encoder_var) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(width)
        if mup is not None:
            # Pinned mup/layer.py:rescale_linear_bias, applied once to fresh tensors.
            with torch.no_grad():
                for layer in self.layers:
                    layer.ffn[0].bias.mul_(math.sqrt(width/base_width))
                    layer.ffn[2].bias.mul_(math.sqrt(ff/base_ffn_width))

    def validate_attention_configuration(self):
        """Check each exact layer's policy, allowing separately owned compositions.

        Shared-wrapper root agreement remains its owner's responsibility; this
        method deliberately does not certify the root reconstruction declaration.
        """
        if not self.layers or any(type(layer) is not EncoderLayer for layer in self.layers):
            raise ValueError('Exact native layers required')
        return tuple(layer.validate_attention_configuration() for layer in self.layers)

    def configuration(self):
        """Explicit versioned native config, required alongside strict tensor state.

        No inference from tensor shapes or migration of historical checkpoints.
        This declaration does not qualify external wrapper checkpoint schemas.
        """
        keys = ('feature_width', 'id_width', 'width', 'heads', 'layers', 'ffn_width',
                'attention', 'tropical_block_size', 'scaling_policy', 'attn_mult')
        if type(self._native_configuration) is not tuple or len(self._native_configuration) not in (10,12):
            raise ValueError('exact native configuration declaration required')
        f,r,w,h,n,ff,variant,block,policy,mult=self._native_configuration[:10]
        init,var=(self._native_configuration[10:] if len(self._native_configuration)==12 else ('legacy',1.))
        _validate_native_policy(w,h,ff,variant,block,policy,mult,init,var)
        if len(self._native_configuration)==12 and init=='legacy':
            raise ValueError('new native configuration requires new initialization')
        if (self.feature.in_features != f or self.feature.out_features != w
                or self.incidence.in_features != 2*r or self.incidence.out_features != w
                or len(self.layers) != n):
            raise ValueError('Native root configuration drift')
        expected = (w, h, ff, variant, block, policy, mult)
        if init!='legacy':expected+=(init,var)
        if any(type(layer) is not EncoderLayer or
               layer.validate_attention_configuration() != expected for layer in self.layers):
            raise ValueError('Native layer configuration drift')
        result=dict(schema='tokengt-native-configuration-v1',**dict(zip(keys,self._native_configuration[:10])))
        if init!='legacy':result.update(schema='tokengt-native-configuration-v2',
                                       initialization_policy=init,encoder_var=var)
        mup=_native_mup_declaration(w,h,ff,init,policy,*self._native_mup)
        if mup is not None:result.update(schema='tokengt-native-configuration-v3',parameterization=mup)
        return result

    @classmethod
    def from_configuration(cls, config):
        required = {'schema', 'feature_width', 'id_width', 'width', 'heads', 'layers',
                    'ffn_width', 'attention', 'tropical_block_size', 'scaling_policy', 'attn_mult'}
        if type(config) is not dict or config.get('schema') not in (
                'tokengt-native-configuration-v1','tokengt-native-configuration-v2','tokengt-native-configuration-v3'):
            raise ValueError('Exact native TokenGT configuration required')
        version2=config['schema']!='tokengt-native-configuration-v1'
        version3=config['schema']=='tokengt-native-configuration-v3'
        wanted=required|{'initialization_policy','encoder_var'} if version2 else required
        if version3:wanted|={'parameterization'}
        if (set(config)!=wanted
                or (version2 and config['initialization_policy']!='mup-transformer-v1')):
            raise ValueError('Exact versioned native initialization configuration required')
        kwargs={k:v for k,v in config.items() if k not in ('schema','parameterization')}
        if version3:
            m=config['parameterization']
            if type(m) is not dict:
                raise ValueError('exact native parameterization configuration required')
            expected=_native_mup_declaration(config['width'],config['heads'],config['ffn_width'],
                config['initialization_policy'],config['scaling_policy'],m.get('policy'),
                m.get('base_width'),m.get('base_ffn_width'))
            if not _same_attention_wire(m,expected):
                raise ValueError('native parameterization configuration mismatch')
            kwargs.update(parameterization_policy=m['policy'],base_width=m['base_width'],
                          base_ffn_width=m['base_ffn_width'])
        return cls(**kwargs)

    def _validate_initialization_checkpoint(self,state_dict,prefix):
        """Pure root and all-layer recipe preflight before any root tensor copy."""
        enabled=len(self._native_configuration)==12
        key=prefix+'_native_initialization'
        if (key in state_dict)!=enabled:
            raise ValueError('native initialization checkpoint policy mismatch')
        if enabled:
            if not _same_attention_wire(state_dict[key],self.configuration()):
                raise ValueError('native initialization checkpoint configuration mismatch')
        for index,layer in enumerate(self.layers):
            name=prefix+f'layers.{index}.attention._attention_initialization'
            recipe=(layer.attention._initialization_configuration()
                    if type(layer.attention) is DenseAttention else None)
            if ((name in state_dict)!=(recipe is not None)
                    or (recipe is not None and not _same_attention_wire(state_dict[name],recipe))):
                raise ValueError('native child initialization checkpoint mismatch')
        return enabled

    def _save_to_state_dict(self,destination,prefix,keep_vars):
        super()._save_to_state_dict(destination,prefix,keep_vars)
        if len(self._native_configuration)==12:
            destination[prefix+'_native_initialization']=copy.deepcopy(self.configuration())

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,
                              missing_keys,unexpected_keys,error_msgs):
        if self._validate_initialization_checkpoint(state_dict,prefix):
            state_dict.pop(prefix+'_native_initialization')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,
                                     missing_keys,unexpected_keys,error_msgs)

    def embed(self, graph):
        x, e, r, p = (graph[k] for k in ('features', 'endpoints', 'roles', 'identifiers'))
        if (x.ndim != 2 or x.shape[1] != self.feature.in_features or x.shape[0] == 0
                or p.ndim != 2 or p.shape[1] * 2 != self.incidence.in_features
                or not 0 < p.shape[0] <= p.shape[1]
                or e.shape != (x.shape[0], 2) or r.shape != (x.shape[0],)
                or e.dtype != torch.long or r.dtype != torch.long
                or x.dtype != self.feature.weight.dtype or p.dtype != x.dtype
                or any(t.device != self.feature.weight.device for t in (x, e, r, p))):
            raise ValueError('Invalid graph tensor dimensions, dtype or device')
        if (not torch.isfinite(x).all() or not torch.isfinite(p).all()
                or (e < 0).any() or (e >= p.shape[0]).any() or ((r != 0) & (r != 1)).any()):
            raise ValueError('Invalid features, endpoints or roles')
        node_edges = e[r == 0]
        if (node_edges.shape[0] != p.shape[0] or not torch.equal(node_edges[:, 0], node_edges[:, 1])
                or not torch.equal(node_edges[:, 0].sort().values, torch.arange(p.shape[0], device=p.device))):
            raise ValueError('Nodes must cover each identifier exactly once with equal endpoints')
        # Strict reference validation intentionally synchronizes and costs O(n²d).
        if not torch.allclose(p @ p.T, torch.eye(p.shape[0], device=p.device, dtype=p.dtype), atol=1e-5, rtol=1e-5):
            raise ValueError('Identifiers must be orthonormal')
        return self.feature(x) + self.incidence(p[e].flatten(1)) + self.role(r)

    def forward(self, graphs, *, attention_mask=None):
        if attention_mask is not None:
            raise ValueError('conditional attention is not implemented')
        if not graphs:
            raise ValueError('At least one nonempty graph required')
        rows = [torch.cat((self.graph_token[None], self.embed(g)), dim=0) for g in graphs]
        x = nn.utils.rnn.pad_sequence(rows, batch_first=True)
        lengths = torch.tensor([len(row) for row in rows], device=x.device)
        valid = torch.arange(x.shape[1], device=x.device)[None] < lengths[:, None]
        for layer in self.layers:
            x = layer(x, valid)
        return self.final_norm(x).masked_fill(~valid[:, :, None], 0), valid
