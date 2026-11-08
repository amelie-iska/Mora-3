"""Source-reviewed ordinary-response width family; no full Mora3/data admission.

This exact owner has complete learned-width graph inputs, fixed two-endpoint
incidence, and contiguous Q/K/V packed rows. Its opt-in constructor records and
applies native attention initialization, hidden-bias/readout base rescaling, and
readout input division. Those source/operator facts qualify these two custom
dispatches here. Shape coincidence alone never qualifies arbitrary wrappers.

Reference: amelie-iska/mup@19814971934ef91dd546f88e913fc963e096d11c,
layer.py, optim.py and examples/Transformer/model.py (MIT). Required Mora3
cooperative heads, tropical operators and whole-corpus gates remain separate.
"""
import copy

from .parameterization import ParameterRole, parameter_ledger, response_parameter_ledger, width_family
from .sciplex import SciPlexSharedModel, _same_configuration


def sciplex_mup_parameter_roles(model):
    """Return the complete exact native v4 role map, read-only and alias checked."""
    if type(model) is not SciPlexSharedModel:
        raise ValueError('exact source-reviewed SciPlex owner required')
    model.validate_configuration()
    if model.config['version']!=4:
        raise ValueError('explicit applied native/readout muP owner policy required')
    spec=model.config['mup'];native=model.backbone.configuration()
    if (spec['policy']!='mup-sciplex-v1' or native['schema']!='tokengt-native-configuration-v3'
            or native['initialization_policy']!='mup-transformer-v1'
            or native['parameterization']['base_width']!=spec['base_width']
            or native['parameterization']['base_ffn_width']!=spec['base_width']):
        raise ValueError('complete matched native/readout base policy required')
    ledger=response_parameter_ledger(model, source_column_positions=True)
    roles={}
    for entry in ledger.parameters:
        # The exact native constructor has no tied Parameters. In-place ties
        # change its operator ownership and require a separately reviewed policy.
        if len(entry.aliases)!=1:
            raise ValueError('native owner alias topology changed')
        name=entry.aliases[0];role=entry.role
        if role=='custom':
            if name=='backbone.incidence.weight':
                if entry.shape!=(model.config['width'],4) or [a.kind for a in entry.axes]!=['width','finite']:
                    raise ValueError('complete fixed ordered endpoint-incidence inputs required')
                role='input'
            elif name in {f'backbone.layers.{i}.attention.qkv.weight' for i in range(2)}:
                w=model.config['width']
                if entry.shape!=(3*w,w) or [a.kind for a in entry.axes]!=['width','width']:
                    raise ValueError('explicit complete contiguous packedQKV geometry required')
                role='hidden'
            else:
                raise ValueError('unreviewed native custom operator')
        roles[name]=ParameterRole(role,entry.axes)
    parameter_ledger(model,roles)
    return roles


def sciplex_mup_width_family(base,delta,target):
    """Bind complete same-axis/same-depth/policy owners; count every native head.

    Axes/configurations are live-model declarations, not external source rights,
    task admission, complete multimodal coverage or a training-readiness proof.
    """
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,sciplex_mup_parameter_roles(m)) for m in models]
    expected=None
    for model in models:
        config=copy.deepcopy(model.get_extra_state())
        for key in ('width',):config.pop(key)
        for key in ('width','feature_width','ffn_width'):config['native_backbone'].pop(key)
        if expected is None:expected=config
        if not _same_configuration(config,expected):
            raise ValueError('native family finite-axis/depth/base/readout/attention policy drift')
    if base.config['width']!=base.config['mup']['base_width']:
        raise ValueError('actual base owner must equal the declared base width')
    return width_family(*ledgers)
