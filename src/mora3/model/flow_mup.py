"""Explicit native response-flow μP family; other cooperative heads stay gated."""
import copy

from .parameterization import parameter_ledger, response_head_parameter_roles, width_family
from .response_flow import ResponseFlowHead, ResponseFlowDenoiserV2
from .sciplex import _same_configuration
from .sciplex_mup import sciplex_mup_parameter_roles,sciplex_mup_width_family


def flow_mup_parameter_roles(model):
    if type(model) not in (ResponseFlowHead,ResponseFlowDenoiserV2):
        raise ValueError('exact reviewed response-flow owner required')
    model.validate_configuration()
    if model.config['version'] not in (3,4):
        raise ValueError('explicit applied response-flow muP policy required')
    spec=model.config['mup'];native=model.native.config['mup']
    if (spec['policy']!='mup-response-flow-v1' or spec['base_width']!=native['base_width']
            or spec['output_mult']!=native['output_mult']):
        raise ValueError('matched native/flow readout/base policies required')
    roles={'native.'+name:role for name,role in sciplex_mup_parameter_roles(model.native).items()}
    roles.update(response_head_parameter_roles(model, source_column_positions=True))
    ledger=parameter_ledger(model,roles)
    if any(len(entry.aliases)!=1 for entry in ledger.parameters):
        raise ValueError('unreviewed cross-flow parameter ties')
    return roles


def flow_mup_width_family(base,delta,target):
    models=(base,delta,target)
    ledgers=[parameter_ledger(m,flow_mup_parameter_roles(m)) for m in models]
    sciplex_mup_width_family(*(m.native for m in models))
    expected=None
    for model in models:
        config=copy.deepcopy(model.get_extra_state());config.pop('native')
        if expected is None:expected=config
        if not _same_configuration(config,expected):
            raise ValueError('flow mode/base/readout/sigma policy drift across widths')
    return width_family(*ledgers)
