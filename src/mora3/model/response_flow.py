"""Full-axis conditional velocity head; no clean response enters context.

Dense noisy-state projection is explicitly lossy, as in the native response
adapter. It uses every ordered gene; it is not gene-token attention. Path and
source bindings belong to the trainer, not this deterministic head.
"""
import copy
import math
import torch
from torch import nn
from .sciplex import SciPlexSharedModel


class ResponseFlowHead(nn.Module):
    def __init__(self,native,*,parameterization_policy='legacy'):
        super().__init__()
        if not isinstance(native, SciPlexSharedModel):
            raise ValueError('native admitted response model required')
        native.validate_configuration()
        if parameterization_policy not in ('legacy','mup-response-flow-v1'):
            raise ValueError('explicit supported response-flow parameterization policy required')
        expected_version=1 if parameterization_policy=='legacy' else 4
        if type(native.config.get('version')) is not int or native.config['version']!=expected_version:
            raise ValueError('legacy response-flow schema requires native owner version1')
        self.native = native
        width = native.edge.numel()
        self.state = nn.Linear(len(native.axis_ids), width)
        self.time = nn.Linear(1, width)
        self.output = nn.Linear(width, len(native.axis_ids))
        self.config = dict(version=1, architecture='conditional-response-velocity-v1',
            native=copy.deepcopy(native.config), space='log1p-counts',
            state='full-axis-dense-lossy', time='scalar-[0,1]',
            context='observed-intervention-dose-coordinate-exposure-only')
        if parameterization_policy!='legacy':
            spec=native.config['mup']
            self.config.update(version=3,architecture='conditional-response-velocity-mup-v1',
                mup=dict(schema='response-flow-parameterization-v1',policy=parameterization_policy,
                    base_width=spec['base_width'],output_mult=spec['output_mult'],
                    readout='finite-output-MuReadout-fresh-SP-rescale-and-input-width-division'))
            with torch.no_grad():
                factor=math.sqrt(width/spec['base_width'])
                self.output.weight.mul_(factor);self.output.bias.mul_(factor)
        self._owner_configuration = copy.deepcopy(self.config)

    def validate_configuration(self):
        """Bind legacy wrapper metadata to the exact native owner declaration."""
        self.native.validate_configuration()
        expected_version=4 if self._owner_configuration['version'] in (3,4) else 1
        if type(self.native.config.get('version')) is not int or self.native.config['version']!=expected_version:
            raise ValueError('legacy response-flow schema requires native owner version1')
        if (self.config != self._owner_configuration
                or self.config['native'] != self.native.get_extra_state()):
            raise ValueError('response flow configuration mismatch')

    def get_extra_state(self):
        self.validate_configuration()
        return copy.deepcopy(self.config)

    def set_extra_state(self, state):
        self.validate_configuration()
        if state != self.config:
            raise ValueError('response flow configuration mismatch')

    def forward(self, tasks, state, time):
        self.validate_configuration()
        p = self.native.edge
        if (not tasks or any(t.input_metadata()['task'] != 'response' for t in tasks)
                or not isinstance(state, torch.Tensor)
                or state.shape != (len(tasks), len(self.native.axis_ids))
                or state.device != p.device or state.dtype != p.dtype
                or not isinstance(time, torch.Tensor) or time.shape != (len(tasks),)
                or time.device != p.device or time.dtype != p.dtype
                or not torch.isfinite(state).all() or not torch.isfinite(time).all()
                or ((time < 0) | (time > 1)).any()):
            raise ValueError('response tasks, finite full-axis state and batch time in [0,1] required')
        context = self.native.backbone([self.native.graph(t) for t in tasks])[0][:, 0]
        return self._velocity_from_context(context, state, time)

    def _velocity_from_context(self, context, state, time):
        """Internal arithmetic shared by validated forward and call-local sampling."""
        hidden=torch.nn.functional.silu(context+self.state(state)+self.time(time[:,None]))
        if self.config['version'] in (3,4):
            spec=self.config['mup']
            hidden=hidden*(spec['output_mult']/(self.native.edge.numel()/spec['base_width']))
        return self.output(hidden)


class ResponseFlowDenoiserV2(ResponseFlowHead):
    """Experimental Eq21 denoiser parameterization, not a quality promotion.

    The learned denoiser retains the bounded shared-context bottleneck, but the
    explicit full-coordinate contraction removes v1's invariant noise subspace.
    Same velocity supervision as v1; no clean target argument or target detach
    inside the learned denoiser. Positive sigma keeps t=1 nonsingular.
    """
    def __init__(self,native,*,sigma_min=0.01,parameterization_policy='legacy'):
        import math
        if (type(sigma_min) not in (int, float) or not math.isfinite(sigma_min)
                or not 0 < sigma_min < 1):
            raise ValueError('v2 requires finite sigma strictly between zero and one')
        super().__init__(native,parameterization_policy=parameterization_policy)
        self.config.update(version=2 if parameterization_policy=='legacy' else 4,
            architecture='conditional-response-denoiser-v2' if parameterization_policy=='legacy'
                         else 'conditional-response-denoiser-mup-v1',
            state='full-axis-analytic-contraction-plus-dense-lossy-denoiser',
            sigma_min=float(sigma_min), parameterization='eq21-learned-denoiser')
        self._owner_configuration = copy.deepcopy(self.config)

    def _velocity_from_context(self, context, state, time):
        denoised = super()._velocity_from_context(context, state, time)
        rate = 1 - self.config['sigma_min']
        return (denoised - rate * state) / (1 - rate * time[:, None])
