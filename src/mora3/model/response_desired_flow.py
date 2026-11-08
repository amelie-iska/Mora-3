"""Experimental desired-interval denoiser; no trained quality claim.

Full-axis channel-major request encoding is explicitly dense/lossy. Requested
and available masks distinguish unknown requests, absent requests and zero bounds.
Only target-free ResponseContexts and immutable user requests enter this head.
No sampler/trainer is implicitly upgraded to this new conditioning contract.
"""
import copy
import torch
from torch import nn
from .response_flow import ResponseFlowDenoiserV2
from .response_sampling import ResponseContext
from .response_desired import DesiredExpression, DesiredExpressionOrdinal


class DesiredResponseFlow(nn.Module):
    def __init__(self, native, *, sigma_min=0.01):
        super().__init__()
        self.flow = ResponseFlowDenoiserV2(native, sigma_min=sigma_min)
        self.request = nn.Linear(4 * len(native.axis_ids), native.edge.numel(), bias=False)
        self.config = dict(version=1, architecture='desired-response-denoiser-v1',
            flow=copy.deepcopy(self.flow.config), space='log1p-counts',
            encoding='full-axis-channel-major-requested-available-lower-upper',
            adaptation='additive-dense-lossy-request-hidden-encoding',
            roles='user-requested-not-observed', context='exact-source-audit-axis-intervention-dose-exposure')

    @property
    def native(self):
        return self.flow.native

    def get_extra_state(self):
        return copy.deepcopy(self.config)

    def set_extra_state(self, state):
        if state != self.config:
            raise ValueError('desired response configuration mismatch')

    def encode_requests(self, contexts, requests):
        try:
            contexts, requests = tuple(contexts), tuple(requests)
        except TypeError as exc:
            raise ValueError('explicit contexts and desired requests required') from exc
        if (not contexts or len(contexts) != len(requests)
                or any(type(c) is not ResponseContext for c in contexts)
                or any(type(d) not in (DesiredExpression, DesiredExpressionOrdinal) for d in requests)
                or any(c != d.context or c.axis_ids != self.native.axis_ids
                       for c, d in zip(contexts, requests))):
            raise ValueError('exact source/context/axis-bound desired requests required')
        p = self.native.edge
        encoded = p.new_zeros((len(contexts), 4, len(self.native.axis_ids)))
        for row, d in enumerate(requests):
            coordinates = (d.intervals if type(d) is DesiredExpressionOrdinal else
                tuple((d.context.axis_ids.index(g), g, lo, hi) for g, lo, hi in d.intervals))
            for i, _, lo, hi in coordinates:
                encoded[row, 0, i] = 1
                if lo is not None:
                    encoded[row, 1, i] = 1
                    # Construct on model dtype/device and reject overflow instead
                    # of clamping or silently dropping extreme supplied bounds.
                    encoded[row, 2:, i] = p.new_tensor([lo, hi])
        if not torch.isfinite(encoded).all():
            raise ValueError('desired bounds not representable in model dtype')
        return encoded.flatten(1)

    def forward(self, contexts, state, time, *, requests):
        contexts = tuple(contexts)
        encoded = self.encode_requests(contexts, requests)
        p = self.native.edge
        if (not isinstance(state, torch.Tensor)
                or state.shape != (len(contexts), len(self.native.axis_ids))
                or state.device != p.device or state.dtype != p.dtype
                or not isinstance(time, torch.Tensor) or time.shape != (len(contexts),)
                or time.device != p.device or time.dtype != p.dtype
                or not torch.isfinite(state).all() or not torch.isfinite(time).all()
                or ((time < 0) | (time > 1)).any()):
            raise ValueError('finite full-axis state and batch time in [0,1] required')
        context = self.native.backbone([c.graph(self.native) for c in contexts])[0][:, 0]
        return self.flow._velocity_from_context(context + self.request(encoded), state, time)
