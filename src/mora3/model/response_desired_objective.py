"""Experimental desired-request CFM objective, without trainer/admission promotion.

Requests are supplied separately, never extracted from measured targets. This
boundary checks identity, not whether a caller selected requests independently.
Hypothetical requests paired with observed responses are a wiring diagnostic,
not evidence those responses satisfy the requests or eligible conditional data.
"""
import copy
import math
from dataclasses import asdict
import torch
from .annotation_trainer import _digest
from .response_desired_flow import DesiredResponseFlow
from .response_flow_trainer import ResponseFlowBatch
from .response_sampling import ResponseContext
from .flow_matching import gaussian_path, velocity_loss


def desired_response_cfm_loss(model, batch, *, requests, noise, time, sigma_min):
    """Full observed log1p-count Gaussian CFM (2210.02747v2 Eq22–23).

    Caller owns noise/time sampling and model mode. No optimizer, RNG, state
    restoration, source admission or learned cache is implicit. Target response
    enters only the intentional noisy path and detached velocity supervision.
    The returned binding is configuration/pool/request identity, not a signature
    or a checkpoint; callers must bind model weights and random state separately.
    """
    if (type(model) is not DesiredResponseFlow or type(batch) is not ResponseFlowBatch
            or type(sigma_min) not in (int, float) or not math.isfinite(sigma_min)
            or not 0 < sigma_min < 1
            or sigma_min != model.flow.config['sigma_min']
            or model.config['flow'] != model.flow.config
            or model.config.get('version') != 1
            or model.config.get('architecture') != 'desired-response-denoiser-v1'
            or model.native.axis_ids != batch.axis_ids):
        raise ValueError('exact desired head, observed pool, axis and path configuration required')
    contexts = tuple(ResponseContext.from_task(t) for t in batch.tasks)
    try:
        requests = tuple(requests)
    except TypeError as exc:
        raise ValueError('explicit independent desired requests required') from exc
    # Validate exact source/context/full-axis requests before reading targets.
    model.encode_requests(contexts, requests)
    p = model.native.edge
    if (not isinstance(noise, torch.Tensor) or noise.dtype != p.dtype
            or noise.device != p.device
            or noise.shape != (len(contexts), len(batch.axis_ids))):
        raise ValueError('model-matched full-axis noise required')
    projections = [t.project(device=p.device, dtype=p.dtype, for_training=True)
                   for t in batch.tasks]
    available = torch.stack([v['target_masks']['expression'] for v in projections])
    counts = torch.stack([v['targets']['expression'] for v in projections])
    if not available.all() or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError('complete finite nonnegative observed source counts required')
    target = torch.log1p(counts)
    state, velocity = gaussian_path(noise, target, time, sigma_min=sigma_min)
    prediction = model(contexts, state, time, requests=requests)
    loss = velocity_loss(prediction, velocity, available)
    if loss is None or not torch.isfinite(loss):
        raise ValueError('finite complete-axis desired CFM loss required')
    configuration = dict(version=1, objective='desired-response-gaussian-cfm-v1',
        model=copy.deepcopy(model.get_extra_state()), pool=batch.binding(),
        requests=[asdict(d) for d in requests],
        request_role='externally-supplied-desired-not-observed',
        supervision_role='source-observed-response-only',
        path=dict(reference='2210.02747v2-eq22-23', sigma_min=float(sigma_min),
                  space='log1p-counts', availability='complete-observed-only'),
        dtype=str(p.dtype), device=str(p.device),
        reduction='mean-all-observed-record-coordinates')
    return dict(loss=loss, configuration=configuration, binding=_digest(configuration))
