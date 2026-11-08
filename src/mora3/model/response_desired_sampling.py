"""Explicit untrained desired-response fixed-Euler inference.

Hashes identify bytes/configuration, not trusted signatures or source admission.
User intervals are conditions, never observed response supervision. Sigma residual
is retained; no trained satisfaction or causal/therapeutic claim is implied.
"""
import copy
from dataclasses import asdict
import hashlib
import json
import math
import torch
from .response_desired_flow import DesiredResponseFlow
from .flow_matching import euler_flow


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _state_digest(model):
    state = []
    for key, value in sorted(model.state_dict().items()):
        if isinstance(value, torch.Tensor):
            if not torch.isfinite(value).all():
                raise ValueError('finite model state required')
            raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            state.append((key, str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest()))
        else:
            state.append((key, value))
    return _digest(state)


@torch.no_grad()
def sample_desired_response(model, contexts, *, requests, steps, seed, sigma_min, negative_policy, reuse_context=False):
    """Sample only exact desired-v1 heads; restore each module's training mode.

    A private device generator leaves caller CPU/CUDA RNG unchanged. All graph
    and request encodings default to recomputation each step. Explicit reuse is
    eval-only and call-local; no learned cache is retained.
    Model/config/prediction provenance is detached from mutable caller objects.
    """
    if type(reuse_context) is not bool:
        raise ValueError("reuse_context must be an explicit bool")
    if (type(model) is not DesiredResponseFlow or type(steps) is not int or steps < 1
            or type(seed) is not int or not 0 <= seed < 2**63
            or type(sigma_min) not in (int, float) or not math.isfinite(sigma_min)
            or not 0 < sigma_min < 1 or negative_policy not in ('preserve', 'clamp-zero')):
        raise ValueError('exact desired head and explicit solver/seed/path/output policy required')
    flow = dict(version=2, architecture='conditional-response-denoiser-v2',
        native=copy.deepcopy(model.native.config), space='log1p-counts',
        state='full-axis-analytic-contraction-plus-dense-lossy-denoiser',
        time='scalar-[0,1]', context='observed-intervention-dose-coordinate-exposure-only',
        sigma_min=float(sigma_min), parameterization='eq21-learned-denoiser')
    expected = dict(version=1, architecture='desired-response-denoiser-v1', flow=flow,
        space='log1p-counts', encoding='full-axis-channel-major-requested-available-lower-upper',
        adaptation='additive-dense-lossy-request-hidden-encoding',
        roles='user-requested-not-observed', context='exact-source-audit-axis-intervention-dose-exposure')
    if model.config != expected or model.flow.config != flow:
        raise ValueError('desired head version/configuration/sigma mismatch')
    try:
        contexts, requests = tuple(contexts), tuple(requests)
    except TypeError as exc:
        raise ValueError('explicit context/request sequences required') from exc
    model.encode_requests(contexts, requests)  # target-free exact bindings before sampling
    model_sha = _state_digest(model)
    p = model.native.edge
    generator = torch.Generator(device=p.device).manual_seed(seed)
    initial = torch.randn((len(contexts), len(model.native.axis_ids)),
        device=p.device, dtype=p.dtype, generator=generator)
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if reuse_context:
            # Preserve forward's addition order and eval semantics exactly. The
            # context/request are invariant across Euler steps for this call.
            encoded = model.encode_requests(contexts, requests)
            context = model.native.backbone([c.graph(model.native) for c in contexts])[0][:, 0]
            conditioned = context + model.request(encoded)
            field = lambda x, t: model.flow._velocity_from_context(conditioned, x, t)
        else:
            field = lambda x, t: model(contexts, x, t, requests=requests)
        result = euler_flow(field, initial, steps=steps)
    finally:
        for module, mode in modes:
            module.training = mode
    counts = None if negative_policy == 'preserve' else torch.expm1(result.clamp_min(0))
    if counts is not None and not torch.isfinite(counts).all():
        raise ValueError('count-space overflow')
    prediction_sha = hashlib.sha256(result.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()).hexdigest()
    config = dict(version=1, sampler='desired-response-fixed-euler-v1',
        model=copy.deepcopy(model.config), model_state_sha256=model_sha,
        contexts=[asdict(c) for c in contexts], requests=[asdict(d) for d in requests],
        roles='user-requested-not-observed', solver='fixed-euler', steps=steps, seed=seed,
        sigma_min=sigma_min, negative_policy=negative_policy, dtype=str(p.dtype), device=str(p.device))
    return dict(log_expression=result, counts=counts, negative_coordinates=int((result < 0).sum()),
        configuration=config, binding=_digest(config), model_state_sha256=model_sha,
        prediction_sha256=prediction_sha, prediction_hash_encoding='cpu-float64-contiguous-bytes',
        terminal_noise_policy='sigma-residual-not-removed', training_admission=False,
        full_dataset_ready=False)


@torch.no_grad()
def assess_desired_sample(sample, requests):
    """Authenticate desired-sampler identities before read-only interval assessment.

    This validates internal consistency, not external source/weight authenticity.
    No model or measured targets are consumed; caller tensors remain untouched.
    """
    from .response_desired import (DesiredExpression, DesiredExpressionOrdinal,
                                  assess_desired_expression)
    try:
        requests = tuple(requests)
        if not requests or any(type(d) not in (DesiredExpression, DesiredExpressionOrdinal)
                               for d in requests):
            raise ValueError('explicit immutable desired requests required')
        cfg = sample['configuration']
        model = cfg['model']
        flow = model['flow']
        prediction = sample['log_expression']
        if (sample['binding'] != _digest(cfg)
                or cfg['version'] != 1 or cfg['sampler'] != 'desired-response-fixed-euler-v1'
                or model['version'] != 1 or model['architecture'] != 'desired-response-denoiser-v1'
                or model['space'] != 'log1p-counts'
                or model['encoding'] != 'full-axis-channel-major-requested-available-lower-upper'
                or model['adaptation'] != 'additive-dense-lossy-request-hidden-encoding'
                or model['roles'] != 'user-requested-not-observed'
                or model['context'] != 'exact-source-audit-axis-intervention-dose-exposure'
                or flow['version'] != 2 or flow['architecture'] != 'conditional-response-denoiser-v2'
                or flow['space'] != 'log1p-counts'
                or flow['state'] != 'full-axis-analytic-contraction-plus-dense-lossy-denoiser'
                or flow['time'] != 'scalar-[0,1]'
                or flow['context'] != 'observed-intervention-dose-coordinate-exposure-only'
                or flow['parameterization'] != 'eq21-learned-denoiser'
                or type(cfg['steps']) is not int or cfg['steps'] < 1
                or type(cfg['seed']) is not int or not 0 <= cfg['seed'] < 2**63
                or type(cfg['sigma_min']) not in (int, float)
                or not math.isfinite(cfg['sigma_min']) or not 0 < cfg['sigma_min'] < 1
                or flow['sigma_min'] != cfg['sigma_min']
                or cfg['roles'] != 'user-requested-not-observed'
                or _digest(cfg['requests']) != _digest([asdict(d) for d in requests])
                or sample['model_state_sha256'] != cfg['model_state_sha256']
                or len(sample['model_state_sha256']) != 64
                or any(c not in '0123456789abcdef' for c in sample['model_state_sha256'])
                or sample['prediction_hash_encoding'] != 'cpu-float64-contiguous-bytes'
                or not isinstance(prediction, torch.Tensor)):
            raise ValueError('desired sample schema/request/state binding mismatch')
        raw = prediction.detach().cpu().to(torch.float64).contiguous().numpy().tobytes()
        if hashlib.sha256(raw).hexdigest() != sample['prediction_sha256']:
            raise ValueError('prediction digest mismatch')
        # Explicit local translation into the legacy mathematical assessor. Never
        # alter its interface or expose the translated binding as source evidence.
        translated = copy.deepcopy(cfg)
        translated['model'] = copy.deepcopy(flow)
        proxy = dict(configuration=translated, binding=_digest(translated), log_expression=prediction)
        report = assess_desired_expression(proxy, requests)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('bound desired sample required') from exc
    report['sample_binding'] = sample['binding']
    report['sampler'] = cfg['sampler']
    return report
