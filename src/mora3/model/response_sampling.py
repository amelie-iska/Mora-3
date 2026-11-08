"""Target-free observational inference, fixed Euler and explicit output policy.

Sigma describes the trained path's terminal residual, not a denoising correction.
V2 authenticates sigma against its bound model configuration; v1 callers must
pair sigma with the training checkpoint. No heldout, physical-dose or causal
admission is inferred.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import torch
from torch.nn import functional as F
from mora3.data.sciplex_admission import ObservationalSciPlexTask, POLICY
from mora3.data.connected_sciplex import INTERVENTIONS


@dataclass(frozen=True)
class ResponseContext:
    axis_ids: tuple
    source_sha256: str
    audit_sha256: str
    intervention: str
    dose_coordinate: float
    exposure_hours: float

    def __post_init__(self):
        if (not isinstance(self.axis_ids,tuple) or not self.axis_ids
                or any(not isinstance(a,str) or not a.strip() for a in self.axis_ids)
                or self.intervention not in INTERVENTIONS
                or any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                       for v in (self.source_sha256,self.audit_sha256))
                or any(type(v) not in (int,float) or not math.isfinite(v) or v<0
                       for v in (self.dose_coordinate,self.exposure_hours))):
            raise ValueError('source-bound finite observed context required')

    @classmethod
    def from_task(cls,task):
        if not isinstance(task,ObservationalSciPlexTask):raise ValueError('admitted task required')
        s=json.loads(task.task.sidecar_bytes)
        if s['task']!='response':raise ValueError('response context required')
        i=s['conditions']['intervention']
        return cls(tuple(s['targets']['expression']['axis_ids']),task.task.source_sha256,
            json.loads(task.receipt_bytes)['audit_sha256'],i['identity'],
            i['source_coordinate'],s['conditions']['context']['exposure_hours'])

    def input_metadata(self):
        return dict(task='response',policy=POLICY,split='train_only')

    def graph(self,native):
        if self.axis_ids!=native.axis_ids:raise ValueError('gene order mismatch')
        w=native.edge
        index=torch.tensor(INTERVENTIONS.index(self.intervention),device=w.device)
        z=torch.cat((F.one_hot(index,5).to(w.dtype),w.new_tensor([self.dose_coordinate])))
        context=native.context(w.new_tensor([self.exposure_hours/24]))
        return dict(features=torch.stack((context,native.intervention(z),w))+native.direction.weight[0],
            endpoints=torch.tensor([[0,0],[1,1],[0,1]],device=w.device),
            roles=torch.tensor([0,0,1],device=w.device),
            identifiers=torch.eye(2,device=w.device,dtype=w.dtype))


@torch.no_grad()
def sample_response(model,contexts,*,steps,seed,sigma_min,negative_policy,reuse_context=False):
    from .response_flow import ResponseFlowHead
    from .flow_matching import euler_flow
    if type(reuse_context) is not bool:
        raise ValueError("reuse_context must be an explicit bool")
    if reuse_context:
        from .response_flow import ResponseFlowDenoiserV2
        if type(model) not in (ResponseFlowHead, ResponseFlowDenoiserV2):
            raise ValueError("context reuse requires an exact supported native head")
    contexts=tuple(contexts)
    if (not isinstance(model,ResponseFlowHead) or not contexts
            or any(not isinstance(c,ResponseContext) for c in contexts)
            or type(steps) is not int or steps<1 or type(seed) is not int or not 0<=seed<2**63
            or type(sigma_min) not in (int,float) or not math.isfinite(sigma_min) or not 0<=sigma_min<1
            or negative_policy not in ('preserve','clamp-zero')):
        raise ValueError('native model, target-free contexts, solver/seed/path/output policy required')
    model.validate_configuration()
    if model.config.get('version') in (2,4) and model.config['sigma_min'] != sigma_min:
        raise ValueError('denoiser and sampling path sigma mismatch')
    if any(c.axis_ids!=model.native.axis_ids for c in contexts):raise ValueError('gene order mismatch')
    p=model.native.edge
    generator=torch.Generator(device=p.device).manual_seed(seed)
    initial=torch.randn((len(contexts),len(model.native.axis_ids)),device=p.device,dtype=p.dtype,generator=generator)
    modes=[(module,module.training) for module in model.modules()]
    try:
        model.eval()
        # Eval-mode context is invariant across solver steps. Never retain it
        # beyond this call, attach it to a module, or reuse training activations.
        if reuse_context:
            context=model.native.backbone([model.native.graph(c) for c in contexts])[0][:,0]
            field=lambda x,t:model._velocity_from_context(context,x,t)
        else:
            field=lambda x,t:model(contexts,x,t)
        result=euler_flow(field,initial,steps=steps)
    finally:
        for module,mode in modes:module.training=mode
    counts=None if negative_policy=='preserve' else torch.expm1(result.clamp_min(0))
    if counts is not None and not torch.isfinite(counts).all():raise ValueError('count-space overflow')
    config=dict(version=1,contexts=[asdict(c) for c in contexts],model=model.config,
        solver='fixed-euler',steps=steps,seed=seed,sigma_min=sigma_min,
        negative_policy=negative_policy,dtype=str(p.dtype),device=str(p.device))
    binding=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    return dict(log_expression=result,counts=counts,negative_coordinates=int((result<0).sum()),
        binding=binding,configuration=config,terminal_noise_policy='sigma-residual-not-removed')
