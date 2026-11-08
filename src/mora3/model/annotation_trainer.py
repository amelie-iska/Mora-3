"""Fixed source-bound annotation batches and full-state native Adam continuation.

Views must originate from read_prediction_views: hashes bind that reader's output,
not a substitute for admission. No generation or heldout eligibility is inferred.
"""
import copy
from dataclasses import dataclass
import hashlib
import json
import torch
from mora3.data.uniprot_annotations import AnnotationPredictionView


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class AnnotationBatch:
    views: tuple
    go_applicable: tuple
    ec_applicable: tuple

    def __post_init__(self):
        for name in ('views','go_applicable','ec_applicable'):
            object.__setattr__(self,name,tuple(getattr(self,name)))
        if not self.views or any(not isinstance(v,AnnotationPredictionView) for v in self.views):
            raise ValueError('nonempty source-validated views required')
        for mask in (self.go_applicable,self.ec_applicable):
            if len(mask)!=len(self.views) or any(type(x) is not bool for x in mask):
                raise ValueError('Boolean applicability per row required')
        for v in self.views:
            p=v.provenance()
            if p.get('split')!='train' or p.get('license')!='CC-BY-4.0':
                raise ValueError('train-only admitted rights required')

    def binding(self):
        return _digest(dict(rows=[dict(accession=v.accession,
            sequence=hashlib.sha256(v.sequence.encode()).hexdigest(),
            target=hashlib.sha256(v.target_bytes).hexdigest(),
            provenance=hashlib.sha256(v.provenance_bytes).hexdigest()) for v in self.views],
            GO=self.go_applicable,EC=self.ec_applicable))

    def applicability(self,device):
        return {k:torch.tensor(v,dtype=torch.bool,device=device)
                for k,v in [('GO',self.go_applicable),('EC',self.ec_applicable)]}


class AnnotationTrainer:
    """One fixed batch/schedule; exclusive checkpoint writing belongs to caller."""
    def __init__(self,model,batch,*,lr=.003,mu_family=None,mu_identifier_axis=None,
                 mu_member='target',weight_decay=0.,decay_aliases=()):
        self.model=model;self.batch=batch;self.steps=0
        config=model.get_extra_state()
        if 'mup' in config:
            if mu_family is None or type(mu_identifier_axis) is not tuple:
                raise ValueError('explicit qualified annotation muP family and identifier axis required')
            from .annotation_mup import annotation_mup_parameter_roles
            from .mu_optimizer import MuAdamW
            self.optimizer=MuAdamW(model,mu_family,
                annotation_mup_parameter_roles(model,identifier_axis=mu_identifier_axis),
                base_lr=lr,weight_decay=weight_decay,decay_aliases=decay_aliases,member=mu_member)
        else:
            if mu_family is not None or mu_identifier_axis is not None or mu_member!='target' or weight_decay!=0 or decay_aliases!=():
                raise ValueError('legacy annotation trainer accepts no muP optimizer policy')
            self.optimizer=torch.optim.Adam(model.parameters(),lr=lr)
        binding=dict(batch=batch.binding(),
            configuration=model.get_extra_state(),payload=model.payload.get_extra_state(),
            GO=model.go.axis_ids,EC=model.ec.axis_ids,
            optimizer='Adam',lr=lr,betas=[.9,.999],eps=1e-8,weight_decay=0,
            device=str(next(model.parameters()).device),dtype=str(next(model.parameters()).dtype))
        if 'mup' in config:
            binding.update(optimizer='MuAdamW',weight_decay=weight_decay,
                           parameterization=self.optimizer.state_dict()['mora3_mu_adamw'])
        self.binding=_digest(binding)

    def step(self):
        self.model.train();self.optimizer.zero_grad(set_to_none=True)
        logits=self.model(self.batch.views)
        losses=self.model.objective(logits,self.batch.views,
            applicable=self.batch.applicability(next(self.model.parameters()).device))
        if not torch.isfinite(losses['total']): raise ValueError('nonfinite annotation loss')
        losses['total'].backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in self.model.parameters()):
            raise ValueError('nonfinite annotation gradient')
        self.optimizer.step();self.steps+=1
        return {k:float(v.detach()) for k,v in losses.items()}

    def state_dict(self):
        return copy.deepcopy(dict(version=1,binding=self.binding,steps=self.steps,
            model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),
            cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()
            if next(self.model.parameters()).is_cuda else None))

    def load_state_dict(self,state):
        if state.get('version')!=1 or state.get('binding')!=self.binding:
            raise ValueError('annotation trainer binding mismatch')
        if type(state.get('steps')) is not int or state['steps']<0:
            raise ValueError('invalid trainer step')
        expected=self.optimizer.state_dict()['param_groups']
        groups=state['optimizer']['param_groups']
        from .mu_optimizer import MuAdamW
        # Exact native MuAdamW prevalidates versioned family/member, common LR
        # ratios and slots in the staged load below. A valid scheduled master LR
        # need not equal a newly constructed optimizer's initial master LR.
        if type(self.optimizer) is not MuAdamW and (len(groups)!=len(expected) or any(
                {k:v for k,v in a.items() if k!='params'} !=
                {k:v for k,v in b.items() if k!='params'}
                for a,b in zip(groups,expected))):
            raise ValueError('annotation optimizer configuration mismatch')
        # Validate model/optimizer together off to the side before changing live state.
        staged_model,staged_optimizer=copy.deepcopy((self.model,self.optimizer))
        staged_model.load_state_dict(state['model'])
        staged_optimizer.load_state_dict(state['optimizer'])
        rng=state['cpu_rng'];probe=torch.Generator();probe.set_state(rng.cpu())
        cuda=state['cuda_rng']
        if next(self.model.parameters()).is_cuda:
            if cuda is None or len(cuda)!=torch.cuda.device_count():
                raise ValueError('CUDA RNG device mismatch')
            for i,value in enumerate(cuda):
                torch.Generator(device=f'cuda:{i}').set_state(value.cpu())
        elif cuda is not None: raise ValueError('unexpected CUDA RNG')
        self.model.load_state_dict(staged_model.state_dict())
        self.optimizer.load_state_dict(staged_optimizer.state_dict());self.steps=state['steps']
        torch.set_rng_state(rng.cpu())
        if cuda is not None: torch.cuda.set_rng_state_all([v.cpu() for v in cuda])
