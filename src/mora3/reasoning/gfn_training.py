"""Bounded on-policy TB diagnostic with normalized unit Gaussian-mixture reward.

Not a semantic reward or full training system. Checkpoints are exclusive, not
crash-atomic; load only trusted locally produced files. No AMP/scheduler/cursor.
"""
import math
import torch
from .continuous_gfn import _finite, diagonal_gaussian_log_density, trajectory_balance
from .gfn_sampling import sample_paths, score_paths


def mixture_log_reward(terminal, centers, horizon):
    """Uniform unit-covariance mixture, uniform mass over terminal time layers."""
    _finite(terminal);_finite(centers)
    if (type(horizon) is not int or horizon<1 or terminal.ndim!=2 or centers.ndim!=2
        or terminal.shape[1]!=centers.shape[1] or terminal.dtype!=centers.dtype
        or terminal.device!=centers.device):
        raise ValueError('Invalid mixture shape/device/dtype or horizon')
    values=terminal[:,None,:].expand(-1,len(centers),-1)
    means=centers[None,:,:].expand_as(values)
    return torch.logsumexp(diagonal_gaussian_log_density(values,means,torch.zeros_like(values)),1)-math.log(len(centers)*horizon)


class OnPolicyTrainer:
    def __init__(self,policy,centers,*,batch_size=128,lr=.003,seed=19):
        _finite(centers)
        if (centers.ndim!=2 or centers.shape[1]!=policy.dimension
            or centers.device!=policy.log_z.device or centers.dtype!=policy.log_z.dtype
            or type(batch_size) is not int or batch_size<1 or not math.isfinite(lr) or lr<=0):
            raise ValueError('Invalid trainer configuration')
        self.policy=policy;self.centers=centers.detach().clone()
        self.batch_size=batch_size;self.lr=lr;self.step_number=0
        self.optimizer=torch.optim.Adam(policy.parameters(),lr=lr)
        self.generator=torch.Generator(device=centers.device).manual_seed(seed)

    def _loss(self,paths):
        x=paths['states'];terminal=x[torch.arange(len(x),device=x.device),paths['lengths']-1]
        reward=mixture_log_reward(terminal,self.centers,self.policy.horizon).detach()
        return trajectory_balance(self.policy.log_z,**score_paths(self.policy,paths),log_reward=reward,mask=paths['mask'])['loss'],terminal

    def step(self):
        paths=sample_paths(self.policy,self.batch_size,generator=self.generator)
        self.optimizer.zero_grad(set_to_none=True)
        loss,_=self._loss(paths);loss.backward()
        gradients=[p.grad for p in self.policy.parameters() if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise ValueError('Nonfinite or absent policy gradients')
        self.optimizer.step();self.step_number+=1
        for p in self.policy.parameters():_finite(p)
        return float(loss.detach())

    @torch.no_grad()
    def evaluate(self,count,*,seed=31):
        generator=torch.Generator(device=self.centers.device).manual_seed(seed)
        paths=sample_paths(self.policy,count,generator=generator)
        loss,terminal=self._loss(paths)
        return dict(tb_loss=float(loss),mean=terminal.mean(0).tolist(),variance=terminal.var(0,unbiased=False).tolist(),
                    layer_frequencies=(torch.bincount(paths['lengths'],minlength=self.policy.horizon+1)[1:]/count).tolist(),log_z=float(self.policy.log_z))

    def _config(self):
        return dict(dimension=self.policy.dimension,horizon=self.policy.horizon,batch_size=self.batch_size,lr=self.lr,
                    dtype=str(self.centers.dtype),device=str(self.centers.device),reward='unit-mixture-uniform-layer-v1')

    def save(self,path):
        payload=dict(version=1,config=self._config(),centers=self.centers,policy=self.policy.state_dict(),
                     optimizer=self.optimizer.state_dict(),step=self.step_number,generator=self.generator.get_state(),
                     cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(self.centers.device) if self.centers.is_cuda else None)
        with open(path,'xb') as stream:torch.save(payload,stream)

    def load(self,path):
        checkpoint=torch.load(path,map_location=self.centers.device,weights_only=True)
        if (checkpoint['version']!=1 or checkpoint['config']!=self._config()
            or not torch.equal(checkpoint['centers'],self.centers)):
            raise ValueError('Checkpoint configuration or target mismatch')
        self.policy.load_state_dict(checkpoint['policy']);self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_number=checkpoint['step'];self.generator.set_state(checkpoint['generator'].cpu())
        torch.set_rng_state(checkpoint['cpu_rng'].cpu())
        if self.centers.is_cuda:torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu(),self.centers.device)
