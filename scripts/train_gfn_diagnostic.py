"""Train-only frozen-real-latent GFN diagnostic; no semantic validation."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mora3.reasoning.gfn_sampling import LayeredGaussianPolicy
from mora3.reasoning.gfn_training import OnPolicyTrainer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--latents',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path)
    p.add_argument('--steps',type=int,default=1000);p.add_argument('--checkpoint-step',type=int,default=500)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args()
    if a.steps<1: p.error('--steps must be positive')
    digest=hashlib.sha256(a.latents.read_bytes()).hexdigest()
    if digest!=a.sha256:p.error('Latent checksum mismatch')
    torch.set_num_threads(2);torch.manual_seed(19);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    raw=torch.load(a.latents,weights_only=True).double()
    centers=((raw-raw.mean(0))/raw.std(0).clamp_min(.1)).to(a.device)
    policy=LayeredGaussianPolicy(raw.shape[1],3,32).to(device=a.device,dtype=torch.float64)
    trainer=OnPolicyTrainer(policy,centers,batch_size=128,lr=.003,seed=19)
    if a.resume:trainer.load(a.resume)
    if trainer.step_number>=a.steps:p.error('Resume already at requested total step')
    a.output.mkdir(parents=True,exist_ok=False)
    initial=trainer.evaluate(4096)
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
    start=time.perf_counter();history=[]
    while trainer.step_number<a.steps:
        loss=trainer.step();history.append(dict(step=trainer.step_number,loss=loss))
        if trainer.step_number==a.checkpoint_step:trainer.save(a.output/f'checkpoint-{trainer.step_number}.pt')
    if a.device=='cuda':torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    final=trainer.evaluate(4096);trainer.save(a.output/'final.pt')
    assert digest==hashlib.sha256(a.latents.read_bytes()).hexdigest()
    target_mean=centers.mean(0);target_variance=1+centers.var(0,unbiased=False)
    report=dict(seed=19,source=str(a.latents),source_sha256=digest,real_records=len(raw),new_source_records=0,
                steps=a.steps,resume=str(a.resume) if a.resume else None,batch_size=128,lr=.003,horizon=3,dtype='float64',
                parameters=sum(x.numel() for x in policy.parameters()),device=a.device,initial=initial,final=final,
                target_mean=target_mean.tolist(),target_variance=target_variance.tolist(),target_layer_frequencies=[1/3]*3,
                mean_rmse=float(((torch.tensor(final['mean'],device=centers.device)-target_mean)**2).mean().sqrt()),
                variance_rmse=float(((torch.tensor(final['variance'],device=centers.device)-target_variance)**2).mean().sqrt()),
                elapsed_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated() if a.device=='cuda' else None,
                reward='Normalized unit-bandwidth mixture over standardized frozen train-only latents, uniform terminal layer',
                semantic_validation=False,full_training_ready=False)
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
