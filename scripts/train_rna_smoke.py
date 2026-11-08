#!/usr/bin/env python
"""Bounded RNA train-slice classification smoke; no held-out or full-model claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from mora3.data.schema import GraphRecord
from mora3.tokenization.shortest_path import ByteTokenizer
from mora3.model.training import RNAClassifier, classification_loss, save_checkpoint, load_checkpoint
from mora3.model.tokengt import parameter_count


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','vocabulary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--steps',type=int,default=200)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--pooling',choices=('positional','bag'),default='positional')
    parser.add_argument('--attention',choices=('vanilla','tropical-code','tropical-blockwise'),default='vanilla')
    parser.add_argument('--tropical-block-size',type=int,default=32)
    args=parser.parse_args()
    if args.tropical_block_size <= 0: raise ValueError('Block size must be positive')
    if args.output.exists(): raise ValueError('Output already exists')
    if not 2<=args.steps<=1000: raise ValueError('Bounded smoke requires 2..1000 steps')
    if shutil.disk_usage(args.manifest.parent).free<200*2**30+20*2**20:
        raise ValueError('Insufficient disk reserve')
    paths=(args.manifest,args.manifest.parent/'records.jsonl',args.vocabulary)
    if any(p.stat().st_size>10*2**20 for p in paths): raise ValueError('Input too large')
    snapshots={p:p.read_bytes() for p in paths}
    sha=lambda raw:hashlib.sha256(raw).hexdigest()
    manifest=json.loads(snapshots[paths[0]])
    if sha(snapshots[paths[1]])!=manifest['checksum_files']['records.jsonl']:
        raise ValueError('Source checksum mismatch')
    records=[GraphRecord.from_bytes(l) for l in snapshots[paths[1]].splitlines()]
    if not 1<=len(records)<=32 or len({r.record_id for r in records})!=len(records):
        raise ValueError('Require 1..32 unique records')
    if any(manifest['split_assignments'].get(r.record_id)!='train' for r in records):
        raise ValueError('Require explicit train assignments')
    v=json.loads(snapshots[paths[2]])
    tokenizer=ByteTokenizer(tuple(bytes.fromhex(t) for t in v['learned_hex']),controls=v['controls'])
    if tokenizer.size!=v['size']: raise ValueError('Vocabulary size mismatch')
    classes=tuple(sorted({m.value for r in records for n in r.nodes for m in n.measurements
                          if m.name=='rna_type' and m.available and isinstance(m.value,str)}))
    torch.set_num_threads(2); torch.manual_seed(19); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    model=RNAClassifier(tokenizer,classes,pooling=args.pooling,attention=args.attention,tropical_block_size=args.tropical_block_size).to(args.device)
    targets=model.targets(records)
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    binding=dict(sources={str(p):sha(b) for p,b in snapshots.items()},classes=classes,
                 pooling=args.pooling,seed=19,lr=.003,feature_width=16,width=32,heads=4,
                 layers=2,id_width=4,device=args.device,torch_version=str(torch.__version__))
    # Preserve existing vanilla checkpoint bindings; tropical is explicitly distinct.
    if args.attention!='vanilla': binding['attention']=args.attention
    if args.attention=='tropical-blockwise': binding['tropical_block_size']=args.tropical_block_size
    start_step=load_checkpoint(args.resume,model,optimizer,binding=binding) if args.resume else 0
    if not 0<=start_step<args.steps: raise ValueError('Checkpoint step outside run')
    # Validate payload bounds and targets before creating output artifacts.
    with torch.no_grad(): initial_logits=model(records)
    initial_loss=float(classification_loss(initial_logits,targets))
    args.output.mkdir(parents=True,exist_ok=False)
    if args.device=='cuda': torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    started=time.perf_counter(); history=[]
    for step in range(start_step+1,args.steps+1):
        optimizer.zero_grad(set_to_none=True)
        logits=model(records); loss=classification_loss(logits,targets)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError('Nonfinite or absent gradient')
        optimizer.step()
        history.append({'step':step,'train_loss':float(loss.detach())})
        if step==args.steps//2:
            save_checkpoint(args.output/f'checkpoint-{step}.pt',model,optimizer,step=step,binding=binding)
    with torch.no_grad(): final_logits=model(records)
    if args.device=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    save_checkpoint(args.output/'final.pt',model,optimizer,step=args.steps,binding=binding)
    if any(p.read_bytes()!=raw for p,raw in snapshots.items()): raise RuntimeError('Source changed')
    report=dict(task='I3-rna-smoke-resume',attention=args.attention,binding=binding,records=len(records),steps=args.steps,
        start_step=start_step,initial_train_loss=initial_loss,
        final_train_loss=float(classification_loss(final_logits,targets)),
        train_accuracy=float((final_logits.argmax(-1)==targets).float().mean()),
        majority_train_accuracy=max(targets.tolist().count(i) for i in range(len(classes)))/len(records),
        parameters=parameter_count(model),elapsed_seconds=elapsed,source_unchanged=True,
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device=='cuda' else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device=='cuda' else None,
        device_name=torch.cuda.get_device_name() if args.device=='cuda' else 'cpu',
        supervised_training=True,full_training_ready=False,evaluation_scope='train slice only',
        command=sys.argv,free_disk_bytes=shutil.disk_usage(args.output).free)
    for name,value in [('report.json',report),('history.json',history),
                       ('predictions.json',{'record_ids':[r.record_id for r in records],
                                            'targets':targets.tolist(),'logits':final_logits.tolist()})]:
        with (args.output/name).open('x') as stream: json.dump(value,stream,indent=2); stream.write('\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
