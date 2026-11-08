"""Catch target leakage, wrong loss reduction, missing optimizer/RNG resume state."""
import copy
import importlib
import math
import sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import pytest
import torch
from mora3.data.schema import GraphRecord, Node, Edge, Measurement, Provenance
from mora3.tokenization.shortest_path import ByteTokenizer


def api():
    return importlib.import_module('mora3.model.training')


def record(label='a', sequence='AC'):
    return GraphRecord('test', (Node('rna',(Measurement('sequence',sequence,None,'observed',True),)),
        Node('rna_annotation',(Measurement('rna_type',label,None,'observed',True),))),
        (Edge(0,1,'has_annotation',True),), (Provenance('test','v1','x','0'*64,'test-only'),))


def model(pooling='positional'):
    torch.manual_seed(19)
    return api().RNAClassifier(ByteTokenizer(controls=('pad','node','edge')), ('a','b'), pooling=pooling,
                               feature_width=8, width=8, heads=2, layers=1, id_width=2)


def test_loss_equation_gradient_and_reject_invalid_targets():
    logits=torch.tensor([[math.log(3),0.],[0.,0.]],dtype=torch.float64,requires_grad=True)
    loss=api().classification_loss(logits,torch.tensor([0,1]))
    assert loss.item()==pytest.approx((-math.log(.75)-math.log(.5))/2)
    loss.backward()
    torch.testing.assert_close(logits.grad,torch.tensor([[-.125,.125],[.25,-.25]],dtype=torch.float64))
    for target in (torch.tensor([-1,0]),torch.tensor([0,2]),torch.tensor([0.0,1.0])):
        with pytest.raises(ValueError): api().classification_loss(logits,target)


def test_annotation_exclusion_and_sequence_gradients():
    m=model().double()
    a,b=record(),record('b')
    ya,yb=m([a]),m([b])
    torch.testing.assert_close(ya,yb,atol=0,rtol=0)
    stripped=replace(b,nodes=(b.nodes[0],Node('rna_annotation')))
    torch.testing.assert_close(ya,m([stripped]),atol=0,rtol=0)
    assert m.targets([a,b]).tolist()==[0,1]
    with pytest.raises(ValueError): m.targets([stripped])
    with pytest.raises(ValueError): m.targets([record('unknown')])
    api().classification_loss(ya,m.targets([a])).backward()
    assert m.payload.embedding.weight.grad[65].abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())


def test_bag_ablation_is_order_invariant_and_positional_is_not():
    a=model('bag').double(); b=model().double()
    torch.testing.assert_close(a.payload.pool([65,67]),a.payload.pool([67,65]),atol=0,rtol=0)
    assert not torch.allclose(b.payload.pool([65,67]),b.payload.pool([67,65]))
    assert sum(p.numel() for p in a.parameters())==sum(p.numel() for p in b.parameters())
    with pytest.raises(ValueError): model('invalid')


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_checkpoint_restores_optimizer_rng_and_exact_trajectory(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    m=model().to(device); opt=torch.optim.Adam(m.parameters(),lr=.003)
    def step(m,opt):
        opt.zero_grad(); logits=m([record(),record('b','CA')])
        loss=api().classification_loss(logits,m.targets([record(),record('b','CA')]))
        loss.backward(); opt.step(); return loss.detach()
    step(m,opt)
    path=tmp_path/'step1.pt'; binding={'source':'abc','pooling':'positional'}
    api().save_checkpoint(path,m,opt,step=1,binding=binding)
    expected_rng=torch.rand(5,device=device); expected_loss=step(m,opt)
    restored=model().to(device); opt2=torch.optim.Adam(restored.parameters(),lr=.003)
    assert api().load_checkpoint(path,restored,opt2,binding=binding)==1
    torch.testing.assert_close(torch.rand(5,device=device),expected_rng,atol=0,rtol=0)
    torch.testing.assert_close(step(restored,opt2),expected_loss,atol=0,rtol=0)
    for p,q in zip(m.parameters(),restored.parameters()): torch.testing.assert_close(p,q,atol=0,rtol=0)
    with pytest.raises(FileExistsError): api().save_checkpoint(path,m,opt,step=2,binding=binding)
    with pytest.raises(ValueError): api().load_checkpoint(path,restored,opt2,binding={'source':'wrong'})


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_gpu_logits_loss_gradient_parity():
    cpu=model(); gpu=copy.deepcopy(cpu).cuda(); rows=[record(),record('b','CA')]
    a,b=cpu(rows),gpu(rows)
    torch.testing.assert_close(a,b.cpu(),atol=2e-6,rtol=2e-5)
    la=api().classification_loss(a,cpu.targets(rows)); lb=api().classification_loss(b,gpu.targets(rows))
    la.backward(); lb.backward()
    for p,q in zip(cpu.parameters(),gpu.parameters()): torch.testing.assert_close(p.grad,q.grad.cpu(),atol=3e-6,rtol=3e-4)


@pytest.mark.parametrize('attention',['vanilla','tropical-code','tropical-blockwise'])
def test_training_cli_artifacts_and_fail_closed_inputs(tmp_path,attention):
    import hashlib, json, os, subprocess
    source=tmp_path/'records.jsonl'; source.write_bytes(record().canonical_bytes()+b'\n')
    manifest=tmp_path/'manifest.json'
    info={'checksum_files':{'records.jsonl':hashlib.sha256(source.read_bytes()).hexdigest()},
          'split_assignments':{'test':'train'}}
    manifest.write_text(json.dumps(info))
    vocab=tmp_path/'vocab.json'; vocab.write_text(json.dumps({'learned_hex':[],'controls':['pad','node','edge'],'size':259}))
    out=tmp_path/'run'
    command=[sys.executable,'scripts/train_rna_smoke.py','--manifest',str(manifest),'--vocabulary',str(vocab),
             '--output',str(out),'--steps','2','--device','cpu','--attention',attention]
    if attention=='tropical-blockwise': command += ['--tropical-block-size','2']
    # Two classes required, add a distinct record and its explicit train assignment.
    second=replace(record('b','CA'),record_id='second')
    source.write_bytes(source.read_bytes()+second.canonical_bytes()+b'\n')
    info['checksum_files']['records.jsonl']=hashlib.sha256(source.read_bytes()).hexdigest()
    info['split_assignments']['second']='train'; manifest.write_text(json.dumps(info))
    env={**os.environ,'OMP_NUM_THREADS':'2'}
    result=subprocess.run(command,text=True,capture_output=True,env=env)
    assert result.returncode==0,result.stderr
    report=json.loads((out/'report.json').read_text())
    assert report['steps']==2 and report['full_training_ready'] is False
    assert report['attention']==attention
    if attention!='vanilla': assert report['binding']['attention']==attention
    if attention=='tropical-blockwise': assert report['binding']['tropical_block_size']==2
    assert (out/'checkpoint-1.pt').stat().st_size>0
    assert (out/'final.pt').stat().st_size>0
    assert subprocess.run(command,capture_output=True,env=env).returncode!=0
    command[command.index('--output')+1]=str(tmp_path/'rejected')
    info['split_assignments']['second']='test'; manifest.write_text(json.dumps(info))
    assert subprocess.run(command,capture_output=True,env=env).returncode!=0
    assert not (tmp_path/'rejected').exists()
