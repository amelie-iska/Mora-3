#!/usr/bin/env python3
"""Explicit CPU-only SCRI diagnostics; never launch an experimental campaign.

small: five total optimizer updates including one in a fresh resume process.
contexts: no updates, synthetic dense radius graphs with reported actual costs.
counts: caller-selected full-factory fixture widths; no experimental readiness.
"""
from pathlib import Path
import argparse,copy,hashlib,json,os,resource,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
for p in ('src','tests/extensions','tests/model','tests/data'):sys.path.insert(0,str(ROOT/p))


def dump(path,value):
    with Path(path).open('x')as f:json.dump(value,f,indent=2,allow_nan=False)


def tensor_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def tiny_family():
    import torch
    from test_scri_native_owner import model,ID_AXIS
    from mora3.extensions.scri.native_owner import native_scri_roles
    from mora3.model.parameterization import parameter_ledger,width_family
    torch.manual_seed(2026110809)
    ms=tuple(model(w)for w in(8,12,16))
    roles=tuple(native_scri_roles(m,identifier_axis=ID_AXIS)for m in ms)
    family=width_family(*(parameter_ledger(m,r)for m,r in zip(ms,roles)))
    if any(sum(p.numel()for p in m.parameters())>32768 for m in ms):raise ValueError('tiny model limit')
    return ms,roles,family


def engine_for(model,roles,family,member):
    from mora3.model.mu_optimizer import MuAdamW
    from mora3.research.primary_training import NativeMuUpdateEngine
    opt=MuAdamW(model,family,roles,base_lr=.0003,weight_decay=0.,decay_aliases=(),member=member)
    return NativeMuUpdateEngine(model,opt,binding_sha256='e'*64,task_cycle=('synthetic-spatial',),source_cursors={'synthetic-spatial':b'0'})


def one_update(engine):
    from test_scri_native_owner import query
    result=engine.model.forward_spatial((query(8),))
    objective=engine.model.objective(result,[[2]],[[True]])
    record=engine.update('synthetic-spatial',objective['loss'],cursor_before=str(engine.steps).encode(),cursor_after=str(engine.steps+1).encode())
    return {'step':record['step'],'loss':record['loss'],'output_mean_abs':float(result.detach().abs().mean())}


def small(out):
    from mora3.extensions.scri.transfer_audit import audit_optimizer_groups
    ms,roles,family=tiny_family();coordinates=[]
    for member,m,r in zip(('base','delta','target'),ms,roles):
        engine=engine_for(m,r,family,member)
        optaudit=audit_optimizer_groups(m,family,r,member=member)
        first=one_update(engine)
        coordinates.append(dict(width=m.core.shared._configuration['width'],parameters=sum(p.numel()for p in m.parameters()),
            first_update=first,optimizer_groups=optaudit))
        if member=='target':
            engine.save_checkpoint(out/'tiny-midpoint.pt')
            second=one_update(engine)
            expected=dict(parameter_sha256=tensor_digest(m),step=engine.steps,cursor=engine.source_cursors['synthetic-spatial'].decode(),second_loss=second['loss'])
            dump(out/'expected-resume.json',expected)
    cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'resume','--output',str(out)]
    p=subprocess.run(cmd,env=os.environ.copy(),capture_output=True,text=True,timeout=30)
    dump(out/'fresh-process.json',dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    if p.returncode:raise ValueError('fresh process diagnostic failed; retained evidence')
    resumed=json.loads((out/'resumed.json').read_text())
    dump(out/'small-qualification.json',dict(schema='Mora3.SCRITinyQualification/v1',coordinates=coordinates,
        actual_optimizer_updates=5,fresh_process_resume_exact=resumed==expected,
        complete_owner_coordinate_qualification=False,synthetic_only=True,source_training_exposures=0))
    if resumed!=expected:raise ValueError('fresh-process continuation mismatch')


def resume(out):
    ms,roles,family=tiny_family();engine=engine_for(ms[2],roles[2],family,'target')
    engine.load_checkpoint(out/'tiny-midpoint.pt');result=one_update(engine)
    dump(out/'resumed.json',dict(parameter_sha256=tensor_digest(engine.model),step=engine.steps,
        cursor=engine.source_cursors['synthetic-spatial'].decode(),second_loss=result['loss']))


def contexts(out):
    import torch
    from test_scri_native_owner import model,query
    m=model();records=[]
    for n in(8,16,32,64):
        begin=time.perf_counter();q=query(n);build=time.perf_counter()-begin
        begin=time.perf_counter();z=m.forward_spatial((q,));loss=m.objective(z,[[2]],[[True]])['loss']
        loss.backward();elapsed=time.perf_counter()-begin
        grads=[p.grad for p in m.parameters()if p.grad is not None]
        records.append(dict(nodes=n,edges=len(q.record.edges),tokens=1+n+len(q.record.edges),
            query_build_seconds=build,forward_backward_seconds=elapsed,graphs_per_second=1/elapsed,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            finite_gradients=all(torch.isfinite(g).all().item()for g in grads),
            incidence_nonzero_gradient=bool(m.graph.incidence.weight.grad.abs().sum()>0)))
        m.zero_grad(set_to_none=True)
    dump(out/'contexts.json',dict(schema='Mora3.SCRIContextMeasurement/v1',device='cpu',threads=1,records=records,
        dense_synthetic_graphs=True,realistic_source_topology_qualified=False,optimizer_updates=0,
        peak_rss_is_cumulative_process_measurement=True,includes_concurrent_host_contention=True))


def counts(out,widths):
    import torch,tempfile
    from dataclasses import replace
    from test_full_owner import actual_core_assets
    from test_lincs_native import cases
    from test_goa_objective import view
    from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner,unique_parameter_inventory
    from mora3.model.full_owner_mup import full_owner_mup_parameter_roles
    from mora3.extensions.scri.model import SCRITargetAxis
    from mora3.extensions.scri.native_owner import SpatialBudget
    bound_cases=cases.__wrapped__()
    class Request:
        def getfixturevalue(self,name):
            if name!='cases':raise ValueError('unknown fixture')
            return bound_cases
    rows=[]
    with tempfile.TemporaryDirectory(prefix='mora-scri-counts-')as tmp:
        assets=actual_core_assets.__wrapped__(Path(tmp),Request())
        assets=replace(assets,goa_packets=(view(Path(tmp)),),scri_axis=SCRITargetAxis('d'*64,(('rna','gene','count'),),'poisson'),
            scri_spatial_budget=SpatialBudget(64,4097,68000000,2000000,2))
        for w in widths:
            torch.manual_seed(2026110809);begin=time.perf_counter()
            m=construct_primary_owner(PrimaryConfiguration(w,8,'a'*64),assets)
            roles=full_owner_mup_parameter_roles(m,identifier_axis=('context','observation'),
                geometry_identifier_axis=tuple('geometry:'+str(i)for i in range(16)),color_identifier_axis=('color:0','color:1'))
            inventory=unique_parameter_inventory(m)
            rows.append(dict(width=w,total_parameters=inventory['total_parameters'],alias_names=inventory['alias_names'],
                unique_parameter_tensors=inventory['unique_parameter_tensors'],declared_roles=len(roles),
                seconds=time.perf_counter()-begin,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
            del m,roles,inventory
        dump(out/'counts.json',dict(schema='Mora3.SCRIRegisteredFixtureOwnerCounts/v1',rows=rows,
            scope='all registered parameters of actual cooperative factory with existing bounded test source assembly; optional expanded lanes not supplied',
            omitted_optional_sources=[k for k,v in vars(assets).items()if v is None],
            complete_expanded_owner_qualified=False,optimizer_updates=0,source_rows_trained=0))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('small','resume','contexts','counts'));p.add_argument('--output',required=True);p.add_argument('--widths',nargs='+',type=int,default=[8,12,16]);a=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('diagnostic requires explicit GPU-hidden execution')
    import torch,shutil
    torch.set_num_threads(1)
    out=Path(a.output).absolute()
    if any(x.is_symlink()for x in(out,*out.parents)):raise ValueError('no symlink outputs')
    if shutil.disk_usage(out.parent).free<200*1024**3+64*1024**2:raise ValueError('200 GiB disk floor')
    if a.mode!='resume':out.mkdir(exist_ok=False)
    start=time.monotonic()
    if a.mode=='small':small(out)
    elif a.mode=='resume':resume(out)
    elif a.mode=='contexts':contexts(out)
    else:
        if any(w<8 or w>2048 or w%2 for w in a.widths):raise ValueError('explicit even widths 8..2048 required')
        counts(out,a.widths)
    print(json.dumps({'mode':a.mode,'elapsed_seconds':time.monotonic()-start,'status':'completed'}))

if __name__=='__main__':main()
