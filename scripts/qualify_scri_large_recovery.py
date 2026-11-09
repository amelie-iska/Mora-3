#!/usr/bin/env python3
"""One bounded CPU registered-owner checkpoint round trip; zero optimizer updates.

The supplied fixture assembly omits unbound optional expanded lanes. This check
never supplies an all-source release, full-head gradient or GPU qualification.
"""
import argparse,hashlib,json,os,resource,shutil,subprocess,sys,tempfile,time
from pathlib import Path
from dataclasses import replace
ROOT=Path(__file__).resolve().parents[1]
for rel in ('src','scripts','tests/model','tests/data','tests/extensions'):sys.path.insert(0,str(ROOT/rel))
from qualify_scri_owner import dump,tensor_digest


def build(width,tmp,seed):
    import torch
    from test_full_owner import actual_core_assets
    from test_lincs_native import cases
    from test_goa_objective import view
    from mora3.model.full_owner import PrimaryConfiguration,construct_primary_owner
    from mora3.extensions.scri.model import SCRITargetAxis
    from mora3.extensions.scri.native_owner import SpatialBudget
    bound=cases.__wrapped__()
    class Request:
        def getfixturevalue(self,name):
            if name!='cases':raise ValueError('unexpected fixture')
            return bound
    assets=actual_core_assets.__wrapped__(tmp,Request())
    assets=replace(assets,goa_packets=(view(tmp),),scri_axis=SCRITargetAxis('d'*64,(('rna','gene','count'),),'poisson'),
        scri_spatial_budget=SpatialBudget(64,4097,68000000,2000000,2))
    torch.manual_seed(seed)
    return construct_primary_owner(PrimaryConfiguration(width,8,'a'*64),assets)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--width',required=True,type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.width not in (80,266)or os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('explicit bounded CPU owner widths only')
    out=Path(a.output).absolute()
    if any(x.is_symlink()for x in(out,*out.parents)):raise ValueError('no symlink output paths')
    if shutil.disk_usage(out.parent).free<200*1024**3+2*1024**3:raise ValueError('200 GiB plus checkpoint budget required')
    if not a.resume:out.mkdir(exist_ok=False)
    import torch
    from test_scri_native_routes import envelope,AXIS
    from mora3.extensions.scri.native_routes import decode_envelope
    from mora3.data.admet import canonical
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);start=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='mora-scri-recovery-')as td:
        model=build(a.width,Path(td),2026110810 if a.resume else 2026110809)
        before=tensor_digest(model)
        q=decode_envelope(canonical(envelope()),task_key='scri-spatial',axis=AXIS)[0]
        if a.resume:
            path=out/'state.pt';expected=json.loads((out/'checkpoint-binding.json').read_text())
            from mora3.extensions.scri.curation import file_digest
            if file_digest(path)!=expected['sha256']:raise ValueError('checkpoint digest mismatch')
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
            with torch.no_grad():z=model.scri.forward_spatial((q,))
            result=dict(parameter_sha256=tensor_digest(model),output=z.tolist(),
                different_initialization_before_load=before!=tensor_digest(model),
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
            dump(out/'resumed.json',result)
        else:
            z=model.scri.forward_spatial((q,));loss=model.scri.objective(z,[[2]],[[True]])['loss'];loss.backward()
            grads=[p.grad for p in model.parameters()if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all().item()for g in grads):raise ValueError('nonfinite registered branch gradients')
            active=len(grads);model.zero_grad(set_to_none=True)
            expected=dict(parameter_sha256=tensor_digest(model),output=z.detach().tolist())
            state=model.state_dict();bad=dict(state)
            key='scri.graph.layers.0.attention.output.weight';bad[key]=bad[key].clone()+1
            try:model.load_state_dict(bad)
            except ValueError:pass
            else:raise ValueError('alias corruption did not refuse')
            if tensor_digest(model)!=before:raise ValueError('bad checkpoint partially changed weights')
            with(out/'state.pt').open('xb')as f:torch.save(state,f)
            from mora3.extensions.scri.curation import file_digest
            dump(out/'checkpoint-binding.json',dict(sha256=file_digest(out/'state.pt'),bytes=(out/'state.pt').stat().st_size))
            p=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(out),'--width',str(a.width),'--resume'],
                env=os.environ.copy(),capture_output=True,text=True,timeout=90)
            dump(out/'fresh-process.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
            if p.returncode:raise ValueError('fresh recovery failed; artifacts retained')
            recovered=json.loads((out/'resumed.json').read_text())
            exact=all(recovered[k]==v for k,v in expected.items())and recovered['different_initialization_before_load']
            dump(out/'receipt.json',dict(schema='Mora3.SCRIRegisteredLargeOwnerRecovery/v1',width=a.width,
                total_parameters=sum(x.numel()for x in model.parameters()),active_gradient_parameter_tensors=active,
                complete_gradient_parameter_tensors=len(tuple(model.parameters())),finite_branch_gradients=True,
                corrupted_alias_refused_before_mutation=True,fresh_process_full_parameter_and_prediction_equality=exact,
                optimizer_updates=0,optimizer_resume_qualified=False,source_training_exposures=0,
                device='cpu',elapsed_seconds=time.monotonic()-start,
                parent_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                child_peak_rss_bytes=recovered['peak_rss_bytes'],
                scope='registered fixture assembly only; optional expanded source heads omitted',full_campaign_ready=False))
            if not exact:raise ValueError('fresh-process parameter/prediction mismatch')
    print(json.dumps(dict(mode='resume'if a.resume else'qualification',width=a.width,elapsed_seconds=time.monotonic()-start)))

if __name__=='__main__':main()
