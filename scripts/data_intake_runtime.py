"""Read-only data-runtime handoff. It implements no model heads or training loop.

Only model_inputs/input_metadata/input_graph may be offered to a neural adapter.
Audit/resume/source IDs, supervision, target axes and target_graph stay separate.
Custom molecule/profile projection is inspection-only until a model owner verifies
and registers a real reader/objective; this does not alter corpus support flags.
"""
from pathlib import Path
import sys,json,hashlib,dataclasses
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from mora3.data.array_tasks import ArrayTask,OrderedArray
from mora3.data.sciplex_admission import ObservationalSciPlexTask
from mora3.data.raster_morphology import RasterMorphologyTask
from data_intake_corpus import CorpusTask,canonical,read_corpus_tasks
from mora3.data.rgd1_consumption import require_corpus_consumption

def _encoded(value):
    if isinstance(value,torch.Tensor):
        x=value.detach().cpu().contiguous()
        return dict(dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    if isinstance(value,dict):return {k:_encoded(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [_encoded(v) for v in value]
    return value

def input_fingerprint(projection):
    graph=projection.get('input_graph')
    semantics=None if graph is None else dict(nodes=[dataclasses.asdict(n) for n in graph.nodes],edges=[dataclasses.asdict(e) for e in graph.edges])
    return hashlib.sha256(canonical(_encoded(dict(inputs=projection['model_inputs'],metadata=projection['input_metadata'],graph_semantics=semantics)))).hexdigest()

def project_corpus_row(row,*,manifest_sha256,for_training=False,allow_profile_inspection=False,device='cpu'):
    require_corpus_consumption(row, purpose='objective' if for_training else 'evaluation')
    if row.get('admitted_for_task') is not True:raise ValueError('task-specific admission required')
    if row.get('group_metadata_status')!='manifest_verified_effective_groups':raise ValueError('verified effective group metadata required')
    if for_training and row['split']!='train':raise ValueError('heldout data cannot be projected for training')
    raw=row['native_payload'];sha=hashlib.sha256(raw).hexdigest()
    if sha!=row['native_payload_sha256']:raise ValueError('canonical native payload hash mismatch')
    if not isinstance(manifest_sha256,str) or len(manifest_sha256)!=64:raise ValueError('manifest binding required')
    schema=row['native_schema'];supported=row.get('model_adapter_supported',schema=='mora3.ObservationalSciPlexTask/v1')
    if for_training and not supported:raise ValueError('verified model adapter is unavailable;inspection is not training admission')
    input_axes={};target_axes={};input_meta={};context_masks={};kind='native_container_runtime';p={}
    if schema=='mora3.ObservationalSciPlexTask/v1':
        task=ObservationalSciPlexTask.from_bytes(raw);p=task.project(device=device,for_training=for_training);input_meta=task.input_metadata();s=json.loads(task.task.sidecar_bytes);role='targets' if s['task']=='response' else 'conditions';axis=s[role]['expression']['axis_ids'];(target_axes if role=='targets' else input_axes)['expression']=axis
        context_masks=dict(cell_line=True,exposure_time=True,physical_dose=False,chemical_structure=False,raw_cell_barcode_lineage=False)
    elif schema=='mora3.ArrayTask/v1':
        task=ArrayTask.from_bytes(raw);p=task.project(device=device);input_meta=task.input_metadata();roles=dict(task.roles)
        for a in task.arrays:
            if roles[a.name] in ('condition','desired'):input_axes[a.name]=list(a.axis_ids)
            if roles[a.name]=='target':target_axes[a.name]=list(a.axis_ids)
        context_masks=dict(cell_line=False,physical_dose=False,exposure_time=False,raw_image=False,instance_mask=False)
    elif schema=='mora3.RasterMorphologyTask/v1':
        task=RasterMorphologyTask.from_bytes(raw);p=task.project(device=device,for_training=for_training);input_meta=dict(direction=task.direction,height=task.height,width=task.width,pixel_size_um=task.pixel_size_um,modality=task.modality,condition_role='source_observed_image' if task.direction=='image_to_mask' else 'source_observed_geometry')
        context_masks=dict(raw_image=task.direction=='image_to_mask',instance_mask=task.direction=='mask_to_image',physical_scale=task.pixel_size_um is not None,expression=False,perturbation=False)
    elif schema=='mora3.CorpusTask/v1' and allow_profile_inspection:
        if for_training:raise ValueError('custom data inspection is not a verified model adapter')
        task=CorpusTask.from_bytes(raw);d=json.loads(task.canonical_bytes());name=d['task']
        if name not in ('source_observational_molecule_to_profile','source_observational_profile_to_molecule'):raise ValueError('unsupported custom runtime objective')
        c=d['conditions'];t=d['targets'];forward=name.endswith('molecule_to_profile');array=t['source_profile_array'] if forward else c['source_profile_array'];a=OrderedArray(**array)
        if a.status!='observed':raise ValueError('observed source profile required;model-produced values need separate inference provenance')
        if forward and list(a.axis_ids)!=c['known_profile_axis']:raise ValueError('source target/query axis mismatch')
        values=torch.tensor(a.values,dtype=torch.float64,device=device);p=dict(inputs={},targets={},target_masks={});kind='custom_data_inspection_only';supported=False
        if forward:
            p['inputs']=dict(c);p['targets']['profile']=values;p['target_masks']['profile']=torch.ones_like(values,dtype=torch.bool);target_axes['profile']=list(a.axis_ids);input_axes['known_profile_axis']=list(c['known_profile_axis'])
        else:
            p['inputs']={k:v for k,v in c.items() if k!='source_profile_array'};p['inputs']['profile']=values;p['targets']['canonical_isomeric_smiles']=t['canonical_isomeric_smiles'];p['target_masks']['canonical_isomeric_smiles']=torch.tensor(True,device=device);input_axes['profile']=list(a.axis_ids)
        input_meta=dict(direction=name,assay_modality=c['assay_modality'],input_profile_axis=input_axes,cell_line=c['cell_line'],profile_unit=None if forward else a.units,physical_dose=None,exposure_time=None)
        context_masks=dict(cell_line=c['cell_line'] is not None,physical_dose=False,exposure_time=False,raw_image=False,instance_mask=False,same_cell_crossassay_pair=False)
    else:raise ValueError('unsupported corpus runtime schema/objective;no silent fallback')
    masks=dict(p.get('target_masks',{}))
    for name,value in p['targets'].items():
        if name not in masks and isinstance(value,torch.Tensor):masks[name]=torch.ones_like(value,dtype=torch.bool)
    binding=dict(manifest_sha256=manifest_sha256,task_id=row['task_id'],native_payload_sha256=sha,source_sha256=row.get('source_sha256'),source_revision=row.get('source_revision'),split=row['split'],effective_groups=sorted(json.loads(row['group_ids_json'])))
    return dict(runtime_kind=kind,model_adapter_supported=bool(supported),model_inputs=p['inputs'],supervision_targets=p['targets'],supervision_masks=masks,input_metadata=input_meta,input_axes=input_axes,target_axes=target_axes,input_graph=p.get('input_graph'),target_graph=p.get('target_graph'),context_masks=context_masks,resume_binding=binding,audit=dict(source_record_id=row['record_id'],source_family=row['source_family'],source_context_json=row.get('condition_context_json'),evidence_class=row.get('evidence_class'),missing_context_json=row.get('missing_context_json')))

def read_runtime_tasks(manifest,*,split='train',source=None,limit=2,for_training=False,allow_profile_inspection=False):
    mp=Path(manifest);sha=hashlib.sha256(mp.read_bytes()).hexdigest()
    for row in read_corpus_tasks(mp,split=split,source=source,supported_only=not allow_profile_inspection,limit=limit):
        yield project_corpus_row(row,manifest_sha256=sha,for_training=for_training,allow_profile_inspection=allow_profile_inspection)

def main():
    import argparse
    a=argparse.ArgumentParser(description='Data-only native/custom inspection projection;no heads,weights or optimizer');a.add_argument('--manifest',required=True);a.add_argument('--source',required=True);a.add_argument('--split',default='train');a.add_argument('--limit',type=int,default=2);a.add_argument('--profile-inspection',action='store_true');p=a.parse_args();result=[]
    for d in read_runtime_tasks(p.manifest,source=p.source,split=p.split,limit=p.limit,allow_profile_inspection=p.profile_inspection):
        result.append(dict(runtime_kind=d['runtime_kind'],model_adapter_supported=d['model_adapter_supported'],inputs=_encoded(d['model_inputs']),targets={k:dict(shape=list(v.shape),dtype=str(v.dtype)) if isinstance(v,torch.Tensor) else 'source_string_target' for k,v in d['supervision_targets'].items()},input_axis_sizes={k:len(v) for k,v in d['input_axes'].items()},target_axis_sizes={k:len(v) for k,v in d['target_axes'].items()},context_masks=d['context_masks'],input_fingerprint=input_fingerprint(d),resume_binding=d['resume_binding']))
    print(json.dumps(dict(projected_rows=len(result),rows=result,optimizer_steps=0,model_weights_loaded=False),indent=2))

if __name__=='__main__':main()
