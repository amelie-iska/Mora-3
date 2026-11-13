"""Streaming metadata-only exposure audit of the complete registered native view.

This preserves all existing roles and filters. It does not fit new partitions,
repair leakage, grant source rights, or release a model for training.
"""
from pathlib import Path
from hashlib import sha256
import json,os,time,resource,shutil,datetime
from mora3.data.finite_union_native_rows import read_exact,decode_disposition
from mora3.data.split_exposure import HeldoutGroupExposure


def _groups(raw, max_groups):
    if type(raw) is not str or len(raw)>8*1_048_576:
        raise ValueError('Complete bounded literal group JSON required')
    value=json.loads(raw)
    if type(value)is not list or len(value)>max_groups or any(type(x)is not str for x in value):
        raise ValueError('Original group list required')
    return tuple(value)


def scan_view_phase(view,root,audit,*,phase,record,checkpoint,max_sidecar_bytes=64*2**20):
    """Consume exact original metadata columns and selected original ordinals."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    if phase not in('heldout','train')or type(max_sidecar_bytes)is not int or max_sidecar_bytes<1:
        raise ValueError('Explicit phase and positive sidecar budget required')
    root=Path(root).resolve();selected_rows=0;sidecars=0;source_bytes=0;commitment=sha256()
    exclusions={}
    for path,ordinal in view._ordinal_exclusions:exclusions.setdefault(path,set()).add(ordinal)
    for index,descriptor in enumerate(view.descriptors):
        if index%64==0:checkpoint()
        source=descriptor['source_shard'];rows=source['rows']
        if type(rows)is not int or not 0<=rows<=2_000_000:
            raise ValueError('Original metadata row budget exceeded')
        # Disposition files are source-role metadata, not targets.
        disposition=descriptor['disposition']
        if disposition.get('kind')=='original-row-ordinal-uint8':
            mask=disposition['reference'];path=(root/mask['path']).resolve()
            if not path.is_relative_to(root)or mask['bytes']!=rows:
                raise ValueError('Exact root-bound ordinal mask required')
        codes=decode_disposition(descriptor,root=root)
        wanted=(1,2)if phase=='heldout'else(0,)
        if not any(code in wanted for code in set(codes)):continue
        omitted=exclusions.get(source['path'],set())
        if any(type(i)is not int or not 0<=i<rows or codes[i]!=0 for i in omitted):
            raise ValueError('Only existing train ordinals can be excluded')
        selected=sum(code in wanted and i not in omitted for i,code in enumerate(codes))
        if not selected:continue
        ref=descriptor['original_group_sidecar'];path=(root/ref['path']).resolve()
        if not path.is_relative_to(root):raise ValueError('Group sidecar outside source root')
        if type(ref['bytes'])is not int or not 0<ref['bytes']<=max_sidecar_bytes or ref['rows']!=rows:
            raise ValueError('Complete sidecar count or resource budget differs')
        raw=read_exact(ref,root=root)
        parquet=pq.ParquetFile(pa.BufferReader(raw))
        names=parquet.schema_arrow.names
        if(len(names)!=len(set(names))or not {'task_id','effective_groups_json'}<=set(names)
                or parquet.metadata.num_rows!=rows):
            raise ValueError('Exact original task/group metadata schema and row count required')
        offset=0;accepted=0;role_counts={'train':0,'validation':0,'test':0}
        for batch in parquet.iter_batches(batch_size=8192,columns=['task_id','effective_groups_json'],use_threads=False):
            identities=batch.column(0).to_pylist();group_text=batch.column(1).to_pylist()
            for local,(identity,text)in enumerate(zip(identities,group_text)):
                ordinal=offset+local;role=codes[ordinal]
                if role not in wanted or ordinal in omitted:continue
                groups=_groups(text,audit.max_groups);coordinate=f'{index}:{ordinal}'
                if phase=='heldout':audit.add_heldout(identity,role,groups,coordinate)
                else:audit.add_train(identity,groups,coordinate)
                accepted+=1;role_counts[('train','validation','test')[role]]+=1
            offset+=batch.num_rows
        if offset!=rows or accepted!=selected:raise ValueError('Complete original ordinal conservation failed')
        item=dict(phase=phase,descriptor_index=index,source_path=source['path'],sidecar_path=ref['path'],
            sidecar_sha256=ref['sha256'],sidecar_bytes=ref['bytes'],original_rows=rows,selected_rows=selected,
            selected_roles=role_counts,applied_train_exclusions=len(omitted))
        record(item);commitment.update(json.dumps(item,sort_keys=True,separators=(',',':')).encode()+b'\n')
        selected_rows+=accepted;sidecars+=1;source_bytes+=len(raw)
    checkpoint()
    return dict(phase=phase,selected_rows=selected_rows,sidecars=sidecars,sidecar_bytes=source_bytes,
        ordered_source_commitment_sha256=commitment.hexdigest())


def run_registered_audit(root,output,*,max_seconds=1200,max_rss_gib=8):
    from mora3.research.cross_source_chemical_roles import _load_group_metadata_view
    from mora3.model.source_assembly_inputs import CURRENT_VIEW_PINS
    root=Path(root).resolve();output=Path(output).resolve();code_root=Path(__file__).resolve().parents[3]
    if(root!=Path.cwd().resolve()or not output.is_relative_to(code_root/'outputs')
            or type(max_seconds)is not int or not 1<=max_seconds<=3600
            or type(max_rss_gib)is not int or not 2<=max_rss_gib<=16):
        raise ValueError('Exact source cwd, continuation-owned output and bounded resource policy required')
    if shutil.disk_usage(root).free<64*2**30+512*2**20:raise ValueError('64 GiB reserve plus output forecast required')
    output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def checkpoint():
        if time.monotonic()-start>max_seconds:raise RuntimeError('Explicit source-audit wall budget reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>max_rss_gib*2**30:raise RuntimeError('Source-audit host-memory budget reached')
        if shutil.disk_usage(root).free<64*2**30:raise RuntimeError('Free disk reserve reached')
    try:
        import pyarrow as pa
        pa.set_cpu_count(1);pa.set_io_thread_count(1)
        view=_load_group_metadata_view(root);checkpoint()
        if len(view.descriptors)>100_000:raise ValueError('Source descriptor budget exceeded')
        expected=dict(view.effective_filter['effective_retained_split_counts'])
        input_ref=CURRENT_VIEW_PINS['effective_source_view']
        index=HeldoutGroupExposure(max_heldout_keys=2_000_000,max_heldout_rows=200_000,max_groups_per_row=65536,max_examples=200)
        with(output/'source-shards.jsonl').open('x')as journal:
            def record(row):
                journal.write(json.dumps(row,sort_keys=True,separators=(',',':'))+'\n')
                if row['descriptor_index']%128==0:journal.flush()
            heldout=scan_view_phase(view,root,index,phase='heldout',record=record,checkpoint=checkpoint)
            index.freeze_heldout()
            print(json.dumps(dict(phase='heldout-index-complete',rows=index.rows,groups=len(index.groups))),flush=True)
            train=scan_view_phase(view,root,index,phase='train',record=record,checkpoint=checkpoint)
            journal.flush();os.fsync(journal.fileno())
        result=index.finish()
        if result['role_rows']!=expected:raise ValueError('Actual full-view role counts differ from pinned current filter')
        if sha256((root/input_ref[0]).read_bytes()).hexdigest()!=input_ref[1]:raise ValueError('Source-view descriptor changed during audit')
        report=dict(schema='Mora3.CompleteRegisteredViewSplitExposure/v1',status='complete-scope-audit',
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_view=dict(path=input_ref[0],sha256=input_ref[1]),
            source_descriptor_count=len(view.descriptors),original_taskviews=sum(d['source_shard']['rows']for d in view.descriptors),
            excluded_original_taskviews=sum(d['source_shard']['rows']for d in view.descriptors)-sum(result['role_rows'].values()),
            current_view_additional_ordinal_exclusions=len(view._ordinal_exclusions),phases=[heldout,train],exposure=result,
            source_group_namespace_equivalence_independently_verified=False,near_duplicate_search_performed=False,
            per_route_later_quarantines_included=False,expanded_sources_outside_registered_view_included=False,
            numeric_condition_or_target_columns_decoded=False,original_splits_changed=False,
            redistribution_approved=False,full_expanded_training_admission=False,optimizer_updates=0,training_runs=0,
            seconds=time.monotonic()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            code_sha256={str(p.relative_to(code_root)):sha256(p.read_bytes()).hexdigest()for p in(
                Path(__file__).resolve(),code_root/'src/mora3/data/split_exposure.py')})
        # code_root is the repository directory; only these exact code files
        # bind this metadata reader. This artifact cannot confer source rights.
        with(output/'result.json').open('x')as f:json.dump(report,f,indent=2);f.flush();os.fsync(f.fileno())
        print(json.dumps({k:v for k,v in report.items()if k not in('code_sha256','exposure')},indent=2),flush=True)
        print(json.dumps({k:v for k,v in result.items()if k!='examples'},indent=2),flush=True)
        return report
    except Exception as error:
        with(output/'FAILED.json').open('x')as f:json.dump(dict(schema='Mora3.IncompleteSplitExposure/v1',complete=False,
            exception=type(error).__name__,reason=str(error),seconds=time.monotonic()-start,training_authorized=False),f,indent=2)
        raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--max-seconds',type=int,default=1200);parser.add_argument('--max-rss-gib',type=int,default=8)
    args=parser.parse_args();run_registered_audit(args.root,args.output,max_seconds=args.max_seconds,max_rss_gib=args.max_rss_gib)
