"""Full-axis original LINCS Parquet QC and lossless native-metadata/source-row index.
No transformations of the authoritative feature payload; no corpus/model admission.
"""
import argparse,collections,datetime,hashlib,json,math,os,pathlib,struct,time
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from data_sources_lincs_download import digest,write_new,mutable,verify_footer
NOW=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def logical_digest(batch):
    h=hashlib.sha256();h.update(batch.schema.serialize().to_pybytes());h.update(struct.pack('<Q',batch.num_rows))
    for arr in batch.columns:
        valid=np.asarray(arr.is_valid().to_numpy(zero_copy_only=False),dtype=np.uint8);h.update(valid.tobytes());mask=valid.astype(bool)
        if pa.types.is_floating(arr.type) or pa.types.is_integer(arr.type) or pa.types.is_boolean(arr.type):
            values=pc.fill_null(arr,False if pa.types.is_boolean(arr.type) else 0).to_numpy(zero_copy_only=False)
            values=np.asarray(values);values=values.astype(values.dtype.newbyteorder('<'),copy=False);h.update(values[mask].tobytes())
        elif pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type):
            for v in arr.to_pylist():
                if v is not None:
                    b=v.encode('utf8');h.update(struct.pack('<Q',len(b)));h.update(b)
        else:raise ValueError('unreviewed native logical type '+str(arr.type))
    return h.hexdigest()
def validate_axis(schema,a):
    if len(schema.names)!=len(set(schema.names)):raise ValueError('duplicate native column names')
    metadata=a['metadata_columns'];features=[n for n in schema.names if n not in metadata]
    if any(n not in schema.names for n in metadata) or len(features)!=a['feature_axis']:raise ValueError('full exact feature axis mismatch')
    is_cp=a['new_asset_id'].endswith('-CP')
    if any(not (n.startswith(('Cells_','Cytoplasm_','Nuclei_')) if is_cp else n.endswith('_at')) for n in features):raise ValueError('author feature namespace mismatch')
    if any(not (pa.types.is_floating(schema.field(n).type) or pa.types.is_integer(schema.field(n).type)) for n in features):raise ValueError('unreviewed feature physical type')
    for field in schema:
        if not (pa.types.is_floating(field.type) or pa.types.is_integer(field.type) or pa.types.is_string(field.type) or pa.types.is_boolean(field.type)):raise ValueError('unreviewed native schema type')
    return features

def inspect_profiles(batch,features):
    counts={k:np.zeros(batch.num_rows,dtype=np.uint16) for k in ['null','nan','positive_infinity','negative_infinity']};stats={}
    for name in features:
        arr=batch.column(batch.schema.get_field_index(name));valid=arr.is_valid().to_numpy(zero_copy_only=False);v=pc.fill_null(arr,0).to_numpy(zero_copy_only=False)
        masks=dict(null=~valid,nan=valid&np.isnan(v),positive_infinity=valid&np.isposinf(v),negative_infinity=valid&np.isneginf(v));finite=valid&np.isfinite(v)
        for k,m in masks.items():counts[k]+=m.astype(np.uint16)
        stats[name]={k:int(m.sum()) for k,m in masks.items()};stats[name].update(finite=int(finite.sum()),finite_min=v[finite].min().item() if finite.any() else None,finite_max=v[finite].max().item() if finite.any() else None)
    counts['finite_complete']=sum(counts.values())==0
    return stats,counts

def tree_bytes(root):
    return sum(p.stat().st_size for p in pathlib.Path(root).rglob('*') if p.is_file() and not p.is_symlink())
def bounded_write(path,data,cap,root):
    if tree_bytes(root)+len(data)>cap:raise ValueError('derived cap before write; all original payloads preserved')
    if os.statvfs(root).f_bavail*os.statvfs(root).f_frsize<200*1024**3+len(data):raise ValueError('free reserve before write')
    with pathlib.Path(path).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def write_index(path,batch,metadata,rows,asset_id,source_sha,start,cap,root):
    arrays=[pa.array([asset_id+':sha256:'+source_sha+':row:'+str(i) for i in range(start,start+batch.num_rows)]),pa.array(range(start,start+batch.num_rows),type=pa.int64())]
    names=['source_occurrence_key','source_row']
    for n in metadata:arrays.append(batch.column(batch.schema.get_field_index(n)));names.append(n)
    for k in ['null','nan','positive_infinity','negative_infinity']:arrays.append(pa.array(rows[k]));names.append('profile_'+k+'_count')
    arrays.append(pa.array(rows['finite_complete']));names.append('finite_complete_profile');table=pa.Table.from_arrays(arrays,names=names)
    sink=pa.BufferOutputStream();pq.write_table(table,sink,compression='zstd',write_page_checksum=True)
    data=sink.getvalue().to_pybytes();bounded_write(path,data,cap,root)
    rb=table.combine_chunks().to_batches()[0]
    return dict(path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),rows=batch.num_rows,source_row_start=start,source_row_stop=start+batch.num_rows,index_logical_sha256=logical_digest(rb))
def context_key(v):
    if isinstance(v,float):
        if not math.isfinite(v):return {'native_float':repr(v)}
        if v==0:return {'native_float':v,'ieee754_hex':struct.pack('<d',v).hex()}
    return v

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args();pp=pathlib.Path(args.plan).resolve();B=pp.parent;plan=json.loads(pp.read_text());raw=json.loads((B/'raw-acquisition-manifest.json').read_text())
    if raw['plan_sha256']!=digest(pp) or raw['raw_bytes']!=474854042 or raw['data_license']!='CC0-1.0' or len(raw['assets'])!=2:raise ValueError('two verified original assets required')
    root=B/'curated/full-native-QC-v1';root.mkdir(exist_ok=False);cap=plan['derived_cap_bytes'];start_time=time.monotonic();out=[]
    for a in plan['assets']:
        if time.monotonic()-start_time>3600:raise ValueError('finite one-hour QC deadline')
        source=pathlib.Path(a['destination']);receipt=next(x for x in raw['assets'] if x['source_asset']['new_asset_id']==a['new_asset_id']);verify_footer(source,a)
        if digest(source)!=receipt['sha256']:raise ValueError('original whole SHA changed')
        assetroot=root/a['new_asset_id'];assetroot.mkdir();pf=pq.ParquetFile(source,page_checksum_verification=True);schema=pf.schema_arrow;features=validate_axis(schema,a)
        if pf.metadata.num_rows!=a['footer_rows'] or pf.num_row_groups!=a['row_groups']:raise ValueError('full original footer count mismatch')
        axis=dict(ordered_feature_names=features,native_schema=[dict(name=f.name,type=str(f.type),nullable=f.nullable) for f in schema],metadata_columns=a['metadata_columns'],source_sha256=receipt['sha256'],feature_units='native author-preprocessed values; no universal physical unit asserted')
        write_new(assetroot/'full-axis-and-schema.json',axis)
        totals={n:dict(null=0,nan=0,positive_infinity=0,negative_infinity=0,finite=0,finite_min=None,finite_max=None) for n in features};contexts={n:collections.Counter() for n in a['metadata_columns']};index=[];digests=[];offset=0;finite_complete=0
        for batch_num,batch in enumerate(pf.iter_batches(batch_size=512,use_threads=False)):
            if time.monotonic()-start_time>3600:raise ValueError('finite one-hour QC deadline; partials retained')
            stats,rows=inspect_profiles(batch,features);finite_complete+=int(rows['finite_complete'].sum())
            for n,record in stats.items():
                for k in ['null','nan','positive_infinity','negative_infinity','finite']:totals[n][k]+=record[k]
                for k,fun in [('finite_min',min),('finite_max',max)]:
                    if record[k] is not None:totals[n][k]=record[k] if totals[n][k] is None else fun(totals[n][k],record[k])
            for n in a['metadata_columns']:
                for v in batch.column(schema.get_field_index(n)).to_pylist():contexts[n][json.dumps(context_key(v),sort_keys=True,allow_nan=False,separators=(',',':'))]+=1
            rec=write_index(assetroot/('row-index-%05d.parquet'%batch_num),batch,a['metadata_columns'],rows,a['new_asset_id'],receipt['sha256'],offset,cap,root);rec['raw_batch_logical_sha256']=logical_digest(batch);index.append(rec);offset+=batch.num_rows
            mutable(B/'live-native-QC.json',dict(updated_utc=NOW(),pid=os.getpid(),phase='full axis native scan and metadata index',asset=a['new_asset_id'],source_rows_processed=offset,expected_rows=a['footer_rows'],derived_bytes=tree_bytes(root),task_admitted_records=0))
        if offset!=a['footer_rows']:raise ValueError('full native row conservation mismatch')
        summary={n:dict(native_distinct_values=len(c),null_rows=c.get('null',0)) for n,c in contexts.items()}
        for n in ['Metadata_pert_dose_micromolar','Metadata_pert_timepoint','Metadata_cell_id','Metadata_pert_type']:
            if n in contexts:summary[n]['exact_native_value_counts']=[dict(value=json.loads(k),rows=v) for k,v in sorted(contexts[n].items())]
        # Independent second full-column decode, plus every typed metadata index readback.
        reread_rows=0
        for i,batch in enumerate(pq.ParquetFile(source,page_checksum_verification=True).iter_batches(batch_size=512,use_threads=False)):
            rec=index[i]
            if logical_digest(batch)!=rec['raw_batch_logical_sha256']:raise ValueError('independent full native value/validity/IEEE bit readback mismatch')
            ix=pathlib.Path(rec['path'])
            if digest(ix)!=rec['sha256'] or ix.stat().st_size!=rec['bytes']:raise ValueError('typed row index file changed')
            table=pq.ParquetFile(ix,page_checksum_verification=True).read().combine_chunks();rb=table.to_batches()[0]
            if logical_digest(rb)!=rec['index_logical_sha256'] or table['source_row'].to_pylist()!=list(range(rec['source_row_start'],rec['source_row_stop'])):raise ValueError('typed metadata/source ordinal readback mismatch')
            for n in a['metadata_columns']:
                original=pa.record_batch([batch.column(schema.get_field_index(n))],names=[n]);copy=pa.record_batch([table[n].chunk(0)],names=[n])
                if logical_digest(original)!=logical_digest(copy):raise ValueError('native context metadata changed')
            reread_rows+=batch.num_rows
        if reread_rows!=offset or digest(source)!=receipt['sha256']:raise ValueError('full original replay count or final whole hash mismatch')
        manifest=dict(schema='mora3.LINCS_Pilot1_full_native_source_QC.v1',completed_utc=NOW(),asset_id=a['new_asset_id'],data_license='CC0-1.0',source_release_entity=a,original_payload=dict(path=str(source),sha256=receipt['sha256'],bytes=source.stat().st_size,rows=offset,columns=len(schema),row_groups=pf.num_row_groups),full_feature_axis=len(features),full_axis_file=str(assetroot/'full-axis-and-schema.json'),full_axis_sha256=digest(assetroot/'full-axis-and-schema.json'),feature_QC=totals,context_QC=summary,finite_complete_profile_rows=finite_complete,profiles_with_native_null_or_nonfinite=offset-finite_complete,row_indexes=index,independent_full_native_and_index_replay_rows=reread_rows,original_values_and_float_bits_preserved=True,source_parent_key=a['source_key'],native_split=None,model_adapter_support=False,task_admitted_records=0,known_gates=['Separate assays, not same cell or paired replicate','Author assigns CP48h versus L1000 24h; no matched-time assumption','Dose is nominal author column label micromolar; native values/sentinels preserved, no conversion','Morphological feature axis has no native image/channel payload','L1000 feature keys are native probe IDs; no automatic gene/protein/sequence annotations','Source available and QC passed are not corpus admission'])
        write_new(assetroot/'manifest.json',manifest);out.append(dict(path=str(assetroot/'manifest.json'),sha256=digest(assetroot/'manifest.json'),rows=offset,feature_axis=len(features),finite_complete_profiles=finite_complete,raw_payload_sha256=receipt['sha256']))
        print('LINCS_FULL_NATIVE_QC',a['new_asset_id'],offset,len(features),finite_complete,flush=True)
    final=dict(schema='mora3.LINCS_Pilot1_two_original_full_native_QC.v1',completed_utc=NOW(),plan=str(pp),plan_sha256=digest(pp),raw_manifest=str(B/'raw-acquisition-manifest.json'),raw_manifest_sha256=digest(B/'raw-acquisition-manifest.json'),components=out,available_native_profile_rows=sum(x['rows'] for x in out),complete_finite_profiles=sum(x['finite_complete_profiles'] for x in out),derived_bytes_before_final_manifest=tree_bytes(root),derived_cap_bytes=cap,task_admitted_records=0,elapsed_seconds=time.monotonic()-start_time)
    write_new(root/'manifest.json',final);mutable(B/'live-native-QC.json',dict(updated_utc=NOW(),pid=None,phase='full native source and index readback complete',manifest=str(root/'manifest.json'),task_admitted_records=0));print('LINCS_FULL_NATIVE_QC_COMPLETE',flush=True)
if __name__=='__main__':main()
