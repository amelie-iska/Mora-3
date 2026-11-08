"""Finite native full STRING scans; immutable source outputs, no corpus admission."""
import argparse,collections,csv,datetime,gzip,hashlib,json,os,pathlib,shutil,time
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from data_sources_fetch import digest
from data_sources_fasta import fasta_records
from data_sources_string_v2 import native_table_equal
from data_sources_string import PHYSICAL_COLUMNS,PHYSICAL_SCHEMA,physical_batch_qc,provider_sequence_qc,count_true

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_new(p,j):
    with p.open('x') as f:json.dump(j,f,indent=2);f.write('\n')
def baseline():
    p=pathlib.Path('outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz').resolve();by_sha=collections.defaultdict(list);accessions=set()
    with gzip.open(p,'rt') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if float(r['Annotation'])<4 or r['Reviewed']!='reviewed':raise ValueError('unqualified UniProtbaseline')
            if r['Entry'] in accessions:raise ValueError('duplicate baselineaccession')
            accessions.add(r['Entry']);by_sha[hashlib.sha256(r['Sequence'].encode()).hexdigest()].append(r['Entry'])
    if len(accessions)!=107778:raise ValueError('baselinecountdrift')
    return by_sha,accessions,dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p),release='2026_03',reviewed_entries=107778,annotation_score_minimum=4,exact_native_sequence_required=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--batch',required=True);ap.add_argument('--mode',choices=['physical','sequences','aliases'],required=True);ap.add_argument('--output-name',required=True);ap.add_argument('--max-rows',type=int);args=ap.parse_args();root=pathlib.Path(args.batch).resolve();out=root/'curated'/args.output_name;out.mkdir(parents=True,exist_ok=False);pa.set_cpu_count(1);pa.set_io_thread_count(1)
    key={'physical':'protein.physical.links.full','sequences':'protein.sequences','aliases':'protein.aliases'}[args.mode];rp=root/'audits/asset-receipts-aria2-string-v1'/('DSE2-STRING-'+key+'.json');r=json.loads(rp.read_text());raw=pathlib.Path(r['path']);cap={'physical':64,'sequences':16,'aliases':8}[args.mode]*1024**3
    if r.get('state')!='verified_available' or not r.get('gzip_crc_full_stream_passed') or raw.stat().st_size!=r['bytes'] or digest(raw)!=r['sha256']:raise ValueError('whole verified native source required')
    status=root/('live-string-native-'+args.output_name+'.json');counts=collections.Counter();payloads=[];derived=0;index=0;started=time.monotonic();source_schema=None;qualified_ref=None;full=False
    def live(phase,error=None):
        status.write_text(json.dumps(dict(updated_utc=now(),pid=os.getpid(),mode=args.mode,phase=phase,source_rows_scanned=index,counts=dict(counts),typed_shards=len(payloads),derived_bytes=derived,elapsed_seconds=time.monotonic()-started,source_rows_per_second=index/max(1,time.monotonic()-started),free_bytes=shutil.disk_usage(root).free,full_source_scan=full,source_connections=0,error=error),indent=2)+'\n')
    def save(table,first,stop):
        nonlocal derived
        if shutil.disk_usage(root).free<200*1024**3+2*1024**3:raise ValueError('free-space reserve')
        p=out/('native-%05d.parquet'%len(payloads));pq.write_table(table,p,compression='zstd',row_group_size=100000)
        decoded=pq.read_table(p)
        if not native_table_equal(table,decoded):raise ValueError('native typed wholepart roundtrip mismatch')
        derived+=p.stat().st_size
        if derived>cap:raise ValueError('derived budget cap; source and partial typed outputs retained')
        payloads.append(dict(path=str(p),rows=table.num_rows,bytes=p.stat().st_size,sha256=digest(p),source_row_start=first,source_row_stop=stop,source_row_span_is_dense=args.mode!='aliases',all_columns_exact_decoded_roundtrip=True));live('full native stream advancing')
    try:
        live('source checksum verified; native stream starting')
        if args.mode=='physical':
            blocks=[];n=0;first=0;source_schema=str(PHYSICAL_SCHEMA)
            with gzip.open(raw,'rb') as f:
                reader=pacsv.open_csv(f,read_options=pacsv.ReadOptions(block_size=16*1024**2,use_threads=False),parse_options=pacsv.ParseOptions(delimiter=' ',quote_char=False,ignore_empty_lines=False),convert_options=pacsv.ConvertOptions(column_types={field.name:field.type for field in PHYSICAL_SCHEMA},strings_can_be_null=False,null_values=[]))
                if reader.schema.names!=PHYSICAL_COLUMNS:raise ValueError('native physical header drift')
                for batch in reader:
                    if args.max_rows is not None:batch=batch.slice(0,min(batch.num_rows,args.max_rows-index))
                    q=physical_batch_qc(batch);counts.update({k:v for k,v in q.items() if k!='channel_nonzero'});counts.update({'nonzero_'+k:v for k,v in q['channel_nonzero'].items()});blocks.append(batch);n+=batch.num_rows;index+=batch.num_rows
                    if n>=1000000:save(pa.Table.from_batches(blocks),first,index);blocks=[];n=0;first=index
                    if args.max_rows is not None and index>=args.max_rows:break
                else:full=True
            if blocks:save(pa.Table.from_batches(blocks),first,index)
        elif args.mode=='sequences':
            qualified,_,qualified_ref=baseline();rows=[];first=0;minlen=None;maxlen=0
            schema=pa.schema([('provider_id',pa.string()),('taxon_id',pa.int64()),('native_header',pa.string()),('native_length',pa.int64()),('native_sequence_sha256',pa.string()),('uppercase_sequence_sha256',pa.string()),('sequence_alphabet_pass',pa.bool_()),('has_stop_symbol',pa.bool_()),('qualified_exact_sequence_accessions',pa.list_(pa.string())),('task_admitted',pa.bool_()),('source_record_index',pa.int64())]);source_schema=str(schema)
            with gzip.open(raw,'rt',encoding='ascii',errors='strict') as f:
                for header,sequence in fasta_records(f):
                    q=provider_sequence_qc(header,sequence,qualified);q['source_record_index']=index;rows.append(q);length=q['native_length'];minlen=length if minlen is None else min(minlen,length);maxlen=max(maxlen,length)
                    counts['native_rows']+=1;counts['native_residues']+=length;counts['sequence_alphabet_pass']+=q['sequence_alphabet_pass'];counts['exact_qualified_sequence_candidate_rows']+=bool(q['qualified_exact_sequence_accessions']);counts['source_rows_with_stop_symbol']+=q['has_stop_symbol'];counts['native_lowercase_rows']+=sequence!=sequence.upper();index+=1
                    if len(rows)>=50000:save(pa.Table.from_pylist(rows,schema=schema),first,index);rows=[];first=index
                    if args.max_rows is not None and index>=args.max_rows:break
                else:full=True
            if rows:save(pa.Table.from_pylist(rows,schema=schema),first,index)
            counts['minimum_native_sequence_length']=minlen;counts['maximum_native_sequence_length']=maxlen
        else:
            _,qualified,qualified_ref=baseline();value_set=pa.array(sorted(qualified));schema=pa.schema([('string_protein_id',pa.string()),('alias',pa.string()),('source',pa.string())]);source_schema=str(schema);origin=collections.Counter()
            with gzip.open(raw,'rb') as f:
                reader=pacsv.open_csv(f,read_options=pacsv.ReadOptions(block_size=8*1024**2,use_threads=False),parse_options=pacsv.ParseOptions(delimiter='\t',quote_char=False,ignore_empty_lines=False),convert_options=pacsv.ConvertOptions(column_types={'#string_protein_id':pa.string(),'alias':pa.string(),'source':pa.string()},strings_can_be_null=False,null_values=[]))
                if reader.schema.names!=['#string_protein_id','alias','source']:raise ValueError('native alias header drift')
                for batch in reader:
                    if args.max_rows is not None:batch=batch.slice(0,min(batch.num_rows,args.max_rows-index))
                    if any(c.null_count for c in batch.columns) or not pc.all(pc.match_substring_regex(batch['#string_protein_id'],r'^[1-9][0-9]*\.\S+$')).as_py() or pc.any(pc.equal(batch['alias'],'')).as_py():raise ValueError('native alias namespace/field violation')
                    counts['native_rows']+=batch.num_rows;counts['missing_provider_origin_rows']+=count_true(pc.equal(batch['source'],''));mask=pc.is_in(batch['alias'],value_set=value_set);selected=pa.Table.from_batches([batch]).filter(mask);count=selected.num_rows;counts['exact_baseline_accession_alias_candidate_rows']+=count
                    if count:
                        selected=selected.append_column('source_row_index',pc.add(pc.indices_nonzero(mask),index));save(selected,index,index+batch.num_rows)
                    for v in pc.value_counts(batch['source']).to_pylist():origin[v['values']]+=v['counts']
                    index+=batch.num_rows
                    if len(payloads)%10==0:live('whole alias stream and exactaccession candidate references advancing')
                    if args.max_rows is not None and index>=args.max_rows:break
                else:full=True
            counts['distinct_native_provider_origin_strings']=len(origin);write_new(out/'native-provider-origin-counts.json',dict(origin))
        if digest(raw)!=r['sha256']:raise ValueError('immutable source changed during scan')
        result=dict(schema='mora3.STRING_native_source_manifest.v1',completed_utc=now(),source_release='STRING12.0',mode=args.mode,source_asset=r,source_receipt=str(rp),source_receipt_sha256=digest(rp),full_source_scan=full,available_source_rows=index,source_quality_pass_rows=counts.get('sequence_alphabet_pass',index),task_admitted_records=0,counts=dict(counts),typed_payloads=payloads,typed_derived_bytes=derived,source_native_schema=source_schema,qualified_baseline=qualified_ref,source_raw_authoritative=True,source_row_order_preserved=True,whole_gzip_CRC_scanned_again=full,provider_namespace='literalNCBItaxon.provider-protein-ID asreleased; provider suffixesareheterogeneous, notgloballyEnsembl; underlyingproviderrevisionnotinferred',sequence_role='nativeFASTArecord withcompleteaxisinraw; exactnativeSHA256/length/header typed, no sequenceannotationqualification fromclusteroralias',physical_role='orientedintegratedphysical-association scoreprofile; computational/integrated/transferred/textminingchannelsareNOTdirectwetbindingtruth ornegative labels',score_units='native0..1000integers separateallchannels; no calibratedprobabilityassertion',alias_role='literalalias/providerorigin+source rowreference; aliasaloneisnotexactsequence, isoformproof, orlicense/annotationtransfer',MINT_curated_collection_reconstructed=False,MINT_missing_artifacts=['originalclu50.tsv50%clustermembership/exactparameters/providersequenceartifact','exactpublicationfilteredpairorderingandclusterheldoutmembership'],original_MINT_seeds_retained=[137,731],global_ID_uniqueness_and_reverse_pair_symmetry_not_verified=True,reader_support=False,corpus_admission_owner='dataset_construction')
        write_new(out/'manifest.json',result);live('complete full native QC' if full else 'bounded probe complete; notfullsource');print('STRING_NATIVE_COMPLETE',args.mode,index,derived,full,flush=True)
    except Exception as e:
        live('stopped; all original and partial derived bytes preserved',type(e).__name__+':'+str(e));raise
if __name__=='__main__':main()
