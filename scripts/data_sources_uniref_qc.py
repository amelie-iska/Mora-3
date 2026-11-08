"""Finite full-native UniRef scans of original checksum-verified source shards."""
import argparse
import collections
import csv
import datetime
import gzip
import hashlib
import json
import pathlib
import pyarrow as pa
import pyarrow.parquet as pq
from data_sources_uniref import qc_record
from data_sources_fetch import digest

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write_new(path,data):
    with path.open('x') as f:json.dump(data,f,indent=2);f.write('\n')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',required=True);parser.add_argument('--manifest-name',required=True);parser.add_argument('--max-assets',type=int,default=130);args=parser.parse_args()
    root=pathlib.Path(args.batch).resolve();out=root/'curated/uniref-native-qc-v1';out.mkdir(parents=True,exist_ok=True)
    baseline=pathlib.Path('outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz').resolve();qualified={}
    with gzip.open(baseline,'rt') as f:
        for r in csv.DictReader(f,delimiter='\t'):
            if float(r['Annotation'])<4 or r['Reviewed']!='reviewed':raise ValueError('unqualified baseline')
            qualified[r['Entry']]=hashlib.sha256(r['Sequence'].encode()).hexdigest()
    if len(qualified)!=107778:raise ValueError('baseline accession count drift')
    baseline_ref=dict(path=str(baseline),sha256=digest(baseline),entries=len(qualified),release='2026_03',identity_policy='representative exact accessionANDexactsequenceSHA256 required; underlying UniRef biological release unproven; candidate only')
    refs=sorted((root/'audits/asset-receipts-fast-uniref50-v1').glob('*.json'))[:args.max_assets];results=[]
    schema=pa.schema([('source_record_id',pa.string()),('source_payload',pa.string()),('source_row_index',pa.int64()),('cluster_id',pa.string()),('sequence_sha256',pa.string()),('native_sequence_length',pa.int64()),('seed_id',pa.string()),('representative_member_id',pa.string()),('representative_accessions',pa.list_(pa.string())),('qualified_exact_accessions',pa.list_(pa.string())),('qualified_mismatch_accessions',pa.list_(pa.string())),('QC_json',pa.string()),('disposition',pa.string())])
    for receipt in refs:
        a=json.loads(receipt.read_text())
        if a.get('state')!='verified_available':continue
        target=out/a['id'];manifest=target/'manifest.json'
        if manifest.exists():results.append(json.loads(manifest.read_text()));continue
        path=pathlib.Path(a['path'])
        if path.stat().st_size!=a['expected_bytes'] or digest(path)!=a['expected_sha256']:raise ValueError('source integrity failure')
        target.mkdir(exist_ok=False);pf=pq.ParquetFile(path);counts=collections.Counter();ids=set();hashes=set();minimum=None;maximum=0;dates=collections.Counter();rows=[];payloads=[];index=0
        def flush():
            if not rows:return
            p=target/('candidate-or-quarantine-%04d.parquet'%len(payloads));table=pa.Table.from_pylist(rows,schema=schema);pq.write_table(table,p,compression='zstd')
            if pq.read_table(p).to_pylist()!=rows:raise ValueError('typed QC reference roundtrip failure')
            payloads.append(dict(path=str(p),rows=len(rows),bytes=p.stat().st_size,sha256=digest(p)));rows.clear()
        for batch in pf.iter_batches(batch_size=16):
            for raw in batch.to_pylist():
                q=qc_record(raw,qualified);length=q['native_sequence_length'];minimum=length if minimum is None else min(minimum,length);maximum=max(maximum,length);dates[raw['updated']]+=1
                counts['native_cluster_rows']+=1;counts['source_member_ID_occurrences']+=raw['member_count'];counts['representative_sequence_bases']+=length
                for key in ['sequence_integrity_pass','sequence_length_match','sequence_alphabet_pass','sequence_xxh128_match','sequence_crc64_format_pass','cluster_metadata_integrity_pass','source_GO_ID_syntax_pass']:counts[key]+=bool(q[key])
                for key,value in q['cluster_flags'].items():counts['cluster_'+key]+=value
                counts['rep_is_seed']+=bool(raw['rep_is_seed']);counts['source_GO_metadata_entries']+=q['source_GO_metadata_count'];counts['qualified_representative_exact_sequence_rows']+=bool(q['qualified_representative_exact_sequence_accessions']);counts['qualified_representative_sequence_mismatch_rows']+=bool(q['qualified_representative_sequence_mismatch_accessions']);counts['within_shard_duplicate_cluster_ID_rows']+=raw['id'] in ids
                ids.add(raw['id']);hashes.add(q['native_sequence_sha256'])
                quality=q['sequence_integrity_pass'] and q['cluster_metadata_integrity_pass'] and q['source_GO_ID_syntax_pass'];counts['all_source_QC_components_pass']+=quality
                if not quality or q['qualified_representative_exact_sequence_accessions'] or q['qualified_representative_sequence_mismatch_accessions']:
                    rows.append(dict(source_record_id=a['source_repo']+':'+a['source_revision']+':'+a['id']+':'+str(index),source_payload=str(path),source_row_index=index,cluster_id=raw['id'],sequence_sha256=q['native_sequence_sha256'],native_sequence_length=length,seed_id=raw['seed_id'],representative_member_id=raw['rep_member_id'],representative_accessions=raw['rep_accessions'],qualified_exact_accessions=q['qualified_representative_exact_sequence_accessions'],qualified_mismatch_accessions=q['qualified_representative_sequence_mismatch_accessions'],QC_json=json.dumps(q,sort_keys=True),disposition='quarantined_source_component' if not quality else 'exact_representative_baseline_join_candidate' if q['qualified_representative_exact_sequence_accessions'] else 'qualified_accession_sequence_mismatch'))
                index+=1
                if len(rows)>=1000:flush()
        flush()
        if index!=pf.metadata.num_rows:raise ValueError('native footer/fullscan count mismatch')
        result=dict(schema='mora3.uniref_native_shard_QC.v1',completed_utc=now(),source_asset=a,source_schema=str(pf.schema_arrow),source_footer_rows=pf.metadata.num_rows,full_row_scan=True,all_original_columns_decoded=True,counts=dict(counts),available_native_cluster_rows=index,quality_pass_native_cluster_rows=counts['all_source_QC_components_pass'],task_admitted_rows=0,source_native_payload=str(path),native_sequence_length_range=[minimum,maximum],distinct_cluster_IDs_within_shard=len(ids),distinct_representative_sequences_within_shard=len(hashes),source_updated_date_histogram=dict(dates),typed_candidate_or_quarantine_payloads=payloads,qualified_baseline=baseline_ref,sequence_crc64_status='source16hex syntax checked; not recomputed',sequence_xxh128_status='actual unseeded xxh128 recomputed for every full native sequence',biological_UniRef_release_unknown=True,source_go_metadata_is_not_evidence_qualified_annotation=True,member_or_seed_annotation_transfer=False,original_member_arrays_roles_and_cluster_indices_preserved=True,new_UniProt_proteins_admitted=0,reader_support='source-native Parquet reader verified; corpus/model adapter false')
        write_new(manifest,result);results.append(result);print('UNIREF_FULL_SHARD_QC',a['id'],index,result['quality_pass_native_cluster_rows'],flush=True)
    result=dict(schema='mora3.uniref_source_native_manifest.v1',created_utc=now(),inventory_id='INV-006',source_repo='ConvergeBio/uniref50',source_revision='4168e14ecc46125bb382c9f86f5ecab9cb2d4570',license='CC-BY-4.0',registered_source_shards=130,verified_full_scanned_shards=len(results),full130shard_source_acquired_and_scanned=len(results)==130,available_native_cluster_rows=sum(x['available_native_cluster_rows'] for x in results),quality_pass_native_cluster_rows=sum(x['quality_pass_native_cluster_rows'] for x in results),task_admitted_rows=0,all_cluster_ids_cross_shard_unique_not_yet_verified=True,shards=results,qualified_baseline=baseline_ref,scope='finite receipt snapshot at process launch; no download watcher or mutation',disposition='source-native sequence and cluster metadata candidates; biological release and score gate not inferred for cluster members',source_cluster_count_is_not_independent_experiments=True,corpus_admission_owner='dataset_construction')
    write_new(out/args.manifest_name,result);print('UNIREF_MANIFEST',str(out/args.manifest_name),result['available_native_cluster_rows'],flush=True)
if __name__=='__main__':main()
