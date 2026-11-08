"""Full native FASTA source scan and typed full-axis payloads; no corpus admission."""
import argparse
import collections
import datetime
import gzip
import json
import pathlib
import re
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from data_sources_fasta import fasta_records,sequence_qc
from data_sources_fetch import digest

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True);parser.add_argument('--output-dir',required=True);parser.add_argument('--source-family',required=True);parser.add_argument('--derived-cap-bytes',type=int,default=8*1024**3);args=parser.parse_args()
    receipt=pathlib.Path(args.receipt).resolve();source=json.loads(receipt.read_text());path=pathlib.Path(source['path']);out=pathlib.Path(args.output_dir).resolve()
    if source.get('state')!='verified_available' or not source.get('gzip_crc_full_stream_passed'):raise ValueError('whole-source verified receipt required')
    if path.stat().st_size!=source['expected_bytes'] or digest(path)!=source['sha256']:raise ValueError('source integrity failure')
    out.mkdir(parents=True,exist_ok=False);counts=collections.Counter();rows=[];payloads=[];row_bytes=0;index=0;min_length=None;max_length=0;derived=0
    schema=pa.schema([('source_record_id',pa.string()),('source_record_index',pa.int64()),('source_payload',pa.string()),('source_identifier',pa.string()),('native_header',pa.string()),('native_sequence',pa.large_string()),('native_length',pa.int64()),('native_sequence_sha256',pa.string()),('uppercase_sequence_sha256',pa.string()),('sequence_alphabet_pass',pa.bool_()),('native_T_count',pa.int64()),('native_U_count',pa.int64()),('ambiguity_symbol_count',pa.int64()),('explicit_Rfam_ID_in_header',pa.string()),('source_role',pa.string()),('task_admitted',pa.bool_())])
    def flush():
        nonlocal row_bytes,derived
        if not rows:return
        if shutil.disk_usage(out).free<200*1024**3+128*1024**2:raise ValueError('free-space reserve')
        p=out/('native-%05d.parquet'%len(payloads));pq.write_table(pa.Table.from_pylist(rows,schema=schema),p,compression='zstd')
        if pq.read_table(p).to_pylist()!=rows:raise ValueError('typed full-sequence roundtrip mismatch')
        derived+=p.stat().st_size
        if derived>args.derived_cap_bytes:raise ValueError('derived cap; originals retained, output quarantine')
        payloads.append(dict(path=str(p),rows=len(rows),bytes=p.stat().st_size,sha256=digest(p),exact_decoded_roundtrip=True));rows.clear();row_bytes=0
    with gzip.open(path,'rt',encoding='utf-8') as f:
        for header,sequence in fasta_records(f):
            q=sequence_qc(header,sequence);length=q['native_length'];min_length=length if min_length is None else min(min_length,length);max_length=max(max_length,length);family=re.search(r'(?<![A-Za-z0-9])RF[0-9]{5}(?![0-9])',header)
            counts['source_records']+=1;counts['sequence_alphabet_pass']+=q['sequence_alphabet_pass'];counts['full_native_letters']+=length;counts['longer_than_legacy8192']+=length>8192;counts['empty_sequence']+=not sequence;counts['native_T_letters']+=q['native_T_count'];counts['native_U_letters']+=q['native_U_count'];counts['ambiguity_symbols']+=q['ambiguity_symbol_count'];counts['explicit_Rfam_header_ID']+=bool(family)
            rows.append(dict(source_record_id=args.source_family+':'+source['id']+':'+str(index),source_record_index=index,source_payload=str(path),source_identifier=q['source_identifier'],native_header=header,native_sequence=sequence,native_length=length,native_sequence_sha256=q['native_sequence_sha256'],uppercase_sequence_sha256=q['uppercase_sequence_sha256'],sequence_alphabet_pass=q['sequence_alphabet_pass'],native_T_count=q['native_T_count'],native_U_count=q['native_U_count'],ambiguity_symbol_count=q['ambiguity_symbol_count'],explicit_Rfam_ID_in_header=family.group(0) if family else None,source_role='released RNAfamily/reference-sequence evidence; exact family/region join requires source metadata',task_admitted=False));index+=1;row_bytes+=len(sequence)+len(header)
            if len(rows)>=50000 or row_bytes>=64*1024**2:flush();print('FASTA_FULL_SCAN_PROGRESS',index,derived,flush=True)
    flush()
    if digest(path)!=source['sha256']:raise ValueError('source changed during scan')
    result=dict(schema='mora3.full_native_fasta_source_QC.v1',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_family=args.source_family,source_receipt=str(receipt),source_asset=source,source_receipt_sha256=digest(receipt),whole_gzip_stream_scanned=True,counts=dict(counts),available_records=index,sequence_quality_pass_records=counts['sequence_alphabet_pass'],task_admitted_records=0,native_length_range=[min_length,max_length],full_native_sequence_axis_preserved=True,no_T_U_conversion=True,source_sequence_case_preserved=True,all_original_headers_preserved=True,record_unit='native FASTA occurrence; no crossfile uniqueness or independentexperiment assertion',native_payloads=payloads,derived_bytes=derived,source_evidence_limits=['RNAfamily hits/reference sequences are not independently validated functional or structural measurements','source headers preserve original accession/version/region; no fabricated genome/cell/perturbation links','missing labels are not negatives','crosssource identity/homology/splits and taskadmission reserved for corpus owner','source-native payload reader verified; model/corpus adapter false'],grouping_keys=['source full header and native identifier','native exact sequenceSHA256 and explicitly uppercase-transformed groupingSHA256','explicit source family/region IDs if supplied'],reader_support=False,corpus_admission_owner='dataset_construction')
    with (out/'manifest.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print('FASTA_NATIVE_MANIFEST',str(out/'manifest.json'),index,counts['sequence_alphabet_pass'],flush=True)
if __name__=='__main__':main()
