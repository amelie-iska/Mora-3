"""Finite threshold-detected protein grouping in new outputs; no index/source rewrite."""
from pathlib import Path
import collections,csv,datetime,gzip,hashlib,json,os,subprocess,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path.cwd();sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import annotation_score
from data_intake_homology_partition import parse_partition
B=Path('outputs/data-intake/20261101-qualified-protein-linclust-v1');B.mkdir(exist_ok=False);(B/'code').mkdir();(B/'native').mkdir();start=time.perf_counter();CAP=8*2**30
for name in ['data_intake_group_qualified_proteins.py','data_intake_homology_partition.py']:(B/'code'/name).write_bytes((R/'scripts'/name).read_bytes())
def ref(p,**kw):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest(),**kw)
def live(stage,**kw):
 (B/'live-build.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,**kw),indent=2)+'\n');print(stage,kw,round(time.perf_counter()-start,2),flush=True)
try:
 p=Path('outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz');assert ref(p)['sha256']=='e9bd53c005179cc68664f920f833273b241afb7196806ca512866869674d5133';entries={};sequences={};symbols=collections.Counter()
 with gzip.open(p,'rt') as f:
  for row in csv.DictReader(f,delimiter='\t'):
   score=annotation_score(row['Annotation']);assert row['Reviewed']=='reviewed' and row['Entry'] not in entries;seq=row['Sequence'];digest=hashlib.sha256(seq.encode()).hexdigest();assert sequences.setdefault(digest,seq)==seq;entries[row['Entry']]=dict(sequence_sha256=digest,annotation_score=score,native_length=len(seq));symbols.update(seq)
 assert len(entries)==107778 and len(sequences)==104176 and max(map(len,sequences.values()))<=65535
 fasta=B/'qualified-canonical-distinct-native.fasta'
 with fasta.open('x') as f:
  for digest,seq in sorted(sequences.items()):f.write('>seq_'+digest+'\n'+seq+'\n')
 tool=Path('/home/iska/Documents/amelie/bio/MMseqs2/mmseqs/bin/mmseqs');version=subprocess.run([str(tool),'version'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True).stdout.strip();helptext=subprocess.run([str(tool),'easy-linclust','-h'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,check=True).stdout;(B/'native-help.txt').open('x').write(helptext)
 output=B/'native/result';tmp=B/'native/new-tmp';argv=[str(tool),'easy-linclust',str(fasta),str(output),str(tmp),'--threads','4','--gpu','0','--min-seq-id','0.3','-c','0.8','--cov-mode','0','--seq-id-mode','0','--alignment-mode','3','--cluster-mode','0','--kmer-per-seq','21','--split-memory-limit','4G','--max-seq-len','65535','--dbtype','1','--shuffle','0','--createdb-mode','0','--remove-tmp-files','0','--force-reuse','0']
 plan=dict(schema='mora3.QualifiedProteinGroupingPlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=20261031,tool_random_seed_not_exposed=True,deterministic_input_order='sortedexactnativeSHA256;shuffle0',source=ref(p),qualified_current_entries=107778,distinct_full_native_sequences=104176,source_sequence_letters_preserved=True,native_symbols=dict(symbols),new_input_fasta=ref(fasta),tool=ref(tool),tool_version=version,argv=argv,derived_cap_bytes=CAP,minimum_free_bytes=200*2**30,CPU_threads=4,GPU_used=False,existing_indices_modified=False,all_new_tmp_retained=True,finite_timeout_seconds=1800,interpretation='Nativeheuristicdetected30percentidentity/80percentqueryandtargetcoveragegroups;notexhaustivehomologyorbiologicalfunctionequivalence',source_MINT_clu50_not_reconstructed=True)
 (B/'build-plan.json').open('x').write(json.dumps(plan,indent=2)+'\n');st=os.statvfs('.');assert st.f_bavail*st.f_frsize>=200*2**30
 log=(B/'native-mmseqs.log').open('x');proc=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);live('native qualified-baseline threshold grouping',native_pid=proc.pid,full_native_sequences=104176)
 try:code=proc.wait(timeout=1800)
 except subprocess.TimeoutExpired:
  os.killpg(proc.pid,__import__('signal').SIGTERM);raise RuntimeError('finite1800secondqualifiedgroupauditcapreached;oursubprocessstoppedandalloutputsretained')
 finally:log.close()
 assert code==0,('nativeMMseqsexit',code)
 cluster=Path(str(output)+'_cluster.tsv');mapping=parse_partition(cluster.read_text(),{'seq_'+x for x in sequences});sizes=collections.Counter(mapping.values());allrows=[]
 for accession,e in sorted(entries.items()):
  native='seq_'+e['sequence_sha256'];representative=mapping[native];allrows.append(dict(accession=accession,canonical_sequence_sha256=e['sequence_sha256'],canonical_native_length=e['native_length'],current_reviewed_annotation_score=e['annotation_score'],current_baseline_release='2026_03',native_input_id=native,native_group_representative=representative,detected_group_id='qualified_MMseqs_linclust30cov80:'+representative[4:],distinct_native_sequences_in_detected_group=sizes[representative],group_evidence='MMseqs2_native_heuristic_threshold_group;not_molecular_identity_or_annotation_transfer',homology_exhaustive=False,new_task_admission=False))
 out=B/'qualified-entry-detected-homology-groups.parquet';pq.write_table(pa.Table.from_pylist(allrows),out,compression='zstd');assert pq.read_table(out).to_pylist()==allrows;assert len({r['accession'] for r in allrows})==107778;used=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());assert used<=CAP
 result=dict(schema='mora3.QualifiedProteinDetectedHomologyGroups/v1',status='native_partition_and_full_qualified_entry_axis_verified',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(p),plan=ref(B/'build-plan.json'),tool_version=version,native_cluster_tsv=ref(cluster),membership=ref(out,rows=len(allrows)),qualified_entries=107778,distinct_full_native_sequences=104176,detected_native_groups=len(sizes),singleton_groups=sum(n==1 for n in sizes.values()),nonsingleton_groups=sum(n>1 for n in sizes.values()),native_sequences_in_nonsingleton_groups=sum(n for n in sizes.values() if n>1),largest_native_group=max(sizes.values()),native_size_histogram=dict(collections.Counter(sizes.values())),all104176nativeIDs_unique_complete_partition=True,all107778qualified_accessions_exact_sequence_bound=True,source_full_lengths_case_and_symbols_unchanged=True,all_membership_Parquet_roundtrip=True,derived_output_bytes=used,no_existing_source_or_index_changes=True,tmp_files_retained=True,current_master_effective_groups_not_rewritten=True,new_corpus_tasks=0,new_protein_sequences=0,new_independent_experiments=0,annotation_or_score_transfer=False,source_MINT_clu50_or_full_UniRef_replacement=False,interpretation='Detectedheuristicthresholdgroupsonly;nativeMMseqsminimumidentity30percentandalignmentcoverage80percentbothsequences;notpairwiseidentityallmembersandexhaustivehomologyunproven',homology_heldout_still_unresolved=True,global_leakage_safe=False,current_train_only_protein_split_unchanged=True,optimizer_steps=0,code_refs=[ref(p) for p in (B/'code').iterdir()],seconds=time.perf_counter()-start)
 (B/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');live('qualified native grouping complete',groups=len(sizes),entries=len(allrows))
except BaseException as exc:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
