"""Finite original RNAcentral native-prior construction with read-only prior reuse."""
from pathlib import Path
import sys,json,hashlib,datetime,sqlite3,collections,time,shutil,os
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_rnacentral_prior import prior_task,ALPHABET
from data_intake_nucleotide_prior import prior_identity,sequence_groups
from data_intake_corpus import CorpusTask
O=R/'outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1';CAP=8*2**30
H=R/'outputs/data-sources/20261031-legacy-mint-v1/handoff-RNAcentral28-full-native-verified-v1.json'
OLD=R/'outputs/data-intake/20261101-nucleotide-corpus-v1/full-v1/task_identity_and_group_index.sqlite3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p;assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p

def main():
 start=time.perf_counter();assert not O.exists() and shutil.disk_usage(R).free>208*2**30;O.mkdir();(O/'code').mkdir();(O/'shards/train').mkdir(parents=True);(O/'effective-groups').mkdir();(O/'occurrences').mkdir();(O/'quarantines').mkdir()
 for n in [Path(__file__).name,'data_intake_rnacentral_prior.py','data_intake_nucleotide_prior.py']:(O/'code'/n).write_bytes((R/'scripts'/n).read_bytes())
 assert sha(H)=='e8694fab174e9df1bb99246cd95e5f23ca41f10fbfe20222b1168be7e12a7567';h=json.loads(H.read_text());selected=[];manifest_refs=[]
 for component,c in h['components'].items():
  mp=Path(c['native_manifest']);assert sha(mp)==c['native_manifest_sha256'];m=json.loads(mp.read_text());manifest_refs.append(ref(mp));files=m['native_payloads'];indices={round((len(files)-1)*i/3) for i in range(4)};longest=(-1,None)
  for i,file in enumerate(files):
   md=pq.ParquetFile(file['path']).metadata;col=pq.ParquetFile(file['path']).schema_arrow.names.index('native_length');maximum=max(md.row_group(j).column(col).statistics.max for j in range(md.num_row_groups))
   if maximum>longest[0]:longest=(maximum,i)
  assert longest[0]==m['native_length_range'][1];indices.add(longest[1]);selected.extend(dict(component=component,native_payload=file,source_manifest=ref(mp),footer_guided_longest_shard=(i==longest[1])) for i,file in enumerate(files) if i in indices)
 assert sum(s['native_payload']['rows'] for s in selected)<=500000
 (O/'build-plan.json').open('x').write(json.dumps(dict(schema='mora3.RNAcentralNativeCorpusPilotPlan/v1',source_handoff=ref(H),selection='fourcomplete-shardquantileranksplusglobalmaximumlengthshardforeachcomponent;notrepresentativepopulationestimate',selected=selected,maximum_source_rows=500000,derived_cap_bytes=CAP,minimum_free_bytes=200*2**30,seed=0,budget_reservation='outputs/monitoring/budget-20261101T034455Z-bounded-RNA-corpus-and-MINT-source-pilots-v1.json',existing_nucleotide_index=dict(path=str(OLD.relative_to(R)),mode='ro',bytes=OLD.stat().st_size),rights='officialCC0release28;derivativeAGPLnottransferred',split='trainonly;noartificialheldouts',model_adapter_supported=False),indent=2)+'\n')
 old=sqlite3.connect('file:'+str(OLD)+'?mode=ro',uri=True);dbp=O/'native-task-and-source-group-index.sqlite3';db=sqlite3.connect(dbp);db.executescript('CREATE TABLE sequences(native_sha TEXT PRIMARY KEY,task_id TEXT NOT NULL,reused_prior INTEGER NOT NULL);CREATE TABLE task_groups(task_id TEXT NOT NULL,group_id TEXT NOT NULL,PRIMARY KEY(task_id,group_id)) WITHOUT ROWID;CREATE TABLE identifiers(source_identifier TEXT NOT NULL,native_sha TEXT NOT NULL,PRIMARY KEY(source_identifier,native_sha)) WITHOUT ROWID;');db.commit();buffer=[];bufferbytes=0;shards=[];lookup={};occurrences=[];quarantines=[];counts=collections.Counter();bycomponent=collections.defaultdict(collections.Counter);lengths=[];produced=0;mutations=collections.Counter()
 def reserve():assert produced+dbp.stat().st_size<CAP and shutil.disk_usage(R).free>200*2**30,'pilot derivedcap/free-reserve gate'
 def flush():
  nonlocal bufferbytes,produced
  if not buffer:return
  p=O/'shards/train'/f'{len(shards):05d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==buffer
  for row in back:
   payload=row['native_payload'];assert hashlib.sha256(payload).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(payload).canonical_bytes()==payload;d=json.loads(payload);assert set(d['conditions'])=={'prior_domain','source_organism'} and d['conditions']['source_organism'] is None;assert set(d['targets'])=={'source_native_sequence'};assert len(d['targets']['source_native_sequence'])==row['full_axis_values'];assert hashlib.sha256(d['targets']['source_native_sequence'].encode()).hexdigest()==json.loads(row['admission_receipt_json'])['native_sequence_sha256']
  shards.append(ref(p,rows=len(back),split='train',source_family='RNAcentral-reference-priors'));produced+=p.stat().st_size;buffer.clear();bufferbytes=0;reserve()
 for number,selection in enumerate(selected):
  file=selection['native_payload'];p=checked(file);rows=pq.read_table(p).to_pylist();assert len(rows)==file['rows'];component=selection['component'];occ=[];bad=[]
  for row_index,row in enumerate(rows):
   seq=row['native_sequence'];digest=hashlib.sha256(seq.encode()).hexdigest();assert digest==row['native_sequence_sha256'] and len(seq)==row['native_length'];assert hashlib.sha256(seq.upper().encode()).hexdigest()==row['uppercase_sequence_sha256'];assert row['source_identifier']==row['native_header'].split()[0];valid=not(set(seq)-ALPHABET) and bool(seq);assert valid==row['sequence_alphabet_pass'];counts['source_rows']+=1;bycomponent[component]['source_rows']+=1;counts['source_native_values']+=len(seq)
   source_ref=dict(native_payload=str(p.relative_to(R)),native_payload_sha256=file['sha256'],native_row_index=row_index,original_record_index=row['source_record_index'],original_FASTA=row['source_payload'],source_identifier=row['source_identifier'],native_header=row['native_header'],source_status=component,source_release='RNAcentral28.0',source_rights='CC0-1.0')
   if not valid:
    counts['quarantined_occurrences']+=1;bycomponent[component]['quarantined_occurrences']+=1;bad.append(dict(source_ref_json=json.dumps(source_ref,sort_keys=True),cause='nonstandard_native_source_symbol;originalretained;norepair',nonstandard_symbols_json=json.dumps(sorted(set(seq)-ALPHABET)),native_sequence_sha256=digest,native_length=len(seq)));continue
   counts['quality_pass_occurrences']+=1;bycomponent[component]['quality_pass_occurrences']+=1;lengths.append(len(seq));sg=sequence_groups(seq);existing=db.execute('SELECT task_id,reused_prior FROM sequences WHERE native_sha=?',(digest,)).fetchone();first=existing is None
   if first:
    rfam_task=prior_identity(seq,'RNA_family_reference_native_symbols',None);reuse=old.execute('SELECT 1 FROM tasks WHERE task_id=? AND source_family=?',(rfam_task,'Rfam-native-reference-priors')).fetchone() is not None
    # Actual old task source family is checked below once, never inferred from headers.
    if not reuse:
     candidate=old.execute('SELECT source_family FROM tasks WHERE task_id=?',(rfam_task,)).fetchone()
     if candidate is not None:
      assert candidate[0].startswith('Rfam');reuse=True
    tid=rfam_task if reuse else prior_identity(seq,'RNA_reference_native_symbols',None);db.execute('INSERT INTO sequences VALUES (?,?,?)',(digest,tid,int(reuse)));counts['distinct_native_sequences']+=1;counts['distinct_reused_existing_Rfam_tasks']+=reuse;counts['distinct_new_task_views']+=not reuse
   else:tid,reuse=existing;counts['duplicate_native_occurrences_within_pilot']+=1
   groups=['nucleotide_native:'+sg['native_sha256'],'nucleotide_upper:'+sg['uppercase_sha256'],'nucleotide_derived_RC_baseorder:'+sg['reverse_complement_base_order_group'],'rnacentral_native_identifier:'+row['source_identifier']];db.executemany('INSERT OR IGNORE INTO task_groups VALUES (?,?)',[(tid,x) for x in groups]);db.execute('INSERT OR IGNORE INTO identifiers VALUES (?,?)',(row['source_identifier'],digest));counts['occurrences_reusing_existing_Rfam_task']+=bool(reuse)
   if first and not reuse:
    task=prior_task(seq,source_ref=source_ref);payload=task.canonical_bytes();payloadsha=hashlib.sha256(payload).hexdigest()
    if mutations[component]<32 or len(seq)>8192 and mutations[component]<48:
     changed=prior_task('A',source_ref=dict(source_identifier='MUTATED',header='MUTATED',status='MUTATED'));assert task.input_bytes()==changed.input_bytes();mutations[component]+=1
    buffer.append(dict(task_id=tid,record_id=row['source_record_id'],source_family='RNAcentral-reference-priors',source_revision='officialRNAcentral28.0active+inactive',source_sha256=file['sha256'],task_name='source_native_nucleotide_sequence_prior',native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=payloadsha,split='train',split_policy='referenceprior_trainonly;homology_andgenuineheldoutunresolved',group_ids_json=json.dumps(groups),evidence_class='releasedRNAreference_sequenceprior;notnewindependentexperiment',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=source_ref,native_sequence_sha256=digest,exactfullaxis=True,source_case_T_U_preserved=True,source_license='CC0-1.0',task_admission='eligible_referenceprior;master_unionindependentauditpending')),missing_context_json=json.dumps(json.loads(payload)['missing']),full_axis_values=len(seq),condition_context_json=json.dumps(dict(source_status_notactivitytruth=True,sourceIDsheadersauditonly=True,all_unlisted_annotations_unknown=True))));bufferbytes+=len(payload)
    if len(buffer)>=2048 or bufferbytes>=8*2**20:flush()
   occ.append(dict(source_ref_json=json.dumps(source_ref,sort_keys=True),task_id=tid,task_already_in_Rfam_corpus=bool(reuse),first_occurrence_in_pilot=first,native_sequence_sha256=digest,uppercase_sequence_sha256=sg['uppercase_sha256'],derived_RC_baseorder_sha256=sg['reverse_complement_base_order_group'],derived_RC_is_molecular_identity=False,native_length=len(seq),source_status_is_not_negative_label=True))
  for name,data,ledger in [('occurrences',occ,occurrences),('quarantines',bad,quarantines)]:
   if data:
    q=O/name/f'{number:03d}.parquet';pq.write_table(pa.Table.from_pylist(data),q,compression='zstd');assert pq.read_table(q).to_pylist()==data;ledger.append(ref(q,rows=len(data)));produced+=q.stat().st_size
  db.commit();reserve();print('RNACENTRAL_PILOT_SHARD',number+1,'source',counts['source_rows'],'newtasks',counts['distinct_new_task_views'],'reusedRfam',counts['distinct_reused_existing_Rfam_tasks'],'seconds',round(time.perf_counter()-start,2),flush=True)
 flush();db.commit()
 for shard in shards:
  taskids=[row['task_id'] for row in pq.read_table(checked(shard),columns=['task_id']).to_pylist()];gr=[dict(task_id=tid,effective_groups_json=json.dumps([r[0] for r in db.execute('SELECT group_id FROM task_groups WHERE task_id=? ORDER BY group_id',(tid,))])) for tid in taskids];p=O/'effective-groups'/Path(shard['path']).name;pq.write_table(pa.Table.from_pylist(gr),p,compression='zstd');assert pq.read_table(p).to_pylist()==gr;lookup[shard['path']]=ref(p,rows=len(gr));produced+=p.stat().st_size
 identifier_conflicts=db.execute('SELECT count(*) FROM (SELECT source_identifier FROM identifiers GROUP BY source_identifier HAVING count(*)>1)').fetchone()[0];group_edges=db.execute('SELECT count(*) FROM task_groups').fetchone()[0];old.close();db.close();reserve();counts=dict(counts);counts.setdefault('quarantined_occurrences',0);assert counts['source_rows']==counts['quality_pass_occurrences']+counts['quarantined_occurrences'];assert sum(s['rows'] for s in shards)==counts['distinct_new_task_views'];result=dict(schema='mora3.RNAcentralNativePriorPilotAdmission/v1',status='source_valid_native_tasks_serialized;independentmasterunionpending',source_handoff=ref(H),source_manifests=manifest_refs,source_whole_QC_inherited_from_verified_handoff=True,selection='completequantileandlongestshards;notunbiasedpopulationestimate',counts=counts,counts_by_component={k:dict(v) for k,v in bycomponent.items()},native_length_range=[min(lengths),max(lengths)],longer_than8192=sum(x>8192 for x in lengths),all_fullaxis_payload_and_Parquet_roundtrips=True,all_exact_native_source_row_SHA_and_lengths_recomputed=True,all_task_conditions_allowlisted_and_header_status_hash_length_excluded=True,actual_target_and_metadata_mutation_checks=dict(mutations),existing_nucleotide_index_read_only=True,existing_exact_Rfam_prior_duplicates_not_serialized_again=True,old_Rfam_payloads_and_groups_not_rewritten=True,all_source_occurrence_group_edges=group_edges,identifier_multiple_native_sequence_groups=identifier_conflicts,all_URS_collisions_retained_without_forcedmerge=True,shards=shards,task_shard_bytes=sum(s['bytes'] for s in shards),effective_group_lookup=lookup,occurrences=occurrences,quarantines=quarantines,new_identity_and_group_index=ref(dbp),total_generated_logical_bytes=produced+dbp.stat().st_size,derived_cap_bytes=CAP,split_counts={'train':counts['distinct_new_task_views']},model_adapter_supported=False,master_corpus_updated=False,mixture_weight=None,full_source_release_count_discrepancy=h['published_release_count_discrepancy'],unknown_homology_taxonomy_heldout=True,source_status_is_not_activity_label=True,new_independent_experiments=0,optimizer_steps=0,seconds=time.perf_counter()-start)
 (O/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',json.dumps({key:result[key] for key in ['counts','native_length_range','task_shard_bytes','total_generated_logical_bytes','seconds']}),flush=True)
if __name__=='__main__':main()
