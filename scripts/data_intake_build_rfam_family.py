"""Full exact-key source census and bounded, family-diverse Rfam annotation priors."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sqlite3,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'scripts'))
from data_intake_corpus import CorpusTask,canonical
from data_intake_nucleotide_prior import prior_identity,sequence_groups
from data_intake_rfam_family import family_task
B=Path('outputs/data-intake/20261101-rfam-family-annotation-v1');O=B/'curated-v1'
START=time.perf_counter();PER_FAMILY=8;CAP=4*2**30

def rec(path,**extra):
 p=Path(path);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest(),**extra)
def check(e):
 p=Path(e['path']);got=rec(p);assert got['bytes']==e['bytes'] and got['sha256']==e['sha256']
 if 'rows' in e:assert pq.ParquetFile(p).metadata.num_rows==e['rows']
 return p
def write_json(p,d):p.open('x').write(json.dumps(d,indent=2)+'\n')
def live(stage,**extra):
 (O/'live-build.json').write_text(json.dumps(dict(pid=os.getpid(),updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stage=stage,elapsed_seconds=time.perf_counter()-START,**extra),indent=2)+'\n');print(stage,extra,round(time.perf_counter()-START,2),flush=True)
def reserve():
 st=os.statvfs('.');assert st.f_bavail*st.f_frsize>=200*2**30;assert sum(p.stat().st_size for p in O.rglob('*') if p.is_file())<CAP

def run():
 B.mkdir(exist_ok=False);O.mkdir();(O/'join-audits').mkdir();(O/'shards').mkdir();(O/'effective-groups').mkdir();(O/'code').mkdir()
 for name in ['data_intake_build_rfam_family.py','data_intake_rfam_family.py']:(O/'code'/name).write_bytes((ROOT/'scripts'/name).read_bytes())
 handoff=Path('outputs/data-sources/20261031-legacy-mint-v1/handoff-rfam-native-context-v1.json');h=json.loads(handoff.read_text());context_path=Path(h['manifest']);assert hashlib.sha256(context_path.read_bytes()).hexdigest()==h['manifest_sha256'];context=json.loads(context_path.read_text())
 native_path=Path('outputs/data-sources/20261031-legacy-mint-v1/curated/rfam15.1-native-fasta-v1/manifest.json');native=json.loads(native_path.read_text());family_ref=context['family_catalog'];clan_ref=context['clan_name_bindings'];families={r['family_accession']:r for r in pq.read_table(check(family_ref)).to_pylist()};clans=collections.defaultdict(set)
 for r in pq.read_table(check(clan_ref)).to_pylist():
  assert r['exact_seed_family_name_bound'] and r['candidate_family_accessions']==[r['family_accession']];clans[r['family_accession']].add(r['clan_id'])
 write_json(B/'build-plan.json',dict(schema='mora3.RfamFamilyIntakePlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=20261031,new_download_bytes=0,derived_cap_bytes=CAP,minimum_free_bytes=200*2**30,context_handoff=rec(handoff),context_manifest=rec(context_path),native_manifest=rec(native_path),full_key_cardinality_census=True,selection='first8distinctnativeidentifierkeysperfamilyinsourceorder;provisionaltrainingprioronly;notperformancebalancedheldout',selection_per_family=PER_FAMILY,predictor_exclusion='CMscores_Evalues_truncation_coordinates_headers_sourceIDs_names_citations_targetlength',evidence='sourceassignedreferencefamilyprior;notindependentexperimentalRNAstructure/function',model_adapter_supported=False))
 regions=collections.Counter();family_keysets=collections.defaultdict(set);selected=set();region_count=0
 for i,e in enumerate(context['full_region_payloads']):
  p=check(e)
  for batch in pq.ParquetFile(p).iter_batches(batch_size=8192,columns=['native_fasta_identifier_key','family_accession','family_catalog_found']):
   for r in batch.to_pylist():
    assert r['family_catalog_found'] and r['family_accession'] in families;key=r['native_fasta_identifier_key'];regions[key]+=1;region_count+=1;sel=family_keysets[r['family_accession']]
    if len(sel)<PER_FAMILY:sel.add(key);selected.add(key)
  if i%20==0:live('complete region exact-key census and deterministic family-diverse selection',region_parts=i+1,region_rows=region_count,selected_keys=len(selected))
 assert region_count==10070931
 fasta={};selected_native=collections.defaultdict(list);source_count=0;native_refs=[]
 for i,e in enumerate(native['native_payloads']):
  p=check(e);native_refs.append(rec(p,rows=e['rows']))
  for batch in pq.ParquetFile(p).iter_batches(batch_size=8192):
   for r in batch.to_pylist():
    seq=r['native_sequence'];assert r['sequence_alphabet_pass'] and len(seq)==r['native_length'] and hashlib.sha256(seq.encode()).hexdigest()==r['native_sequence_sha256'];key=r['source_identifier'];sha=r['native_sequence_sha256'];old=fasta.get(key);fasta[key]=(old[0]+1,old[1],old[2] or old[1]!=sha) if old else (1,sha,False);source_count+=1
    if key in selected:selected_native[key].append(dict(source_record_id=r['source_record_id'],source_identifier=key,source_record_index=r['source_record_index'],native_payload=str(p),native_payload_sha256=e['sha256'],native_sequence_sha256=sha,native_length=len(seq),sequence=seq))
  if i%20==0:live('complete FASTA exact-key/sequence multiplicity census',fasta_parts=i+1,fasta_occurrences=source_count,selected_native_occurrences=sum(len(v) for v in selected_native.values()))
 assert source_count==10164372
 census_counts=collections.Counter();census_parts=[];buffer=[]
 def flush_census():
  if not buffer:return
  p=O/'join-audits'/f'key-cardinality-{len(census_parts):04d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');assert pq.read_table(p).to_pylist()==buffer;census_parts.append(rec(p,rows=len(buffer)));buffer.clear()
 for key,n in regions.items():
  f=fasta.get(key);nf=f[0] if f else 0;census_counts['region_keys']+=1;census_counts['matched_keys' if f else 'region_only_keys']+=1;census_counts['matched_region_rows' if f else 'unmatched_region_rows']+=n
  if f:census_counts['matched_fasta_occurrences']+=nf
  if n>1:census_counts['repeated_region_keys']+=1
  if f and nf>1:census_counts['repeated_matched_fasta_keys']+=1
  if f and f[2]:census_counts['matched_key_multiple_native_sequences']+=1
  buffer.append(dict(native_identifier_key=key,region_rows=n,fasta_occurrences=nf,native_sequence_conflict=f[2] if f else None))
  if len(buffer)>=50000:flush_census()
 for key,f in fasta.items():
  if key in regions:continue
  census_counts['fasta_only_keys']+=1;census_counts['unmatched_fasta_occurrences']+=f[0];buffer.append(dict(native_identifier_key=key,region_rows=0,fasta_occurrences=f[0],native_sequence_conflict=f[2]))
  if len(buffer)>=50000:flush_census()
 flush_census();assert census_counts['matched_region_rows']+census_counts['unmatched_region_rows']==region_count;assert census_counts['matched_fasta_occurrences']+census_counts['unmatched_fasta_occurrences']==source_count;live('full key join cardinality complete',counts=dict(census_counts))
 eligible={k for k in selected if k in fasta and not fasta[k][2]};quarantine=[]
 for key in sorted(selected-eligible):quarantine.append(dict(native_identifier_key=key,reason='no_exact_FASTA_identifier_match' if key not in fasta else 'one_identifier_has_multiple_native_sequences',region_rows=regions[key],fasta_occurrences=fasta[key][0] if key in fasta else 0))
 # A second complete region pass retains every matching assertion, even when a key was selected by a later family.
 assertions=[];key_families=collections.defaultdict(set)
 for e in context['full_region_payloads']:
  p=Path(e['path'])
  for batch in pq.ParquetFile(p).iter_batches(batch_size=8192):
   for r in batch.to_pylist():
    k=r['native_fasta_identifier_key']
    if k in eligible:
     key_families[k].add(r['family_accession']);assertions.append(dict(native_identifier_key=k,source_record_id=r['source_record_id'],family_accession=r['family_accession'],native_payload=str(p),native_payload_sha256=e['sha256'],source_row_index=r['source_row_index'],source_native_fields_json=canonical(r).decode()))
 ar=O/'selected-exact-region-assertions.parquet';pq.write_table(pa.Table.from_pylist(assertions),ar,compression='zstd');assert pq.read_table(ar).to_pylist()==assertions
 nr=[];observations={}
 for key in sorted(eligible):
  rs=selected_native[key];assert len(rs)==fasta[key][0];assert {r['native_sequence_sha256'] for r in rs}=={fasta[key][1]};seq=rs[0]['sequence'];sha=fasta[key][1];obs=observations.setdefault(sha,dict(sequence=seq,families=set(),keys=set(),native_occurrences=0))
  assert obs['sequence']==seq;obs['families'].update(key_families[key]);obs['keys'].add(key);obs['native_occurrences']+=len(rs)
  for r in rs:nr.append({k:v for k,v in r.items() if k!='sequence'})
 np=O/'selected-native-FASTA-occurrence-refs.parquet';pq.write_table(pa.Table.from_pylist(nr),np,compression='zstd');assert pq.read_table(np).to_pylist()==nr
 q=O/'quarantine.parquet';pq.write_table(pa.Table.from_pylist(quarantine) if quarantine else pa.table({'native_identifier_key':pa.array([],type=pa.string()),'reason':pa.array([],type=pa.string())}),q,compression='zstd')
 del regions,fasta,selected_native
 dbpath=ROOT/'outputs/data-intake/20261101-nucleotide-corpus-v1/full-v1/task_identity_and_group_index.sqlite3';conn=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True);shards=[];lookup={};rows=[];group_rows=[];mutation=0;family_coverage=set();native_letters=0;task_count=0
 def flush_tasks():
  if not rows:return
  ix=len(shards);p=O/'shards'/f'{ix:05d}.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==rows;shards.append(rec(p,rows=len(rows),split='train',source_family='Rfam15.1-source-family-prior'))
  g=O/'effective-groups'/f'{ix:05d}.parquet';pq.write_table(pa.Table.from_pylist(group_rows),g,compression='zstd');assert pq.read_table(g).to_pylist()==group_rows;lookup[str(p)]=rec(g,rows=len(group_rows));rows.clear();group_rows.clear();reserve()
 for sha,obs in observations.items():
  seq=obs['sequence'];fs=sorted(obs['families']);cs=sorted({c for f in fs for c in clans[f]});family_coverage.update(fs);native_letters+=len(seq);prior=prior_identity(seq,'RNA_family_reference_native_symbols',None);gs=[r[0] for r in conn.execute('SELECT group_id FROM task_groups WHERE task_id=? ORDER BY group_id',(prior,))];assert gs and 'nucleotide_native:'+sha in gs;gs=sorted(set(gs+['rfam_model:Rfam15.1:'+f for f in fs]));audit=dict(exact_source_identifier_keys=sorted(obs['keys']),native_occurrences=obs['native_occurrences'],source_annotation_assertions=rec(ar),native_occurrence_refs=rec(np),family_catalog_ref=family_ref,clan_binding_ref=clan_ref,source_key_join_policy='exactnativeaccession.version_and_orderedinterval;no_positionalzip;unlistedfamilyunknown',source_quality_metadata_is_input=False)
  for direction in ['prediction','generation']:
   t=family_task(seq,fs,cs,audit,direction);raw=t.canonical_bytes();assert CorpusTask.from_bytes(raw).canonical_bytes()==raw;d=json.loads(raw)
   if direction=='prediction':mutated=family_task(seq,['RF99999'],[],{'all':'changed'},direction)
   else:mutated=family_task(('A' if seq[0]!='A' else 'C')+seq[1:],fs,cs,{'all':'changed'},direction)
   assert t.input_bytes()==mutated.input_bytes();mutation+=1;tid=hashlib.sha256(canonical([d['task'],d['conditions'],d['targets']])).hexdigest();task_count+=1
   rows.append(dict(task_id=tid,record_id=sha,source_family='Rfam15.1-source-family-prior',source_revision=context['source_release'],source_sha256=hashlib.sha256(context_path.read_bytes()).hexdigest(),task_name=d['task'],native_schema='mora3.CorpusTask/v1',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),split='train',split_policy='boundedfamilydiversesourceprior_trainonly;fullalloccurrenceparentgroups_fromnucleotidecomponent;homology_unknown',group_ids_json=json.dumps(gs),evidence_class='source_covariance_model_assigned_reference_family_annotation;notexperimentaltruth',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(exact_identifier_join=True,all_native_sequence_collisions_for_key_quarantined=True,source_license='CC0-1.0',source_family_assignments_not_experiment=True,negative_family_targets=False)),missing_context_json=json.dumps(d['missing']),full_axis_values=len(seq),condition_context_json=json.dumps(dict(segmentation_applicability=False,profile_or_image_pair=False,source_confidence_or_name_or_headers_in_predictors=False,target_length_audit_only=True))))
   group_rows.append(dict(task_id=tid,effective_groups_json=json.dumps(gs)))
   if len(rows)>=512:flush_tasks()
  if task_count%4096==0:live('bounded source-valid family annotation/generation serialization',tasks=task_count,observations=task_count//2)
 flush_tasks();conn.close()
 manifest=dict(schema='mora3.RfamFamilyAnnotationAdmission/v1',status='complete_fullsource_key_census_and_bounded_sourceprior_admission',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=20261031,context_manifest=rec(context_path),native_manifest=rec(native_path),context_handoff=rec(handoff),source_license='CC0-1.0',source_release=context['source_release'],fullsource_native_occurrences=source_count,fullsource_region_rows=region_count,fullsource_family_models=len(families),fullsource_key_cardinality_counts=dict(census_counts),key_cardinality_shards=census_parts,selected_keys=len(selected),eligible_exact_keys=len(eligible),quarantined_keys=len(quarantine),quarantine=rec(q,rows=len(quarantine)),family_selection_per_family=PER_FAMILY,selection_policy='first8distinctkeys_per_native_family_sourceorder;boundedtrainingprior;notrepresentative_evaluation',selected_native_occurrences=len(nr),selected_region_assertions=len(assertions),unique_native_sequence_observations=len(observations),families_with_admitted_assertions=len(family_coverage),family_ids=sorted(family_coverage),serialized_tasks=task_count,directions=2,new_independent_experiments=0,source_family='Rfam15.1-source-family-prior',shards=shards,task_shard_bytes=sum(e['bytes'] for e in shards),effective_group_lookup=lookup,selected_region_assertions_ref=rec(ar,rows=len(assertions)),selected_native_occurrence_refs=rec(np,rows=len(nr)),full_source_native_payloads=native_refs,full_source_context_payloads=context['full_region_payloads'],all_source_payload_hashes_and_native_sequence_checks=True,all_task_Parquet_payload_codec_roundtrips=True,target_and_audit_mutation_checks=mutation,all_original_symbols_case_TU_length_axes_preserved=True,unique_observation_native_letters=native_letters,all_native_occurrence_parent_groups_inherited_from_complete_prior=True,negative_labels_available=False,source_confidence_metadata_in_predictors=False,prediction_condition_fields=['source_native_sequence'],generation_condition_fields=['desired_source_Rfam_family_ids','desired_source_clan_ids'],evidence='sourceassigned_covariance_model_reference_prior;notindependentfunctionalexperiment_or_experimentalRNAstructure',split_counts=dict(train=task_count,validation=0,test=0),model_adapter_supported=False,homology_parent_aliases_taxonomy_heldout_unresolved=True,global_leakage_safe=False,final_weights=None,optimizer_steps=0,code_refs=[rec(p) for p in (O/'code').iterdir()],elapsed_seconds=time.perf_counter()-START)
 reserve();write_json(O/'manifest.json',manifest);live('bounded family annotation component complete',tasks=task_count,observations=len(observations),families=len(family_coverage));print('COMPLETE',task_count,manifest['task_shard_bytes'],flush=True)
if __name__=='__main__':
 try:run()
 except BaseException as e:
  if O.exists():write_json(O/'failure.json',dict(error=repr(e),elapsed_seconds=time.perf_counter()-START))
  raise
