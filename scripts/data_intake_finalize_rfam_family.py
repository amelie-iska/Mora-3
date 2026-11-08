"""Independent source-pointer admission and immutable corpus union for Rfam priors."""
from pathlib import Path
import collections,datetime,hashlib,json,sqlite3,sys,time
import pyarrow.parquet as pq
R=Path.cwd();sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask,canonical
from data_intake_nucleotide_prior import prior_identity
from data_intake_rfam_family import family_task
B=Path('outputs/data-intake/20261101-rfam-family-annotation-v1');O=B/'independent-union-v1';O.mkdir(exist_ok=False);start=time.perf_counter()
mp=B/'curated-v1/manifest.json';m=json.loads(mp.read_text());bp=Path('outputs/data-intake/20261101-nucleotide-corpus-v1/master-corpus-manifest-v13-whole-nucleotide-priors-v1.json');base=json.loads(bp.read_text())
def rec(p,**kw):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest(),**kw)
def check(e):
 got=rec(e['path']);assert got['bytes']==e['bytes'] and got['sha256']==e['sha256']
 if 'rows' in e:assert pq.ParquetFile(e['path']).metadata.num_rows==e['rows']
 return Path(e['path'])
def live(stage,**kw):
 (O/'live-verification.json').write_text(json.dumps(dict(stage=stage,seconds=time.perf_counter()-start,**kw),indent=2)+'\n');print(stage,kw,round(time.perf_counter()-start,2),flush=True)
try:
 context=json.loads(check(m['context_manifest']).read_text());family={r['family_accession']:r for r in pq.read_table(check(context['family_catalog'])).to_pylist()};clans=collections.defaultdict(set)
 for r in pq.read_table(check(context['clan_name_bindings'])).to_pylist():clans[r['family_accession']].add(r['clan_id'])
 native_refs=pq.read_table(check(m['selected_native_occurrence_refs'])).to_pylist();byfile=collections.defaultdict(dict)
 for r in native_refs:byfile[r['native_payload']][r['source_record_id']]=r
 sequences={};keys={};nativematches=0
 for path,wanted in byfile.items():
  expected_sha=next(iter(wanted.values()))['native_payload_sha256'];p=Path(path);assert rec(p)['sha256']==expected_sha
  for batch in pq.ParquetFile(p).iter_batches(batch_size=8192,columns=['source_record_id','source_record_index','source_identifier','native_sequence','native_sequence_sha256','native_length']):
   for r in batch.to_pylist():
    e=wanted.get(r['source_record_id'])
    if e is None:continue
    assert e['source_record_index']==r['source_record_index'] and e['source_identifier']==r['source_identifier'] and e['native_sequence_sha256']==r['native_sequence_sha256'] and e['native_length']==r['native_length'];seq=r['native_sequence'];sha=hashlib.sha256(seq.encode()).hexdigest();assert sha==r['native_sequence_sha256'];assert sequences.setdefault(sha,seq)==seq;assert keys.setdefault(r['source_identifier'],sha)==sha;nativematches+=1
 assert nativematches==len(native_refs)==26895 and len(sequences)==18423
 assertions=pq.read_table(check(m['selected_region_assertions_ref'])).to_pylist();byfile=collections.defaultdict(dict);fs=collections.defaultdict(set)
 for r in assertions:byfile[r['native_payload']][r['source_record_id']]=r
 regionmatches=0
 for path,wanted in byfile.items():
  assert rec(path)['sha256']==next(iter(wanted.values()))['native_payload_sha256']
  for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
   for r in batch.to_pylist():
    e=wanted.get(r['source_record_id'])
    if e is None:continue
    assert canonical(r).decode()==e['source_native_fields_json'];assert r['native_fasta_identifier_key']==e['native_identifier_key'];assert r['family_accession']==e['family_accession'];assert r['family_accession'] in family;fs[keys[r['native_fasta_identifier_key']]].add(r['family_accession']);regionmatches+=1
 assert regionmatches==26895
 live('all selected native sequence and family assertions independently matched to original typed source rows',native_occurrences=nativematches,region_assertions=regionmatches)
 counts=collections.Counter();directions=collections.Counter();seen=set();seenobs=collections.defaultdict(set);newbytes=0;conn=sqlite3.connect('file:'+str((R/'outputs/data-intake/20261101-nucleotide-corpus-v1/full-v1/task_identity_and_group_index.sqlite3'))+'?mode=ro',uri=True);priorgroups={}
 for sha,seq in sequences.items():
  prior=prior_identity(seq,'RNA_family_reference_native_symbols',None);priorgroups[sha]={r[0] for r in conn.execute('SELECT group_id FROM task_groups WHERE task_id=?',(prior,))};assert 'nucleotide_native:'+sha in priorgroups[sha]
 conn.close()
 for e in m['shards']:
  p=check(e);newbytes+=e['bytes'];g=m['effective_group_lookup'][e['path']];grows=pq.read_table(check(g)).to_pylist();gm={r['task_id']:r['effective_groups_json'] for r in grows};rows=pq.read_table(p).to_pylist();assert len(rows)==len(gm)==g['rows']
  for r in rows:
   assert r['task_id'] not in seen;seen.add(r['task_id']);sha=r['record_id'];seq=sequences[sha];f=sorted(fs[sha]);cs=sorted({c for fid in f for c in clans[fid]});direction='prediction' if r['task_name']=='Rfam_source_family_prediction_prior' else 'generation';d=json.loads(r['native_payload']);expected=family_task(seq,f,cs,d['audit']['source_context'],direction);assert expected.canonical_bytes()==r['native_payload'];assert hashlib.sha256(r['native_payload']).hexdigest()==r['native_payload_sha256'];assert r['native_schema']=='mora3.CorpusTask/v1' and r['model_adapter_supported'] is False and r['admitted_for_task'] is True and r['split']=='train';assert r['full_axis_values']==len(seq);assert set(json.loads(gm[r['task_id']]))==priorgroups[sha]|{'rfam_model:Rfam15.1:'+x for x in f};assert r['group_ids_json']==gm[r['task_id']];directions[direction]+=1;seenobs[sha].add(direction);counts[r['source_family']]+=1
 assert len(seen)==m['serialized_tasks']==36846 and set(directions.values())=={18423};assert all(v=={'prediction','generation'} for v in seenobs.values());assert newbytes==m['task_shard_bytes']
 census=collections.Counter();censusrows=0
 for e in m['key_cardinality_shards']:
  for batch in pq.ParquetFile(check(e)).iter_batches(batch_size=10000):
   for r in batch.to_pylist():
    censusrows+=1;nr,nf=r['region_rows'],r['fasta_occurrences'];assert nr>=0 and nf>=0 and nr+nf>0
    if nr:
     census['region_keys']+=1;census['matched_keys' if nf else 'region_only_keys']+=1;census['matched_region_rows' if nf else 'unmatched_region_rows']+=nr
     if nf:census['matched_fasta_occurrences']+=nf
     if nr>1:census['repeated_region_keys']+=1
     if nf>1:census['repeated_matched_fasta_keys']+=1
     if nf and r['native_sequence_conflict']:census['matched_key_multiple_native_sequences']+=1
    else:census['fasta_only_keys']+=1;census['unmatched_fasta_occurrences']+=nf
 assert dict(census)==m['fullsource_key_cardinality_counts']
 # Revalidate every referenced old complete data/group byte stream and footer; scientific semantics inherit their bound prior receipts.
 oldbytes=0
 for i,e in enumerate(base['shards']):
  check(e);check(base['effective_group_lookup'][e['path']]);oldbytes+=e['bytes']
  if i%1000==0:live('complete prior corpus data/group byte hash and row-footer verification',shards=i+1)
 assert oldbytes==base['total_shard_bytes'];master=json.loads(bp.read_text());master.update(build_id='20261101-Rfam-source-family-priors-v14',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supersedes_for_consumption=str(bp),prior_manifest_sha256=rec(bp)['sha256'],additional_Rfam_family_admission_manifest=str(mp),verification=str(O/'receipt.json'));master['shards']+=m['shards'];master['effective_group_lookup'].update(m['effective_group_lookup']);master['serialized_tasks']+=36846;master['total_shard_bytes']+=newbytes;master['split_counts']['train']+=36846;master['native_schema_counts']['mora3.CorpusTask/v1']+=36846;master['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=36846;master['verification_layers'].append(str(O/'receipt.json'));master['source_ledger'].append(dict(source_family=m['source_family'],inventory_id='INV-003',release=m['source_release'],license='CC0-1.0',raw_available_region_rows=10070931,source_native_quality_pass_region_rows=10070931,admitted_source_assertions=26895,admitted_unique_native_sequences=18423,serialized_tasks=36846,task_directions=2,new_independent_experiments=0,count_unit='sourcecomputedfamilyassignments_and_existing_exactRNAsequence_views;notnewnative_sequences',quarantine_selected_identifier_keys=57,source_prior_overlap='all18423targetsequencesalreadyinthecompleteRfamsequenceprior',model_adapter_supported=False));master['serialized_distinct_record_counts_by_source'][m['source_family']]=18423;master['contexts'][m['source_family']]=dict(rows=36846,unique_native_sequence_observations=18423,families=4091,full_native_letters=2263066,context_units='full_native_nucleotide_symbols_and_observed_source_family/clan_assignments',model_token_count=None);master['source_task_context_policy'][m['source_family']]=dict(evidence=m['evidence'],conditions=dict(prediction=['full_native_sequence'],generation=['desired_source_Rfam_family_ids','desired_source_clan_ids']),negative_labels='unlisted_families_unknown;no_dense_negative_BCE',excluded_predictors=['CMscores_Evalues_truncation_coordinates_headers_names_citations_targetlength'],source_group_policy='complete_parentgroups_of_allnativeoccurrences;exactmodelgroupsadded',split='train_only;nohomology_parentalias_taxonheldoutclaim');master['mixture_proposal']['Rfam_family_annotation_allocation']=dict(status='unactivatedprovisional;overlapsnativeRNAsequenceprior',weight=None,requires='nativevariablelengthfamilyreader/decoder;positiveonlyunknownlabelmasks;unconditional_vs_familypriorbalancing_and_nonoverlappingevaluation');master['open_gates'].append('Rfamfamilypriorsarecomputedreferenceannotations;unlistedlabelsunknown;57selectedkeysunmatched;136modelshavenoboundedadmittedexamples;currentfamilypacketselectionnotpopulationrepresentationorheldoutvalidation')
 assert master['serialized_tasks']==9481434 and len(master['shards'])==9436 and master['total_shard_bytes']==6112055305;assert sum(master['split_counts'].values())==master['serialized_tasks'];newmp=B/'master-corpus-manifest-v14-Rfam-family-priors-v1.json'
 receipt=dict(schema='mora3.RfamFamilyIndependentUnionVerification/v1',status='passed',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_master=rec(bp),new_component=rec(mp),all_selected_source_native_rows_independent_hash_ID_sequence_reconstruction=nativematches,all_selected_source_family_assertions_exact_original_native_row_compare=regionmatches,all36846newpayloads_exact_source_reconstruction_and_fullaxes=True,newtasks=len(seen),newdirections=dict(directions),newuniqueexisting_sequence_observations=len(seenobs),newexperiments=0,complete_key_census_rows=censusrows,complete_key_cardinality_counts=dict(census),all_new_effective_groups_exact_complete_prior_groups_plus_source_family_groups=True,known_heldout_intersections='zero_inheritedthroughcompletepriorgroupsandv13verification;newrfam_modelgroupsonlytrain',all9364prior_data_and_group_bytehashes_and_rowfooters_revalidated=True,prior_semantic_proof_inherited_from_bound_receipts=True,global_leakage_safe=False,new_native_model_support=False,new_optimizer_steps=0,master_path=str(newmp),master_totals=dict(tasks=master['serialized_tasks'],shards=len(master['shards']),data_bytes=master['total_shard_bytes'],splits=master['split_counts']),seconds=time.perf_counter()-start)
 (O/'receipt.json').open('x').write(json.dumps(receipt,indent=2)+'\n');newmp.open('x').write(json.dumps(master,indent=2)+'\n');live('v14 full-source-joined family corpus published',tasks=master['serialized_tasks'],shards=len(master['shards']))
except BaseException as exc:
 (O/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
