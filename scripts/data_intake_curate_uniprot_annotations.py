from pathlib import Path
import json,gzip,csv,hashlib,collections,time,copy,sys,argparse,shutil,datetime
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_uniprot_annotations import validate_entry,typed_assertions,full_source_table,annotation_prediction_task,annotation_conditioned_sequence_task,normalize_feature,normalize_go,ec_assertions,physical_facts
from data_intake_corpus import canonical,CorpusTask
ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--output-name',required=True);args=ap.parse_args()
B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';V=B/'paged-acquisition-v2';out=B/args.output_name;out.mkdir();(out/'full-source-parquet').mkdir();(out/'assertions').mkdir();(out/'tasks/train').mkdir(parents=True);(out/'effective-groups').mkdir();(out/'code').mkdir();start=time.perf_counter();assert shutil.disk_usage(R).free>200*2**30+32*2**30
source_code=R/'scripts/data_intake_uniprot_annotations.py';(out/'code/data_intake_uniprot_annotations.py').write_bytes(source_code.read_bytes())
with gzip.open(R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as f:baseline={r['Entry']:r for r in csv.DictReader(f,delimiter='\t')}
if args.full:
 source=json.loads((B/'acquisition-manifest.json').read_text());assert source['source_entries']==107778 and source['status']=='full107778_lossless_records_verified';receipts=source['page_receipts'];expected_count=107778
else:
 receipts=[]
 for name in ['length-7-333','length-334-549','length-550-35213']:
  options=[p for p in (V/'page-receipts').glob(name+'-page0000-attempt*.json') if 'failure' not in p.name];assert len(options)==1;receipts.append(json.loads(options[0].read_text()))
 expected_count=1500
receipts.sort(key=lambda r:(r['cohort'],r['page_index']));seen=set();shards=[];groups_lookup={};full_sources=[];assertion_files=[];source_rows=[];task_buffer=[];task_counts=collections.Counter();category_entries=collections.defaultdict(set);category_rows=collections.Counter();topkeys=collections.Counter();feature_types=collections.Counter();evidence_codes=collections.Counter();coordinate_status=collections.Counter();go_aspects=collections.Counter();GO_evidence=collections.Counter();ec_complete=collections.Counter();phys_roles=collections.Counter();native_bytes=0;entry_tasks=0;generation_omitted=0;mutation_checks=0;newids=set();physical_budget=32*2**30;projection_rows=0

def rec(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**kw)
def resolve(e,pointer):
 value=e
 for token in pointer.split('/')[1:]:
  token=token.replace('~1','/').replace('~0','~');value=value[int(token)] if isinstance(value,list) else value[token]
 return value

def flush():
 global native_bytes
 if not task_buffer:return
 i=len(shards);p=out/'tasks/train'/f'{i:05d}.parquet';pq.write_table(pa.Table.from_pylist(task_buffer),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==task_buffer
 for row in back:assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(row['native_payload']).canonical_bytes()==row['native_payload']
 shards.append(rec(p,rows=len(back),split='train',source_family=back[0]['source_family'] if len({r['source_family'] for r in back})==1 else None));native_bytes+=p.stat().st_size;assert native_bytes<physical_budget
 gp=out/'effective-groups'/f'{i:05d}.parquet';gr=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in back];pq.write_table(pa.Table.from_pylist(gr),gp,compression='zstd');assert pq.read_table(gp).to_pylist()==gr;groups_lookup[str(p.relative_to(R))]=rec(gp,rows=len(gr));task_buffer.clear()
for page_number,receipt in enumerate(receipts):
 p=R/receipt['path'];raw=p.read_bytes();assert len(raw)==receipt['bytes'] and hashlib.sha256(raw).hexdigest()==receipt['sha256'];body=gzip.decompress(raw) if receipt['compression']=='gzip' else raw;entries=json.loads(body)['results'];assert len(entries)==receipt['entries'];assert receipt['source_release']=='2026_03';all_assertions=[];entry_by_id={e['primaryAccession']:e for e in entries}
 for j,e in enumerate(entries):
  acc=e['primaryAccession'];identity=validate_entry(e,baseline[acc]);assert acc not in seen;seen.add(acc);topkeys.update(e.keys());ref=dict(raw_file=receipt['path'],raw_file_sha256=receipt['sha256'],results_index=j,source_release='2026_03',entry_version=identity['entry_version'],sequence_version=identity['sequence_version']);rows=typed_assertions(e)
  for a in rows:
   value=resolve(e,a['source_pointer']);assert json.loads(a['source_value_json'])==value;assert hashlib.sha256(canonical(value)).hexdigest()==a['source_assertion_sha256'];category_rows[a['category']]+=1;category_entries[a['category']].add(acc)
   for binding in json.loads(a['evidence_bindings_json']):
    assert resolve(e,binding['source_pointer'])['evidences']==binding['evidences']
    evidence_codes.update(v.get('evidenceCode','unspecified') for v in binding['evidences'] if isinstance(v,dict))
  all_assertions+=rows;projection_rows+=len(rows)
  for f in e.get('features',[]):feature_types[f.get('type','unspecified')]+=1;coordinate_status[normalize_feature(f,e['sequence']['value'],acc)['coordinate_status']]+=1
  for x in e.get('uniProtKBCrossReferences',[]):
   if x.get('database')=='GO':
    g=normalize_go(x);go_aspects[str(g['aspect'])]+=1;GO_evidence[str(g['source_evidence_code'])]+=1
  for x in ec_assertions(e):ec_complete[str(x['normalized']['complete'])]+=1
  phys_roles.update(f['evidence_role'] for f in physical_facts(e))
  groups=['accession:'+acc,'sequence:'+identity['sequence_sha256']];tasks=[annotation_prediction_task(e,source_ref=ref)];gen=annotation_conditioned_sequence_task(e,source_ref=ref)
  if gen:tasks.append(gen)
  else:generation_omitted+=1
  for task in tasks:
   d=json.loads(task.canonical_bytes());direction=d['task'];family='UniProt-full-record-annotation-prediction' if direction=='source_full_UniProt_annotation_prediction' else 'UniProt-full-record-annotation-conditioned-sequence';payload=task.canonical_bytes();digest=hashlib.sha256(payload).hexdigest();taskid=hashlib.sha256(canonical([acc,direction,digest])).hexdigest();assert taskid not in newids;newids.add(taskid);task_counts[direction]+=1;entry_tasks+=1
   if direction=='source_full_UniProt_annotation_prediction':
    q=copy.deepcopy(e);q['proteinDescription']={'recommendedName':{'fullName':{'value':'TARGET_MUTATION'}}};q['comments']=[];q['features']=[];q['references']=[];q['uniProtKBCrossReferences']=[];other=annotation_prediction_task(q,source_ref=ref)
   else:
    q=copy.deepcopy(e);seq=e['sequence']['value'];mutated=('A' if seq[0]!='A' else 'C')+seq[1:];q['sequence']=dict(value=mutated,length=len(mutated),molWeight=999999,md5=hashlib.md5(mutated.encode()).hexdigest());q['primaryAccession']='AUDIT_ONLY_MUTATION';q['references']=[]
    def no_citation(value):
     if isinstance(value,dict):
      for k,v in value.items():
       if k=='evidences':
        for ev in v:
         if isinstance(ev,dict):ev['id']='MUTATED_CITATION'
       else:no_citation(v)
     elif isinstance(value,list):
      for item in value:no_citation(item)
    no_citation(q);other=annotation_conditioned_sequence_task(q,source_ref=ref)
   assert other is not None and task.input_bytes()==other.input_bytes();mutation_checks+=1
   task_buffer.append(dict(task_id=taskid,record_id=acc,source_family='UniProt-full-record-annotations' if args.full else family,source_revision='UniProt2026_03;entryVersion'+str(identity['entry_version'])+';sequenceVersion'+str(identity['sequence_version']),source_sha256=receipt['sha256'],task_name=direction,native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=digest,split='train',split_policy='existingqualifiedentry_trainonly;exactseq_accessiongroups;homologyheldoutunresolved',group_ids_json=json.dumps(groups),evidence_class='source_curated_mixed_evidence_annotations;notallreviewedannotationsareexperimental',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=ref,annotation_score=identity['annotation_score'],reviewed=True,canonical_sequence_exact_baseline=True,license='CC-BY-4.0',source_scope='KBfullrecord;GOAandexternaloverlaycompanionseparate',new_independent_observation=False)),missing_context_json=json.dumps(d['missing']),full_axis_values=len(e['sequence']['value']),condition_context_json=json.dumps(dict(segmentation_applicability='proteinannotation_contextonly;noimagepair',GO_closed_world=False,alllinkedresources_acquired=False,canonical_only=True,isoform_label_transfer=False,model_generated_labels=False))))
   if len(task_buffer)==64:flush()
  source_rows.append(dict(primary_accession=acc,source_ref_json=canonical(ref).decode(),source_sequence_sha256=identity['sequence_sha256'],raw_annotation_score=identity['annotation_score'],source_features=len(e.get('features',[])),source_comments=len(e.get('comments',[])),source_crossrefs=len(e.get('uniProtKBCrossReferences',[])),source_references=len(e.get('references',[])),conditional_generation_eligible=gen is not None))
 table=full_source_table(entries);fp=out/'full-source-parquet'/f'{page_number:04d}.parquet';pq.write_table(table,fp,compression='zstd');back=pq.read_table(fp).to_pylist();assert [json.loads(r['full_source_json']) for r in back]==entries;assert all(hashlib.sha256(canonical(e)).hexdigest()==r['source_structured_sha256'] for e,r in zip(entries,back));full_sources.append(rec(fp,rows=len(entries)))
 apath=out/'assertions'/f'{page_number:04d}.parquet';pq.write_table(pa.Table.from_pylist(all_assertions),apath,compression='zstd');assert pq.read_table(apath).to_pylist()==all_assertions;assertion_files.append(rec(apath,rows=len(all_assertions)))
 print('TYPED_PAGE',page_number+1,'entries',len(seen),'assertion_views',projection_rows,'tasks',entry_tasks,'seconds',round(time.perf_counter()-start,2),flush=True)
flush();assert len(seen)==expected_count
sp=out/'entry-source-and-task-status.parquet';pq.write_table(pa.Table.from_pylist(source_rows),sp,compression='zstd');assert pq.read_table(sp).to_pylist()==source_rows
manifest=dict(schema='mora3.UniProt_full_annotation_typed_curation.v2',status='full_source_typed_projection_data_tasks_verified' if args.full else 'first1500verified_fullsource_typed_snapshot;full107778sourceintegrationinprogress',source_release='2026_03',source_entries=len(seen),expected_complete_scope=107778,source_scope_complete=bool(args.full),source_pages=len(receipts),source_page_receipts=receipts,full_source_parquet=full_sources,assertion_files=assertion_files,source_status=rec(sp,rows=len(source_rows)),full_nested_source_objects_exact_Arrow_roundtrip=True,all_assertion_source_pointers_and_values_exact=True,all_local_evidence_bindings_exact=True,unrecognized_source_fields_retained=True,typed_assertion_views=projection_rows,assertion_views_are_not_independent_experiments=True,category_entry_counts={k:len(v) for k,v in category_entries.items()},category_assertion_view_counts=dict(category_rows),raw_top_level_field_entries=dict(topkeys),feature_type_counts=dict(feature_types),feature_coordinate_status=dict(coordinate_status),evidence_code_occurrences_with_view_duplication=dict(evidence_codes),GO_aspect_assertion_counts=dict(go_aspects),GO_source_evidence_counts=dict(GO_evidence),EC_complete_partial_assertion_counts=dict(ec_complete),physical_source_role_counts=dict(phys_roles),source_pI_status='notinvented_from_pH;unparsedphysicaltextretained;numericpIrequiresownsourceoralgorithm',GO_completeness='KBfilteredxrefsonly;sourceworkerGOAcompanionversionindependent;NOT/forms/extensionseparate',isoform_status='rawalternativeproducts/featuresretained;unverifiedisoformcoordinatesnotsupervised;fullisoformsequencecompaniongap',external_linked_database_content='referencesonlyunlessseparatelyacquired;noautomaticlicenseorpairedmeasurementclaim',source_family='UniProt-full-record-annotations' if args.full else 'partial_logical_views',task_direction_counts=dict(task_counts),serialized_tasks=entry_tasks,task_shards=shards,task_shard_bytes=native_bytes,effective_group_lookup=groups_lookup,all_task_payloads_roundtrip=True,all_task_target_mutation_input_invariance_checks=mutation_checks,conditioned_generation_omitted_no_eligible_conditions=generation_omitted,model_adapter_supported=False,loss_policy='observedassertionsonly;missingnotnegative;sourcecomputedmassnotmeasured;maximumVelocitynotrelabeledkcat;exactcanonicalcoordinategates',new_independent_observations=0,optimizer_steps=0,license='CC-BY-4.0',source_code=rec(out/'code/data_intake_uniprot_annotations.py'),source_only_train_split=True,homology_and_source_parent_heldout_unresolved=True,seconds=time.perf_counter()-start)
with (out/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('TYPED_COMPLETE',json.dumps({k:manifest[k] for k in ['source_entries','typed_assertion_views','serialized_tasks','task_direction_counts','task_shard_bytes','all_task_target_mutation_input_invariance_checks','seconds']}),flush=True)
