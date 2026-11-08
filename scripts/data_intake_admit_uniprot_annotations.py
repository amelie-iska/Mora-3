"""Immutable UniProt full-annotation task admission and v9 master assembly."""
from pathlib import Path
import sys,json,gzip,csv,hashlib,collections,datetime,time,copy,shutil,argparse
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_uniprot_annotations import validate_entry,annotation_prediction_task,annotation_conditioned_sequence_task
from data_intake_uniprot_generation_v2 import generation_task
from data_intake_corpus import canonical,CorpusTask

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def receipt(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def verify(rec):
 p=R/rec['path'];assert p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'];return p

def resolved_shard_rows(source_path,groups_path,metadata):
 source=pq.read_table(source_path,columns=['task_id','source_family','split']).to_pylist();groups=pq.read_table(groups_path,columns=['task_id','effective_groups_json']).to_pylist();gm={r['task_id']:r['effective_groups_json'] for r in groups}
 if len(gm)!=len(groups) or len(source)!=len(groups) or {r['task_id'] for r in source}!=set(gm):raise ValueError('effective-group task identities differ from source envelope')
 result=[]
 for row in source:
  if row['split'] not in ['train','validation','test']:raise ValueError('unrecognized source-envelope split')
  for key in ['split','source_family']:
   if metadata.get(key) is not None and metadata[key]!=row[key]:raise ValueError('shard summary disagrees with source envelope')
  result.append(dict(row,effective_groups_json=gm[row['task_id']]))
 return result

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output-name',required=True);args=ap.parse_args();B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';out=B/args.output_name;out.mkdir();(out/'shards/train').mkdir(parents=True);(out/'effective-groups').mkdir();(out/'code').mkdir();start=time.perf_counter();assert shutil.disk_usage(R).free>200*2**30+8*2**30
 for name in ['data_intake_admit_uniprot_annotations.py','data_intake_uniprot_annotations.py','data_intake_uniprot_supplement.py','data_intake_uniprot_generation_v2.py']:(out/'code'/name).write_bytes((R/'scripts'/name).read_bytes())
 raw_manifest=json.loads((B/'acquisition-manifest.json').read_text());typed_path=B/'typed-full-v1/manifest.json';typed=json.loads(typed_path.read_text());supplement_path=B/'physical-form-supplement-v2/manifest.json';supplement=json.loads(supplement_path.read_text());assert typed['source_scope_complete'] and typed['source_entries']==107778 and typed['all_task_payloads_roundtrip'];assert supplement['source_entries']==107778
 typed_by_raw={r['path']:p for r,p in zip(typed['source_page_receipts'],typed['full_source_parquet'])}
 # Complete-byte verification is distinct from subsequent entire source/task semantic checks.
 for rec in typed['full_source_parquet']+typed['assertion_files']+supplement['files']:verify(rec)
 basepath=R/'outputs/data-intake/20261031-rosetta-molecule-profile-v1/master-corpus-manifest-v8-source-evidence-v1.json';base=json.loads(basepath.read_text());with_source=json.loads(json.dumps(base));heldout=set();oldprotein_groups_byfamily=collections.defaultdict(set);oldrows_byfamily=collections.Counter();base_tasks=0
 for sh in base['shards']:
  rec=base['effective_group_lookup'][sh['path']];gp=verify(rec)
  for row in resolved_shard_rows(verify(sh),gp,sh):
   groups=json.loads(row['effective_groups_json']);base_tasks+=1
   if row['split']!='train':heldout.update(groups)
   else:
    ps=[g for g in groups if g.startswith(('accession:','sequence:'))]
    if ps:oldprotein_groups_byfamily[row['source_family']].update(ps);oldrows_byfamily[row['source_family']]+=1
 assert base_tasks==base['serialized_tasks']
 with gzip.open(R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as f:baseline={r['Entry']:r for r in csv.DictReader(f,delimiter='\t')}
 counts=collections.Counter();shards=[];lookup={};buffer=[];seen=set();seen_tasks=set();seen_sequences=set();source_status=[];changes=[];heldout_conflicts=[];source_seq_lengths=[];newprotein_groups=set();payloadsizes=[];max_payload=0;mutation_checks=0;source_bytes=0
 def flush():
  if not buffer:return
  i=len(shards);p=out/'shards/train'/f'{i:05d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==buffer
  for row in back:assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(row['native_payload']).canonical_bytes()==row['native_payload']
  shards.append(receipt(p,rows=len(back),split='train',source_family='UniProt-full-record-annotations'));g=out/'effective-groups'/f'{i:05d}.parquet';rows=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json'],evidence_scope='identity/leakagegroupsonly;notpredictorfeaturesorsamecell/effectproof') for r in back];pq.write_table(pa.Table.from_pylist(rows),g,compression='zstd');assert pq.read_table(g).to_pylist()==rows;lookup[str(p.relative_to(R))]=receipt(g,rows=len(rows));buffer.clear()
 for n,rec in enumerate(raw_manifest['page_receipts']):
  p=verify(rec);raw=p.read_bytes();source_bytes+=len(raw);entries=json.loads(gzip.decompress(raw) if rec['compression']=='gzip' else raw)['results'];assert len(entries)==rec['entries'] and rec['source_release']=='2026_03'
  # Independent whole-record comparison against lossless Arrow representation, not a first-row probe.
  typed_rec=typed_by_raw[rec['path']]
  source_back=pq.read_table(R/typed_rec['path']).to_pylist();assert [json.loads(r['full_source_json']) for r in source_back]==entries
  for j,e in enumerate(entries):
   acc=e['primaryAccession'];identity=validate_entry(e,baseline[acc]);assert acc not in seen;seen.add(acc);seen_sequences.add(identity['sequence_sha256']);source_seq_lengths.append(identity['sequence_length']);groups=['accession:'+acc,'sequence:'+identity['sequence_sha256']];newprotein_groups.update(groups);conflicts=sorted(set(groups)&heldout)
   if conflicts:heldout_conflicts.append(dict(accession=acc,groups=conflicts));continue
   ref=dict(raw_file=rec['path'],raw_file_sha256=rec['sha256'],results_index=j,source_release='2026_03',entry_version=identity['entry_version'],sequence_version=identity['sequence_version']);pred=annotation_prediction_task(e,source_ref=ref);gen=generation_task(e,source_ref=ref);oldgen=annotation_conditioned_sequence_task(e,source_ref=ref)
   if oldgen and (gen is None or oldgen.input_bytes()!=gen.input_bytes()):changes.append(dict(accession=acc,previous_conditions_json=json.loads(oldgen.canonical_bytes())['conditions'],new_conditions_json=json.loads(gen.canonical_bytes())['conditions'] if gen else None,cause='exclude_unverified_product_specific_kinetic_or_EC_conditions;sourceunchanged'))
   tasks=[pred]+([gen] if gen else [])
   for task in tasks:
    obj=json.loads(task.canonical_bytes());direction=obj['task'];payload=task.canonical_bytes();digest=hashlib.sha256(payload).hexdigest();tid=hashlib.sha256(canonical([acc,direction,digest])).hexdigest();assert tid not in seen_tasks;seen_tasks.add(tid);counts[direction]+=1;payloadsizes.append(len(payload));max_payload=max(max_payload,len(payload))
    # Explicit condition allowlists and target/identity/citation mutation on every final task.
    if direction=='source_full_UniPro_annotation_prediction':raise AssertionError('misspelled task')
    if direction=='source_full_UniProt_annotation_prediction':
     assert obj['conditions']==dict(canonical_sequence=e['sequence']['value'])
     q=copy.deepcopy(e);q['comments']=[];q['features']=[];q['proteinDescription']={};q['references']=[];q['uniProtKBCrossReferences']=[];other=annotation_prediction_task(q,source_ref=ref)
    else:
     q=copy.deepcopy(e);seq=e['sequence']['value'];mut=('A' if seq[0]!='A' else 'C')+seq[1:];q['sequence']=dict(value=mut,length=len(mut),molWeight=999999,md5=hashlib.md5(mut.encode()).hexdigest());q['primaryAccession']='MUTATED_ID';q['proteinDescription']={k:v for k,v in q.get('proteinDescription',{}).items() if k not in ['submittedName']};q['references']=[]
     def nocites(x):
      if isinstance(x,dict):
       for k,v in x.items():
        if k=='evidences':
         for ev in v:
          if isinstance(ev,dict):ev['id']='MUTATED_CITATION'
        else:nocites(v)
      elif isinstance(x,list):
       for v in x:nocites(v)
     nocites(q);other=generation_task(q,source_ref=ref)
    assert other is not None and task.input_bytes()==other.input_bytes();mutation_checks+=1
    buffer.append(dict(task_id=tid,record_id=acc,source_family='UniProt-full-record-annotations',source_revision='UniProt2026_03;entryVersion'+str(identity['entry_version'])+';sequenceVersion'+str(identity['sequence_version']),source_sha256=rec['sha256'],task_name=direction,native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=digest,split='train',split_policy='sourcevalid_existingqualifiedentry_trainonly;exactsequence_accessiongroups;homologyheldoutunresolved',group_ids_json=json.dumps(groups),evidence_class='source_curated_mixed_evidence_annotations;notallreviewedannotationsareexperimental',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=ref,reviewed=True,annotation_score=identity['annotation_score'],canonical_sequence_exact_baseline=True,source_full_nested_roundtrip=True,license='CC-BY-4.0',scope_safe_generation_v2=True,new_independent_observation=False)),missing_context_json=json.dumps(obj['missing']),full_axis_values=identity['sequence_length'],condition_context_json=json.dumps(dict(segmentation_applicability='proteinannotation_contextonly;noimagepair',GO_closed_world=False,alllinkedresources_acquired=False,canonical_only=True,isoform_label_transfer=False,model_generated_labels=False))))
    if len(buffer)==64:flush()
   source_status.append(dict(accession=acc,sequence_sha256=identity['sequence_sha256'],sequence_length=identity['sequence_length'],annotation_prediction=True,canonical_generation=gen is not None,source_ref_json=canonical(ref).decode(),new_independent_observation=False))
  print('ADMISSION_PAGE',n+1,'entries',len(seen),'tasks',sum(counts.values()),'scope_corrections',len(changes),'seconds',round(time.perf_counter()-start,2),flush=True)
 flush();assert len(seen)==107778 and not heldout_conflicts
 st=out/'source-and-task-status.parquet';pq.write_table(pa.Table.from_pylist(source_status),st,compression='zstd');assert pq.read_table(st).to_pylist()==source_status
 ch=out/'product-scope-condition-corrections.json';ch.write_text(json.dumps(changes,indent=2)+'\n')
 overlap={fam:dict(matching_exact_protein_groups=len(gs&newprotein_groups),existing_task_rows_with_protein_groups=oldrows_byfamily[fam]) for fam,gs in oldprotein_groups_byfamily.items()};op=out/'protein-identity-and-split-audit.json';op.write_text(json.dumps(dict(qualified_accessions=len(seen),distinct_canonical_sequences=len(seen_sequences),extra_accession_aliases=len(seen)-len(seen_sequences),source_overlap=overlap,all_new_tasks_train=True,new_protein_group_intersections_with_existing_valtest=len(newprotein_groups&heldout),heldout_task_groups=len(heldout),protein_homology_unresolved=True,chemical_equivalence_and_rawannotation_chemical_identity_links='notcomplete;no_global_leakage_safe_claim',shared_GO_is_not_identity=True,new_independent_observations=0),indent=2)+'\n')
 total=sum(counts.values());data_bytes=sum(s['bytes'] for s in shards);receipt_path=out/'receipt.json';result=dict(schema='mora3.UniProtFullAnnotationAdmission/v2',source_entries=len(seen),unique_canonical_sequences=len(seen_sequences),raw_complete_pages=217,raw_verified_bytes=source_bytes,full_nested_source_Arrow_comparison_all_entries=True,source_qualified_and_exact_baseline_all_entries=True,admitted_records=len(source_status),serialized_tasks=total,task_direction_counts=counts,shards=shards,data_bytes=data_bytes,effective_group_lookup=lookup,all_final_task_roundtrips=True,target_mutation_input_invariance_checks=mutation_checks,source_status=receipt(st,rows=len(source_status)),scope_correction_receipt=receipt(ch,rows=len(changes)),identity_and_split_audit=receipt(op),source_release='2026_03',license='CC-BY-4.0',model_adapter_supported=False,source_train_only=True,final_mixture=False,new_independent_observations=0,optimizer_steps=0,typed_source_manifest=receipt(typed_path),physical_form_projection=receipt(supplement_path),GOA_companion='sourceworker_fullsubsetinprogress;independentJuly28snapshot;notyetadmitted',isoform_full_sequence_companion='pending;noinheritance',full_payload_bytes=dict(min=min(payloadsizes),max=max_payload,mean=sum(payloadsizes)/len(payloadsizes),sum=sum(payloadsizes)),sequence_axis=dict(min=min(source_seq_lengths),max=max(source_seq_lengths),mean=sum(source_seq_lengths)/len(source_seq_lengths)),model_tokens='notmeasured;no truncation',seconds=time.perf_counter()-start)
 receipt_path.write_text(json.dumps(result,indent=2)+'\n')
 m=with_source;m['build_id']='20261031-uniprot-full-annotation-corpus-v9';m['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['prior_manifest_sha256']=sha(basepath);m['supersedes_for_consumption']=str(basepath.relative_to(R));m['shards']+=shards;m['serialized_tasks']+=total;m['total_shard_bytes']+=data_bytes;m['split_counts']['train']+=total;m['native_schema_counts']['mora3.CorpusTask/v1']+=total;m['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=total;m['effective_group_lookup'].update(lookup);m['serialized_distinct_record_counts_by_source']['UniProt-full-record-annotations']=len(source_status);m['source_ledger'].append(dict(source_family='UniProt-full-record-annotations',inventory_id='INV-005',raw_available=107778,quality_pass=107778,admitted_records=len(source_status),unique_canonical_sequences=len(seen_sequences),serialized_tasks=total,count_unit='existingUniProtentryannotationviews;notnewindependentexperiments',release='UniProt2026_03',license='CC-BY-4.0',note='Sourcevalid annotationprediction/fullpacket and locallyexperimentallysupported canonicalgeneration;productscope corrected;trainonly;missing annotations unknown;homology/GOA/isoform/native decoder gates explicit'))
 m['additional_full_uniprot_annotation_admission_receipt']=str(receipt_path.relative_to(R));m['source_curation_manifests'].append(str(typed_path.relative_to(R)));m['source_curation_manifests'].append(str(supplement_path.relative_to(R)));m['verification_layers'].append(str(receipt_path.relative_to(R)));m['contexts']['UniProt-full-record-annotations']=dict(rows=total,context_units='fullaminoacidsequence+losslesssourceannotationpacket;notmodeltokens',**result['sequence_axis'],model_token_count=None,max_full_payload_bytes=max_payload);m['mixture_proposal']['full_UniProt_annotation_allocation']=dict(status='provisional_unactivated;overlap_existingsequence_function_kineticviews',weight=None,requires='native annotationreader/objectives;label/evidence masks;fullcontextbudget;proteinheldoutandmixtureevaluation');m['governing_contracts'].append('planning/uniprot_annotation_contract.md');m['open_gates'].append('FullKBrawandtypedretained;GOA/isoformcompanionsinprogress;newannotationneuraldecoderunsupported;proteinunseenheldoutnotestablished;physicalv2numericobjectivesunadmitted')
 target=B/'master-corpus-manifest-v9-uniprot-full-v1.json';target.open('x').write(json.dumps(m,indent=2)+'\n');print('COMPLETE',str(target),total,len(shards),data_bytes,flush=True)
if __name__=='__main__':main()
