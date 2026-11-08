"""Admit source-protocol Tox21 eval subset only when all known group splits agree."""
from pathlib import Path
import json,sys,hashlib,collections,time,functools,datetime
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_tox21_panel import tasks,row_panel,ASSAYS
from data_intake_corpus import canonical,CorpusTask
B=R/'outputs/data-intake/20261101-tox21-original-admission-v1';O=B/'evaluation-and-union-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=R/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
@functools.lru_cache(maxsize=4)
def source_rows(path,digest):
 p=R/path;assert sha(p)==digest;return [json.loads(s) for s in p.read_text().splitlines()]

def main():
 O.mkdir();(O/'shards/validation').mkdir(parents=True);(O/'shards/test').mkdir();(O/'effective-groups').mkdir();(O/'code').mkdir();(O/'code'/Path(__file__).name).write_bytes(Path(__file__).read_bytes());start=time.perf_counter();basepath=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1/master-corpus-manifest-v11-uniprot-GOA-expanded-v1.json';base=json.loads(basepath.read_text());admit=json.loads((B/'manifest.json').read_text());groups=collections.defaultdict(set)
 for sh in base['shards']:
  splits=set(pq.read_table(check(sh),columns=['split']).column('split').to_pylist());assert len(splits)==1;split=next(iter(splits));gp=check(base['effective_group_lookup'][sh['path']])
  for s in pq.read_table(gp,columns=['effective_groups_json']).column('effective_groups_json').to_pylist():groups[split].update(json.loads(s))
 candidate=pq.read_table(check(admit['evaluation_candidate_table'])).to_pylist();candidate.sort(key=lambda r:(0 if r['source_split']=='source_final_evaluation_structures' else 1,r['source_name'],r['source_row']));selected=[];excluded=[];selectedgroups=collections.defaultdict(set);phenokeys=set();selection_counts=collections.Counter();labelstats=collections.defaultdict(collections.Counter)
 for row in candidate:
  split='test' if row['source_split']=='source_final_evaluation_structures' else 'validation';gs=set(json.loads(row['groups_json']));other='validation' if split=='test' else 'test';cause=None
  if gs&groups['train']:cause='known_existing_training_molecule_scaffold_overlap'
  elif gs&groups[other]:cause='cross_existing_evaluation_split_group_conflict'
  elif gs&selectedgroups[other]:cause='cross_source_protocol_eval_role_group_conflict;officialfinaltestpriority'
  if cause:excluded.append(dict(source_name=row['source_name'],source_split=row['source_split'],cause=cause,source_ref_json=row['source_ref_json']));continue
  ref=json.loads(row['source_ref_json']);raw=source_rows(ref['native_jsonl'],ref['native_jsonl_sha256'])[ref['source_row']];assert raw['source_name']==row['source_name'];vals=json.loads(row['call_values_json']);key=hashlib.sha256(canonical([split,row['canonical_isomeric_smiles'],vals,row['observed_call_mask']])).hexdigest()
  if key in phenokeys:excluded.append(dict(source_name=row['source_name'],source_split=row['source_split'],cause='duplicate_same_role_structure_and_full_callpattern;originalsource_row_retained',source_ref_json=row['source_ref_json']));continue
  phenokeys.add(key);selectedgroups[split].update(gs);selected.append((split,row,raw,ref));selection_counts[split]+=1
  for assay,v,observed in zip(ASSAYS,vals,row['observed_call_mask']):labelstats[split+':'+assay][str(v) if observed else 'unknown']+=1
 assert not(selectedgroups['validation']&selectedgroups['test']);assert not((selectedgroups['validation']|selectedgroups['test'])&groups['train'])
 shards=[];lookup={};buffers=collections.defaultdict(list);seen=set();directions=collections.Counter()
 def flush(split):
  rows=buffers[split]
  if not rows:return
  p=O/'shards'/split/f'{sum(s["split"]==split for s in shards):04d}.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==rows;shards.append(rec(p,rows=len(rows),split=split,source_family='Tox21-original-source-call-prior'));g=O/'effective-groups'/f'{len(shards):04d}.parquet';gr=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in rows];pq.write_table(pa.Table.from_pylist(gr),g,compression='zstd');assert pq.read_table(g).to_pylist()==gr;lookup[str(p.relative_to(R))]=rec(g,rows=len(gr));rows.clear()
 for split,row,raw,ref in selected:
  pnl=dict(axis=list(ASSAYS),values=json.loads(row['call_values_json']),observed_mask=row['observed_call_mask'],observed_calls=row['observed_calls'])
  for t in tasks(raw,pnl,source_ref=ref):
   assert t is not None;payload=t.canonical_bytes();d=json.loads(payload);digest=hashlib.sha256(payload).hexdigest();tid=hashlib.sha256(canonical([row['canonical_isomeric_smiles'],d['task'],pnl['values'],pnl['observed_mask']])).hexdigest();assert tid not in seen;seen.add(tid);directions[split+':'+d['task']]+=1
   buffers[split].append(dict(task_id=tid,record_id=raw['source_name'],source_family='Tox21-original-source-call-prior',source_revision='originalNIH2014challenge;contentpinned20261031',source_sha256=ref['native_jsonl_sha256'],task_name=d['task'],native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=digest,split=split,split_policy='officialleaderboard_asvalidation_officialfinal_astest;knownchemicalgroupscoldagainstv11andToxtrain;crossroleconflictsexcluded;priorpublic/model_exposureunknown',group_ids_json=row['groups_json'],evidence_class='source_original_experimental_HTS_active_inactive_annotation;notclinicaltoxicity_or_continuousresponse',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=ref,source_license='originalUSgovTox21publicdomain',source_protocol_split=True,known_chemical_train_overlap=False,prior_model_and_benchmark_exposure_unknown=True,not_final_unseen_biological_validation=True)),missing_context_json=json.dumps(d['missing']),full_axis_values=12,condition_context_json=json.dumps(dict(raw_image=False,mask=False,physicaldose=None,exposure_time=None,cellline=None,allassays_samecell=False,coordinates_2Ddepiction_only=True))))
   if len(buffers[split])==256:flush(split)
 for split in list(buffers):flush(split)
 # Every newly serialized panel is compared to original source values and exact SampleID joins.
 scorefile=R/'outputs/data-intake/20261031-tox21-original-and-evaluation-v1/raw/final-evaluation-outcomes.txt';import csv
 with scorefile.open() as f:labels={r['Sample ID']:{k:r[k] for k in ASSAYS} for r in csv.DictReader(f,delimiter='\t')}
 verified=0;all_new=admit['task_shards']+shards;all_lookup=dict(admit['effective_group_lookup'],**lookup)
 for sh in all_new:
  for row in pq.read_table(check(sh)).to_pylist():
   p=row['native_payload'];assert hashlib.sha256(p).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(p).canonical_bytes()==p;d=json.loads(p);ref=d['audit']['source_ref'];raw=source_rows(ref['native_jsonl'],ref['native_jsonl_sha256'])[ref['source_row']];pnl=row_panel(raw,labels);assert (d['targets']['source_call_values']==pnl['values'] and d['targets']['source_call_mask']==pnl['observed_mask']) if d['task']=='source_Tox21_molecule_to_observed_assay_calls' else d['conditions']['desired_source_call_values']==pnl['values'] and d['conditions']['desired_source_call_mask']==pnl['observed_mask'];assert d['conditions']['canonical_isomeric_smiles']==raw['derived_canonical_isomeric_smiles'] if d['task']=='source_Tox21_molecule_to_observed_assay_calls' else d['targets']['canonical_isomeric_smiles']==raw['derived_canonical_isomeric_smiles'];verified+=1
   gs=set(json.loads(row['group_ids_json']));assert not(gs&groups['train']) if row['split']!='train' else not(gs&(groups['validation']|groups['test']));groups[row['split']].update(gs)
 # Group keys from all new rows are now included in the same full split audit.
 intersections={a+'__'+b:len(groups[a]&groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]};assert not any(intersections.values())
 ep=O/'evaluation_exclusions.json';ep.write_text(json.dumps(excluded,indent=2)+'\n');receipt=dict(schema='mora3.Tox21SourceProtocolSubsetAdmission/v1',status='selectedsourceprotocol_eval_and_allnewpanelpayloads_verified',candidate_source_records=len(candidate),selected_source_phenopackets=dict(selection_counts),new_eval_tasks=len(seen),all_new_Tox21_payloads_source_readback=verified,eval_task_direction_counts=dict(directions),per_assay_source_call_counts={k:dict(v) for k,v in labelstats.items()},eval_exclusion_receipt=rec(ep,rows=len(excluded)),eval_shards=shards,effective_group_lookup=lookup,known_full_corpus_group_intersections=intersections,evaluation_scope='originalsourceprotocolsubset;chemicalgroupselectiononly;notnewblindbenchmark;unknownpriorpublic/model_exposure',labels_not_used_to_balance_or_optimize_selection=True,physical_dose_time_context_missing=True,molecule_activity_sourcecall_prior_notclinicalprediction=True,all_raw_and_excluded_source_records_preserved=True,model_adapter_supported=False,optimizer_steps=0,seconds=time.perf_counter()-start)
 rp=O/'receipt.json';rp.open('x').write(json.dumps(receipt,indent=2)+'\n');m=base;m['build_id']='20261101-tox21-original-source-call-corpus-v12';m['created_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['prior_manifest_sha256']=sha(basepath);m['supersedes_for_consumption']=str(basepath.relative_to(R));m['shards']+=all_new;m['effective_group_lookup'].update(all_lookup);total=admit['serialized_tasks']+len(seen);m['serialized_tasks']+=total;m['total_shard_bytes']+=sum(s['bytes'] for s in all_new);m['split_counts']['train']+=admit['serialized_tasks']
 for sp,n in selection_counts.items():m['split_counts'][sp]+=2*n
 m['native_schema_counts']['mora3.CorpusTask/v1']+=total;m['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=total;m['serialized_distinct_record_counts_by_source']['Tox21-original-source-call-prior']=len(admit['task_shards']) and admit['unique_admitted_training_phenopackets']+sum(selection_counts.values());m['source_ledger'].append(dict(source_family='Tox21-original-source-call-prior',raw_available=12707,quality_pass=12702,admitted_training_source_occurrences=admit['eligible_source_training_occurrences'],admitted_training_unique_phenopackets=admit['unique_admitted_training_phenopackets'],admitted_eval_source_phenopackets=dict(selection_counts),serialized_tasks=total,count_unit='sample/panelcallpriors;2directionsnot2experiments',license='originalUSgovTox21publicdomain;source-specificNCATSreview',release='originalNIH2014challenge;contentpinned20261031',note='12assayaxesandunknownmasks;all5invalidgraphsretainedquarantined;source-specificknownchemicalgroupsplit;unknownpriorbenchmarkexposure/assayconditions;notclinicaltoxicity'))
 m['additional_Tox21_original_admission_manifest']=str((B/'manifest.json').relative_to(R));m['additional_Tox21_source_protocol_eval_admission']=str(rp.relative_to(R));m['verification_layers'].append(str(rp.relative_to(R)));m['verification']=str(rp.relative_to(R));m['source_curation_manifests'].append(dict(path=str((B/'manifest.json').relative_to(R)),sha256=sha(B/'manifest.json')));m['contexts']['Tox21-original-source-call-prior']=dict(rows=total,context_units='fullnativecompoundgraph/string+12originalassaycalls;missingmasked',axis=list(ASSAYS),model_token_count=None);m['mixture_proposal']['Tox21_original_callprior_allocation']=dict(status='unactivatedprovisional;source experimentalderivedcallprior;missingdose/time;notclinicalresponse',weight=None,requires='native12axisreaders,targetexclusion,sourceheldout/evidencequalityandcontext/throughput');m['open_gates'].append('Tox21evalissourceprotocolsubset;unknownpriorbenchmark/model exposure;all12assayphysicaldose/time/cellcontextmissing;notdirectbinding,continuousresponseorclinicaltoxicity')
 target=B/'master-corpus-manifest-v12-tox21-source-protocol-v1.json';target.open('x').write(json.dumps(m,indent=2)+'\n');print('COMPLETE',json.dumps(dict(master=str(target),selected=selection_counts,new_tasks=total,union_tasks=m['serialized_tasks'],splits=m['split_counts'])),flush=True)
if __name__=='__main__':main()
