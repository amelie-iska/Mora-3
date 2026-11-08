"""Full original12-axis Tox21 sample alignment, split/group audit and eligible train priors."""
from pathlib import Path
import sys,json,csv,hashlib,collections,datetime,time,copy
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_tox21_panel import ASSAYS,panel,row_panel,tasks
from data_intake_corpus import canonical,chemical_identity,CorpusTask
B=R/'outputs/data-intake/20261031-tox21-original-and-evaluation-v1';O=R/'outputs/data-intake/20261101-tox21-original-admission-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def main():
 O.mkdir();(O/'shards/train').mkdir(parents=True);(O/'effective-groups').mkdir();(O/'code').mkdir();start=time.perf_counter()
 for n in ['data_intake_curate_tox21_original.py','data_intake_tox21_panel.py']:(O/'code'/n).write_bytes((R/'scripts'/n).read_bytes())
 source=json.loads((B/'source-native-audit-v2/receipt.json').read_text());scorepath=B/'raw/final-evaluation-outcomes.txt';assert sha(scorepath)=='293438f8407c996a0212bedce86d7b9a278f5548df7e7b45ade2126c6dd2770d';labels={}
 with scorepath.open() as f:
  reader=csv.DictReader(f,delimiter='\t');assert reader.fieldnames==['Sample ID']+list(ASSAYS)
  for row in reader:assert row['Sample ID'] not in labels;labels[row['Sample ID']]={k:row[k] for k in ASSAYS}
 assert len(labels)==647
 rows=[];rawstats=collections.Counter();labelcounts=collections.defaultdict(collections.Counter);molgroups={};query_calls=collections.defaultdict(lambda:collections.defaultdict(set));invalid=[];finalids=set()
 for s in source['source_stats']:
  r=s['native_jsonl'];p=R/r['path'];assert sha(p)==r['sha256'] and p.stat().st_size==r['bytes'];n=0
  with p.open() as f:
   for line in f:
    row=json.loads(line);n+=1;split=row['source_split'];pnl=row_panel(row,labels);rawstats[split]+=1
    if split=='source_final_evaluation_structures':finalids.add(row['source_properties']['Sample ID'])
    cid=None;error=row['structure_error']
    if row['derived_canonical_isomeric_smiles']:
     try:cid=chemical_identity(row['derived_canonical_isomeric_smiles']);assert cid['canonical_isomeric_smiles']==row['derived_canonical_isomeric_smiles']
     except Exception as exc:error=type(exc).__name__+':'+str(exc)
    if error or cid is None:invalid.append(dict(source_split=split,source_row=row['source_row'],source_name=row['source_name'],cause=error or 'unparseable_source_structure'));cid=None
    source_ref=dict(native_jsonl=r['path'],native_jsonl_sha256=r['sha256'],source_row=row['source_row'],source_archive=row['source_archive'],source_member=row['source_member'],originalSDFandallpropertiesretained=True,final_label_file=str(scorepath.relative_to(R)) if split=='source_final_evaluation_structures' else None,final_label_file_sha256=sha(scorepath) if split=='source_final_evaluation_structures' else None)
    rr=dict(source=row,panel=pnl,identity=cid,source_ref=source_ref);rows.append(rr)
    for assay,v,mask in zip(ASSAYS,pnl['values'],pnl['observed_mask']):labelcounts[split+':'+assay][str(v) if mask else 'unknown']+=1
    if cid:
     mol=cid['canonical_isomeric_smiles'];molgroups[mol]=cid
     for assay,v,mask in zip(ASSAYS,pnl['values'],pnl['observed_mask']):
      if mask:query_calls[(split,mol)][assay].add(v)
  assert n==s['source_records']
 assert finalids==set(labels)
 evalgroups=set();official_eval=[]
 for rr in rows:
  if rr['identity'] and rr['source']['source_split']!='source_training':evalgroups.update([rr['identity']['molecule_group'],rr['identity']['scaffold_group']]);official_eval.append(rr)
 masterpath=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1/master-corpus-manifest-v11-uniprot-GOA-expanded-v1.json';master=json.loads(masterpath.read_text());existing_heldout=set();existing_train=set()
 for sh in master['shards']:
  p=R/sh['path'];splitset=set(pq.read_table(p,columns=['split']).column('split').to_pylist());target=existing_train if splitset=={'train'} else existing_heldout
  g=master['effective_group_lookup'][sh['path']];gp=R/g['path'];assert sha(gp)==g['sha256']
  for x in pq.read_table(gp,columns=['effective_groups_json']).column('effective_groups_json').to_pylist():target.update(k for k in json.loads(x) if k.startswith(('molecule:','scaffold:')))
 allowed=[];exclusions=[];alltyped=[];candidate_eval=[];conflicts=[];seen_phenopackets={};sourceocc=[];candidate_with_oldtrain=0
 for (split,mol),calls in query_calls.items():
  for assay,values in calls.items():
   if len(values)>1:conflicts.append(dict(source_split=split,canonical_isomeric_smiles=mol,assay=assay,source_values=sorted(values),resolution='all source sample calls retained;no averaging;condition/lotmissing;sourceannotationprioronly,notdeterministicbiophysicalresponse'))
 for rr in rows:
  r=rr['source'];cid=rr['identity'];pnl=rr['panel'];groups=[cid['molecule_group'],cid['scaffold_group']] if cid else [];split=r['source_split'];reason=None
  if cid is None:reason='invalid_original_source_structure'
  elif not pnl['observed_calls']:reason='no_observed_assaycalls;separatelyeligible_structurepriorcandidate'
  elif split!='source_training':reason='officialsource_evaluationcandidate;globaltrainingchemicalreservationandprotocolreviewrequired'
  elif set(groups)&existing_heldout:reason='intersects_existing_v11_validation_test_molecule_or_scaffold_reservation'
  elif set(groups)&evalgroups:reason='intersects_original_Tox21_leaderboard_or_final_molecule_scaffold_reservation'
  typed=dict(source_split=split,source_name=r['source_name'],source_row=r['source_row'],source_ref_json=canonical(rr['source_ref']).decode(),source_properties_json=canonical(r['source_properties']).decode(),coordinate_role=r['coordinate_role'],canonical_isomeric_smiles=cid['canonical_isomeric_smiles'] if cid else None,groups_json=json.dumps(groups),exact_stereochemical_structure_sha256=hashlib.sha256(cid['canonical_isomeric_smiles'].encode()).hexdigest() if cid else None,assay_axis=list(ASSAYS),call_values_json=json.dumps(pnl['values']),observed_call_mask=pnl['observed_mask'],observed_calls=pnl['observed_calls'],source_call_scope='sample/lot + independentassay endpoints;notonecommondose/time/cellcontext',task_admission=reason is None,exclusion_or_candidate_reason=reason)
  alltyped.append(typed)
  if reason:
   exclusions.append(dict(source_split=split,source_row=r['source_row'],source_name=r['source_name'],cause=reason))
   if cid and split!='source_training':
    inold=sorted(set(groups)&existing_train);candidate_with_oldtrain+=bool(inold);candidate_eval.append(dict(typed,existing_training_group_conflicts=inold))
   continue
  key=hashlib.sha256(canonical([cid['canonical_isomeric_smiles'],pnl['values'],pnl['observed_mask']])).hexdigest();sourceocc.append(dict(source_ref_json=canonical(rr['source_ref']).decode(),source_name=r['source_name'],phenopacket_key=key,new_unique_packet=key not in seen_phenopackets))
  if key not in seen_phenopackets:seen_phenopackets[key]=rr;allowed.append(rr)
 # Schema union is explicit, and all source properties remain in immutable original native rows.
 typedp=O/'all-original-source-panels.parquet';pq.write_table(pa.Table.from_pylist(alltyped),typedp,compression='zstd');assert pq.read_table(typedp).to_pylist()==alltyped
 ep=O/'official-evaluation-candidates.parquet';pq.write_table(pa.Table.from_pylist(candidate_eval),ep,compression='zstd');assert pq.read_table(ep).to_pylist()==candidate_eval
 cp=O/'source-call-multiplicity-and-conflicts.json';cp.write_text(json.dumps(conflicts,indent=2)+'\n');ex=O/'exclusions-and-candidates.json';ex.write_text(json.dumps(exclusions,indent=2)+'\n');oc=O/'admitted-source-occurrences.parquet';pq.write_table(pa.Table.from_pylist(sourceocc),oc,compression='zstd');assert pq.read_table(oc).to_pylist()==sourceocc
 buffer=[];shards=[];lookup={};directions=collections.Counter();mutation=0
 def flush():
  if not buffer:return
  p=O/'shards/train'/f'{len(shards):04d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');assert pq.read_table(p).to_pylist()==buffer
  for row in buffer:assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(row['native_payload']).canonical_bytes()==row['native_payload']
  shards.append(rec(p,rows=len(buffer),source_family='Tox21-original-source-call-prior',split='train'));g=O/'effective-groups'/p.name;gr=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in buffer];pq.write_table(pa.Table.from_pylist(gr),g,compression='zstd');assert pq.read_table(g).to_pylist()==gr;lookup[str(p.relative_to(R))]=rec(g,rows=len(gr));buffer.clear()
 for rr in allowed:
  r=rr['source'];pnl=rr['panel'];cid=rr['identity'];forward,inverse=tasks(r,pnl,source_ref=rr['source_ref']);assert forward and inverse
  changed=copy.deepcopy(r);changed['source_properties']={k:('0' if v==1 else '1') for k,v,mask in zip(ASSAYS,pnl['values'],pnl['observed_mask']) if mask};changed['source_name']='MUTATED';fp=row_panel(changed);other=tasks(changed,fp,source_ref={'mutation':True})[0];assert forward.input_bytes()==other.input_bytes();changed['derived_canonical_isomeric_smiles']='CCCC';other=tasks(changed,pnl,source_ref={'mutation':True})[1];assert inverse.input_bytes()==other.input_bytes();mutation+=2
  groups=[cid['molecule_group'],cid['scaffold_group']];assert not(set(groups)&(existing_heldout|evalgroups))
  for task in [forward,inverse]:
   payload=task.canonical_bytes();d=json.loads(payload);digest=hashlib.sha256(payload).hexdigest();tid=hashlib.sha256(canonical([cid['canonical_isomeric_smiles'],d['task'],pnl['values'],pnl['observed_mask']])).hexdigest();directions[d['task']]+=1
   buffer.append(dict(task_id=tid,record_id=r['source_name'],source_family='Tox21-original-source-call-prior',source_revision='originalNIH2014challenge;contentpinned20261031',source_sha256=rr['source_ref']['native_jsonl_sha256'],task_name=d['task'],native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=digest,split='train',split_policy='source_trainingonly;excludev11heldoutandoriginalTox21evalmolecule/scaffoldgroups;evaluationcandidatesnotadmitted',group_ids_json=json.dumps(groups),evidence_class='source_original_experimental_HTS_active_inactive_annotation;notclinicaltoxicity/continuous_doseresponse',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=rr['source_ref'],license='US-government-public-domainoriginalTox21data;officialNCATSproviderstatement',all12assayaxes_retained=True,physical_dose_time_missing=True,measuredcallannotation_prioronly=True,duplicate_source_occurrences=oc.name)),missing_context_json=json.dumps(d['missing']),full_axis_values=12,condition_context_json=json.dumps(dict(raw_image=False,mask=False,perassayphysicaldose=None,exposure_time=None,cellline=None,allassays_samecell=False,coordinates_2D_depiction_only=True))))
   if len(buffer)==256:flush()
 flush();result=dict(schema='mora3.OriginalTox21PanelAdmission/v1',status='all_original_panels_joined_and_selected_train_callpriors_verified',source_records=dict(rawstats),source_total_records=len(rows),all_final647SampleIDs_bijective=True,full12assayaxis=list(ASSAYS),source_call_counts={k:dict(v) for k,v in labelcounts.items()},source_invalid_graphs=invalid,source_native_allfields_preserved_by_refs=True,rights_review=dict(status='substantivelyreviewed_originalgovdataset',grant='NCATSofficialstatesTox21data_publicdomain;originalchallengeassetsfromNIH',evidence=[str((B/'raw/ncats-public-domain-2020.pdf').relative_to(R)),str((B/'raw/nih-copyright-faq.html').relative_to(R)),str((B/'raw/challenge-data.html').relative_to(R))],scope='originalgovernmentHTScalls/chemicalfacts;no thirdpartycompiledsourcegrantassumed'),typed_source_panels=rec(typedp,rows=len(alltyped)),eligible_source_training_occurrences=len(sourceocc),unique_admitted_training_phenopackets=len(allowed),duplicate_training_phenopackets_retained=len(sourceocc)-len(allowed),serialized_tasks=sum(directions.values()),task_direction_counts=dict(directions),task_shards=shards,task_shard_bytes=sum(s['bytes'] for s in shards),effective_group_lookup=lookup,source_occurrence_ledger=rec(oc,rows=len(sourceocc)),official_eval_candidate_records=len(candidate_eval),official_eval_with_existing_train_group_conflicts=candidate_with_oldtrain,official_eval_tasks_admitted=0,evaluation_candidate_table=rec(ep,rows=len(candidate_eval)),source_label_conflict_groups=len(conflicts),source_conflict_receipt=rec(cp),conflict_policy='keepreportedper-samplecallsanduncertainty;noaveraging;sourcecallpriornotdeterministicresponse',exclusions=rec(ex,rows=len(exclusions)),all_task_codec_and_Arrow_roundtrips=True,all_target_mutation_checks=mutation,existing_v11_heldout_group_intersections=0,original_Tox21_eval_group_intersections=0,prior_master=str(masterpath.relative_to(R)),prior_master_sha256=sha(masterpath),full_global_unseen_eval_claim=False,tautomer_protomer_salt_equivalence='unresolved;stereochemicalsourcespreserved;nonstereogroupsandscaffoldarederivedconservativeleakagekeys',model_adapter_supported=False,new_independent_experiments_claim=None,optimizer_steps=0,seconds=time.perf_counter()-start)
 (O/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',len(rows),len(allowed),sum(directions.values()),'evalcandidates',len(candidate_eval),'conflictingoldtrain',candidate_with_oldtrain,flush=True)
if __name__=='__main__':main()
