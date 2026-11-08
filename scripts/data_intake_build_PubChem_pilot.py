"""Finite <=100k-source-row/512MiB native molecular corpus pilot; no model edits."""
from pathlib import Path
import sys,json,hashlib,collections,datetime,time,shutil,os
import pyarrow as pa,pyarrow.parquet as pq,pyarrow.compute as pc
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import canonical,CorpusTask
from data_intake_PubChem_tasks import validate_native,smiles_prior,codec_task
O=R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1';S=R/'outputs/data-sources/20261031-legacy-mint-v1/curated/PubChem10m-standardSELFIES-full-v1/manifest.json';M=R/'outputs/data-intake/20261101-NCI-AIDS-original-v1/independent-master-union-v19/master-corpus-manifest-v19-original-NCI-assays-v1.json';Q=R/'outputs/data-sources/20261031-legacy-mint-v1/curated/PubChem10m-standardSELFIES-quarantine-index-v1/quarantined-source-occurrence-refs.parquet';CAP=512*2**20
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def write(p,x):p.open('x').write(json.dumps(x,indent=2)+'\n')
def guard():
 n=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());assert n<CAP,(n,CAP);assert shutil.disk_usage(R).free>200*2**30;return n
def main():
 start=time.perf_counter();O.mkdir(exist_ok=True);(O/'shards').mkdir();(O/'groups').mkdir();(O/'code').mkdir();source=json.loads(S.read_text());source_ref=ref(S);source_sha=source_ref['sha256'];m=json.loads(M.read_text());assert sha(M)=='4fa73767f29703d851a487690e16327537af4029ac3bad654d41c382821c1d7e';assert sha(S)=='00d6cb2cd0239c8c03e5a43671dccc432d6b692043d35290abf494e69703c33e';budget=R/'outputs/monitoring/budget-20261101T052809Z-PubChem-corpus-pilot-512MiB-v1.json';assert budget.exists()
 for name in [Path(__file__).name,'data_intake_PubChem_tasks.py']:(O/'code'/name).write_bytes((R/'scripts'/name).read_bytes())
 chunks=[];all_payloads=[];selected_files={};native_validation_hashes=set();longest={};maxima={};total=0
 for ch in source['chunks']:
  p=Path(ch['path']);assert sha(p)==ch['sha256'];x=json.loads(p.read_text());chunks.append(x);all_payloads+=x['payloads'];rows=x['payloads']
  if x['job']['source_asset']['source_native_split']=='train':
   index=int.from_bytes(hashlib.sha256(('20261031:'+x['job']['id']).encode()).digest()[:8],'big')%len(rows);selected_files[rows[index]['path']]=rows[index]
  else:
   for i in range(9):index=round(i*(len(rows)-1)/8);selected_files[rows[index]['path']]=rows[index]
 assert len(selected_files)==18
 index={r['path']:r for r in all_payloads};native_scan_refs=[]
 for k,file in enumerate(all_payloads):
  p=check(file);cols=['native_smiles','derived_atom_count','standard_selfies_token_count','source_current_canonical_graph_sha256','source_native_split'];t=pq.read_table(p,columns=cols);assert len(t)==file['rows'];total+=len(t);native_scan_refs.append(dict(path=file['path'],sha256=file['sha256'],rows=file['rows']))
  if t['source_native_split'][0].as_py()=='validation':native_validation_hashes.update(x for x in t['source_current_canonical_graph_sha256'].to_pylist() if x is not None)
  for name,arr in [('native_SMILES_characters',pc.utf8_length(t['native_smiles'])),('standard_SELFIES_tokens',t['standard_selfies_token_count']),('source_derived_atoms',t['derived_atom_count'])]:
   maximum=pc.max(arr).as_py()
   if maximum is not None and maximum>maxima.get(name,-1):maxima[name]=maximum;longest[name]=dict(payload=file,local_row=pc.index(arr,maximum).as_py())
  if k%200==0:print('PUBCHEM_NATIVE_AXES',k+1,'rows',total,'seconds',round(time.perf_counter()-start,2),flush=True)
 assert total==9999960
 # Complete selected native payloads plus every task-specific codec failure.
 selected={};cache={};reasons=collections.defaultdict(set)
 def load(file):
  if file['path'] not in cache:
   cache.clear();cache[file['path']]=pq.read_table(check(file)).to_pylist()
  return cache[file['path']]
 def add(file,i,reason):
  row=load(file)[i];rid=row['source_record_id'];selected.setdefault(rid,dict(row=row,source_ref=dict(path=str(Path(file['path']).relative_to(R)),bytes=file['bytes'],sha256=file['sha256'],row_index=i)));reasons[rid].add(reason)
 for file in selected_files.values():
  for i in range(file['rows']):add(file,i,'systematic_complete_native_payload')
 qt=pq.read_table(Q).to_pylist();assert len(qt)==843
 for q in qt:
  file=index[q['native_typed_payload']];assert file['sha256']==q['native_payload_sha256'];add(file,q['native_payload_local_row'],'all_source_SELFIES_quarantines')
 for name,x in longest.items():add(x['payload'],x['local_row'],'global_longest_'+name)
 assert len(selected)<=100000
 write(O/'plan.json',dict(schema='mora3.PubChemBoundedCorpusPlan/v1',source=ref(S),master=ref(M),budget_ack=ref(budget),source_row_cap=100000,new_derived_cap_bytes=CAP,selected_original_source_rows=len(selected),selection='one_seeded5kpayload_per9trainchunks+nineuniform5kvalidationpayloads+all843codecquarantineoriginalrefs+globalmaxcharacters/tokens/atoms;deduplicateliteraloriginalsourceIDs',complete_native_full_axis_columns_scanned=total,full_source_max_axes=maxima,all_native_validation_exact_graph_hashes=len(native_validation_hashes),seed=20261031,native_random_split_seed=None,admission='native_source_train_only;excludeall_global_native_validation_exact_graphs+existingmastervaltestmolecule/scaffolds+selectednativevalidationcoarsegroups;not_a_blind_nativeevalclaim',no_full10M_corpus_build=True,no_experimental_response_image_or_sequence_joins=True,final_mixture_weights=None))
 # Reuse all old effective allocations; no new heldout assignment or old writes.
 anchors={};anchor_refs=[]
 for i,sh in enumerate(m['shards']):
  gr=m['effective_group_lookup'][sh['path']];p=check(gr);t=pq.read_table(p,columns=['task_id','effective_groups_json']);col=t['effective_groups_json'];mask=pc.or_(pc.match_substring(col,'molecule:'),pc.match_substring(col,'scaffold:'))
  if not pc.any(mask).as_py():continue
  splits={r['task_id']:r['split'] for r in pq.read_table(R/sh['path'],columns=['task_id','split']).to_pylist()};anchor_refs.append(gr)
  for row in t.filter(mask).to_pylist():
   for group in json.loads(row['effective_groups_json']):
    if group.startswith(('molecule:','scaffold:')):assert group not in anchors or anchors[group]==splits[row['task_id']];anchors[group]=splits[row['task_id']]
  if i%1000==0:print('PUBCHEM_MASTER_GROUPS',i+1,'anchors',len(anchors),flush=True)
 validated={};validation_groups=set();failures={}
 for k,(rid,item) in enumerate(selected.items()):
  try:v=validate_native(item['row']);validated[rid]=v
  except Exception as exc:failures[rid]=dict(cause='native_graph_or_scaffold_independent_derivation_failed',exception=repr(exc));continue
  if item['row']['source_native_split']=='validation':validation_groups.update(v['identity'][key] for key in ['molecule_group','scaffold_group'])
  if k%10000==0:print('PUBCHEM_SELECTED_QC',k+1,'valid',len(validated),'seconds',round(time.perf_counter()-start,2),flush=True)
 ledger=[];seen_graph={};tasks=[];task_counts=collections.Counter();quarantines=collections.Counter();retained=collections.Counter();fullaxes=collections.Counter();taskgroups={};selected_order=sorted(selected,key=lambda rid:(selected[rid]['row']['source_native_split']!='train',selected[rid]['row']['source_native_file_row_index']))
 for rid in selected_order:
  item=selected[rid];row=item['row'];v=validated.get(rid);gate=[]
  if v is None:gate.append(failures[rid]['cause'])
  if row['source_native_split']!='train':gate.append('native_source_validation_retained_as_audit_candidate_only')
  if v is not None:
   ident=v['identity'];groups=[ident['molecule_group'],ident['scaffold_group'],'chemical_exact_isomeric:'+row['source_current_canonical_graph_sha256']];prior={anchors[g] for g in groups if g in anchors};iso=row['source_current_canonical_graph_sha256']
   if prior-{'train'}:gate.append('authoritative_master_heldout_chemical_or_scaffold')
   if iso in native_validation_hashes:gate.append('global_source_native_validation_exact_graph')
   if validation_groups&set(groups):gate.append('selected_native_validation_chemical_or_scaffold')
   if iso in seen_graph:gate.append('pilot_exact_isomeric_graph_duplicate')
  else:groups=[];prior=set();iso=None
  src=dict(native_source=source_ref,native_record=item['source_ref'],source_record_id=rid,source_native_split=row['source_native_split'],source_native_row=row['source_native_file_row_index'])
  rowtasks=[]
  if not gate:
   seen_graph[iso]=rid;rowtasks=[('PubChem-native-SMILES-reference-prior','reference_prior',smiles_prior(row,src,validated=v))]
   if v['selfies_eligible']:rowtasks += [('PubChem-standardSELFIES-codec','smiles_to_selfies',codec_task(row,'smiles_to_selfies',src,validated=v)),('PubChem-standardSELFIES-codec','selfies_to_smiles',codec_task(row,'selfies_to_smiles',src,validated=v))]
   retained['admitted_unique_isomeric_graphs']+=1;retained['valid_SMILES_without_SELFIES_admitted']+=not v['selfies_eligible']
  for reason in gate:quarantines[reason]+=1
  rowids=[]
  for family,direction,task in rowtasks:
   raw=task.canonical_bytes();d=json.loads(raw);tid=hashlib.sha256(canonical([family,rid,direction])).hexdigest();taskgroups[tid]=groups;rowids.append(tid);task_counts[family+':'+direction]+=1
   tasks.append(dict(task_id=tid,record_id=rid,source_family=family,source_revision=source['source_revision'],source_sha256=source_sha,task_name=d['task'],native_schema='mora3.CorpusTask/v1',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),split='train',split_policy='native_source_train_only;existingmasterheldout+globalnativevalidationexact+selectedvalidationcoarseguards_excluded;no_newheldouts',group_ids_json=json.dumps(sorted(groups)),evidence_class='source_reference_structure' if direction=='reference_prior' else 'computed_lossless_representation_not_experimental_response',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=src,source_license='Apache-2.0',target_role='source_reference_SMILES' if direction=='reference_prior' else 'computed_standard_SELFIES_codec',full_native_axes=True,no_3D_or_experimental_response_pair=True)),missing_context_json=json.dumps(d['missing']),full_axis_values=len(next(iter(d['targets'].values()))),condition_context_json=json.dumps(dict(assay=None,physical_dose=None,time=None,image_available=False,expression_available=False,segmentation_available=False))))
  ledger.append(dict(source_record_id=rid,source_native_split=row['source_native_split'],source_native_file_row_index=row['source_native_file_row_index'],source_ref_json=json.dumps(item['source_ref']),selection_reasons_json=json.dumps(sorted(reasons[rid])),source_status=row['status'],native_parse_pass=row['native_current_RDKit_parse_pass'],SELFIES_graph_roundtrip_pass=row['graph_roundtrip_pass'],independently_validated_SMILES=v is not None,independently_validated_SELFIES=v['selfies_eligible'] if v else False,source_canonical_graph_sha256=iso,derived_identity_json=json.dumps(v['identity']) if v else None,gate_causes_json=json.dumps(gate),prior_allocation_splits_json=json.dumps(sorted(prior)),admitted_task_ids_json=json.dumps(rowids),all_native_source_columns_json=json.dumps(row),independent_derivation_failure_json=json.dumps(failures.get(rid)),selected_native_characters=len(row['native_smiles']) if row['native_smiles'] else None,selected_SELFIES_token_count=row['standard_selfies_token_count']))
 p=O/'selected-source-admission-ledger.parquet';pq.write_table(pa.Table.from_pylist(ledger),p,compression='zstd');assert pq.read_table(p).to_pylist()==ledger;ledgerref=ref(p,rows=len(ledger));shards=[];lookup={}
 for family in sorted({r['source_family'] for r in tasks}):
  rows=[r for r in tasks if r['source_family']==family]
  for offset in range(0,len(rows),1024):
   batch=rows[offset:offset+1024];p=O/'shards'/f'{len(shards):05d}.parquet';pq.write_table(pa.Table.from_pylist(batch),p,compression='zstd');assert pq.read_table(p).to_pylist()==batch
   for r in batch:assert CorpusTask.from_bytes(r['native_payload']).canonical_bytes()==r['native_payload']
   shards.append(ref(p,rows=len(batch),source_family=family,split='train'));g=O/'groups'/p.name;gr=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in batch];pq.write_table(pa.Table.from_pylist(gr),g,compression='zstd');assert pq.read_table(g).to_pylist()==gr;lookup[str(p.relative_to(R))]=ref(g,rows=len(gr));guard()
 p=O/'native-source-column-scan-receipt.json';write(p,dict(schema='mora3.PubChemNativeAxisAndValidationExactScan/v1',all_payloads=native_scan_refs,native_occurrences=total,selected_source_rows=len(ledger),native_validation_unique_valid_source_graph_hashes=len(native_validation_hashes),maximum_axes=maxima,scope='fullsourcecompletebyteshash+fivecolumns/fullaxes;not_newallsourcegraphreQC;fullsemanticproofinheritedsourcehandoff'))
 selected_status=collections.Counter(r['status'] for r in [v['row'] for v in selected.values()]);all_quarantine_ids={r['source_record_id'] for r in qt};assert all_quarantine_ids<=set(selected);actual=guard();result=dict(schema='mora3.PubChemBoundedCorpusAdmission/v1',status='serialized_pending_independent_raw_native_task_and_union_audit',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=ref(S),prior_master=ref(M),budget_ack=ref(budget),selection_plan=ref(O/'plan.json'),available_source_occurrences=9999960,fullsource_native_parse_pass=9999899,fullsource_standardSELFIES_graph_candidates=9999117,selected_source_occurrences=len(ledger),selected_status_counts=dict(selected_status),selected_native_parse_pass=sum(x['row']['native_current_RDKit_parse_pass'] for x in selected.values()),selected_independent_graph_quality_pass=len(validated),full843SELFIES_source_quarantines_preserved=True,all782parsepass_unsupported_SELFIES_rows_selected=True,selected_admitted_counts=dict(retained),task_counts=dict(task_counts),admitted_task_views=len(tasks),split_counts={'train':len(tasks)},new_heldout_views=0,source_native_splits_audit_only_not_blind_evaluation=True,global_nativevalidation_exact_source_graph_exclusion=True,selected_nativevalidation_coarse_guard_exclusion=True,master_existing_allocations_unchanged=True,quarantine_or_exclusion_counts=dict(quarantines),selected_source_ledger=ledgerref,native_axis_scan_receipt=ref(O/'native-source-column-scan-receipt.json'),source_quarantine_index=ref(Q),shards=shards,effective_group_lookup=lookup,task_shard_bytes=sum(r['bytes'] for r in shards),all_new_derived_bytes_so_far=actual,new_derived_cap_bytes=CAP,source_row_cap=100000,full_native_axes_preserved=True,global_native_max_axes=maxima,source_license='Apache-2.0 data grant',source_release='HF@f83219601635a0a80fc99c13a9ca37f99ef34f0a;biologicalPubChemrelease_unknown',standardSELFIES_not_originalGroupSELFIES=True,model_adapter_supported=False,new_independent_experiments=0,optimizer_steps=0,final_mixture_weights=None,seconds=time.perf_counter()-start)
 write(O/'manifest.json',result);guard();print('COMPLETE',json.dumps({k:result[k] for k in ['selected_source_occurrences','selected_independent_graph_quality_pass','selected_admitted_counts','task_counts','admitted_task_views','task_shard_bytes','all_new_derived_bytes_so_far','quarantine_or_exclusion_counts','seconds']}),flush=True)
if __name__=='__main__':main()
