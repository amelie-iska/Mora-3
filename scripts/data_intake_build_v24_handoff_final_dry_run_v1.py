from pathlib import Path
import datetime,hashlib,importlib.util,json,os,sys
R=Path('/home/iska/Documents/amelie/bio/Mora-3');os.chdir(R);P=R/'planning/research/data_intake/20261101-ExpansionRx-v24-internal-handoff-preparation-v1';D=P/'dry-run-v2-final-context-v1';D.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('v24_preview_builder',P/'publisher_guard.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(R/'scripts'));from data_intake_reader_api_inventory import inventory
A=R/'outputs/data-intake/20261101-ADMET-corpus-v1';U=A/'independent-master-union-v24-ExpansionRx-v1';OLD=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/model-handoff-v23';M=U/'master-corpus-manifest-v24-ExpansionRx-native-v1.json';BASE=R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/independent-master-union-v23/master-corpus-manifest-v23-whole-RNAcentral-native-v1.json';H='outputs/data-intake/20261101-ADMET-corpus-v1/model-handoff-v24'
def load(p):return json.loads(p.read_text())
def ref(p):return m.file_reference(p)
def write(p,obj):p.open('xb').write(m.json_bytes(obj))
component=A/'ExpansionRx-native-readout-component-v1/manifest.json';sourceproof=A/'ExpansionRx-independent-original-source-task-proof-v1/receipt.json';proof=U/'receipt.json';rootclose=R/'outputs/monitoring/closure-20261101T202009Z-ExpansionRx-v24-exact-staging-v1/closure.json';policy=R/'planning/rgd1-v6-energy-objective-quarantine-20261101T195010Z-v1.json';registry=A/'source-endpoint-and-joint-benchmark-review-v3/endpoint-review-registry.json';affected=R/'planning/research/data_intake/20261101-expanded-scope-v4/ExpansionRx-post-RNA-exact-union-preparation-v1/inherited-RGD1-v6-affected-scope-v1/affected-source-task-objective-inventory.json'
refs=dict(master=ref(M),base_master=ref(BASE),component=ref(component),source_task_proof=ref(sourceproof),union_proof=ref(proof),root_union_closure=ref(rootclose),policy=ref(policy),registry=ref(registry),RGD1_inventory=ref(affected),old_readiness=ref(OLD/'reader-readiness.json'),old_handoff=ref(OLD/'executable-model-handoff.json'),old_native_units=ref(OLD/'native-unit-and-count-ledger.json'),readiness_path=H+'/reader-readiness.json',handoff_path=H+'/executable-model-handoff.json',publication_receipt_path='outputs/data-intake/20261101-ADMET-corpus-v1/internal-pointer-publication-v24-run-v1/receipt.json')
bundle={k:load(p) for k,p in dict(master=M,base=BASE,component=component,union_proof=proof,root_closure=rootclose,policy=policy,registry=registry,old_readiness=OLD/'reader-readiness.json',old_handoff=OLD/'executable-model-handoff.json').items()}
m.validate_metadata(bundle['master'],bundle['base'],bundle['component'],bundle['union_proof'],bundle['root_closure'],bundle['policy'],bundle['registry'],refs)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();api=inventory(R,refs['master'],bundle['master']['serialized_distinct_record_counts_by_source']);api.update(snapshot_time_scope='preparation-only candidate API inspection; no modules imported, no runtime/head/loss support inferred')
write(D/'prepared-reader-API-input.json',api)
docs=m.build_artifacts(bundle,refs,now,api,H);m.validate_handoff_documents(docs,refs)
refs['protocol']=m.byte_reference(m.json_bytes(docs['ExpansionRx-source-protocol-and-eligibility.json']),H+'/ExpansionRx-source-protocol-and-eligibility.json')
(D/'handoffs').mkdir();previews={};destinations={}
for name,obj in docs.items():
 p=D/'handoffs'/name;write(p,obj);previews[name]=ref(p);destinations[name]=m.byte_reference(p.read_bytes(),H+'/'+name)
p=load(R/m.MUTABLE[0]);q=load(R/m.MUTABLE[1]);pn,qn=m.pointer_projection(p,q,refs,now);m.validate_pointer_destinations(pn,qn,refs)
write(D/'data_intake_progress-after.json',pn);write(D/'dataset_acquisition_queue-after.json',qn)
(D/'before').mkdir();(D/'before/data_intake_progress.json').open('xb').write((R/m.MUTABLE[0]).read_bytes());(D/'before/dataset_acquisition_queue.json').open('xb').write((R/m.MUTABLE[1]).read_bytes())
def changes(old,new,path=[]):
 if isinstance(old,dict) and isinstance(new,dict):
  out=[]
  for key in list(old)+[k for k in new if k not in old]:
   if key not in old:out.append(dict(path=path+[key],change='add',new=new[key]))
   elif key not in new:out.append(dict(path=path+[key],change='remove',old=old[key]))
   else:out.extend(changes(old[key],new[key],path+[key]))
  return out
 if m.canonical(old)==m.canonical(new):return []
 return [dict(path=path,change='replace',old=old,new=new)]
write(D/'exact-intended-pointer-mutations.json',{m.MUTABLE[0]:changes(p,pn),m.MUTABLE[1]:changes(q,qn)})
write(D/'dry-run-receipt.json',dict(schema='mora3.V24InternalHandoffDryRun/v1',created_utc=now,refs=refs,current_pointers_unchanged=True,source_entry_count=114,all114_entries_order_and_fields_unchanged=True,source_metadata_count_validation=True,handoff_previews=previews,handoff_destination_byte_refs=destinations,pointer_previews={m.MUTABLE[0]:ref(D/'data_intake_progress-after.json'),m.MUTABLE[1]:ref(D/'dataset_acquisition_queue-after.json')},before={m.MUTABLE[0]:ref(D/'before/data_intake_progress.json'),m.MUTABLE[1]:ref(D/'before/dataset_acquisition_queue.json')},API_preparation_snapshot=ref(D/'prepared-reader-API-input.json'),intended_mutations=ref(D/'exact-intended-pointer-mutations.json'),no_native_source_rows_read=True,no_index_queries_or_hashes=True,no_actual_publication=True,full_dataset_ready=False,model_use=False,training=False))
assert (R/m.MUTABLE[0]).read_bytes()==(D/'before/data_intake_progress.json').read_bytes();assert (R/m.MUTABLE[1]).read_bytes()==(D/'before/dataset_acquisition_queue.json').read_bytes()
print(json.dumps(ref(D/'dry-run-receipt.json')));print('METADATA_BYTES',sum(x['bytes'] for x in previews.values()));print('MUTABLE_AFTER_BYTES',len(m.json_bytes(pn))+len(m.json_bytes(qn)))
