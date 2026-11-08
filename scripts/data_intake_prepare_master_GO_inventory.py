"""Bounded native heldout-set/schema inventory; NOT a master target audit."""
from pathlib import Path
import collections,datetime,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_heldout_GO_policy import native_split_GO_sets,GO_mentions,task_role_surfaces,ZERO
M=R/'outputs/data-intake/20261101-LINCS-corpus-v1/independent-master-union-v22/master-corpus-manifest-v22-LINCS-qualified-source-coordinate-v1.json';B=R/'outputs/data-intake/20261101-ProCyon-native-corpus-v1';O=B/'master-wide-heldout-GO-preparation-v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while c:=f.read(2**20):h.update(c)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(v):
 p=Path(v['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'];return p
def main():
 start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);assert not O.exists();O.mkdir();m=json.loads(M.read_text());prior=B/'qualified-canonical-GOA-target-exposure-v2/manifest.json';pmanifest=json.loads(prior.read_text());index=checked(pmanifest['exposure_index']);source_proto=json.loads(checked(pmanifest['inputs']['ProCyon_identity_protocol_audit']).read_text());source=json.loads(checked(source_proto['inputs']['source_QC_manifest']).read_text());db=sqlite3.connect('file:'+str(index)+'?mode=ro&immutable=1',uri=True);protocols=[];membership=collections.defaultdict(set)
 def guard():
  st=os.statvfs(R);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;other=Path('/proc/3756183/statm');rna=int(other.read_text().split()[1])*os.sysconf('SC_PAGE_SIZE') if other.exists() else 0
  assert time.monotonic()-start<180 and rss<512*2**20 and rss+rna+128*2**20<8*2**30 and st.f_bavail*st.f_frsize>64*2**30
 guard();protein=[dict(text_id=row[0],split=row[1]) for row in db.execute('SELECT GO_ID,source_split FROM procyon_rows')];sets=native_split_GO_sets(protein);protocols.append(dict(protocol='ProCyon:protein_GO:sample_aware_ontology_go_centric',source_revision='2b99a397917282b3184d616bf01825e340488c88',source_readonly_index=pmanifest['exposure_index'],source_rows=len(protein),all_source_rows_regardless_current_protein_qualification=True,**sets));del protein
 domain=[f for f in source['CSV_files'] if f['source_path'].endswith('/domain_go_relations.unified.csv')];assert len(domain)==1;f=domain[0];rows=[]
 for batch in pq.ParquetFile(checked(f['payload'])).iter_batches(batch_size=4096):
  for r in batch.to_pylist():
   d=json.loads(r['native_fields_json']);assert d['split']==r['native_split'];rows.append(dict(text_id=d['text_id'],split=d['split']))
  guard()
 assert len(rows)==36974;protocols.append(dict(protocol='ProCyon:domain_GO:sample_aware_ontology_go_centric',source_revision='2b99a397917282b3184d616bf01825e340488c88',source_native_relation_file=f['source'],source_native_relation_payload=f['payload'],source_rows=len(rows),all_source_rows_regardless_domain_parent_gate=True,**native_split_GO_sets(rows)));del rows;db.close()
 for protocol in protocols:
  for split,ids in protocol['native_role_ID_sets'].items():
   for go in ids:membership[go].add((protocol['protocol'],split,split in ZERO))
 write=O/'official-native-GO-role-ID-sets.json';write.open('x').write(json.dumps(dict(schema='mora3.OfficialProCyonNativeGORoleIDSets/v1',protocols=protocols,native_official_heldouts_not_redefined=True,union_reservation_only_proposed=True),indent=2)+'\n');members=[dict(GO_ID=go,all_original_protocol_roles_json=json.dumps(sorted(values)),in_proposed_cross_protocol_zero_shot_reservation=any(v[2] for v in values)) for go,values in sorted(membership.items())];p=O/'native-GO-ID-role-membership.parquet';pq.write_table(pa.Table.from_pylist(members),p,compression='zstd');assert pq.read_table(p).to_pylist()==members;setref=ref(p,rows=len(members));union=sorted(go for go,v in membership.items() if any(x[2] for x in v));u=O/'proposed-union-of-unmodified-native-zero-shot-GO-IDs.json';u.open('x').write(json.dumps(dict(proposal_only=True,GO_IDs=union,native_source_roles_retained_by_membership_table=True,not_a_new_or_redefined_official_evaluation_set=True),indent=2)+'\n')
 # Master metadata enumerates all admitted rows. Sampled payloads establish
 # PRESENT roles only; sampled absence never proves family-wide absence.
 buckets=collections.defaultdict(list)
 for sh in m['shards']:buckets[sh.get('source_family','NO_SOURCE_FAMILY_METADATA')].append(sh)
 inventory=[];sampled=0
 for metadata_family,files in buckets.items():
  guard();chosen=[]
  for i in sorted({0,len(files)-1}):chosen.append(files[i])
  for sh in chosen:
   p=R/sh['path'];assert p.stat().st_size==sh['bytes'];pf=pq.ParquetFile(p);assert pf.metadata.num_rows==sh['rows'];columns=[k for k in ['task_id','source_family','native_schema','task_name','native_payload','native_payload_sha256'] if k in pf.schema_arrow.names];sample=next(pf.iter_batches(columns=columns,batch_size=min(4,sh['rows']))).to_pylist()
   for row in sample:
    raw=row['native_payload'];assert hashlib.sha256(raw).hexdigest()==row['native_payload_sha256'];d=json.loads(raw);record=dict(master_metadata_family=metadata_family,native_family=row.get('source_family'),native_schema=row['native_schema'],native_task_name=row['task_name'],task_id=row['task_id'],source_shard=sh,sample_scope='closed first<=4 rows from first/last metadata-family shard; presence evidence only',native_payload_SHA256=row['native_payload_sha256'],payload_root_fields=list(d));sampled+=1
    try:
     surfaces=task_role_surfaces(d,row['native_schema']);record['modeled_role_roots']={role:(list(v) if isinstance(v,dict) else type(v).__name__) for role,v in surfaces};record['GO_namespace_role_mentions']=[dict(role=role,**mention) for role,v in surfaces for mention in GO_mentions(v)];record['schema_role_contract_supported']=True
    except (ValueError,KeyError,TypeError) as error:record['schema_role_contract_supported']=False;record['role_contract_review_needed']=str(error)
    record['sample_no_GO_is_not_full_family_absence_proof']=True;inventory.append(record)
 guard();ip=O/'bounded-master-native-schema-role-inventory.json';ip.open('x').write(json.dumps(dict(schema='mora3.BoundedMasterNativeGORoleInventory/v1',master=ref(M),all_admitted_master_tasks=m['serialized_tasks'],all_admitted_master_shards=len(m['shards']),all_admitted_task_shard_bytes=m['total_shard_bytes'],metadata_family_coverage={str(k):dict(shards=len(v),task_views=sum(s['rows'] for s in v)) for k,v in buckets.items()},sampled_native_rows=sampled,samples=inventory,all_source_families_included_regardless_generic_adapter_flag=True,scope='source/schema role inventory only; not whole-master target namespace audit'),indent=2)+'\n');found=collections.defaultdict(set)
 for row in inventory:
  for x in row.get('GO_namespace_role_mentions',[]):found[row['native_family']].add(x['role'])
 manifest=dict(schema='mora3.MasterWideHeldoutGOPreparation/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=ref(M),prior_exact_pair_exposure=ref(prior),official_native_role_ID_sets=ref(write),native_ID_role_membership=setref,proposed_cross_protocol_reservation=ref(u,GO_IDs=len(union)),master_role_inventory=ref(ip),families_with_sampled_present_GO_namespace_roles={k:sorted(v) for k,v in found.items()},source_protocol_GO_ID_counts={p['protocol']:{split:len(ids) for split,ids in p['native_role_ID_sets'].items()} for p in protocols},actual_model_optimizer_exposure=None,new_task_admission=False,no_full_master_namespace_absence_or_masking_claim=True,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);mp=O/'manifest.json';mp.open('x').write(json.dumps(manifest,indent=2)+'\n');print(json.dumps(dict(manifest=ref(mp),proposed_heldout_union_GO_IDs=len(union),sampled_native_rows=sampled,present_GO_roles=manifest['families_with_sampled_present_GO_namespace_roles'],seconds=manifest['seconds'])))
if __name__=='__main__':main()
