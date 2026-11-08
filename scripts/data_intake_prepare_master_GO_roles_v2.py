"""Correct eight bounded SciPlex role probes; no repeated whole-master pass."""
from pathlib import Path
import collections,datetime,hashlib,json,os,resource,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_master_GO_scan import row_exposure
B=R/'outputs/data-intake/20261101-ProCyon-native-corpus-v1';I=B/'master-wide-heldout-GO-preparation-v1';O=B/'master-wide-heldout-GO-preparation-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def main():
 start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);assert not O.exists();O.mkdir();old=json.loads((I/'bounded-master-native-schema-role-inventory.json').read_text());protected=set(json.loads((I/'proposed-union-of-unmodified-native-zero-shot-GO-IDs.json').read_text())['GO_IDs']);unresolved=[r for r in old['samples'] if not r['schema_role_contract_supported']];assert len(unresolved)==8;replacements=[];probes=[]
 selected=[r for r in old['samples'] if not r['schema_role_contract_supported'] or r['native_family'] in ['UniProt-function-annotations','UniProt-full-record-annotations','UniProt-GOA-assertion-packets','Reactome98-static-pathway-prior']]
 bypath=collections.defaultdict(list)
 for r in selected:bypath[r['source_shard']['path']].append(r)
 for path,records in bypath.items():
  p=R/path;expected=records[0]['source_shard'];assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256'];pf=pq.ParquetFile(p);rows=next(pf.iter_batches(batch_size=4)).to_pylist();byID={r['task_id']:r for r in rows}
  for previous in records:
   row=byID[previous['task_id']];raw=row['native_payload'];assert hashlib.sha256(raw).hexdigest()==row['native_payload_sha256']==previous['native_payload_SHA256'];packet=json.loads(raw);result=row_exposure(packet,row['native_schema'],row['source_family'],protected);evidence=dict(task_id=row['task_id'],source_shard=expected,native_payload_SHA256=row['native_payload_sha256'],native_schema=row['native_schema'],source_family=row['source_family'],result=result,nonadditive_bounded_probe=True)
   probes.append(evidence)
   if not previous['schema_role_contract_supported']:replacements.append(evidence)
  assert time.monotonic()-start<120 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<512*2**20
 assert len(replacements)==8 and len(probes)==40
 p=O/'corrected-role-and-atomic-projection-probes.json';p.open('x').write(json.dumps(dict(replaces_only_eight_failed_v1_SciPlex_role_probes=True,all_other_v1_metadata_and_native_role_evidence_inherited=True,whole_master_absence_or_policy_activation_claim=False,probes=probes),indent=2)+'\n')
 manifest=dict(schema='mora3.MasterWideHeldoutGOBoundedRoleAddendum/v2',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_preparation=ref(I/'manifest.json'),unmodified_native_role_ID_sets=ref(I/'official-native-GO-role-ID-sets.json'),proposed_cross_protocol_zero_shot_IDs=ref(I/'proposed-union-of-unmodified-native-zero-shot-GO-IDs.json'),source_role_membership=ref(I/'native-GO-ID-role-membership.parquet'),prior_role_inventory=ref(I/'bounded-master-native-schema-role-inventory.json'),corrected_bounded_probes=ref(p),previous_SciPlex_role_errors_resolved=len(replacements),actual_rows_probed=len(probes),scope='bounded closed-row role and proposal probes only; no new whole-master scan or source admission',new_task_admission=False,policy_activated=False,master_publication=False,model_use=False,model_training=False,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 p=O/'manifest.json';p.open('x').write(json.dumps(manifest,indent=2)+'\n');print(json.dumps(ref(p),indent=2))
if __name__=='__main__':main()
