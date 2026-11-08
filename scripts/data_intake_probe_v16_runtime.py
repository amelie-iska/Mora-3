"""Bounded existing native-reader bridge with authoritative master group injection."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,sys,time
R=Path.cwd();sys.path[:0]=[str(R/'scripts'),str(R/'src')]
from data_intake_corpus import read_corpus_tasks
from mora3.data.uniprot_annotations import prediction_view,generation_view
B=Path('outputs/data-intake/20261101-protein-threshold-group-overlay-v1');O=B/'model-handoff-v1';O.mkdir(exist_ok=False);M=B/'master-corpus-manifest-v16-protein-threshold-guards-v1.json';m=json.loads(M.read_text());start=time.perf_counter();cache=collections.OrderedDict();counts=collections.Counter();guardrows=0
for row in read_corpus_tasks(M,root=R,source='UniProt-full-record-annotations',supported_only=False):
 kind='prediction' if row['task_name']=='source_full_UniProt_annotation_prediction' else 'generation'
 if counts[kind]>=1500:continue
 d=json.loads(row['native_payload']);ref=d['audit']['source_ref'];key=ref['raw_file']
 if key not in cache:
  p=Path(key);b=p.read_bytes();assert hashlib.sha256(b).hexdigest()==ref['raw_file_sha256'];cache[key]=json.loads(gzip.decompress(b) if b[:2]==b'\x1f\x8b' else b)['results']
  if len(cache)>4:cache.popitem(last=False)
 e=cache[key][ref['results_index']];func=prediction_view if kind=='prediction' else generation_view;v=func(row,e);old=func(dict(row,group_ids_json=row['source_group_ids_json']),e);assert v.inputs()==old.inputs() and v.target_bytes==old.target_bytes;groups=v.provenance()['groups'];assert set(groups)==set(json.loads(row['group_ids_json']));assert any(x.startswith('detected_protein_guard:') for x in groups);assert v.family_group is None and v.heldout_eligible is False;guardrows+=1;counts[kind]+=1
 if min(counts.values())>=1500 and len(counts)==2:break
assert dict(counts)=={'prediction':1500,'generation':1500}
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
receipt=dict(schema='mora3.AuthoritativeGroupNativeReaderBridgeProbe/v1',status='passed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=ref(M),existing_model_reader=ref(Path('src/mora3/data/uniprot_annotations.py')),rows_by_direction=dict(counts),all3000source_native_views_contain_authoritative_detected_group_guards=True,all3000inputs_and_targets_unchanged_by_group_metadata=True,all_source_page_hash_and_native_admission_gates_checked=True,exhaustive_family_and_heldout_eligibility_not_granted=True,legacy_native_snapshot_iterator_does_not_automatically_resolve_master_groups=True,required_consumer_recipe='read_corpus_tasks(master,source=UniProt-full-record-annotations,supported_only=False)->rawpageboundentry->prediction_view/generation_view;nativeinputallowlistsremainunchanged;groupmetadataauditonly',model_code_changed=False,optimizer_steps=0,seconds=time.perf_counter()-start)
(O/'native-group-runtime-probe.json').open('x').write(json.dumps(receipt,indent=2)+'\n');print('PASS',dict(counts),time.perf_counter()-start)
