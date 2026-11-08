"""Read-only full admitted-row compatibility with the existing native model views."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,os,sys,time
import pyarrow.parquet as pq
R=Path.cwd();sys.path.insert(0,str(R/'src'))
from mora3.data.uniprot_annotations import prediction_view,generation_view
from mora3.data.uniprot_residue import residue_targets
B=R/'outputs/data-intake/20261101-nucleotide-corpus-v1/model-handoff-v1';O=B/'full-native-contract-v1';O.mkdir(exist_ok=False);(O/'code').mkdir();start=time.perf_counter()
mp=B/'uniprot-admitted-native-reader-view.json';m=json.loads(mp.read_text());pages={r['path']:r for r in m['source_page_receipts']};cache=collections.OrderedDict();counts=collections.Counter();features=collections.Counter();nativebytes=collections.Counter();accessions=collections.defaultdict(set);sourcefiles={}
for name in ['uniprot_annotations.py','uniprot_residue.py']:
 p=R/'src/mora3/data'/name;raw=p.read_bytes();(O/'code'/name).write_bytes(raw);sourcefiles[str(p.relative_to(R))]=hashlib.sha256(raw).hexdigest()
def check(e):
 p=R/e['path'];b=p.read_bytes();assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256'];return b
def live(stage,**extra):
 (O/'live-verification.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,**extra),indent=2)+'\n');print(stage,extra,round(time.perf_counter()-start,1),flush=True)
try:
 for ix,s in enumerate(m['task_shards']):
  check(s);rows=pq.read_table(R/s['path']).to_pylist();assert len(rows)==s['rows']
  for row in rows:
   task=json.loads(row['native_payload']);ref=task['audit']['source_ref'];key=ref['raw_file']
   if key not in cache:
    p=pages[key];body=check(p);entries=json.loads(gzip.decompress(body) if p['compression']=='gzip' else body)['results'];assert len(entries)==p['entries'];cache[key]=entries
    if len(cache)>4:cache.popitem(last=False)
   entry=cache[key][ref['results_index']];kind='prediction' if row['task_name']=='source_full_UniProt_annotation_prediction' else 'generation';view=(prediction_view if kind=='prediction' else generation_view)(row,entry);assert view.accession not in accessions[kind];accessions[kind].add(view.accession);assert view.heldout_eligible is False;counts[kind]+=1;nativebytes[kind]+=len(view.target_bytes)
   if kind=='prediction':
    assert view.inputs()=={'canonical_sequence':entry['sequence']['value']}
    for f in residue_targets(view):features[f.coordinate_status]+=1
   else:
    assert set(view.inputs())=={'source_experimentally_supported_function_texts','source_experimental_associated_GO_ids','source_explicit_experimental_NOT_GO_ids','source_experimentally_supported_EC_ids','source_kinetic_conditions','organism_taxon_id'}
  if ix%100==0:live('full current-native-view source binding and target verification',shards=ix+1,counts=dict(counts))
 assert dict(counts)=={'prediction':107778,'generation':77082};assert accessions['generation']<=accessions['prediction']
 nowhash={str((R/k).relative_to(R)):hashlib.sha256((R/k).read_bytes()).hexdigest() for k in sourcefiles};stable=nowhash==sourcefiles
 receipt=dict(schema='mora3.FullAdmittedUniProtNativeViewCompatibility/v1',status='passed',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_path=str(mp.relative_to(R)),manifest_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),all184860admittedrows_pass_existing_native_prediction_or_generation_view=True,unique_accessions_by_direction={k:len(v) for k,v in accessions.items()},counts=dict(counts),complete_native_target_bytes=dict(nativebytes),all107778canonical_entry_residue_projections_verified=True,residue_source_coordinate_status=dict(features),model_source_hashes_at_import=sourcefiles,model_sources_unchanged_at_end=stable,model_source_hashes_at_end=nowhash,model_code_changed_by_data_worker=False,source_train_only=True,homology_heldout_unresolved=True,generic_historical_model_adapter_flags_unchanged=False,existing_iterator_diagnostic_limit=1500,scope='Full data-model-view compatibility;not native full-target objective coverage,neural training or heldoutquality',new_optimizer_steps=0,seconds=time.perf_counter()-start)
 (O/'receipt.json').open('x').write(json.dumps(receipt,indent=2)+'\n');live('all corrected admitted UniProt native contracts verified',counts=dict(counts))
except BaseException as exc:
 (O/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
