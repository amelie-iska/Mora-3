"""Bind real sources, pilot conservation, finite caps and masks before a full build."""
from pathlib import Path
import sys,json,hashlib,datetime,time,collections,shutil,os,subprocess,importlib.metadata
import pyarrow.parquet as pq,pyarrow.compute as pc
import selfies as sf
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import CorpusTask
from data_intake_chemical_projection_scope import project_source_graph
from data_intake_full_pubchem_census import validate_rows
from data_intake_full_pubchem_guards import BuildGuard,immutable_json
O=R/'outputs/data-intake/20261101-PubChem-native-corpus-v1';B=R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1';M=B/'independent-master-union-v20/master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json';S=R/'outputs/data-sources/20261031-legacy-mint-v1/curated/PubChem10m-standardSELFIES-full-v1/manifest.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def main():
 start=time.perf_counter();O.mkdir();(O/'preflight').mkdir();(O/'code').mkdir();deadline=time.time()+8*3600;guard=BuildGuard(O,census_cap=2*2**30,total_cap=12*2**30,deadline_epoch=deadline);guard.check('census',reserve=64*2**20);assert shutil.disk_usage(R).free>=212*2**30;assert sha(M)=='fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f';assert sha(S)=='00d6cb2cd0239c8c03e5a43671dccc432d6b692043d35290abf494e69703c33e';ack=R/'outputs/monitoring/budget-20261101T055515Z-PubChem-full12GiB-and-MINT-pilot8GiB-v1.json';assert sha(ack)=='3d6eb2a7e50d598cd5219acb0ada9766409bf0f37a35f8a1c53ae84cd0853a8c';source=json.loads(S.read_text());m=json.loads(M.read_text());assert {n:importlib.metadata.version(n) for n in source['versions']}==source['versions'] and sf.get_semantic_constraints()==source['semantic_constraints'];assert Path('/usr/bin/timeout').is_file()
 for name in [Path(__file__).name,'data_intake_full_pubchem_census.py','data_intake_full_pubchem_guards.py','data_intake_PubChem_tasks.py','data_intake_chemical_projection_scope.py','data_intake_corpus.py']:(O/'code'/name).write_bytes((R/'scripts'/name).read_bytes())
 jobs=[];rawassets={}
 for chunkref in source['chunks']:
  cp=Path(chunkref['path']);assert sha(cp)==chunkref['sha256'];chunk=json.loads(cp.read_text());asset=chunk['job']['source_asset'];mode=asset['source_native_split'];rawassets[mode]=asset
  for f in chunk['payloads']:
   assert Path(f['path']).exists() and Path(f['path']).stat().st_size==f['bytes'];assert pq.ParquetFile(f['path']).metadata.num_rows==f['rows'];jobs.append(dict(mode=mode,source_payload=f))
 for mode,count in [('train',8999964),('validation',999996)]:
  ranges=sorted((j['source_payload']['source_row_start'],j['source_payload']['source_row_stop']) for j in jobs if j['mode']==mode);position=0
  for begin,end in ranges:assert begin==position and end>begin;position=end
  assert position==count;asset=rawassets[mode];check(asset);assert sha(Path(asset['path']))==asset['expected_sha256']
 jobs.sort(key=lambda j:(j['mode']!='validation',j['source_payload']['source_row_start']));assert len(jobs)==2000
 heldout=set()
 for i,sh in enumerate(m['shards']):
  g=m['effective_group_lookup'][sh['path']];t=pq.read_table(check(g),columns=['task_id','effective_groups_json']);mask=pc.or_(pc.match_substring(t['effective_groups_json'],'molecule:'),pc.match_substring(t['effective_groups_json'],'scaffold:'))
  if not pc.any(mask).as_py():continue
  splits={row['task_id']:row['split'] for row in pq.read_table(R/sh['path'],columns=['task_id','split']).to_pylist()}
  for row in t.filter(mask).to_pylist():
   if splits[row['task_id']] in ['validation','test']:heldout.update(group for group in json.loads(row['effective_groups_json']) if group.startswith(('molecule:','scaffold:')))
  if i%1000==0:print('FULL_PREFLIGHT_MASTER',i+1,'heldoutguards',len(heldout),flush=True)
 pcpath=B/'manifest.json';pilot=json.loads(pcpath.read_text());ledger=pq.read_table(check(pilot['selected_source_ledger'])).to_pylist();frozen=set();pilot_by_id={};probes=[]
 for row in ledger:
  if row['source_native_split']=='validation' and row['independently_validated_SMILES']:
   ident=json.loads(row['derived_identity_json']);frozen.update(ident[k] for k in ['molecule_group','scaffold_group'])
  if json.loads(row['admitted_task_ids_json']):pilot_by_id[row['source_record_id']]=row
 for i,row in enumerate(pilot_by_id.values()):
  native=json.loads(row['all_native_source_columns_json']);projection=project_source_graph(native['native_smiles']);assert projection['exact_projection_eligible'];groups={ident[k] for kind,ident in projection['graphs'] for k in ['molecule_group','scaffold_group']};assert not groups&heldout and not groups&frozen,'newcomponentguardsconflictwitholdpilot;stopbeforebulk';assert projection['graphs'][0][1]['canonical_isomeric_smiles']==native['source_canonical_isomeric_smiles'];assert hashlib.sha256(native['source_canonical_isomeric_smiles'].encode()).hexdigest()==native['source_current_canonical_graph_sha256']
  if i%5000==0:print('FULL_PREFLIGHT_PILOT',i+1,flush=True)
 assert len(pilot_by_id)==27319;ids=set();views=0;mutations=0
 for sh in pilot['shards']:
  path=check(sh);rows=pq.read_table(path).to_pylist();assert len(rows)==sh['rows']
  for row in rows:
   assert row['record_id'] in pilot_by_id and row['task_id'] not in ids;ids.add(row['task_id']);task=CorpusTask.from_bytes(row['native_payload']);assert task.canonical_bytes()==row['native_payload'] and hashlib.sha256(task.canonical_bytes()).hexdigest()==row['native_payload_sha256'];d=json.loads(task.canonical_bytes());assert CorpusTask(d['task'],d['conditions'],{'changed_hidden_target':'X'},{'sourceID_or_target_length':'changed'},['changed_missing']).input_bytes()==task.input_bytes();mutations+=1;views+=1
 assert views==81155
 # Bounded source probes include invalid, unsupported-codec, longest and regular rows.
 regular=pq.read_table(jobs[-1]['source_payload']['path']).slice(0,32).to_pylist();unsupported=[json.loads(r['all_native_source_columns_json']) for r in ledger if r['native_parse_pass'] and not r['SELFIES_graph_roundtrip_pass']][:8];invalid=[json.loads(r['all_native_source_columns_json']) for r in ledger if not r['native_parse_pass']][:8];longest=max((json.loads(r['all_native_source_columns_json']) for r in ledger),key=lambda r:len(r['native_smiles'] or ''));probes=regular+unsupported+invalid+[longest];out,gs,errors,counts=validate_rows(probes,'train');assert sum(r['derived_native_graph_pass'] for r in out)==len(regular)+len(unsupported)+1;assert all(out[len(regular)+i]['derived_native_graph_pass'] and not out[len(regular)+i]['derived_SELFIES_graph_pass'] for i in range(len(unsupported)));assert len(errors)==len(invalid)
 p=O/'preflight/receipt.json';immutable_json(p,dict(schema='mora3.PubChemFullBuildPreflight/v1',status='passed',master=ref(M),source=ref(S),source_raw_original_LFS_hashes_rechecked=[ref(Path(a['path'])) for a in rawassets.values()],all2000native_payload_footer_byte_ranges_conserve_fullsource=True,pilot_original_graphs_strict_projection_and_component_guards_checked=27319,pilot_task_IDs_and_bytes_hash_conserved=81155,pilot_hidden_target_audit_mutation_invariance=mutations,native_source_probe_rows=len(probes),native_source_probe_counts=counts,source_invalid_preserved=len(invalid),source_unsupported_codec_SMILES_retained=len(unsupported),old_heldout_chemical_guards=sorted(heldout),frozen_pilot_validation_chemical_guards=sorted(frozen),software_distribution_versions={n:importlib.metadata.version(n) for n in source['versions']},source_semantic_constraints_exact=True,strict_scope_independent_proof=ref(B/'model-handoff-v2/strict-scope-independent-readback.json'),old_sources_and_pilot_unmodified=True,seconds=time.perf_counter()-start));plan=dict(schema='mora3.PubChemFullNativeBuildPlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),output_namespace=str(O.relative_to(R)),master=ref(M),source=ref(S),pilot_component=ref(pcpath),source_full_proof_handoff=ref(R/'outputs/data-sources/20261031-legacy-mint-v1/handoff-PubChem10m-full-standardSELFIES-verified-v1.json'),strict_scope=ref(B/'model-handoff-v2/strict-scope-independent-readback.json'),preflight=ref(p),budget_ack=ref(ack),source_payload_jobs=jobs,deadline_epoch=deadline,deadline_utc=datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).isoformat(),max_CPU_affinity=2,GPU=0,network=False,census_phase_cap_bytes=2*2**30,materialized_phase_cap_bytes=10*2**30,total_new_derived_cap_bytes=12*2**30,minimum_free_bytes=200*2**30,seed=20261031,source_train_rows=8999964,source_validation_rows_audit_only=999996,new_heldouts=False,model_adapter_supported=False,source_biological_release_unknown=True,source_CIDs_missing=True,standardSELFIES_not_originalGroupSELFIES=True,task_specific_masks='SMILESvalidSFunsupportedkeepsreferenceprior;sourceinvalid/lossystereoprojectionquarantinesretainedbyoriginalrefs;noassayproperty3Dimageexpressionlabels',gate_code_bits={'1':'native_or_schema_projection_quarantine','2':'validation_audit_only','4':'global_native_validation_exact_graph','8':'oldmaster_valtest_whole_or_component_molecule/scaffold','16':'frozen_pilot_selected_validation_whole_or_component_guard','32':'lossy_stereo_sourceprojection'},checkpoint_plan='commitnewSQLitepercomplete5knativesourcechunk;exclusiveimmutableledgerParquet+chunkreceipt;actualmutablelive-progressonly;noreset/delete/cleanup;oncap/deadlinestoppreservingpartials;phaseBonlyafterwholecensusandindependentproof',source_observation_unit='originalsourceoccurrences;referencegraphdedupseparate;codecviewsnotbiologicalexperiments',existing_pilot_IDs_payloads_and_masks_preserved=True,final_mixture_weights=None,optimizer_steps=0);immutable_json(O/'build-plan.json',plan);guard.check('census');print('FULL_PREFLIGHT_PASS',json.dumps(dict(plan=ref(O/'build-plan.json'),receipt=ref(p),pilot_graphs=27319,pilot_views=views,source_probes=len(probes),deadline_utc=plan['deadline_utc'],seconds=time.perf_counter()-start)),flush=True)
if __name__=='__main__':main()
