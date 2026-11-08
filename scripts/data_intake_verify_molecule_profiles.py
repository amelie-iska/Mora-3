from pathlib import Path
import sys,json,hashlib,collections,time,copy
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import chemical_identity,canonical,read_corpus_tasks
B=R/'outputs/data-intake/20261031-rosetta-molecule-profile-v1';out=B/'independent-source-evidence-v1';out.mkdir();start=time.perf_counter();base=R/'outputs/data-intake/20261031-subagent/curated/cpg0003-rosetta-profile-links-v1';primary=json.loads((base/'manifest.json').read_text());metadata={};samples=collections.defaultdict(list);sourcefiles=[]
def rec(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),**kw)
for mod in ['CellPainting','L1000']:
 p=base/f'CDRPBIO-BBBC036-Bray--{mod}--metadata.parquet';expected=primary['artifacts'][p.name];assert p.stat().st_size==expected['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'];sourcefiles.append(rec(p))
 for row in pq.read_table(p).to_pylist():
  metadata[row['record_id']]=row
  if mod=='L1000' and not row['control']:
   md=json.loads(row['source_metadata_json']);smiles=md.get('CPD_SMILES');sample=md.get('pert_id')
   if smiles and sample:samples[sample].append(dict(source_L1000_record_id=row['record_id'],source_L1000_row=row['source_row'],source_CPD_SMILES=smiles,canonical_isomeric_smiles=chemical_identity(smiles)['canonical_isomeric_smiles'],source_metadata_file=str(p.relative_to(R)),source_metadata_sha256=expected['sha256']))
assert all(len({r['canonical_isomeric_smiles'] for r in refs})==1 for refs in samples.values())
prior=R/'outputs/data-intake/20261031-chembl-logd-context-v1/master-corpus-manifest-v7.json';arrays={}
for family in ['Rosetta-CellPainting','Rosetta-L1000']:
 for row in read_corpus_tasks(prior,source=family,supported_only=True):
  if row['record_id'].startswith('CDRPBIO-BBBC036-Bray:'):
   d=json.loads(row['native_payload']);a=d['arrays'][0];arrays[row['record_id']]=dict(hash=hashlib.sha256(canonical(a)).hexdigest(),axis=len(a['values']),source_native_sha256=row['native_payload_sha256'])
v8=B/'master-corpus-manifest-v8.json';manifest=json.loads(v8.read_text());newrec=json.loads((B/'curated-corpus-v1/receipt.json').read_text());checked=0;counts=collections.Counter();byrecord=collections.defaultdict(list);evidence={}
for shard in newrec['shards']:
 p=R/shard['path'];data=p.read_bytes();assert len(data)==shard['bytes'] and hashlib.sha256(data).hexdigest()==shard['sha256'];rows=pq.read_table(p).to_pylist();assert len(rows)==shard['rows']
 for row in rows:
  raw=row['native_payload'];assert hashlib.sha256(raw).hexdigest()==row['native_payload_sha256'];d=json.loads(raw);record=row['record_id'];mdrow=metadata[record];md=json.loads(mdrow['source_metadata_json']);ctx=json.loads(mdrow['source_context_json']);mod=mdrow['modality'];sample=md['Metadata_broad_sample'] if mod=='CellPainting' else md['pert_id'];refs=samples[sample];iso=refs[0]['canonical_isomeric_smiles']
  if mod=='L1000':assert any(r['source_L1000_record_id']==record for r in refs)
  conditions=d['conditions'];targets=d['targets'];forward=row['task_name']=='molecule_to_profile';array=(targets if forward else conditions)['source_profile_array'];molecule=(conditions if forward else targets)['canonical_isomeric_smiles'];assert molecule==iso;assert conditions['cell_line']==ctx.get('cell_line');assert hashlib.sha256(canonical(array)).hexdigest()==arrays[record]['hash']==row['original_array_sha256'];assert len(array['values'])==arrays[record]['axis']==(1783 if mod=='CellPainting' else 977)
  assert 'physical_dose' in d['missing'] and 'exposure_time' in d['missing'];assert 'dose' not in conditions and 'exposure_time' not in conditions
  allowed={'assay_modality','cell_line','canonical_isomeric_smiles','known_profile_axis'} if forward else {'assay_modality','cell_line','source_profile_array'};assert set(conditions)==allowed
  assert row['original_source_native_payload_sha256']==arrays[record]['source_native_sha256'];checked+=1;counts[mod+':'+row['task_name']]+=1;byrecord[record].append(row['task_name'])
  evidence[record]=dict(source_profile_record_id=record,modality=mod,source_chemical_sample_id=sample,canonical_isomeric_smiles=iso,molecular_reference_source_json=json.dumps(refs,sort_keys=True),exact_source_profile_native_payload_sha256=arrays[record]['source_native_sha256'],full_profile_array_sha256=arrays[record]['hash'],full_axis=arrays[record]['axis'],explicit_cell_line=ctx.get('cell_line'),physical_dose=None,exposure_time=None,identity_join_scope='molecular_identity_metadata_only;not_assay_response_pair',same_cell_or_matched_CP_L1000=False)
assert checked==newrec['serialized_tasks'] and all(sorted(v)==['molecule_to_profile','profile_to_molecule'] for v in byrecord.values())
p=out/'profile-molecular-source-evidence.parquet';pq.write_table(pa.Table.from_pylist(list(evidence.values())),p,compression='zstd');assert pq.read_table(p).to_pylist()==list(evidence.values())
receipt=dict(schema='mora3.independent_full_molecule_profile_source_readback.v1',verified=True,new_tasks_checked=checked,distinct_existing_source_profiles=len(byrecord),counts=dict(counts),full_axis_hashes_match_original_native_arrays=True,all_molecular_conditions_or_targets_match_original_CPD_SMILES=True,exact_L1000_source_chemical_metadata_refs_preserved=True,all_explicit_cell_lines_match_source_without_plate_parsing=True,no_physicaldose_time_or_samecell_joins_invented=True,predictor_keys_exactly_allowlisted=True,complete_source_metadata_hashes_verified=sourcefiles,evidence=rec(p,rows=len(evidence)),no_raw_download=True,model_optimizer_steps=0,seconds=time.perf_counter()-start)
with (out/'receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
enriched=copy.deepcopy(manifest);enriched['independent_molecule_profile_source_evidence']=rec(out/'receipt.json');enriched['verification_layers'].append(str((out/'receipt.json').relative_to(R)));enriched['prior_manifest_sha256']=hashlib.sha256(v8.read_bytes()).hexdigest();enriched['supersedes_for_consumption']=str(v8.relative_to(R));enriched['build_id']+='-source-evidence-v1'
with (B/'master-corpus-manifest-v8-source-evidence-v1.json').open('x') as f:json.dump(enriched,f,indent=2)
print(json.dumps({k:receipt[k] for k in ['verified','new_tasks_checked','distinct_existing_source_profiles','counts','seconds']}))
