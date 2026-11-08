"""Complete available OpenADMET chemistry/group exposure audit, no admissions.

Shares the constructor's existing CPU26-27 budget; source/native task writers
and every existing index remain unchanged. Experimental equivalence is not
inferred from reference graphs or conservative leakage-group matches.
"""
from pathlib import Path
import collections,datetime,hashlib,json,os,resource,sqlite3,sys,time
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_admet_graph_scope import source_graph_scope
B=R/'outputs/data-intake/20261101-ADMET-corpus-v1';O=B/'chemical-and-protocol-group-audit-v2';M=R/'outputs/data-intake/20261101-LINCS-corpus-v1/independent-master-union-v22/master-corpus-manifest-v22-LINCS-qualified-source-coordinate-v1.json';H=R/'outputs/data-sources/20261101-ADMET-cross-modality-v1/handoff-OpenADMET-full-native-v1.json'

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(4*2**20):h.update(x)
 return h.hexdigest()
def ref(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def checked(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p
 assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p
def write(p,v):p.open('x').write(json.dumps(v,indent=2)+'\n')
def main():
 start=time.monotonic();os.sched_setaffinity(0,{26,27});os.nice(10);pa.set_cpu_count(1);pa.set_io_thread_count(1);assert not O.exists();O.mkdir();(O/'code').mkdir()
 for name in [Path(__file__).name,'data_intake_admet_graph_scope.py']:(O/'code'/name).open('xb').write((R/'scripts'/name).read_bytes())
 def guard():
  st=os.statvfs(R);rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;other=Path('/proc/3501176/status');other_rss=0
  if other.exists():
   for line in other.read_text().splitlines():
    if line.startswith('VmRSS:'):other_rss=int(line.split()[1])*1024
  if st.f_bavail*st.f_frsize<65*2**30 or time.monotonic()-start>1800 or rss>512*2**20 or rss+other_rss+128*2**20>8*2**30:raise RuntimeError('shared hardware/CPU/RSS/time guard; preserve partial audit')
 guard();h=json.loads(H.read_text());assert sha(H)=='b54afa8be4c25eaec66fb249ae96cfb2704cc64b3e4849217c46b0e7fc0a0eb5';sm=json.loads(checked(h['manifest']).read_text());m=json.loads(M.read_text());assert sha(M)=='82ef478d53ccbd2bb8a87607b1ce5dd35a6a0f58f6afa7d912aa214129c1e6f2';pubplan=json.loads((R/'outputs/data-intake/20261101-PubChem-native-corpus-v1/build-plan.json').read_text());pubpre=json.loads(checked(pubplan['preflight']).read_text());frozen=set(pubpre['frozen_pilot_validation_chemical_guards']);held=set(json.loads((R/'outputs/data-intake/20261101-RNAcentral-full-native-corpus-v1/preflight/actual-baseline-heldout-groups.json').read_text())['group_ids']);pubc=json.loads((R/'outputs/data-intake/20261101-PubChem-native-corpus-v1/census/manifest.json').read_text());pi=checked(pubc['graph_index']);db=sqlite3.connect('file:'+str(pi)+'?mode=ro&immutable=1',uri=True);native_parent={};queries=[];rawrefs=[];source_folds=collections.Counter();source_entity_counts=collections.Counter();graph_masks=collections.Counter()
 for f in sm['source_files']:
  if f['source']['path'].endswith(('expansion_data_train.csv','expansion_data_test.csv')):
   for row in pq.read_table(checked(f['payload'])).to_pylist():
    data=json.loads(row['native_row_json']);id=row['native_entity_id'];assert id not in native_parent;native_parent[id]='source_train' if f['source']['path'].endswith('train.csv') else 'source_test'
 for f in sm['source_files']:
  if not f['source']['path'].endswith(('expansion_data_raw.csv','ASAP/ADMET.csv')):continue
  p=checked(f['payload']);rawrefs.append(f['source']);rows=pq.read_table(p).to_pylist();assert len(rows)==f['rows'];payload_ref=ref(p)
  for i,row in enumerate(rows):
   native=json.loads(row['native_row_json']);family=row['native_entity_namespace'];original_split=native.get('Set');fold=native_parent.get(row['native_entity_id'],'source_unallocated') if family=='ExpansionRx' else {'Train':'source_train','Test':'source_test','Validation':'source_validation','Valid':'source_validation'}.get(original_split,'source_unallocated');literal=native['SMILES' if family=='ExpansionRx' else 'CXSMILES'];scope=source_graph_scope(literal);groups=set(scope['conservative_leakage_groups']);iso=hashlib.sha256(scope['whole_legacy_isomeric_SMILES'].encode()).digest();in_pub=db.execute('SELECT 1 FROM graphs WHERE iso=?',(iso,)).fetchone() is not None;in_audit=db.execute('SELECT 1 FROM validation_graphs WHERE iso=?',(iso,)).fetchone() is not None
   queries.append(dict(query_id=family+':'+str(i),source_family=family,native_entity_id=row['native_entity_id'],native_row_index=i,native_source_payload=payload_ref,native_original_source=f['source'],native_parent_protocol=fold,native_original_split=original_split,source_graph_scope_json=json.dumps(scope,sort_keys=True),coarse_graph_groups=sorted(groups),known_current_heldout_group_matches=sorted(groups&held),frozen_pilot_reservation_matches=sorted(groups&frozen),matches_full_admitted_PubChem_plain_isomeric_graph=bool(in_pub),PubChem_plain_exact_claim_eligible=scope['legacy_plain_SMILES_exact_comparison_eligible'],matches_PubChem_native_validation_audit_pool=in_audit,oldtrain_group_matches=[],source_heldout_parent_group_matches=[],global_training_admission=False,source_observation_or_material_join=False));source_entity_counts[family]+=1;source_folds[family+':'+fold]+=1;graph_masks[family+':enhanced_stereo']+=scope['source_enhanced_stereo_group_count']>0;graph_masks[family+':legacy_exact_graph_eligible']+=scope['legacy_plain_SMILES_exact_comparison_eligible'];graph_masks[family+':native_predictor_identity_candidate']+=scope['native_predictor_identity_candidate']
   if i%256==0:guard()
 assert len(queries)==8178;db.close();assert ref(pi)==pubc['graph_index'];target_groups=set(g for q in queries for g in q['coarse_graph_groups']);query_by_group=collections.defaultdict(list)
 for index,q in enumerate(queries):
  for group in q['coarse_graph_groups']:query_by_group[group].append(index)
 # Native protocol test/validation subjects remain reserved; shared scaffolds are explicit.
 source_held=collections.defaultdict(set)
 for q in queries:
  if q['native_parent_protocol'] in ('source_test','source_validation'):source_held[q['source_family']].update(q['coarse_graph_groups'])
 for q in queries:q['source_heldout_parent_group_matches']=sorted(set(q['coarse_graph_groups'])&source_held[q['source_family']])
 hits=collections.defaultdict(set);counts=collections.Counter();scanrefs=[];seen=set();pub_prior='PubChem-native-SMILES-reference-prior';families={'FreeSolv','ReactZyme','RGD1-computed-energy','BioGRID','Rosetta-CDRP-molecule-profile-CellPainting','Rosetta-CDRP-molecule-profile-L1000','Tox21-original-source-call-prior','NCI-AIDS-source-conclusion-prior','NCI-AIDS-source-endpoint-prior','NCI-AIDS-paired-assay-profile-prior','ChEMBL37-AZ-experimental-logD7_4','Reactome98-static-pathway-prior','Rosetta-LINCS-Pilot1-molecule-profile-CellPainting','Rosetta-LINCS-Pilot1-molecule-profile-L1000',pub_prior}
 for number,sh in enumerate(m['shards']):
  if sh.get('source_family') not in families|{None}:continue
  guard();gr=m['effective_group_lookup'][sh['path']];gp=checked(gr);assert pq.ParquetFile(gp).metadata.num_rows==sh['rows'];split=sh.get('split');per_id=None
  if split is None:
   data=checked(sh);allocation=pq.read_table(data,columns=['task_id','split']).to_pylist();per_id={r['task_id']:r['split'] for r in allocation};assert len(per_id)==sh['rows']
  scanrefs.append(dict(data_shard=sh,effective_groups=gr,data_sha_evidence='actual full hash when allocation mixed; otherwise unchanged prior master full proof',allocation=split));counts['covered_task_views']+=sh['rows'];counts['group_files_scanned']+=1
  for batch in pq.ParquetFile(gp).iter_batches(columns=['task_id','effective_groups_json'],batch_size=8192):
   for r in batch.to_pylist():
    allocation=per_id[r['task_id']] if per_id is not None else split;matches=target_groups&set(json.loads(r['effective_groups_json']))
    if matches:
     for group in matches:hits[group].add(allocation)
  if number%1000==0:print('OPENADMET_CURRENT_GROUP_SCAN',number+1,'covered',counts['covered_task_views'],'seconds',round(time.monotonic()-start,1),flush=True)
 for q in queries:q['oldtrain_group_matches']=sorted(g for g in q['coarse_graph_groups'] if 'train' in hits.get(g,set()))
 summary=collections.defaultdict(collections.Counter)
 for q in queries:
  k=q['source_family']+':'+q['native_parent_protocol'];s=summary[k];s['source_entity_occurrences']+=1;s['known_current_heldout_group_collision']+=bool(q['known_current_heldout_group_matches']);s['frozen_pilot_reservation_collision']+=bool(q['frozen_pilot_reservation_matches']);s['oldtrain_coarse_or_scaffold_exposed']+=bool(q['oldtrain_group_matches']);s['PubChem_admitted_literal_plain_graph_match']+=q['matches_full_admitted_PubChem_plain_isomeric_graph'] and q['PubChem_plain_exact_claim_eligible'];s['PubChem_native_validation_audit_only_graph_match']+=q['matches_PubChem_native_validation_audit_pool'];s['same_source_official_heldout_scaffold_or_component_overlap']+=bool(q['source_heldout_parent_group_matches'])
  scope=json.loads(q['source_graph_scope_json']);s['native_predictor_identity_candidates']+=scope['native_predictor_identity_candidate'];s['candidate_train_after_protected_and_source_protocol_guards']+=q['native_parent_protocol']=='source_train' and scope['native_predictor_identity_candidate'] and not q['known_current_heldout_group_matches'] and not q['frozen_pilot_reservation_matches'] and not q['source_heldout_parent_group_matches'];s['candidate_cold_source_test_after_oldtrain_and_protected_guards']+=q['native_parent_protocol'] in ('source_test','source_validation') and scope['native_predictor_identity_candidate'] and not q['oldtrain_group_matches'] and not q['known_current_heldout_group_matches'] and not q['frozen_pilot_reservation_matches']
 # Lists are serialized losslessly as JSON columns; never infer a new paired assay.
 output=[]
 for q in queries:
  r={k:v for k,v in q.items() if k not in ['coarse_graph_groups','known_current_heldout_group_matches','frozen_pilot_reservation_matches','oldtrain_group_matches','source_heldout_parent_group_matches','native_source_payload','native_original_source']}
  for k in ['coarse_graph_groups','known_current_heldout_group_matches','frozen_pilot_reservation_matches','oldtrain_group_matches','source_heldout_parent_group_matches','native_source_payload','native_original_source']:r[k+'_json']=json.dumps(q[k],sort_keys=True)
  output.append(r)
 path=O/'source-identity-and-protocol-group-review.parquet';pq.write_table(pa.Table.from_pylist(output),path,compression='zstd');assert pq.read_table(path).to_pylist()==output;guard();receipt=dict(schema='mora3.OpenADMETCompleteSourceChemicalProtocolGroupReview/v2',status='full_available_native_chemistry_and_current_known_group_exposure_audited;not_admitted',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),master=ref(M),source_handoff=ref(H),source_manifest=h['manifest'],all_native_source_entity_occurrences=dict(source_entity_counts),native_protocol_entity_counts=dict(source_folds),native_graph_scope_counts=dict(graph_masks),per_source_protocol_group_review={k:dict(v) for k,v in summary.items()},source_identity_group_review=ref(path,rows=len(output)),current_group_scan=dict(counts),current_group_scan_refs=scanrefs,full_PubChem_graph_index=pubc['graph_index'],frozen_pilot_source_reservations=pubplan['preflight'],source_test_validation_are_not_training=True,original_protocol_not_automatically_blind=True,native_enhanced_stereo_preserved_not_plain_exact_identity=True,coarse_component_or_scaffold_match_is_not_same_material_or_assay=True,original_source_observations_not_duplicated_by_ML_exports=True,new_source_pairing_or_task_admission=False,global_training_views_added=0,model_training=False,remaining_gates=['endpoint/readout/protocol/unit/evidence task-specific review','source-specific masks and train-only normalization/calibration','provisional compatible allocations with native benchmark and existing frozen reservations preserved','actual model native enhanced-stereo/context/censor/hidden-target bridge','broader peptide/protein/RNA/DNA ADMET source construction and full-readiness signoff'],source_originals_and_indexes_immutable=True,seconds=time.monotonic()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024);write(O/'manifest.json',receipt);print('COMPLETE_OPENADMET_GROUP_REVIEW',json.dumps(dict(source_counts=dict(source_entity_counts),summary={k:dict(v) for k,v in summary.items()},seconds=receipt['seconds'])),flush=True)
if __name__=='__main__':main()
