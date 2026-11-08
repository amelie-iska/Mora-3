"""New audit-only guard sidecars; native task data and allocations stay byte-identical."""
from pathlib import Path
import collections,datetime,hashlib,json,os,sys,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path.cwd();sys.path.insert(0,str(R/'scripts'))
from data_intake_protein_group_overlay import augment_groups,PROTEIN_SOURCES
B=Path('outputs/data-intake/20261101-protein-threshold-group-overlay-v1');B.mkdir(exist_ok=False);(B/'effective-groups').mkdir();(B/'code').mkdir();start=time.perf_counter()
for name in ['data_intake_build_protein_group_overlay.py','data_intake_protein_group_overlay.py']:(B/'code'/name).write_bytes((R/'scripts'/name).read_bytes())
M=Path('outputs/data-intake/20261031-uniprot-full-annotations-v1/master-corpus-manifest-v15-GOA-expanded-sourceannotations-v1.json');m=json.loads(M.read_text());H=Path('outputs/data-intake/20261101-qualified-protein-linclust-v1/manifest.json');h=json.loads(H.read_text())
def ref(p,**kw):
 p=Path(p);b=p.read_bytes();return dict(path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),**kw)
def check(e):
 p=Path(e['path']);got=ref(p);assert got['bytes']==e['bytes'] and got['sha256']==e['sha256'];return p
def live(stage,**kw):
 (B/'live-build.json').write_text(json.dumps(dict(pid=os.getpid(),stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.perf_counter()-start,**kw),indent=2)+'\n');print(stage,kw,round(time.perf_counter()-start,2),flush=True)
try:
 hm=ref(H);guardprefix='detected_protein_guard:'+hm['sha256'][:16]+':'
 mapping={}
 for row in pq.read_table(check(h['membership'])).to_pylist():
  digest=row['canonical_sequence_sha256'];guard=guardprefix+row['detected_group_id'];assert mapping.setdefault(digest,guard)==guard
 assert len(mapping)==104176
 (B/'build-plan.json').open('x').write(json.dumps(dict(schema='mora3.ExactQualifiedProteinGuardOverlayPlan/v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_master=ref(M),detected_native_groups=hm,allowed_protein_sources=sorted(PROTEIN_SOURCES),identity_gate='Exact sequence:SHA guard present in existing protein-specific task group set AND current107778qualifiedcanonical baseline;noaccession/gene/nameguess',all_native_payloads_and_splits_unchanged=True,original_effective_group_sidecars_immutable=True,new_groups_only_audit_not_predictors=True,global_homology_exhaustive=False,minimum_free_bytes=200*2**30,derived_cap_bytes=2*2**30),indent=2)+'\n')
 lookup=dict(m['effective_group_lookup']);changed_sources=collections.Counter();unmatched=collections.Counter();newgroups=collections.defaultdict(set);total=0;changedshards=0;newedges=0;held=collections.defaultdict(set);rowsby=collections.Counter();newrefs=[]
 for i,s in enumerate(m['shards']):
  p=Path(s['path']);assert p.stat().st_size==s['bytes'];rows=pq.read_table(p,columns=['task_id','source_family','split']).to_pylist();assert len(rows)==s['rows'];oldref=m['effective_group_lookup'][s['path']];gr=pq.read_table(check(oldref)).to_pylist();gm={r['task_id']:r for r in gr};assert len(gm)==len(rows)==oldref['rows'];assert set(gm)=={r['task_id'] for r in rows};newrows=[];changed=False
  for row in rows:
   old=gm[row['task_id']];before=json.loads(old['effective_groups_json']);after=augment_groups(before,row['source_family'],mapping);added=set(after)-set(before);assert set(before)<=set(after);total+=1;rowsby[row['split']]+=1
   if added:
    assert row['source_family'] in PROTEIN_SOURCES and row['split']=='train';expected={mapping[g[len('sequence:'):]] for g in before if g.startswith('sequence:') and g[len('sequence:'):] in mapping};assert added==expected;changed_sources[row['source_family']]+=1;newedges+=len(added);changed=True;newrows.append(dict(old,effective_groups_json=json.dumps(after)))
    for group in added:newgroups[group].add(row['source_family'])
   else:
    newrows.append(old)
    if row['source_family'] in PROTEIN_SOURCES:unmatched[row['source_family']]+=1
   if row['split'] in ['validation','test']:held[row['split']].update(after)
  if changed:
   q=B/'effective-groups'/f'{i:06d}.parquet';pq.write_table(pa.Table.from_pylist(newrows),q,compression='zstd');back=pq.read_table(q).to_pylist();assert back==newrows;assert {r['task_id'] for r in back}==set(gm);lookup[s['path']]=ref(q,rows=len(newrows));newrefs.append(lookup[s['path']]);changedshards+=1
  if i%500==0:live('exact qualified protein guard overlay and complete allocation audit',shards=i+1,tasks=total,changed_rows=sum(changed_sources.values()))
 assert total==m['serialized_tasks'] and dict(rowsby)==m['split_counts'];assert not set(newgroups)&held['validation'] and not set(newgroups)&held['test'];assert not held['validation']&held['test'];assert len(newgroups)==39589
 overlaps=[dict(detected_group=g,source_families=sorted(v),evidence='samequalifiedsequence/detectedthresholdguardacrosssourceviews;notindependentexperiment') for g,v in sorted(newgroups.items()) if len(v)>1];op=B/'cross-source-detected-protein-guard-overlaps.parquet';pq.write_table(pa.Table.from_pylist(overlaps),op,compression='zstd');assert pq.read_table(op).to_pylist()==overlaps
 newm=json.loads(M.read_text());newm.update(build_id='20261101-qualified-protein-threshold-group-corpus-v16',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supersedes_for_consumption=str(M),prior_manifest_sha256=ref(M)['sha256'],effective_group_lookup=lookup,verification=str(B/'verification.json'),additional_qualified_protein_detected_group_overlay=str(B/'verification.json'));newm['group_allocation']['qualified_protein_detected_threshold_guard']=dict(source_manifest=hm,overlay_receipt=str(B/'verification.json'),exact_current_sequence_gate=True,not_exhaustive_homology=True,no_identity_or_annotation_transfer=True);newm['effective_group_policy']='authoritativeversionedgroups_v15plus_exactqualifiedprotein_detectedguard;originalsource/envelopegroupspreserved;no_data_or_split_rewrite;thresholdgroups_not_exhaustive_homology';newm['verification_layers'].append(str(B/'verification.json'));newm['open_gates'].append('CurrentqualifiedproteinLinclust30identity80bidirectionalcoverageguardsareheuristicdetectedgroups,notallhomology/familyprooforblindheldout;proteincontextsremaintrainonly;sourceIDswithoutqualifiedexactsequencesnotlinked');assert newm['shards']==m['shards'] and newm['serialized_tasks']==m['serialized_tasks'] and newm['split_counts']==m['split_counts'] and newm['native_schema_counts']==m['native_schema_counts'] and newm['total_shard_bytes']==m['total_shard_bytes']
 used=sum(p.stat().st_size for p in B.rglob('*') if p.is_file());assert used<2*2**30;st=os.statvfs('.');assert st.f_bavail*st.f_frsize>=200*2**30
 target=B/'master-corpus-manifest-v16-protein-threshold-guards-v1.json';receipt=dict(schema='mora3.ExactQualifiedProteinGroupOverlayVerification/v1',status='passed',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_master=ref(M),prior_whole_data_and_semantics_verification=ref(Path(m['verification'])),qualified_group_source=hm,total_native_task_metadata_rows=total,all_prior_effective_group_bytes_and_task_allocations_revalidated=True,changed_task_rows=sum(changed_sources.values()),changed_rows_by_source=dict(changed_sources),eligible_source_rows_without_current_canonical_sequence_match=dict(unmatched),changed_group_sidecars=changedshards,new_group_sidecar_refs=newrefs,detected_groups_added=len(newgroups),new_guard_edges=newedges,cross_source_guard_overlap=ref(op,rows=len(overlaps)),no_prior_groups_removed=True,all_changed_groups_exact_current_sequence_evidence=True,all_new_sidecar_roundtrips=True,task_data_refs_bytes_and_allocations_unchanged=True,old_task_data_byte_and_semantic_checks_inherited_from_bound_v15=True,known_added_group_vs_validation_intersections=0,known_added_group_vs_test_intersections=0,known_old_trainheldout_intersections_inherited_zero_from_v15=True,protein_homology_exhaustive=False,source_parent_aliases_and_blind_evaluation_unresolved=True,new_tasks=0,new_sequences=0,new_experiments=0,optimizer_steps=0,master_path=str(target),master_totals=dict(tasks=newm['serialized_tasks'],shards=len(newm['shards']),data_bytes=newm['total_shard_bytes'],split_counts=newm['split_counts']),new_derived_bytes=used,code_refs=[ref(p) for p in (B/'code').iterdir()],seconds=time.perf_counter()-start)
 (B/'verification.json').open('x').write(json.dumps(receipt,indent=2)+'\n');target.open('x').write(json.dumps(newm,indent=2)+'\n');live('new protein guard corpus metadata published',master=str(target),changed_rows=sum(changed_sources.values()))
except BaseException as exc:
 (B/'failure.json').open('x').write(json.dumps(dict(error=repr(exc),seconds=time.perf_counter()-start),indent=2)+'\n');raise
