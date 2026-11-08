"""Full canonical GOA replacement: immutable payloads, independent all-row admission audit."""
from pathlib import Path
import sys,json,gzip,csv,hashlib,collections,datetime,time,functools,shutil
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import canonical,CorpusTask
from data_intake_admit_uniprot_annotations import resolved_shard_rows
from data_intake_goa_projection import annotation_projection
from data_intake_protein_group_overlay import augment_groups
B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1'
O=R/'outputs/data-intake/20261101-full-GOA-corpus-union-v1'
PRIOR=R/'outputs/data-intake/20261101-protein-threshold-group-overlay-v1/master-corpus-manifest-v16-protein-threshold-guards-v1.json'
SOURCE='UniProt-GOA-assertion-packets'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rec(path,**kw):return dict(path=str(path.relative_to(R)),bytes=path.stat().st_size,sha256=sha(path),**kw)
def check(record):
 path=R/record['path'];assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'];return path
@functools.lru_cache(maxsize=8)
def raw_page(path,digest):
 p=R/path;data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
 return json.loads(gzip.decompress(data) if data[:2]==b'\x1f\x8b' else data)['results']

def main():
 start=time.perf_counter();assert shutil.disk_usage(R).free>210*2**30;O.mkdir();(O/'code').mkdir();(O/'effective-groups').mkdir()
 for name in [Path(__file__).name,'data_intake_protein_group_overlay.py','data_intake_goa_projection.py']:(O/'code'/name).write_bytes((R/'scripts'/name).read_bytes())
 m=json.loads(PRIOR.read_text());prior_hash=sha(PRIOR);oldpath=R/m['additional_GOA_snapshot_admission_manifest'];old=json.loads(oldpath.read_text());gpath=B/'GOA-typed-complete-v5/manifest.json';g=json.loads(gpath.read_text())
 assert g['full107778coverage'] and g['canonical_queries']==107778 and g['source_license']=='CC-BY-4.0';assert set(old['query_accessions'])<=set(g['query_accessions'])
 handoff=R/'outputs/data-sources/20261031-goa-companion-v1/handoff-manifest-quickgo-parallel-v2.json';assert sha(handoff)=='2e425d5438aa801f90314b7c70556ed2ad52e4f1d9ac555e74332fc8e33b73d9'
 with gzip.open(R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as stream:baseline={r['Entry']:r['Sequence'] for r in csv.DictReader(stream,delimiter='\t')}
 assert set(g['query_accessions'])==set(baseline) and len(baseline)==107778
 # Source-bound form references remain separate from current canonical entry sequences.
 fm=json.loads((B/'physical-form-supplement-v2/manifest.json').read_text());forms={}
 for file in fm['files']:
  if file['kind']=='forms':
   for row in pq.read_table(check(file)).to_pylist():
    form=json.loads(row['record_json']);forms[(form['accession'],form['form_id'])]=form
 bm=json.loads((B/'form-sequence-coordinate-bindings-v1/manifest.json').read_text())
 for file in bm['binding_files']:
  for row in pq.read_table(check(file)).to_pylist():
   form=json.loads(row['record_json']);forms[(form['primary_accession'],form['declared_form_id'])]=form
 ambiguous=set(json.loads((B/'form-sequence-coordinate-bindings-v1/global_form_ID_multiple_sequence_hashes.json').read_text()))
 isoforms={};im=json.loads((B/'isoform-sequences-v2/manifest.json').read_text());native_isoform_files=im['native_shards'][:]
 extra=json.loads((B/'isoform-exact-parents-v1/manifest.json').read_text())
 for partref in extra['complete_batches']:native_isoform_files.append(json.loads(check(partref).read_text())['native_sequences'])
 for file in native_isoform_files:
  for row in pq.read_table(check(file)).to_pylist():
   if not row['is_canonical_record']:
    sid=row['source_accession'];assert sid not in isoforms or isoforms[sid]==row['sequence'];isoforms[sid]=row['sequence']
 hp=R/'outputs/data-intake/20261101-qualified-protein-linclust-v1/manifest.json';h=json.loads(hp.read_text());guard_prefix='detected_protein_guard:'+sha(hp)[:16]+':'
 seqgroups={}
 for row in pq.read_table(check(h['membership'])).to_pylist():
  digest=row['canonical_sequence_sha256'];guard=guard_prefix+row['detected_group_id'];assert digest not in seqgroups or seqgroups[digest]==guard;seqgroups[digest]=guard
 assert len(seqgroups)==104176
 # Old task identities and payload hashes must be conserved, without appended duplicates.
 prior_packets={}
 for sh in old['task_shards']:
  for row in pq.read_table(check(sh),columns=['task_id','native_payload_sha256']).to_pylist():
   assert row['task_id'] not in prior_packets;prior_packets[row['task_id']]=row['native_payload_sha256']
 assert len(prior_packets)==old['serialized_task_packets']
 oldpaths={sh['path'] for sh in old['task_shards']};assert oldpaths=={sh['path'] for sh in m['shards'] if sh.get('source_family')==SOURCE}
 m['shards']=[sh for sh in m['shards'] if sh['path'] not in oldpaths];m['effective_group_lookup']={key:value for key,value in m['effective_group_lookup'].items() if key not in oldpaths}
 delta=g['serialized_task_packets']-old['serialized_task_packets'];m['serialized_tasks']+=delta;m['total_shard_bytes']+=g['task_shard_bytes']-old['task_shard_bytes'];m['split_counts']['train']+=delta;m['native_schema_counts']['mora3.CorpusTask/v1']+=delta;m['adapter_support_counts']['mora3.CorpusTask/v1;model_adapter_supported=False']+=delta;m['serialized_distinct_record_counts_by_source'][SOURCE]=g['serialized_task_packets']
 overlay_rows=0;unmatched_overlay_rows=0
 for index,sh in enumerate(g['task_shards']):
  original=pq.read_table(check(g['effective_group_lookup'][sh['path']])).to_pylist();rows=[]
  for row in original:
   groups=json.loads(row['effective_groups_json']);augmented=augment_groups(groups,SOURCE,seqgroups);overlay_rows+=len(augmented)>len(groups);unmatched_overlay_rows+=len(augmented)==len(groups);rows.append(dict(task_id=row['task_id'],effective_groups_json=json.dumps(augmented)))
  target=O/'effective-groups'/f'{index:05d}.parquet';pq.write_table(pa.Table.from_pylist(rows),target,compression='zstd');assert pq.read_table(target).to_pylist()==rows;m['effective_group_lookup'][sh['path']]=rec(target,rows=len(rows))
 m['shards']+=g['task_shards'];m['source_ledger']=[x for x in m['source_ledger'] if x.get('source_family')!=SOURCE]
 m['source_ledger'].append(dict(source_family=SOURCE,inventory_id='INV-005',raw_available=g['raw_available_annotations'],quality_pass=g['quality_pass_annotations'],admitted_annotation_views=g['raw_available_annotations']-g['quarantined_annotations'],serialized_task_packets=g['serialized_task_packets'],admitted_records=g['serialized_task_packets'],quarantined_annotations=g['quarantined_annotations'],quarantined_subjects=g['quarantined_subjects'],count_unit='sourceassertions_and_form_specific_task_packets;notnewexperiments',release='GOAannotations2026-07-28;qualifiedcanonical/formsequencesKB2026_03',license='CC-BY-4.0',note='All107778canonicalqueryscopecomplete;allpossibleisoformfiltercoverageNOTasserted;historicalGOAsubjectsequenceversionunknown;GPADrequirednullshortcodesretained;missingunknown;noinheritance;trainonly;neuralreaderunsupported'))
 # Quarantines remain source available/QC rather than silently disappearing.
 quarantine=json.loads(check(g['quarantine_receipt']).read_text());qids={aid for q in quarantine for aid in q['source_annotation_ids']};assert len(qids)==g['quarantined_annotations'];qseen=set()
 for file in g['assertion_files']:
  table=pq.read_table(check(file));assert len(table)==file['rows']
  for row in table.to_pylist():
   if row['annotation_id'] in qids:
    ref=json.loads(row['source_ref_json']);a=raw_page(ref['raw_file'],ref['raw_sha256'])[ref['results_index']];assert a['id']==row['annotation_id'];assert json.loads(row['projection_json'])==annotation_projection(a);qseen.add(a['id'])
 assert qseen==qids
 ids=set();groups=collections.defaultdict(set);counts=collections.Counter();schema=collections.Counter();adapters=collections.Counter();source_records=collections.defaultdict(set);source_tasks=collections.Counter();data_bytes=group_bytes=0;packets=annotations=preserved=0;bindings=collections.Counter();annotation_ids=set();form_source_checks=0
 for index,sh in enumerate(m['shards']):
  p=check(sh);data_bytes+=p.stat().st_size;gp=check(m['effective_group_lookup'][sh['path']]);group_bytes+=gp.stat().st_size;meta=resolved_shard_rows(p,gp,sh);native=sh.get('source_family')==SOURCE
  columns=['task_id','record_id','source_family','split','native_schema','model_adapter_supported']
  if native:columns+=['native_payload','native_payload_sha256','admitted_for_task','full_axis_values']
  rows=pq.read_table(p,columns=columns).to_pylist();assert len(rows)==sh['rows']==len(meta)
  for row,group_row in zip(rows,meta):
   assert row['task_id']==group_row['task_id'] and row['task_id'] not in ids;ids.add(row['task_id']);counts[row['split']]+=1;schema[row['native_schema']]+=1;adapters[row['native_schema']+';model_adapter_supported='+str(row['model_adapter_supported'])]+=1;source_records[row['source_family']].add(row['record_id']);source_tasks[row['source_family']]+=1;rowgroups=json.loads(group_row['effective_groups_json']);groups[row['split']].update(rowgroups)
   if not native:continue
   assert row['split']=='train' and row['admitted_for_task'] and not row['model_adapter_supported'];payload=row['native_payload'];assert hashlib.sha256(payload).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(payload).canonical_bytes()==payload;d=json.loads(payload);assert d['task']=='source_GOA_assertion_packet_prediction';assert set(d['conditions'])=={'protein_sequence'}
   subject=row['record_id'];assert subject.startswith('UniProtKB:');sid=subject[len('UniProtKB:'):];binding=d['audit']['binding'];parent=binding['parent_accession'];seq=d['conditions']['protein_sequence'];assert parent in baseline and binding['subject']==subject and binding['canonical_labels_inherited'] is False and sid not in ambiguous
   kind=binding['sequence_binding'];bindings[kind]+=1
   if kind=='exact_current_canonical_source':assert sid==parent and seq==baseline[parent]
   elif kind=='exact_canonical_interval_primary_sequence':
    assert sid.startswith(parent+'-PRO_');form=forms[(parent,sid[len(parent)+1:])];assert seq==form['verified_sequence'];start0,end0=form['canonical_interval0'];assert seq==baseline[parent][start0:end0];form_source_checks+=1
   elif kind=='exact_official_same_release_isoform_FASTA':
    assert sid.startswith(parent+'-') and seq==isoforms[sid];form=forms[(parent,sid)];assert seq==form['verified_sequence'];form_source_checks+=1
   elif kind=='sourceDisplayed_alias_exact_canonical_sequence':
    form=forms[(parent,sid)];assert seq==form['verified_sequence']==baseline[parent];assert form.get('source_sequence_status',form.get('sequence_status'))=='Displayed';form_source_checks+=1
   else:raise AssertionError('Unqualified GOA sequence binding')
   digest=hashlib.sha256(seq.encode()).hexdigest();assert row['full_axis_values']==len(seq);assert {'accession:'+parent,'sequence:'+digest,'protein_form:'+subject}<=set(rowgroups)
   expected_guard=seqgroups.get(digest);actual_guard={x for x in rowgroups if x.startswith('detected_protein_guard:')};assert actual_guard==({expected_guard} if expected_guard else set())
   source=d['targets']['source_GO_annotations'];typed=d['targets']['typed_GO_annotations'];refs=d['audit']['source_refs'];assert len(source)==len(typed)==len(refs) and d['audit']['source_snapshot']=='2026-07-28 10:08';assert any(t['source_positive_GO_head_mask'] or t['source_negative_GO_head_mask'] for t in typed)
   for a,t,ref in zip(source,typed,refs):
    original=raw_page(ref['raw_file'],ref['raw_sha256'])[ref['results_index']];assert a==original and a['geneProductId']==subject and a['id'] not in annotation_ids and a['id'] not in qids;annotation_ids.add(a['id']);assert t==annotation_projection(a) and json.loads(t['full_source_json'])==a and t['missing_is_negative'] is False;annotations+=1
   packets+=1
   if row['task_id'] in prior_packets:assert row['native_payload_sha256']==prior_packets[row['task_id']];preserved+=1
  if index%500==0:print('FULL_GOA_UNION',index+1,'tasks',len(ids),'packets',packets,'annotations',annotations,'seconds',round(time.perf_counter()-start,2),flush=True)
 assert len(ids)==m['serialized_tasks'] and dict(counts)==m['split_counts'] and dict(schema)==m['native_schema_counts'] and dict(adapters)==m['adapter_support_counts'] and data_bytes==m['total_shard_bytes'];assert {key:len(value) for key,value in source_records.items()}==m['serialized_distinct_record_counts_by_source'];assert preserved==len(prior_packets) and packets==g['serialized_task_packets'] and annotations==g['raw_available_annotations']-g['quarantined_annotations'];assert len(annotation_ids)+len(qids)==g['raw_available_annotations']
 intersections={a+'__'+b:len(groups[a]&groups[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]};assert not any(intersections.values());assert sha(PRIOR)==prior_hash
 receipt=dict(schema='mora3.FullCanonicalGOACorpusIndependentVerification/v1',status='verified',prior_manifest=rec(PRIOR),source_handoff=rec(handoff),GOA_component=rec(gpath),canonical_queries=len(baseline),all_canonical_queries_exact_disjoint_complete=True,all_possible_isoform_filter_coverage_asserted=False,all_master_data_and_effective_group_bytes_hashed=True,all_envelope_metadata_read=True,all_new_task_payloads_hashed_and_roundtripped=packets,all_admitted_source_annotations_independently_equal_raw=annotations,all_quarantined_source_annotations_independently_equal_raw=len(qids),source_binding_counts=dict(bindings),exact_noncanonical_source_sequence_bindings_checked=form_source_checks,prior_GOA_packet_id_and_payload_hash_conserved_once=preserved,detected_guard_overlay_rows=overlay_rows,not_in_current_canonical_detected_guard_map=unmatched_overlay_rows,detected_guard_note='unmatchedformsequencesnotforcedtoborrowcanonicalgroup;heuristicgroupsnotexhaustivehomology',all_missing_is_unknown_and_evidence_NOT_native_projection=True,serialized_tasks=len(ids),shards=len(m['shards']),data_bytes=data_bytes,effective_group_bytes=group_bytes,split_counts=dict(counts),native_schema_counts=dict(schema),adapter_support_counts=dict(adapters),source_tasks=dict(source_tasks),distinct_records_by_source={key:len(value) for key,value in source_records.items()},known_effective_group_split_intersections=intersections,old_payload_semantics='inheritedboundpriorreceipts;wholebytesandmetadata/groupsrechecked',prior_generic_payload_missing_full_canonical_query_mask='immutablehistoricalsnapshot;resolvedexternallyforthisexactcanonicalscopeonly;notallformsorhistoricalsequenceversion',current_homology_heldout_unresolved=True,global_leakage_safe=False,new_independent_experiments=0,optimizer_steps=0,seconds=time.perf_counter()-start)
 rp=O/'receipt.json';rp.open('x').write(json.dumps(receipt,indent=2)+'\n');m.update(build_id='20261101-full-canonical-GOA-corpus-v17',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),prior_manifest_sha256=prior_hash,supersedes_for_consumption=str(PRIOR.relative_to(R)),source_components_base_manifest=str(PRIOR.relative_to(R)),GOA_component_replaces_prior_snapshot_without_duplicate_packets=True,additional_GOA_snapshot_admission_manifest=str(gpath.relative_to(R)),verification=str(rp.relative_to(R)),effective_group_policy='authoritativev16groupsplusfullcanonicalGOAreplacementsourceform/sequenceandqualifieddetectedguards;payloadsandsplitsofpriorsourcesunchanged;notexhaustivehomology')
 m['verification_layers'].append(str(rp.relative_to(R)));m['source_curation_manifests'].append(rec(gpath));m['open_gates']=[gate for gate in m['open_gates'] if not gate.startswith('GOAexpandedquerysnapshotpartial;')];m['open_gates'].append('Full107778canonicalGOAqueryscopecomplete;allpossibleisoformfiltercoverageNOTasserted;21sourceannotations/11subjectsquarantined;historicalGOAsequenceversionsandnativeGOAdecoderstillgated;ontologyreferencesseparatelyversionednotpropagated');m['contexts'][SOURCE]=dict(rows=packets,source_annotation_views=annotations,raw_available_annotation_views=g['raw_available_annotations'],canonical_query_scope_complete=True,all_possible_isoform_filter_coverage_asserted=False,context_units='fullnativeproteinsequenceandform-specificGOAassertionpackets;notmodeltokens',full_payloads_retained=True,model_token_count=None);m['mixture_proposal']['GOA_annotation_allocation']=dict(status='provisional_unactivated;overlapsKBlabels;separatesource/evidence/formstratarequired',weight=None,requires='nativeGOAdecoderandmasking;historicalsequence/formqualification;genuinefamilyheldout');target=O/'master-corpus-manifest-v17-full-canonical-GOA-v1.json';target.open('x').write(json.dumps(m,indent=2)+'\n');print('COMPLETE',str(target.relative_to(R)),len(ids),len(m['shards']),data_bytes,flush=True)
if __name__=='__main__':main()
