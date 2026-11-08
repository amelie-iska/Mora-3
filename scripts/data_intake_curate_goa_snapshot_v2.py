"""Finite immutable snapshot from completed GOA batches; no network/source mutation."""
from pathlib import Path
import sys,json,hashlib,gzip,csv,collections,datetime,time,copy,importlib.util,argparse
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_goa_projection import annotation_projection,bind_subject,prediction_task
from data_intake_corpus import canonical,CorpusTask

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rec(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p;assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p

def validate_native_gaf_annotations(annotations,gaf,source):
 if collections.Counter(source.native_gaf_key(a) for a in annotations)!=collections.Counter(source.gaf_semantic_keys(gaf)):raise ValueError('native annotation and GAF semantic multisets differ')

def validate_native_exports(rows,gaf,gpad,accessions,source):
 return source.crosscheck_exports(rows,accessions,gaf,gpad)

def completed_manifest(m):
 if m.get('schema')=='mora3.goa_companion_native_handoff.v1':
  q=m['QC'];assert q['GAF_native_semantic_multiset_equal'] and q['GPAD_native_semantic_multiset_equal']
  parents=[x[len('UniProtKB:'):] for x in q['subject_counts'] if '-' not in x[len('UniProtKB:'):]];assert len(parents)==1
  return dict(m,query_canonical_accessions=parents,all_raw_payloads=m['raw_payloads'],full_batch_scan=True)
 return m

def run():
 ap=argparse.ArgumentParser();ap.add_argument('--output-name',required=True);ap.add_argument('--max-batches',type=int,default=100);args=ap.parse_args();B=R/'outputs/data-intake/20261031-uniprot-full-annotations-v1';S=R/'outputs/data-sources/20261031-goa-companion-v1';out=B/args.output_name;out.mkdir();(out/'assertions').mkdir();(out/'shards/train').mkdir(parents=True);(out/'effective-groups').mkdir();(out/'code').mkdir();start=time.perf_counter()
 for n in ['data_intake_curate_goa_snapshot_v2.py','data_intake_goa_projection.py','data_sources_goa.py','data_sources_goa_v2.py']:(out/'code'/n).write_bytes((R/'scripts'/n).read_bytes())
 spec=importlib.util.spec_from_file_location('frozen_goa_source',out/'code/data_sources_goa.py');source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source);sys.modules['data_sources_goa']=source
 spec2=importlib.util.spec_from_file_location('frozen_goa_source_v2',out/'code/data_sources_goa_v2.py');source2=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(source2)
 candidates=[S/'handoff-manifest-sample-v1.json',S/'curated/manifest-pilot100-v1-batch-0000.json']+sorted((S/'curated').glob('manifest-full-part-*.json'));inputs=candidates[:args.max_batches];assert inputs and len(inputs)==len(set(inputs));input_receipts=[rec(p) for p in inputs];(out/'input-snapshot.json').open('x').write(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed_source_manifests=input_receipts,full107778scope=False,selection='firstfrozencompletedmanifestspilotthenlexicalpart/batch;trainonly;noheldoutclaim'),indent=2)+'\n')
 with gzip.open(R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz','rt') as f:baseline={r['Entry']:r['Sequence'] for r in csv.DictReader(f,delimiter='\t')}
 fm=json.loads((B/'physical-form-supplement-v2/manifest.json').read_text());forms={}
 for f in fm['files']:
  if f['kind']=='forms':
   p=check(f)
   for row in pq.read_table(p).to_pylist():
    x=json.loads(row['record_json']);forms[(x['accession'],x['form_id'])]=x
 bm=json.loads((B/'form-sequence-coordinate-bindings-v1/manifest.json').read_text())
 for f in bm['binding_files']:
  for row in pq.read_table(check(f)).to_pylist():
   r=json.loads(row['record_json']);forms[(r['primary_accession'],r['declared_form_id'])]=r
 ambiguous_form_ids=set(json.loads((B/'form-sequence-coordinate-bindings-v1/global_form_ID_multiple_sequence_hashes.json').read_text()))
 im=json.loads((B/'isoform-sequences-v2/manifest.json').read_text());isoforms={}
 for f in im['native_shards']:
  for row in pq.read_table(check(f)).to_pylist():
   if not row['is_canonical_record']:isoforms[row['source_accession']]=row['sequence']
 extra=json.loads((B/'isoform-exact-parents-v1/manifest.json').read_text())
 for mr in extra['complete_batches']:
  part=json.loads(check(mr).read_text())
  for row in pq.read_table(check(part['native_sequences'])).to_pylist():
   if not row['is_canonical_record']:
    sid=row['source_accession'];seq=row['sequence'];assert sid not in isoforms or isoforms[sid]==seq;isoforms[sid]=seq
 def bind(subject):
  b=bind_subject(subject,baseline,forms,isoforms)
  if subject.split(':',1)[1] in ambiguous_form_ids:b.update(sequence=None,sequence_binding='ambiguous_global_source_form_ID_requires_queryparent_disambiguation')
  return b
 assertions=[];shards=[];lookup={};buffer=[];queries=set();seen_annotations=set();seen_subjects=set();taskids=set();task_counts=collections.Counter();evidence_counts=collections.Counter();missing=collections.Counter();sequence_bindings=collections.Counter();annotations_total=0;quarantines=[];mutation_checks=0;sourcefiles={};allrawbytes=0
 def flush():
  if not buffer:return
  p=out/'shards/train'/f'{len(shards):05d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==buffer
  for r in back:assert hashlib.sha256(r['native_payload']).hexdigest()==r['native_payload_sha256'];assert CorpusTask.from_bytes(r['native_payload']).canonical_bytes()==r['native_payload']
  shards.append(rec(p,rows=len(back),split='train',source_family='UniProt-GOA-assertion-packets'));g=out/'effective-groups'/p.name;gr=[dict(task_id=r['task_id'],effective_groups_json=r['group_ids_json']) for r in back];pq.write_table(pa.Table.from_pylist(gr),g,compression='zstd');assert pq.read_table(g).to_pylist()==gr;lookup[str(p.relative_to(R))]=rec(g,rows=len(gr));buffer.clear()
 for batchnum,manifest_path in enumerate(inputs):
  m=completed_manifest(json.loads(manifest_path.read_text()));assert m['source_snapshot']=='2026-07-28 10:08' and m['full_batch_scan'] and m['license']=='CC-BY-4.0';assert not queries.intersection(m['query_canonical_accessions']);queries.update(m['query_canonical_accessions']);native=[];refs={};table=[];gaf=None;gpad=None
  for f in m['all_raw_payloads']:
   p=check(f);sourcefiles[str(p)]=f
   if p.name.endswith('complete.gaf'):gaf=p.read_text()
   if p.name.endswith('complete.gpad'):gpad=p.read_text()
   if (p.name.startswith('page-') or '-json-page-' in p.name) and p.suffix=='.json':
    page=json.loads(p.read_text())
    for index,a in enumerate(page['results']):
     assert a['id'] not in seen_annotations;seen_annotations.add(a['id']);native.append(a);refs[a['id']]=dict(raw_file=str(p.relative_to(R)),raw_sha256=f['sha256'],results_index=index,completed_manifest=str(manifest_path.relative_to(R)))
  assert len(native)==m['available_annotations'] and gaf is not None
  proof=validate_native_exports(native,gaf,gpad,m['query_canonical_accessions'],source2);missing['GPAD_full_native_proof_batches']+=proof['GPAD_full_native_semantic_multiset_equal'];missing['source_NULL_GO_shortcode_rows']+=proof['GAF_unrepresentable_native_rows']
  subjects=collections.defaultdict(list)
  for a in native:
   p=annotation_projection(a);binding=bind(a['geneProductId']);subjects[a['geneProductId']].append(a);annotations_total+=1;evidence_counts[p['evidence_class']]+=1;missing['term_name_missing']+=p['term_name'] is None;missing['explicit_NOT']+=p['negated'] is True;missing['with_from_present']+=bool(p['with_from_or_of_and']);missing['extensions_present']+=bool(p['extension_or_of_and']);table.append(dict(annotation_id=a['id'],subject=a['geneProductId'],projection_json=canonical(p).decode(),binding_json=canonical({k:v for k,v in binding.items() if k!='sequence'}).decode(),source_ref_json=canonical(refs[a['id']]).decode()))
  p=out/'assertions'/f'{batchnum:04d}.parquet';pq.write_table(pa.Table.from_pylist(table),p,compression='zstd');assert pq.read_table(p).to_pylist()==table;assertions.append(rec(p,rows=len(table)))
  for subject,annotations in subjects.items():
   assert subject not in seen_subjects;seen_subjects.add(subject);binding=bind(subject);sequence_bindings[binding['sequence_binding']]+=1;task=prediction_task(binding,annotations,source_snapshot=m['source_snapshot'],source_refs=[refs[a['id']] for a in annotations])
   if task is None:quarantines.append(dict(subject=subject,annotations=len(annotations),cause='unresolved_exact_form_sequence' if binding['sequence'] is None else 'onlyND_or_unknownrelation_no_semanticGOheadlabels',source_annotation_ids=[a['id'] for a in annotations]));continue
   # Actual entire target packets mutate while predictor bytes must remain invariant.
   changed=copy.deepcopy(annotations)
   for a in changed:a.update(goId='GO:9999999',goName='MUTATED_NAME',qualifier='NOT|involved_in',reference='PMID:MUTATED',symbol='MUTATED_SYMBOL',assignedBy='MUTATED_PROVIDER')
   other=prediction_task(binding,changed,source_snapshot=m['source_snapshot'],source_refs=[]);assert other is not None and task.input_bytes()==other.input_bytes();mutation_checks+=1;payload=task.canonical_bytes();digest=hashlib.sha256(payload).hexdigest();tid=hashlib.sha256(canonical([subject,'GOAJuly28',digest])).hexdigest();assert tid not in taskids;taskids.add(tid);task_counts[binding['subject_kind']]+=1;groups=['accession:'+binding['parent_accession'],'sequence:'+hashlib.sha256(binding['sequence'].encode()).hexdigest(),'protein_form:'+subject]
   buffer.append(dict(task_id=tid,record_id=subject,source_family='UniProt-GOA-assertion-packets',source_revision='GOAannotation2026-07-28;exactcurrentKBsequence2026_03',source_sha256=sha(manifest_path),task_name='source_GOA_assertion_packet_prediction',native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=digest,split='train',split_policy='sourcequalifiedexistingprotein_trainonly;exactaccession_sequence_formgroups;homologyunresolved',group_ids_json=json.dumps(groups),evidence_class='sourceassertedGOannotationprior;mixedexperimental_inferred_NDtypedseparately',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_snapshot=m['source_snapshot'],source_KB_sequence_release='2026_03',qualified_parent_entry=True,exact_form_sequence=binding['sequence_binding'],semantic_not_residue_or_experimental_response_task=True,canonical_labels_inherited=False,license='CC-BY-4.0',historical_GOA_sequence_version='notexplicit',new_independent_observation=False)),missing_context_json=json.dumps(json.loads(payload)['missing']),full_axis_values=len(binding['sequence']),condition_context_json=json.dumps(dict(GO_closed_world=False,source_genProduct_subject=subject,independent_annotation_sequence_versions=True,segmentation_applicability='proteinannotationcontextonly;noimagepair'))))
   if len(buffer)==64:flush()
  print('GOA_TYPED_BATCH',batchnum+1,'canonicalqueries',len(queries),'annotationviews',annotations_total,'taskpackets',sum(task_counts.values()),'seconds',round(time.perf_counter()-start,2),flush=True)
 flush();qp=out/'form-or-placeholder-quarantines.json';qp.write_text(json.dumps(quarantines,indent=2)+'\n');result=dict(schema='mora3.GOATypedSnapshotAdmission/v2',status='boundedcompleted_source_snapshot_semantic_annotation_prediction_admission',source_snapshot='2026-07-28 10:08',source_KB_sequence_release='2026_03',source_snapshots_equal=False,full107778coverage=(queries==set(baseline)),canonical_queries=len(queries),query_accessions=sorted(queries),raw_available_annotations=annotations_total,quality_pass_annotations=annotations_total,typed_annotation_views=annotations_total,serialized_task_packets=sum(task_counts.values()),task_counts_by_subject_kind=task_counts,source_subjects=len(seen_subjects),quarantined_subjects=len(quarantines),quarantined_annotations=sum(q['annotations'] for q in quarantines),quarantine_receipt=rec(qp),evidence_counts=evidence_counts,missingness_and_context_counts=missing,sequence_binding_counts=sequence_bindings,source_raw_payloads=list(sourcefiles.values()),assertion_files=assertions,task_shards=shards,task_shard_bytes=sum(s['bytes'] for s in shards),effective_group_lookup=lookup,all_annotation_JSON_unknown_fields_retained=True,all_entire_native_GAF_semantic_multisets_equal=(missing['source_NULL_GO_shortcode_rows']==0),all_representable_GAF_and_required_full_GPAD_semantics_equal=True,null_shortcodes_preserved_without_inference=True,all_Arrow_and_native_task_roundtrips=True,all_target_mutation_invariance_checks=mutation_checks,source_license='CC-BY-4.0',ontology_terms='sourceGOnameoftenmissing;sourceworkerownsseparatelyversionedofficialontologycompanion;nofabrication',parent_score_rule='sameexistingreviewed>=4entry;isoform-specificscore_notinvented',task_semantics='sourceannotationpriorpackets;notdirectbinding/causal/physicalmeasurements',canonical_labels_inherited=False,homology_heldout='unresolved;trainonly',model_adapter_supported=False,full_directions='predictiononly;nogenadmissionhere',new_independent_experiments=0,optimizer_steps=0,seconds=time.perf_counter()-start)
 (out/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');print('COMPLETE',len(queries),annotations_total,sum(task_counts.values()),len(quarantines),flush=True)
if __name__=='__main__':run()
