"""Finite native DNA/Rfam prior curation, content dedup, complete occurrence/group ledgers."""
from pathlib import Path
import sys,json,hashlib,sqlite3,datetime,time,collections,shutil,argparse,re
import pyarrow as pa,pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_nucleotide_prior import prior_task,prior_identity,sequence_groups,rfam_parent
from data_intake_corpus import canonical,CorpusTask
B=R/'outputs/data-intake/20261101-nucleotide-corpus-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check(r):
 p=Path(r['path']);p=p if p.is_absolute() else R/p;assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];return p

def run():
 ap=argparse.ArgumentParser();ap.add_argument('--output-name',required=True);ap.add_argument('--max-per-source',type=int,default=0);a=ap.parse_args();o=B/a.output_name;o.mkdir();(o/'shards/train').mkdir(parents=True);(o/'effective-groups').mkdir();(o/'occurrences').mkdir();(o/'code').mkdir();start=time.perf_counter()
 for n in ['data_intake_build_nucleotide_corpus.py','data_intake_nucleotide_prior.py']:(o/'code'/n).write_bytes((R/'scripts'/n).read_bytes())
 plan=json.loads((B/'build-plan.json').read_text());hp=R/plan['source_handoff'];assert sha(hp)==plan['source_handoff_sha256'];h=json.loads(hp.read_text());components={c['inventory_id']:c for c in h['components'] if c['inventory_id'] in ['INV-001','INV-003']};source_manifests={}
 for key,c in components.items():
  p=Path(c['manifest']['path']);assert sha(p)==c['manifest']['sha256'];source_manifests[key]=json.loads(p.read_text())
 conn=sqlite3.connect(o/'task_identity_and_group_index.sqlite3');conn.executescript('CREATE TABLE tasks(task_id TEXT PRIMARY KEY, source_family TEXT NOT NULL); CREATE TABLE task_groups(task_id TEXT NOT NULL, group_id TEXT NOT NULL, PRIMARY KEY(task_id,group_id)) WITHOUT ROWID; CREATE TABLE group_sources(group_id TEXT NOT NULL, source_family TEXT NOT NULL, PRIMARY KEY(group_id,source_family)) WITHOUT ROWID;');conn.commit();buffer=[];buffer_bytes=0;shards=[];lookup={};occfiles=[];counts=collections.Counter();seen_source=collections.Counter();lengths=collections.defaultdict(list);letters=collections.Counter();inputfiles=[];parent_missing=collections.Counter();knownchecks=0;checks_by_source=collections.Counter();produced_bytes=0;budget=plan['derived_cap_bytes'];native_rows=collections.Counter();unique_tasks=collections.Counter();source_seen=set()
 def rec(p,**kw):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p),**kw)
 def live(stage,**kw):
  (o/'live-build.json').write_text(json.dumps(dict(updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=__import__('os').getpid(),stage=stage,source_occurrences=dict(seen_source),serialized_unique_tasks=dict(unique_tasks),shards=len(shards),data_bytes=sum(x['bytes'] for x in shards),elapsed_seconds=time.perf_counter()-start,max_per_source=a.max_per_source,**kw),indent=2)+'\n')
 def reserve():
  size=produced_bytes+(o/'task_identity_and_group_index.sqlite3').stat().st_size;assert size<budget and shutil.disk_usage(R).free>plan['minimum_free_bytes'],'derivedcap/free-reserve gate'
 def flush():
  nonlocal buffer_bytes,produced_bytes
  if not buffer:return
  p=o/'shards/train'/f'{len(shards):06d}.parquet';pq.write_table(pa.Table.from_pylist(buffer),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==buffer
  for row in back:
   assert hashlib.sha256(row['native_payload']).hexdigest()==row['native_payload_sha256'];d=json.loads(CorpusTask.from_bytes(row['native_payload']).canonical_bytes());assert set(d['conditions'])=={'prior_domain','source_organism'} and set(d['targets'])=={'source_native_sequence'};assert hashlib.sha256(d['targets']['source_native_sequence'].encode()).hexdigest()==json.loads(row['admission_receipt_json'])['native_sequence_sha256']
  rec0=rec(p,rows=len(back),split='train',source_family=back[0]['source_family'] if len({r['source_family'] for r in back})==1 else None);shards.append(rec0);produced_bytes+=p.stat().st_size;buffer.clear();buffer_bytes=0;reserve()
 def register(seq,domain,organism,family,source_id,revision,src_ref,parent,original_source):
  nonlocal buffer_bytes,knownchecks
  sg=sequence_groups(seq);tid=prior_identity(seq,domain,organism);first=conn.execute('INSERT OR IGNORE INTO tasks(task_id,source_family) VALUES (?,?)',(tid,family)).rowcount==1;groups=['nucleotide_native:'+sg['native_sha256'],'nucleotide_upper:'+sg['uppercase_sha256'],'nucleotide_derived_RC_baseorder:'+sg['reverse_complement_base_order_group']]+parent['groups'];conn.executemany('INSERT OR IGNORE INTO task_groups VALUES (?,?)',[(tid,g) for g in groups]);conn.executemany('INSERT OR IGNORE INTO group_sources VALUES (?,?)',[(g,family) for g in groups]);seen_source[family]+=1;letters[family]+=len(seq);lengths[family].append(len(seq));parent_missing[family]+=parent['reference_id'] is None
  if first:
   t=prior_task(seq,domain=domain,organism=organism,source_ref=src_ref);payload=t.canonical_bytes();digest=hashlib.sha256(payload).hexdigest();d=json.loads(payload);assert d['targets']['source_native_sequence']==seq;unique_tasks[family]+=1
   if checks_by_source[family]<64:
    changed=prior_task(('A' if seq[0]!='A' else 'C')+seq[1:],domain=domain,organism=organism,source_ref={'mutated_sourceID':'MUTATED','length':999999});assert t.input_bytes()==changed.input_bytes();knownchecks+=1;checks_by_source[family]+=1
   buffer.append(dict(task_id=tid,record_id=source_id,source_family=family,source_revision=revision,source_sha256=src_ref['native_payload_sha256'],task_name='source_native_nucleotide_sequence_prior',native_schema='mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=digest,split='train',split_policy='unpairedreferenceprior_trainonly;all_sourceoccurrencesandparentgroupsretained;homology_and_blindheldoutunresolved',group_ids_json=json.dumps(sorted(set(groups))),evidence_class='released_reference_sequence_prior;notindependentexperimentalmeasurement',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(source_ref=src_ref,native_sequence_sha256=sg['native_sha256'],full_axis=True,native_case_TU_preserved=True,source_QC='entiresourcenativepassed;exactrowhashrecheckedhere',source_license='CC-BY-4.0' if family.startswith('DNA') else 'CC0-1.0',new_uniprot_sequence_admission=False)),missing_context_json=json.dumps(d['missing']),full_axis_values=len(seq),condition_context_json=json.dumps(dict(source_organism=organism,target_length_audit_only=True,source_parent=parent,segmentation_applicability='noimages/masks;unpairedsequenceprioronly',raw_image=False,mask=False,cell_perturbation_response=False))));buffer_bytes+=len(payload)
   if len(buffer)>=2048 or buffer_bytes>=16*2**20:flush()
  return dict(source_record_id=source_id,task_id=tid,new_unique_task=first,source_family=family,source_ref_json=canonical(src_ref).decode(),source_parent_json=canonical(parent).decode(),native_sequence_sha256=sg['native_sha256'],uppercase_sequence_sha256=sg['uppercase_sha256'],native_length=len(seq),original_source_metadata_json=canonical(original_source).decode())
 def save_occ(rows):
  nonlocal produced_bytes
  p=o/'occurrences'/f'{len(occfiles):05d}.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');assert pq.read_table(p).to_pylist()==rows;occfiles.append(rec(p,rows=len(rows)));produced_bytes+=p.stat().st_size;reserve()
 def qcrows(shard):
  for q in shard['qc_payloads']:
   p=check(q)
   for batch in pq.ParquetFile(p).iter_batches(batch_size=4096,columns=['source_record_id','source_row_index','source_payload_path','source_sequence_sha256','uppercase_sequence_sha256','source_parent_group','dna_sequence_quality_pass']):yield from batch.to_pylist()
 dna=source_manifests['INV-001'];family='DNA-native-reference-prior';domain='source_DNA_sequence';expectedDNA=1677609
 for si,s in enumerate(dna['shards']):
  if a.max_per_source and seen_source[family]>=a.max_per_source:break
  p=check(s['source_asset']);inputfiles.append(rec(p,source_family=family,source_rows=s['available_records']));qc=qcrows(s);rowindex=0;occ=[]
  for batch in pq.ParquetFile(p).iter_batches(batch_size=4096,columns=['accession','organism','sequence']):
   for row in batch.to_pylist():
    q=next(qc);assert q['source_row_index']==rowindex and q['source_payload_path']==str(p) and q['dna_sequence_quality_pass'];seq=row['sequence'];assert hashlib.sha256(seq.encode()).hexdigest()==q['source_sequence_sha256'] and hashlib.sha256(seq.upper().encode()).hexdigest()==q['uppercase_sequence_sha256'];assert q['source_parent_group']=='genbank:'+row['accession'];acc=row['accession'];parent=dict(reference_id=acc,groups=[q['source_parent_group'],'source_nucleotide_reference:'+acc,'genomic_accession_unversioned:'+re.sub(r'[.][0-9]+$','',acc)],source_interval_1based_closed=None,parent_scope='publishedGenBank-derivedrecord;underlyingrecordnotrefetched;sourcewindow/cropnotassumedwholegenome',observed_strand=None);org=row['organism'] if isinstance(row['organism'],str) and row['organism'] else None;ref=dict(native_payload=str(p.relative_to(R)),native_payload_sha256=s['source_asset']['sha256'],source_row_index=rowindex,source_QC_manifest=str(Path(components['INV-001']['manifest']['path']).relative_to(R)),source_QC_manifest_sha256=components['INV-001']['manifest']['sha256']);occ.append(register(seq,domain,org,family,q['source_record_id'],dna['source_revision'],ref,parent,dict(accession=acc,organism=row['organism'],alloriginalfeatures_andproteins='immutable_native_row_ref_only;notadmittedlabels')));rowindex+=1
    if a.max_per_source and seen_source[family]>=a.max_per_source:break
   conn.commit();save_occ(occ);occ=[];live('DNA fullnativeprior serialization',active_input_shard=si);print('DNA_ROWS',seen_source[family],'unique',unique_tasks[family],'seconds',round(time.perf_counter()-start,2),flush=True)
   if a.max_per_source and seen_source[family]>=a.max_per_source:break
  if not a.max_per_source:assert rowindex==s['available_records'];assert next(qc,None) is None
 rf=source_manifests['INV-003'];family='Rfam15.1-native-reference-prior';domain='RNA_family_reference_native_symbols';expectedRF=10164372
 for si,s in enumerate(rf['native_payloads']):
  if a.max_per_source and seen_source[family]>=a.max_per_source:break
  p=check(s);inputfiles.append(rec(p,source_family=family,source_rows=s['rows']));occ=[];rowindex=0
  for batch in pq.ParquetFile(p).iter_batches(batch_size=4096):
   for row in batch.to_pylist():
    assert row['sequence_alphabet_pass'];seq=row['native_sequence'];assert len(seq)==row['native_length'] and hashlib.sha256(seq.encode()).hexdigest()==row['native_sequence_sha256'] and hashlib.sha256(seq.upper().encode()).hexdigest()==row['uppercase_sequence_sha256'];parent=rfam_parent(row['source_identifier']);ref=dict(native_payload=str(p.relative_to(R)),native_payload_sha256=s['sha256'],source_row_index=rowindex,upstream_source_record_index=row['source_record_index'],source_QC_manifest=str(Path(components['INV-003']['manifest']['path']).relative_to(R)),source_QC_manifest_sha256=components['INV-003']['manifest']['sha256']);occ.append(register(seq,domain,None,family,row['source_record_id'],'Rfam15.1;January2026;originalCC0FASTA',ref,parent,dict(native_header=row['native_header'],native_identifier=row['source_identifier'],explicit_Rfam_ID_in_header=row['explicit_Rfam_ID_in_header'],T_count=row['native_T_count'],U_count=row['native_U_count'],ambiguity_symbol_count=row['ambiguity_symbol_count'])));rowindex+=1
    if a.max_per_source and seen_source[family]>=a.max_per_source:break
   conn.commit();save_occ(occ);occ=[]
   if a.max_per_source and seen_source[family]>=a.max_per_source:break
  live('Rfam fullnativeprior serialization',active_input_shard=si);print('RFAM_ROWS',seen_source[family],'unique',unique_tasks[family],'seconds',round(time.perf_counter()-start,2),flush=True)
  if not a.max_per_source:assert rowindex==s['rows']
 flush();conn.commit();assert seen_source['DNA-native-reference-prior']==(min(a.max_per_source,expectedDNA) if a.max_per_source else expectedDNA);assert seen_source['Rfam15.1-native-reference-prior']==(min(a.max_per_source,expectedRF) if a.max_per_source else expectedRF)
 # Final groups include every duplicate occurrence's source parents, not only the first task row.
 for index,sh in enumerate(shards):
  rows=pq.read_table(R/sh['path'],columns=['task_id']).to_pylist();gr=[]
  for r in rows:
   gs=[x[0] for x in conn.execute('SELECT group_id FROM task_groups WHERE task_id=? ORDER BY group_id',(r['task_id'],))];assert gs;gr.append(dict(task_id=r['task_id'],effective_groups_json=json.dumps(gs),evidence_scope='allnativeoccurrenceidentity/sourceparentgroups;notbiologicalsamecellorchemicalequivalence'))
  p=o/'effective-groups'/f'{index:06d}.parquet';pq.write_table(pa.Table.from_pylist(gr),p,compression='zstd');assert pq.read_table(p).to_pylist()==gr;lookup[sh['path']]=rec(p,rows=len(gr));produced_bytes+=p.stat().st_size
  if index%100==0:live('final complete occurrence/group union',group_shards=index+1);reserve()
 # The PK sorts by group, so the cross-source audit needs no reindexing of existing data.
 overlaps=[]
 for gid,num in conn.execute('SELECT group_id,COUNT(*) FROM group_sources GROUP BY group_id HAVING COUNT(*)>1'):overlaps.append(dict(group_id=gid,source_families=['DNA-native-reference-prior','Rfam15.1-native-reference-prior'],scope='identity/leakagegroup_only;derivedRCgroupnotbiologicalidentity;sourcereference_notpairedassay'))
 op=o/'cross-source-identity-and-parent-overlaps.parquet';pq.write_table(pa.Table.from_pylist(overlaps) if overlaps else pa.table({'group_id':pa.array([],type=pa.string())}),op,compression='zstd');reserve();task_count=conn.execute('SELECT COUNT(*) FROM tasks').fetchone()[0];edge_count=conn.execute('SELECT COUNT(*) FROM task_groups').fetchone()[0];conn.close()
 result=dict(schema='mora3.NativeNucleotidePriorAdmission/v1',status='boundedpilot_verified' if a.max_per_source else 'whole_two_source_nativeprior_admission_verified',source_occurrences=dict(seen_source),source_available_full_counts=dict(DNA=expectedDNA,Rfam15_1=expectedRF),source_QC_pass_full_counts='inheritedwhole_sourceworker_nativeQC;our selected/fullrowhashmatchescountedseparately',rows_revalidated_and_retained=sum(seen_source.values()),serialized_unique_tasks=task_count,unique_tasks_by_source=dict(unique_tasks),duplicate_occurrences_retained=sum(seen_source.values())-task_count,source_original_native_rows_retained=True,all_native_case_TU_axes_preserved=True,source_symbols_seen=sum(letters.values()),source_symbols_by_source=dict(letters),native_length_stats={k:dict(min=min(v),max=max(v),mean=sum(v)/len(v)) for k,v in lengths.items()},source_parent_unparsed_occurrences=dict(parent_missing),task_group_edges=edge_count,cross_source_overlapping_groups=len(overlaps),cross_source_overlap_receipt=rec(op,rows=len(overlaps)),shards=shards,task_shard_bytes=sum(s['bytes'] for s in shards),effective_group_lookup=lookup,source_occurrence_ledger=occfiles,input_native_payloads=inputfiles,all_selected_source_native_hashes_revalidated=True,all_serialized_task_payload_hashes_and_codec_roundtrips=True,target_mutation_predictor_checks=knownchecks,target_mutation_checks_by_source=dict(checks_by_source),fullaxis_no8192truncation=True,source_input_refs_preserve_all_withheld_nativefields=True,split_counts=dict(train=task_count,validation=0,test=0),model_adapter_supported=False,source_evidence='releasedreference_sequence_prior;notindependentexperiment/causal/biophysicaltruth',new_UniProt_protein_admissions=0,translation_or_proteinlabels_admitted=False,Rfam_family_or_confidence_labels_admitted=False,homology_heldout_unresolved=True,unknown_source_parent_aliases_unresolved=True,global_leakage_safe=False,final_mixture_weights=None,new_independent_experiments=0,optimizer_steps=0,source_handoff=str(hp.relative_to(R)),source_handoff_sha256=plan['source_handoff_sha256'],code_refs=[rec(p) for p in (o/'code').iterdir()],generated_identity_index=rec(o/'task_identity_and_group_index.sqlite3'),derived_cap_bytes=budget,seconds=time.perf_counter()-start)
 (o/'manifest.json').open('x').write(json.dumps(result,indent=2)+'\n');live('finite source/task/group verification complete',manifest=str(o/'manifest.json'),all_tasks_train=True);print('COMPLETE',sum(seen_source.values()),task_count,'shards',len(shards),'bytes',result['task_shard_bytes'],'seconds',result['seconds'],flush=True)
if __name__=='__main__':run()
