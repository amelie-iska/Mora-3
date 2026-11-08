import pathlib,json,csv,gzip,hashlib,time,sys,re,collections
import pyarrow as pa
import pyarrow.parquet as pq
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import protein_function_task,gene_reference_ids,annotation_score,CorpusTask
B=R/'outputs/data-intake/20261031-protein-function-context-v1';C=B/'curated-v1';C.mkdir(exist_ok=False);(C/'shards/train').mkdir(parents=True);start=time.perf_counter();receipt=json.loads((B/'acquisition-receipt.json').read_text());assert receipt['status']=='downloaded_verified_release_rows' and receipt['release']=='2026_03';raw=pathlib.Path(receipt['path']);assert hashlib.sha256(raw.read_bytes()).hexdigest()==receipt['sha256'];oldp=R/'outputs/data-intake/20261031-subagent/raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz';old={}
with gzip.open(oldp,'rt') as f:
 for row in csv.DictReader(f,delimiter='\t'):old[row['Entry']]=row
assert len(old)==107778
rows=[];shards=[];qc=[];edges=[];seen=set();seen_accessions=set();duplicates=[];genes=collections.defaultdict(set);accinfo={};fieldcounts=collections.Counter();eco=collections.Counter();alias_missing=collections.Counter();eligible=0;serialized=0;num=0

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def rel(p):return str(pathlib.Path(p).relative_to(R))
def flush():
 global rows
 if not rows:return
 p=C/'shards/train'/f'{len(shards):05d}.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==rows
 for r in back:
  raw=r['native_payload'];assert sha_bytes(raw)==r['native_payload_sha256'];t=CorpusTask.from_bytes(raw);assert t.canonical_bytes()==raw;d=json.loads(raw);changed=CorpusTask(d['task'],d['conditions'],{'mutated_annotation':'test-only'},{'mutated_source':True},d['missing']);assert t.input_bytes()==changed.input_bytes()
 shards.append(dict(path=rel(p),source_family='UniProt-function-annotations',split='train',rows=len(rows),bytes=p.stat().st_size,sha256=sha(p)));rows=[]
def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()
with gzip.open(raw,'rt') as f:
 for row in csv.DictReader(f,delimiter='\t'):
  num+=1;acc=row['Entry'];assert acc not in seen_accessions;seen_accessions.add(acc);score=annotation_score(row['Annotation']);assert row['Reviewed']=='reviewed';oldrow=old.get(acc);assert oldrow is not None and oldrow['Sequence']==row['Sequence'] and annotation_score(oldrow['Annotation'])==score and oldrow['Reviewed']==row['Reviewed'];sequence_hash=sha_bytes(row['Sequence'].encode());taxon=row['Organism (ID)'];assert taxon.isdigit() and int(taxon)>0;groups=['accession:'+acc,'sequence:'+sequence_hash];gene_error=None
  try:geneids=gene_reference_ids(row['GeneID'])
  except ValueError as err:geneids=();gene_error=str(err)
  for gid in geneids:
   edge=dict(accession=acc,sequence_sha256=sequence_hash,gene_id=gid,taxon=taxon,annotation_score=score,source_release=receipt['release'],source_sha256=receipt['sha256'],evidence='source_asserted_UniProt_GeneID_reference_exact_source_sequence_checked',scope='gene/proteinreference_not_assayedisoform_or_samecell');edges.append(edge);genes[gid].add((taxon,acc,sequence_hash));groups+=['gene:'+taxon+':'+gid,'ncbi_gene:'+gid]
  accinfo[acc]=dict(sequence_sha256=sequence_hash,taxon=taxon,gene_ids=geneids)
  for key in ['Function [CC]','Catalytic activity','Gene Ontology IDs','GeneID','Gene Names (primary)']:
   if row[key].strip():fieldcounts[key]+=1
  for key in ['Function [CC]','Catalytic activity']:
   for code in set(re.findall(r'ECO:\d{7}',row[key])):eco[key+':'+code]+=1
  for rh in sorted(set(re.findall(r'RHEA:(\d+)',row['Catalytic activity']))):groups.append('rhea:RHEA:'+rh)
  go=[x.strip() for x in row['Gene Ontology IDs'].split(';') if x.strip()];assert all(re.fullmatch(r'GO:\d{7}',g) for g in go)
  try:task=protein_function_task(row)
  except ValueError as err:
   qc.append(dict(accession=acc,full_sequence_match=True,annotation_score=score,eligible_function=False,serialized=False,reason='no_function_catalytic_GO_supervision;existing_sequence_task_remains_eligible',gene_reference_error=gene_error));continue
  eligible+=1;d=json.loads(task.canonical_bytes());semantic=sha_bytes(json.dumps(dict(conditions=d['conditions'],targets=d['targets']),sort_keys=True).encode())
  if semantic in seen:
   duplicates.append(dict(accession=acc,semantic_task_hash=semantic,reason='exact_condition_target_annotation_duplicate'));qc.append(dict(accession=acc,full_sequence_match=True,annotation_score=score,eligible_function=True,serialized=False,reason='exact_condition_target_annotation_duplicate',gene_reference_error=gene_error));continue
  seen.add(semantic);rawtask=task.canonical_bytes();h=sha_bytes(rawtask);rows.append(dict(task_id=sha_bytes(('UniProt2026_03:function:'+acc).encode()+rawtask),record_id=acc,source_family='UniProt-function-annotations',source_revision='UniProt2026_03',source_sha256=receipt['sha256'],task_name='sequence_to_curated_function_annotations',native_schema='mora3.CorpusTask/v1',native_payload=rawtask,native_payload_sha256=h,split='train',split_policy='source_valid_curated_annotation_train_only_pending_homology_v1',group_ids_json=json.dumps(sorted(set(groups))),evidence_class='source_curated_mixed_ECO_function_annotations_not_blanket_experimental_truth',admitted_for_task=True,model_adapter_supported=False,admission_receipt_json=json.dumps(dict(reviewed=True,annotation_score=score,exact_old_sequence_match=True,release='2026_03',source_only_annotation_scope=True)),missing_context_json=json.dumps(d['missing']),full_axis_values=len(row['Sequence']),condition_context_json=json.dumps(dict(organism_taxon=taxon))));serialized+=1;qc.append(dict(accession=acc,full_sequence_match=True,annotation_score=score,eligible_function=True,serialized=True,reason='source_valid_curated_function_task',gene_reference_error=gene_error))
  if len(rows)>=512:flush()
  if num%10000==0:print(json.dumps(dict(reviewed_rows=num,eligible=eligible,serialized=serialized,gene_edges=len(edges),seconds=time.perf_counter()-start)),flush=True)
flush();assert num==107778 and seen_accessions==set(old)
pq.write_table(pa.Table.from_pylist(qc),C/'quality-and-admission.parquet',compression='zstd');pq.write_table(pa.Table.from_pylist(edges),C/'uniprot-gene-reference-edges.parquet',compression='zstd')
if duplicates:pq.write_table(pa.Table.from_pylist(duplicates),C/'duplicate-annotation-tasks.parquet',compression='zstd')
# Join source-asserted gene reference identities only. No protein isoform,construct orsamecell claim.
linkrows=[];linkcounts=collections.Counter();bio=R/'outputs/data-intake/20261031-subagent/curated/biogrid-chemicals-v1/records.parquet'
for r in pq.read_table(bio).to_pylist():
 s=json.loads(r['source_row_json']);gid=s['Entrez Gene ID'];tax=s['Organism ID'];matches=[x for x in genes.get(gid,set()) if x[0]==tax];seqs={x[2] for x in matches};rights_eligible=s['Curated By']=='BIOGRID' and s['Chemical Source']!='DRUGBANK';linkrows.append(dict(source_family='BioGRID',record_id=r['record_id'],source_gene_field='Entrez Gene ID',source_gene_id=gid,source_taxon=tax,source_rights_component_eligible=rights_eligible,uniprot_accessions=sorted(x[1] for x in matches),exact_sequence_reference_count=len(seqs),scope='gene_reference_only;sourceassayedisoformnotestablished',cell_or_assay_pairing=False));linkcounts['BioGRID:matched' if matches else 'BioGRID:unmatched']+=1
RC=R/'outputs/data-intake/20261031-subagent/curated/cpg0003-rosetta-profile-links-v1'
for p in RC.glob('*metadata.parquet'):
 for r in pq.read_table(p).to_pylist():
  if not r['study'].startswith(('TA-ORF','LUAD')):continue
  md=json.loads(r['source_metadata_json']);field='Metadata_GeneID' if 'Metadata_GeneID' in md else 'pert_univ_id';gid=md.get(field,'').strip();matches=list(genes.get(gid,set())) if gid.isdigit() else [];seqs={x[2] for x in matches};linkrows.append(dict(source_family='Rosetta-genetic-reference',record_id=r['record_id'],source_gene_field=field,source_gene_id=gid,source_taxon=None,source_rights_component_eligible=True,uniprot_accessions=sorted(x[1] for x in matches),exact_sequence_reference_count=len(seqs),scope='source_gene_reference_only;clone_or_mutant_sequence_notestablished',cell_or_assay_pairing=False));linkcounts['Rosetta-genetic:matched' if matches else 'Rosetta-genetic:unmatched']+=1
pq.write_table(pa.Table.from_pylist(linkrows),C/'cross-source-gene-reference-audit.parquet',compression='zstd')
manifest=dict(schema='mora3.uniprot_function_context_curation.v1',source_receipt=rel(B/'acquisition-receipt.json'),source_sha256=receipt['sha256'],source_release='2026_03',license='CC-BY-4.0',raw_available=107778,full_row_quality_pass=107778,full_same_release_accession_sequence_score_matches=107778,eligible_function_rows=eligible,admitted_serialized_function_tasks=serialized,source_rows_without_functional_supervision=num-eligible,exact_condition_target_duplicate_tasks=len(duplicates),shards=shards,serialized_tasks=serialized,total_shard_bytes=sum(x['bytes'] for x in shards),source_gene_reference_edges=len(edges),source_gene_identifiers=len(genes),source_annotation_field_availability=dict(fieldcounts),source_ECO_code_counts=dict(eco),cross_source_gene_reference_rows=len(linkrows),cross_source_gene_match_counts=dict(linkcounts),role_scope='functionannotations+gene/proteinreference;not experimentaloutcomes,assayedisoforms,samecell orcausalresponse',split='train_only;homologyandgene-familyheldoutunresolved',model_adapter_supported=False,all_payload_hashes_roundtrips_and_target_audit_mutation_invariance=True,maximum_sequence_chars=max(len(x['Sequence']) for x in old.values()),minimum_sequence_chars=min(len(x['Sequence']) for x in old.values()),source_raw_immutable=True,optimizer_steps=0,seconds=time.perf_counter()-start)
with (C/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('CURATION_COMPLETE',json.dumps({k:manifest[k] for k in ['raw_available','eligible_function_rows','admitted_serialized_function_tasks','total_shard_bytes','source_gene_reference_edges','cross_source_gene_match_counts','seconds']}),flush=True)
