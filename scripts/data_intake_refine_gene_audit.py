import pathlib,json,collections,sys,hashlib
import pyarrow as pa
import pyarrow.parquet as pq
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import rosetta_gene_reference
B=R/'outputs/data-intake/20261031-protein-function-context-v1';C=B/'curated-v1';edges=pq.read_table(C/'uniprot-gene-reference-edges.parquet').to_pylist();genes=collections.defaultdict(set)
for e in edges:genes[e['gene_id']].add((e['taxon'],e['accession'],e['sequence_sha256']))
rows=[r for r in pq.read_table(C/'cross-source-gene-reference-audit.parquet').to_pylist() if r['source_family']=='BioGRID'];counts=collections.Counter();RC=R/'outputs/data-intake/20261031-subagent/curated/cpg0003-rosetta-profile-links-v1'
for p in sorted(RC.glob('*metadata.parquet')):
 for r in pq.read_table(p).to_pylist():
  if not r['study'].startswith(('TA-ORF','LUAD')):continue
  ref=rosetta_gene_reference(json.loads(r['source_metadata_json']));gid=ref['gene_id'];matches=list(genes.get(gid,set())) if gid else [];taxa={x[0] for x in matches};rows.append(dict(source_family='Rosetta-genetic-reference',record_id=r['record_id'],source_gene_field='|'.join(ref['source_fields']),source_gene_id=gid,source_taxon=next(iter(taxa)) if len(taxa)==1 else None,source_rights_component_eligible=True,uniprot_accessions=sorted(x[1] for x in matches),exact_sequence_reference_count=len({x[2] for x in matches}),scope='source_gene_reference_only;clone_mutant_or_assayedisoform_sequence_notestablished',cell_or_assay_pairing=False,source_gene_fields_json=json.dumps(ref['source_fields'],sort_keys=True),reference_status=ref['status'],normalization='knownnumericGeneIDfield;integer-valueddecimaltextnormalized;no symbollookup'))
for r in rows:
 counts[r['source_family']+(':rights_eligible' if r['source_rights_component_eligible'] else ':rights_pending')+(':matched' if r['uniprot_accessions'] else ':unmatched')]+=1
p=C/'cross-source-gene-reference-audit-v2.parquet';pq.write_table(pa.Table.from_pylist(rows),p,compression='zstd')
receipt=dict(schema='mora3.typed_gene_reference_audit.v2',supersedes='cross-source-gene-reference-audit.parquet retained initial audit',path=str(p.relative_to(R)),rows=len(rows),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),counts=dict(counts),source_reference_not_samecell_orassayedisoform=True,source_values_immutable=True,source_normalization='knownsourceGeneIDfields only;integral decimaldisplay;missing/conflict invalidnot guessed',unresolved_status_counts=dict(collections.Counter(r.get('reference_status','BioGRIDtypedgene') for r in rows)))
with (C/'gene-reference-audit-v2.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
