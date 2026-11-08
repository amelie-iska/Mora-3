from pathlib import Path
import sys,json,collections,hashlib,datetime,time
import pyarrow as pa
import pyarrow.parquet as pq
R=Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_corpus import chemical_identity
B=R/'outputs/data-intake/20261031-rosetta-molecule-profile-v1/identity-audit-v2';B.mkdir();base=R/'outputs/data-intake/20261031-subagent/curated/cpg0003-rosetta-profile-links-v1';start=time.perf_counter();m=json.loads((base/'manifest.json').read_text());records={};source_sample_smiles=collections.defaultdict(set);source_sample_refs=collections.defaultdict(list);audit=[]
for mod in ['CellPainting','L1000']:
 p=base/f'CDRPBIO-BBBC036-Bray--{mod}--metadata.parquet';expected=m['artifacts'][p.name];assert p.stat().st_size==expected['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'];records[mod]=pq.read_table(p).to_pylist()
for row in records['L1000']:
 md=json.loads(row['source_metadata_json']);sample=md.get('pert_id');smiles=md.get('CPD_SMILES');canonical=None;error=None
 if not row['control'] and smiles:
  try:canonical=chemical_identity(smiles)['canonical_isomeric_smiles']
  except ValueError as e:error=str(e)
 if canonical and sample:source_sample_smiles[sample].add(canonical);source_sample_refs[sample].append(row['record_id'])
 audit.append(dict(record_id=row['record_id'],modality='L1000',control=row['control'],source_sample_id=sample,canonical_isomeric_smiles=canonical,molecular_identity_source='source_CPDSMILES_in_same_L1000record' if canonical else None,error=error))
conflicts={s for s,v in source_sample_smiles.items() if len(v)!=1}
for row in records['CellPainting']:
 md=json.loads(row['source_metadata_json']);sample=md.get('Metadata_broad_sample');identity=source_sample_smiles.get(sample);canonical=next(iter(identity)) if identity and sample not in conflicts and not row['control'] else None
 audit.append(dict(record_id=row['record_id'],modality='CellPainting',control=row['control'],source_sample_id=sample,canonical_isomeric_smiles=canonical,molecular_identity_source='exact_full_chemical_sample_ID_to_L1000source_chemical_metadata;not_assay_pair' if canonical else None,error='source_sample_structure_conflict' if sample in conflicts else None))
p=B/'source-identity-audit.parquet';pq.write_table(pa.Table.from_pylist(audit),p,compression='zstd');assert pq.read_table(p).to_pylist()==audit
receipt=dict(schema='mora3.Rosetta_exact_source_sample_chemical_identity_audit.v1',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_CDRP_profiles={mod:len(v) for mod,v in records.items()},eligible_noncontrol_profile_identities=dict(collections.Counter(r['modality'] for r in audit if r['canonical_isomeric_smiles'])),source_chemical_samples=len(source_sample_smiles),conflicting_sample_identities=len(conflicts),molecular_identity_source='exact original L1000 source CPD_SMILES;CP joins by full source chemical sample ID only',identity_join_scope='source molecule metadata only;no CP/L1000samecell,samewell,dose,time ormatchedassay assertion',source_rows_all_retained=True,prior_corpus_heldout_and_admission_filter='notyetapplied;nextstage',no_downloads=True,no_source_mutation=True,admitted_new_tasks=0,seconds=time.perf_counter()-start)
with (B/'identity-audit-receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
