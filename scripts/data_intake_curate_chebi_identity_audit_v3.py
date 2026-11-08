import pathlib,json,hashlib,sys,collections,re,datetime
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_chebi import chebi_reference
from data_intake_corpus import audit_table
B=R/'outputs/data-intake/20261031-pathway-semantics-v2';C=B/'chebi-context-v1';receipt=json.loads((C/'acquisition-receipt.json').read_text());assert receipt['raw_complete']==2031;out=C/'curated-identity-v3';out.mkdir(exist_ok=False);rows=[];lookup={};counts=collections.Counter();held=set()
for r in pq.read_table(R/'outputs/data-intake/20261031-final-corpus-build-v1/multisource-v3/audits/freesolv-split-allocation.parquet').to_pylist():
 if r['split']!='train':held.update(r['groups'])
for source in receipt['records']:
 p=pathlib.Path(source['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==source['sha256'];data=json.loads(p.read_text());ref=chebi_reference(data,source['id']);ref.update(source_file=str(p.relative_to(R)),source_sha256=source['sha256'],source_name=data.get('name'),source_definition=data.get('definition'),source_retrieved_file_write_utc=datetime.datetime.fromtimestamp(p.stat().st_mtime,datetime.timezone.utc).isoformat());chemical={x for x in [ref['molecule_group'],ref['scaffold_group']]+ref['fragment_groups'] if x};collision=sorted(held&chemical);ref['heldout_chemical_or_scaffold_matches']=collision;rows.append(ref);lookup[source['id']]=ref;counts[ref['status']]+=1
pq.write_table(audit_table(rows),out/'compound-or-class-reference-decisions.parquet',compression='zstd');assert pq.read_table(out/'compound-or-class-reference-decisions.parquet').to_pylist()==audit_table(rows).to_pylist();pathways=[]
with (B/'source-native-pathway-models.jsonl').open() as f:
 for line in f:
  graph=json.loads(line);ids=set()
  for entity in graph['species'].values():
   for qualifier,urls in entity['qualifiers'].items():
    if qualifier not in ('is','hasPart'):continue
    for url in urls:
     if 'chebi' in url.lower():
      m=re.search(r'CHEBI[:_](\d+)',url,re.I)
      if m:ids.add('CHEBI:'+m.group(1))
  refs=[lookup[x] for x in sorted(ids)];collisions=sorted({x for r in refs for x in r['heldout_chemical_or_scaffold_matches']});specificity_unresolved=[r['requested_id'] for r in refs if r['status'] not in ('curated_single_compound_structure','curated_single_compound_InChI_reference_after_SMILES_failure','generic_R_group_not_single_compound','wildcard_structure_not_single_compound')];generic=[r['requested_id'] for r in refs if r['status'] in ('generic_R_group_not_single_compound','wildcard_structure_not_single_compound')];compound_groups=sorted({x for r in refs if r['status'] in ('curated_single_compound_structure','curated_single_compound_InChI_reference_after_SMILES_failure') for x in [r['molecule_group'],r['scaffold_group']]+r['fragment_groups']});pathways.append(dict(record_id=graph['source_member'],source_chebi_reference_ids=sorted(ids),reviewed_compound_structure_refs=sum(r['status'] in ('curated_single_compound_structure','curated_single_compound_InChI_reference_after_SMILES_failure') for r in refs),generic_structure_references=generic,unresolved_specificity_reference_ids=specificity_unresolved,heldout_conflicting_groups=collisions,chemical_groups=compound_groups,source_static_annotation_eligible=True,chemical_holdout_review_pass=not collisions and not specificity_unresolved,reason='heldout_chemical_or_scaffold_overlap' if collisions else ('chemical_reference_specificity_or_quality_unresolved' if specificity_unresolved else 'defined_concrete_chemical_keys_checked_generic_Rgroups_not_counted_as_singlemolecules')))
pq.write_table(audit_table(pathways),out/'pathway-chemical-holdout-review.parquet',compression='zstd');summary=dict(schema='mora3.Chebi_source_identity_and_holdout_audit.v3',source_snapshot='unversioned official ChEBI API snapshot;perentrymodified_on+retrievaltime+SHA256 retained',reference_ids=2031,complete_source_files_hash_verified=2031,reference_status_counts=dict(counts),specific_concrete_reference_heldout_matches=sum(bool(r['heldout_chemical_or_scaffold_matches']) for r in rows),pathways=2889,pathway_chemical_holdout_review_pass=sum(p['chemical_holdout_review_pass'] for p in pathways),pathways_with_heldout_collisions=sum(bool(p['heldout_conflicting_groups']) for p in pathways),pathways_with_unresolved_reference_specificity=sum(bool(p['unresolved_specificity_reference_ids']) for p in pathways),compound_decisions=str((out/'compound-or-class-reference-decisions.parquet').relative_to(R)),pathway_decisions=str((out/'pathway-chemical-holdout-review.parquet').relative_to(R)),identity_policy='originalsourceSMILESretained;derivedisomericSMILES;nonstereoSMILES/Murckoscaffoldgroups;separatefragmentaudit;noformalcharge/protomer/tautomer/saltstateequivalenceasserted',generic_policy='Rgroups/wildcards arenot singlecompoundgraphs;sourcecuratedclassannotationsremaincandidatecontext,notatomisticstructures',no_experimental_property_or_assay_validation=True,license='CC-BY-4.0',primary_terms_url='https://www.ebi.ac.uk/chebi/about',global_leakage_safe=False,source_raw_immutable=True)
with (out/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
print(json.dumps(summary),flush=True)
