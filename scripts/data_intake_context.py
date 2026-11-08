#!/usr/bin/env python3
"""Pinned public kinetics and pathway data; no causal response or clinical claims."""
import argparse,csv,hashlib,json,pathlib,re,tarfile,xml.etree.ElementTree as ET
import pyarrow as pa
import pyarrow.parquet as pq
from data_intake_curate import digest,fresh_output,write_json,provenance,annotation_quality

def kinetic_measurements(entry):
 if entry.get('entryType')!='UniProtKB reviewed (Swiss-Prot)' or entry.get('annotationScore',0)<4:raise ValueError('Require reviewed annotation>=4')
 seq=entry.get('sequence',{}).get('value','')
 if not re.fullmatch('[ACDEFGHIKLMNPQRSTVWYBXZJUO]+',seq):raise ValueError('Invalid sequence')
 records=[]
 comments=[c for c in entry.get('comments',[]) if c.get('commentType')=='BIOPHYSICOCHEMICAL PROPERTIES']
 import math
 for ci,c in enumerate(comments):
  params=c.get('kineticParameters',{})
  for key,label,field in [('michaelisConstants','Km','constant'),('maximumVelocities','Vmax','velocity')]:
   for ki,p in enumerate(params.get(key,[])):
    value=float(p[field]);unit=p.get('unit')
    if not math.isfinite(value) or value<0 or not isinstance(unit,str) or not unit:raise ValueError('Invalid kinetic quantity/unit')
    records.append(dict(record_id=f"UniProt2026_03:{entry['primaryAccession']}:{ci}:{key}:{ki}",accession=entry['primaryAccession'],sequence=seq,sequence_sha256=hashlib.sha256(seq.encode()).hexdigest(),annotation_score=int(entry['annotationScore']),measurement_type=label,value=value,unit=unit,substrate_alias=p.get('substrate') or p.get('enzyme'),substrate_smiles=None,assay_ph=None,assay_temperature=None,assay_temperature_unit=None,assay_exposure_time=None,evidence_json=json.dumps(p.get('evidences',[]),sort_keys=True),source_measurement_json=json.dumps(p,sort_keys=True),protein_biophysiochemistry_json=json.dumps(comments,sort_keys=True),evidence_type='curated kinetic annotation; inspect ECO and publication for experimental basis',context_rule='Protein-wide pH/temperature notes are not assigned to this measurement without shared assay evidence.',split='unassigned_sequence_substrate_publication_groups',training_ready=False,predictor_fields_json='["sequence","substrate_alias"]',target_fields_json='["measurement_type","value","unit"]'))
 return records

def resource_ids(element):
 return sorted({v for x in element.iter() for k,v in x.attrib.items() if k.endswith('resource')})

def parse_sbml(raw,member):
 root=ET.fromstring(raw);model=root.find('./{*}model')
 if model is None:raise ValueError('SBML model missing')
 species=[];declared=set();reactions=[]
 for el in model.findall('./{*}listOfSpecies/{*}species'):
  id=el.get('id')
  if not id or id in declared:raise ValueError('Missing/duplicate species ID')
  declared.add(id);species.append(dict(source_member=member,model_id=model.get('id'),pathway_name=model.get('name'),species_id=id,name=el.get('name'),compartment=el.get('compartment'),sbo_term=el.get('sboTerm'),source_attributes_json=json.dumps(el.attrib,sort_keys=True),resource_ids_json=json.dumps(resource_ids(el)),source_xml=ET.tostring(el,encoding='unicode')))
 for el in model.findall('./{*}listOfReactions/{*}reaction'):
  groups=[]
  for section,tag in [('listOfReactants','speciesReference'),('listOfProducts','speciesReference'),('listOfModifiers','modifierSpeciesReference')]:
   nodes=[]
   for p in el.findall(f'./{{*}}{section}/{{*}}{tag}'):
    if p.get('species') not in declared:raise ValueError('Undeclared reaction participant')
    nodes.append(dict(species=p.get('species'),stoichiometry=p.get('stoichiometry'),sbo_term=p.get('sboTerm'),source_attributes=dict(p.attrib)))
   groups.append(nodes)
  law=el.find('./{*}kineticLaw')
  reactions.append(dict(source_member=member,model_id=model.get('id'),pathway_name=model.get('name'),reaction_id=el.get('id'),reaction_name=el.get('name'),compartment=el.get('compartment'),reversible_source=el.get('reversible'),reactants=groups[0],products=groups[1],modifiers=groups[2],resource_ids=resource_ids(el),kinetic_law_xml=ET.tostring(law,encoding='unicode') if law is not None else None,source_xml=ET.tostring(el,encoding='unicode')))
 return species,reactions

def kinetics(base):
 out=fresh_output(base/'curated/uniprot-kinetics-v1');p=base/'source-metadata/uniprot-kinetics-100.json';receipt=json.loads((base/'source-metadata/uniprot-kinetics-100.receipt.json').read_text())
 if digest(p)!=receipt['sha256'] or receipt['release']!='2026_03':raise ValueError('Kinetics source checksum/release mismatch')
 data=json.loads(p.read_text());rows=[];reject=[];proteins=[]
 for e in data['results']:
  try:rs=kinetic_measurements(e);rows+=rs;proteins.append(e['primaryAccession'])
  except (ValueError,KeyError) as exc:reject.append(dict(accession=e.get('primaryAccession'),reason=str(exc)))
 pq.write_table(pa.Table.from_pylist(rows),out/'measurements.parquet',compression='zstd')
 write_json(out/'rejections.json',reject)
 types={k:sum(x['measurement_type']==k for x in rows) for k in ['Km','Vmax']}
 evidence=sum(any(e.get('evidenceCode')=='ECO:0000269' for e in json.loads(x['evidence_json'])) for x in rows)
 manifest=dict(schema_version=1,source=receipt,license='CC-BY-4.0 (UniProt2026_03)',available_query_entries=receipt['total_results'],downloaded_protein_entries=len(data['results']),reviewed_annotation4_5_proteins=len(proteins),retained_measurements=len(rows),measurement_types=types,measurements_with_experimental_ECO_0000269=evidence,rejected_protein_entries=reject,training_ready=False,missing_context=['verified substrate molecular identity','measurement-linked pH and temperature where absent','kcat not inferred from Vmax','homology/substrate/publication split allocation','Mora3 task adapter integration'],selection='first100 REST-default relevance order; nonrandom source slice',artifact_sha256=digest(out/'measurements.parquet'))
 write_json(out/'manifest.json',manifest);print('kinetics',json.dumps({k:manifest[k] for k in ['reviewed_annotation4_5_proteins','retained_measurements','measurement_types','measurements_with_experimental_ECO_0000269']}),flush=True)

def species_table(rows):
 fields=['source_member','model_id','pathway_name','species_id','name','compartment','sbo_term','source_attributes_json','resource_ids_json','source_xml']
 return pa.Table.from_pylist(rows,schema=pa.schema([(k,pa.string()) for k in fields]))

def reaction_table(rows):
 fields=['source_member','model_id','pathway_name','reaction_id','reaction_name','compartment','reversible_source','reactants_json','products_json','modifiers_json','resource_ids_json','source_xml','kinetic_law_xml','evidence_type','role','split','training_ready']
 return pa.Table.from_pylist(rows,schema=pa.schema([(k,pa.bool_() if k=='training_ready' else pa.string()) for k in fields]))

def pathways(base):
 out=fresh_output(base/'curated/reactome98-human-sbml-v2');src=provenance(base,'raw/reactome98/homo_sapiens.3.1.sbml.tgz')
 # Parse members in memory; never unpack, execute, or modify raw archive.
 species_writer=reaction_writer=None;ns=nr=nlaw=0;member_index=[];regulatory=[];canonical_ids=set()
 with tarfile.open(src['path']) as archive:
  members=[x for x in archive.getmembers() if x.isfile()]
  for m in members:
   if m.size>32*1024**2 or not m.name.endswith('.sbml'):raise ValueError('Unexpected archive member budget/type')
   raw=archive.extractfile(m).read();species,reactions=parse_sbml(raw,m.name)
   srows=species;rrows=[]
   for x in reactions:
    ids=[v.rsplit('/',1)[-1] for v in x['resource_ids'] if 'reactome' in v]
    canonical_ids.update(ids)
    r=dict(source_member=x['source_member'],model_id=x['model_id'],pathway_name=x['pathway_name'],reaction_id=x['reaction_id'],reaction_name=x['reaction_name'],compartment=x['compartment'],reversible_source=x['reversible_source'],reactants_json=json.dumps(x['reactants'],sort_keys=True),products_json=json.dumps(x['products'],sort_keys=True),modifiers_json=json.dumps(x['modifiers'],sort_keys=True),resource_ids_json=json.dumps(x['resource_ids']),source_xml=x['source_xml'],kinetic_law_xml=x['kinetic_law_xml'] or '',evidence_type='Reactome curated pathway reaction; not an intervention-response experiment',role='pathway_structure_candidate',split='unassigned_pathway_reaction_entity_publication_groups',training_ready=False)
    rrows.append(r)
    if re.search(r'transcription|gene expression',x['reaction_name'] or '',re.I):regulatory.append({**r,'regulation_scope':'source-named transcription/gene-expression event; not automatically a verified TF->gene edge'})
    nlaw+=int(x['kinetic_law_xml'] is not None)
   if srows:
    table=species_table(srows)
    if species_writer is None:species_writer=pq.ParquetWriter(out/'species.parquet',table.schema,compression='zstd')
    species_writer.write_table(table)
   if rrows:
    table=reaction_table(rrows)
    if reaction_writer is None:reaction_writer=pq.ParquetWriter(out/'reactions.parquet',table.schema,compression='zstd')
    reaction_writer.write_table(table)
   ns+=len(srows);nr+=len(rrows);member_index.append(dict(member=m.name,bytes=m.size,sha256=hashlib.sha256(raw).hexdigest(),species=len(srows),reactions=len(rrows)))
 if species_writer:species_writer.close()
 if reaction_writer:reaction_writer.close()
 if regulatory:pq.write_table(pa.Table.from_pylist(regulatory),out/'transcription-event-candidates.parquet',compression='zstd')
 write_json(out/'archive-members.json',member_index)
 relations=[]
 with (base/'raw/reactome98/ReactomePathwaysRelation.txt').open() as f:
  for row in csv.reader(f,delimiter='\t'):
   if len(row)!=2:raise ValueError('Malformed pathway hierarchy')
   if all(x.startswith('R-HSA-') for x in row):relations.append(dict(parent_pathway=row[0],child_pathway=row[1],relation='pathway-has-subpathway; not biochemical sequence order'))
 pq.write_table(pa.Table.from_pylist(relations),out/'human-pathway-hierarchy.parquet',compression='zstd')
 manifest=dict(schema_version=1,release='Reactome98',license='CC0-1.0 (annotation/data files)',source=src,pathway_archive_members=len(member_index),uncompressed_member_bytes=sum(x['bytes'] for x in member_index),species_occurrences=ns,reaction_occurrences=nr,unique_reactome_resource_identifiers=len(canonical_ids),reactions_with_kinetic_law_xml=nlaw,source_named_transcription_event_occurrences=len(regulatory),human_pathway_hierarchy_edges=len(relations),training_ready=False,count_note='Occurrences repeat across nested/overlapping pathways; not independent reactions or measured response examples. Deduplication and entity-role review pending.',scientific_limits=['Reaction direction/stoichiometry/modifier source annotations retained','No rate constants, cell-specific causal outcomes, reaction order across pathways, gene targets or kinetics inferred','Pathway hierarchy is parent/subpathway only','Transcription subset is a name-based review candidate, not an admitted causal GRN'],artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir()})
 write_json(out/'manifest.json',manifest);print('pathways',json.dumps({k:manifest[k] for k in ['pathway_archive_members','species_occurrences','reaction_occurrences','source_named_transcription_event_occurrences','human_pathway_hierarchy_edges']}),flush=True)

def admission_decision(current,source_sequence,molecular_supported):
 score,eligible=annotation_quality(current or {},source_sequence)
 reason='verified' if eligible else 'sequence_mismatch' if current and current.get('Sequence')!=source_sequence else 'not_in_current_reviewed_annotation4_5_export'
 return dict(annotation_score_current=score,sequence_annotation_quality_pass=eligible,molecular_pair_quality_pass=bool(eligible and molecular_supported),quality_reason=reason)

def admission(base):
 import gzip
 out=fresh_output(base/'curated/reactzyme-full-annotation-admission-v1')
 src=provenance(base,'raw/uniprot2026_03/reviewed-annotation4-5-sequences.tsv.gz');current={}
 with gzip.open(src['path'],'rt') as f:
  reader=csv.DictReader(f,delimiter='\t')
  if reader.fieldnames!=['Entry','Annotation','Reviewed','Sequence']:raise ValueError('Unexpected full quality export schema')
  for row in reader:
   score,valid=annotation_quality(row,row['Sequence'])
   if not valid or row['Entry'] in current:raise ValueError('Full quality export has duplicate or ineligible entry')
   current[row['Entry']]=row
 valid_table=pq.read_table(base/'curated/reactzyme-original-v2/records.parquet',columns=['entry','rhea_id','sequence_sha256'])
 supported={(x['entry'],x['rhea_id'],x['sequence_sha256']) for x in valid_table.to_pylist()}
 reaction_sources={}
 with (base/'raw/reactzyme/rhea_molecules.tsv').open() as f:
  for row in csv.DictReader(f,delimiter='\t'):reaction_sources[row['Rhea ID']]=row
 rows=[];seen=set();reason_counts={};quality_pairs=quality_molecular=0;quality_accessions=set();quality_sequences=set()
 with (base/'raw/reactzyme/cleaned_uniprot_rhea.tsv').open() as f:
  for i,row in enumerate(csv.DictReader(f,delimiter='\t')):
   entry=row['Entry'];seq=row['Sequence'];sha=hashlib.sha256(seq.encode()).hexdigest()
   for rid in row['Rhea ID'].split(';'):
    key=(entry,rid,sha);decision=admission_decision(current.get(entry),seq,key in supported)
    duplicate=key in seen;seen.add(key)
    reaction=reaction_sources.get(rid,{})
    quality_pairs+=int(decision['sequence_annotation_quality_pass'] and not duplicate)
    quality_molecular+=int(decision['molecular_pair_quality_pass'] and not duplicate)
    if decision['sequence_annotation_quality_pass']:quality_accessions.add(entry);quality_sequences.add(sha)
    reason_counts[decision['quality_reason']]=reason_counts.get(decision['quality_reason'],0)+1
    rows.append(dict(record_id=f'ReactZyme2024:{entry}:{rid}',source_row=i,entry=entry,rhea_id=rid,sequence=seq,sequence_sha256=sha,ec_number_source=row['EC number'],source_rhea_ids=row['Rhea ID'],date_of_creation_source=row['Date of creation'],substrate_smiles_source=reaction.get('substrate'),product_smiles_source=reaction.get('product'),molecular_source_status='validated_molecular_pair' if key in supported else 'molecular_conversion_unsupported' if reaction else 'reaction_molecules_missing',duplicate_source_pair=duplicate,**decision,training_ready=False,split='unassigned_connected_protein_homology_reaction_chemical_groups',eligibility_note='Quality pass is task-specific admission evidence; rights attribution, leakage allocation and Mora3 adapter remain gates. Unsupported molecular pairs may retain verified sequence/Rhea annotation tasks.'))
 table=pa.Table.from_pylist(rows);pq.write_table(table,out/'pair-quality-decisions.parquet',compression='zstd')
 from pyarrow import compute as pc
 eligible=table.filter(pc.and_(table['molecular_pair_quality_pass'],pc.invert(table['duplicate_source_pair'])))
 pq.write_table(eligible,out/'molecular-quality-candidates.parquet',compression='zstd')
 manifest=dict(schema_version=1,source_quality_release=src,raw_reactzyme_sources=[provenance(base,'raw/reactzyme/'+x) for x in ['cleaned_uniprot_rhea.tsv','rhea_molecules.tsv']],quality_export_reviewed_score4_5_entries=len(current),source_enzyme_reaction_pairs=len(rows),unique_source_pair_identities=len(seen),quality_verified_sequence_rhea_pairs=quality_pairs,quality_verified_molecular_pairs=quality_molecular,quality_verified_unique_accessions=len(quality_accessions),quality_verified_unique_sequence_hashes=len(quality_sequences),quality_reasons=reason_counts,training_ready=False,source_sequence_policy='Full original sequence must exactly match UniProt2026_03 reviewed annotation4/5 export. Missing or mismatched entries fail closed; no silent sequence replacement.',task_specific_eligibility='Molecular unsupported rows retain verified sequence/Rhea annotation candidates where quality passes. Physical kinetics/transition-state/assay outcomes remain missing.',split_policy='No final train/validation/test or homology clustering; biological identities and connected groups require review.',artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir()})
 write_json(out/'manifest.json',manifest);print('admission',json.dumps({k:manifest[k] for k in ['quality_export_reviewed_score4_5_entries','source_enzyme_reaction_pairs','quality_verified_sequence_rhea_pairs','quality_verified_molecular_pairs','quality_verified_unique_accessions','quality_reasons']}),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--source',choices=['kinetics','pathways','admission'],required=True);args=p.parse_args();globals()[args.source](args.base)
if __name__=='__main__':main()
