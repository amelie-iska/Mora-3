#!/usr/bin/env python3
"""Immutable inspection/curation of pinned public intake sources; never trains a model.

All original fields survive as source_row_json. Output directories must be fresh.
Property benchmarks are evaluation-reserved; no final training split is assigned.
"""
import argparse,csv,functools,gzip,hashlib,json,math,pathlib,re,zipfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
from rdkit import Chem,RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.error')
RDLogger.DisableLog('rdApp.warning')

def digest(path):
 h=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()

def fresh_output(path):
 path=pathlib.Path(path);path.mkdir(parents=True,exist_ok=False);return path

def label_value(value):
 if value is None or str(value).strip().lower() in ('','nan','na','none'):return None
 value=float(value)
 if not math.isfinite(value):raise ValueError('Nonfinite label')
 return value

@functools.lru_cache(maxsize=100000)
def molecular_identity(smiles):
 mol=Chem.MolFromSmiles(smiles) if smiles else None
 if mol is None or not mol.GetNumAtoms() or any(a.GetAtomicNum()==0 for a in mol.GetAtoms()):raise ValueError('Invalid/empty/wildcard SMILES')
 iso=Chem.MolToSmiles(mol,isomericSmiles=True,canonical=True)
 non=Chem.MolToSmiles(mol,isomericSmiles=False,canonical=True)
 # Scaffold grouping intentionally ignores stereo; copy preserves source identity.
 scaffold_mol=Chem.Mol(mol);Chem.RemoveStereochemistry(scaffold_mol)
 scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=scaffold_mol,includeChirality=False)
 return dict(canonical_isomeric_smiles=iso,molecule_group='molecule:'+hashlib.sha256(non.encode()).hexdigest(),scaffold_group='scaffold:'+hashlib.sha256((scaffold or non).encode()).hexdigest(),stereochemistry_policy='preserve source; stereoisomers share leakage identity')

def annotation_quality(score,sequence):
 try:
  value=float(score.get('Annotation','nan'))
  value=int(value) if math.isfinite(value) and value.is_integer() and 0<=value<=5 else None
 except (TypeError,ValueError):value=None
 return value, bool(value is not None and value>=4 and score.get('Sequence')==sequence and score.get('Reviewed')=='reviewed')

def validate_count_chunk(indices,values,*,start,n_genes):
 if values.ndim!=2 or values.shape[1]!=n_genes or len(indices)!=len(values):raise ValueError('Invalid dimensions')
 if not np.array_equal(indices,np.arange(start,start+len(values))):raise ValueError('Noncontiguous source indices')
 if not np.isfinite(values).all() or (values<0).any() or (values!=np.floor(values)).any() or (values>np.iinfo(np.int32).max).any():raise ValueError('Counts must be finite nonnegative int32')
 return values.astype(np.int32)

def write_json(path,value):
 with pathlib.Path(path).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def write_table(path,rows):
 pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')

def provenance(base,relative):
 path=base/relative
 receipt=base/'raw'/(relative.removeprefix('raw/').replace('/','__')+'.receipt.json')
 result=json.loads(receipt.read_text())
 if result['status'] not in ('downloaded_verified','downloaded_hash_pinned') or result['sha256']!=digest(path):raise ValueError('Source incomplete, failed, or checksum changed')
 return {k:result.get(k) for k in ['path','url','revision','license','bytes','sha256','md5','expected_md5','status']}

def property_label_contract(source,values):
 if source.startswith('FreeSolv'):
  return dict(default_target_fields=['expt'],computed_target_opt_in_fields=['calc'],audit_uncertainty_fields=[k for k in ['d_expt','d_calc'] if k in values],label_evidence={k:('experimental' if k=='expt' else 'computed' if k=='calc' else 'experimental_uncertainty_may_be_default' if k=='d_expt' else 'computed_uncertainty') for k in values})
 return dict(default_target_fields=list(values),computed_target_opt_in_fields=[],audit_uncertainty_fields=[],label_evidence={k:'source_assay_or_property_label' for k in values})

def properties(base):
 out=fresh_output(base/'curated/molecular-properties-v2')
 specs=[('ESOL','delaney-processed.csv','smiles',['measured log solubility in mols per litre'],'log10(mol/L)'),('FreeSolv-MoleculeNet','SAMPL.csv','smiles',['expt','calc'],'kcal/mol'),('Lipophilicity','Lipophilicity.csv','smiles',['exp'],'logD dimensionless; assay pH source review pending'),('BACE','bace.csv','mol',['Class','pIC50'],'Class binary; pIC50 source-defined'),('BBBP','BBBP.csv','smiles',['p_np'],'binary'),('HIV','HIV.csv','smiles',['HIV_active'],'binary'),('Tox21','tox21.csv.gz','smiles',None,'binary'),('SIDER','sider.csv.gz','smiles',None,'binary')]
 summaries=[];all_rows=[];quarantine=[]
 for name,filename,smiles_field,labels,unit in specs:
  p=base/'raw/moleculenet'/filename
  src=provenance(base,'raw/moleculenet/'+filename)
  opener=gzip.open if p.suffix=='.gz' else open
  accepted=[];rejected=[];missing=0
  with opener(p,'rt',newline='') as f:
   reader=csv.DictReader(f);headers=reader.fieldnames
   if len(set(headers))!=len(headers):raise ValueError('Duplicate headers')
   tasklabels=labels if labels is not None else [x for x in headers if x not in ('smiles','mol_id')]
   for i,row in enumerate(reader):
    try:
     identity=molecular_identity(row[smiles_field]);values={k:label_value(row[k]) for k in tasklabels}
     if all(v is None for v in values.values()):raise ValueError('All task labels missing')
     missing+=sum(v is None for v in values.values())
     accepted.append(dict(record_id=f'{name}:{i}',source=name,source_row=i,source_sha256=src['sha256'],source_split=row.get('Model'),**identity,label_contract_json=json.dumps(property_label_contract(name,values)),labels_json=json.dumps(values,sort_keys=True,allow_nan=False),source_row_json=json.dumps(row,ensure_ascii=False,sort_keys=True),label_units=unit,role='provisional_evaluation_reserve',split='unassigned_evaluation_reserve',training_eligible=False,eligibility_reason='benchmark reserved; source-specific rights and cross-corpus exclusion review pending',predictor_fields_json='["canonical_isomeric_smiles"]',missing_context_json=json.dumps(['cell_context','dose','time','protein_target','expression','morphology','3d_structure'])))
    except (ValueError,KeyError) as e:rejected.append(dict(source=name,source_row=i,reason=str(e),source_row_json=json.dumps(row,sort_keys=True)))
  all_rows.extend(accepted);quarantine.extend(rejected)
  summaries.append(dict(source=name,headers=headers,task_labels=property_label_contract(name,{k:None for k in tasklabels})['default_target_fields'],available_label_fields=tasklabels,computed_optional_task_labels=property_label_contract(name,{k:None for k in tasklabels})['computed_target_opt_in_fields'],available_rows=len(accepted)+len(rejected),retained_rows=len(accepted),rejected_rows=len(rejected),missing_label_values=missing,provenance=src,training_eligible_rows=0))
 # Original FreeSolv release retains experimental vs calculated values and uncertainty provenance.
 p=base/'source-metadata/freesolv-database.json';data=json.loads(p.read_text());pin=json.loads((base/'source-metadata/freesolv-commit.json').read_text())['sha'];original=[]
 for key,row in data.items():
  ident=molecular_identity(row['smiles'])
  original.append(dict(record_id=key,source='FreeSolv-original',**ident,label_contract_json=json.dumps(property_label_contract('FreeSolv-original',{k:row[k] for k in ['expt','d_expt','calc','d_calc']})),labels_json=json.dumps({k:row[k] for k in ['expt','d_expt','calc','d_calc']}),label_units='kcal/mol',source_row_json=json.dumps(row,sort_keys=True),role='provisional_evaluation_reserve',split='unassigned_evaluation_reserve',training_eligible=False,eligibility_reason='evaluation-reserved; CC-BY-4.0 with source caveats',source_sha256=digest(p),source_row=None,source_split=None,predictor_fields_json='["canonical_isomeric_smiles"]',missing_context_json=json.dumps(['cell_context','dose','time','protein_target','expression','morphology','3d_structure'])))
 all_rows.extend(original)
 summaries.append(dict(source='FreeSolv-original',task_labels=['expt'],computed_optional_task_labels=['calc'],audit_uncertainty_fields=['d_expt','d_calc'],available_rows=len(data),retained_rows=len(original),rejected_rows=0,training_eligible_rows=0,license='CC-BY-4.0; upstream caveat retained',revision=pin,source_sha256=digest(p),source_path=str(p),experimental_uncertainty_policy='source d_expt may be assigned default; preserve notes and do not claim all are measured errors'))
 write_table(out/'records.parquet',all_rows)
 write_json(out/'quarantine.json',quarantine)
 groups={}
 for row in all_rows:groups.setdefault(row['molecule_group'],set()).add(row['source'])
 overlaps=[dict(molecule_group=k,sources=sorted(v)) for k,v in groups.items() if len(v)>1]
 write_json(out/'cross-source-identity-overlaps.json',overlaps)
 manifest=dict(schema_version=1,seed=20261031,source_summaries=summaries,retained_rows=len(all_rows),unique_molecule_groups=len(groups),cross_source_overlap_groups=len(overlaps),rejected_rows=len(quarantine),training_eligible_rows=0,training_ready=False,split_policy='Property benchmark rows provisionally evaluation-reserved; final training/evaluation allocation remains undecided. Global molecule/scaffold exclusions must propagate to training corpora.',predictor_policy='Only molecular identity. Source descriptors/predictions/other labels remain audit-only; FreeSolv calc requires a separate explicitly opted-in computed objective; expt is its default target.',output_sha256=digest(out/'records.parquet'))
 write_json(out/'manifest.json',manifest);print('properties',json.dumps({k:manifest[k] for k in ['retained_rows','unique_molecule_groups','rejected_rows','cross_source_overlap_groups']}),flush=True)

def sciplex(base):
 out=fresh_output(base/'curated/gperturb-sciplex2-full-v1');src=provenance(base,'raw/gperturb/SciPlex2_count.csv')
 root=base.parents[2];old=root/'outputs/implementation/I6-gperturb-sciplex-access'
 paths={n:old/n for n in ['SciPlex2_cell_info.csv','SciPlex2_gene_name.csv','SciPlex2_perturbation.csv']}
 article=json.loads((base/'source-metadata/gperturb-v3.json').read_text());import hashlib
 sources=[src]
 for name,p in paths.items():
  expected=next(x for x in article['files'] if x['name']==name)
  if hashlib.md5(p.read_bytes()).hexdigest()!=expected['computed_md5']:raise ValueError('Existing metadata MD5 mismatch')
  sources.append(dict(path=str(p),bytes=p.stat().st_size,sha256=digest(p),md5=expected['computed_md5'],revision='figshare:26491588.v3',reuse='read-only reference; no copy'))
 info=pd.read_csv(paths['SciPlex2_cell_info.csv']);genes=pd.read_csv(paths['SciPlex2_gene_name.csv']);pert=pd.read_csv(paths['SciPlex2_perturbation.csv'])
 if list(info.columns)!=['Unnamed: 0','n_genes','n_counts'] or list(genes.columns)!=['0'] or list(pert.columns)!=['Unnamed: 0','Dex','Nutlin','BMS','Vehicle','SAHA']:raise ValueError('Unexpected metadata schema')
 if len(info)!=len(pert) or not info['Unnamed: 0'].is_unique or not np.array_equal(pert.iloc[:,0],np.arange(len(info))):raise ValueError('Metadata identities/positions invalid')
 n_genes=len(genes);n_cells=len(info);blocks=[];start=0
 columns=None
 for block in pd.read_csv(base/'raw/gperturb/SciPlex2_count.csv',chunksize=512):
  if list(block.columns)!=['Unnamed: 0']+[str(i) for i in range(n_genes)]:raise ValueError('Count header/gene order mismatch')
  val=validate_count_chunk(block.iloc[:,0].to_numpy(),block.iloc[:,1:].to_numpy(),start=start,n_genes=n_genes)
  blocks.append(sparse.csr_matrix(val));start+=len(val)
 if start!=n_cells:raise ValueError('Incomplete count matrix')
 mat=sparse.vstack(blocks,format='csr');sparse.save_npz(out/'counts-csr.npz',mat,compressed=True)
 intervention=pert.iloc[:,1:].to_numpy(dtype=float)
 if not np.isfinite(intervention).all() or (intervention<0).any() or not ((intervention>0).sum(axis=1)==1).all():raise ValueError('Invalid interventions')
 records=[]
 for i in range(n_cells):
  col=int(np.flatnonzero(intervention[i]>0)[0]);drug=pert.columns[1+col]
  records.append(dict(record_id=f'GSM4150377:{info.iloc[i,0]}',source_row=i,cell_barcode=str(info.iloc[i,0]),intervention_alias=drug,source_scaled_dose=float(intervention[i,col]),physical_dose=None,physical_dose_unit=None,vehicle_control=drug=='Vehicle',cell_line='A549',exposure_hours=24,expression_array='counts-csr.npz',expression_row=i,gene_order='genes.json',baseline_matched_cell=None,post_assay_n_genes=int(info.iloc[i,1]),post_assay_n_counts=float(info.iloc[i,2]),leakage_group='study:GSE139944:GSM4150377',split='unassigned_study_group',eligible_task_candidates_json=json.dumps(['expression_reconstruction','observational_alias_response_with_source_coordinate']),physical_dose_conditioned_eligible=False,training_ready=False,missing_context_json=json.dumps(['physical_dose_mapping','well','replicate','donor','matched_baseline','chemical_structure','morphology']),prediction_input_fields_json=json.dumps(['intervention_alias','source_scaled_dose','vehicle_control','cell_line','exposure_hours']),expression_target_fields_json='["expression_array","expression_row"]',audit_only_fields_json='["cell_barcode","post_assay_n_genes","post_assay_n_counts","record_id"]'))
 write_table(out/'records.parquet',records)
 write_json(out/'genes.json',[dict(position=i,source_symbol=str(s)) for i,s in enumerate(genes['0'])])
 reloaded=sparse.load_npz(out/'counts-csr.npz')
 if (reloaded!=mat).nnz:raise ValueError('Array serialization changed values')
 manifest=dict(schema_version=1,seed=20261031,source_release='figshare:26491588.v3',license='MIT (versioned processed release)',sources=sources,available_cells=n_cells,retained_cells=n_cells,gene_columns=n_genes,count_values=n_cells*n_genes,nonzero_count_values=mat.nnz,max_count=int(mat.max()),min_count=0,source_headers={'count':['Unnamed: 0']+[str(i) for i in range(n_genes)],'cell_info':list(info.columns),'perturbation':list(pert.columns),'genes':list(genes.columns)},intervention_counts=pert.iloc[:,1:].gt(0).sum().to_dict(),counts_validated='entire matrix finite nonnegative integer, all source positions and 5000 named columns, lossless CSR round trip',pair_evidence='Author positional CSV export contract reviewed at commit05d5a431413ff9dc443c82b994b571ecc563426d; complete file official MD5 verified. No independent raw barcode validation.',context_evidence='Original sci-Plex2 transcriptomics A549/24h; planning/research/datasets-perturbnet-sciplex.md',missing_physical_dose=True,training_ready=False,task_eligibility='Expression reconstruction and observational source-coordinate response candidates retained; physical dose-conditioned tasks ineligible. Source validation is not model integration.',split_policy='One study group; no random cell split or held-out claim.',outstanding_gates=['source raw barcode cross-check','task/schema integration','study/perturbation leakage-safe evaluation','physical dose mapping for physical-dose tasks'],artifacts={p.name:dict(bytes=p.stat().st_size,sha256=digest(p)) for p in out.iterdir()})
 write_json(out/'manifest.json',manifest);print('sciplex',json.dumps({k:manifest[k] for k in ['available_cells','retained_cells','gene_columns','count_values','nonzero_count_values']}),flush=True)

def reactzyme(base):
 out=fresh_output(base/'curated/reactzyme-original-v2')
 sources=[provenance(base,'raw/reactzyme/'+n) for n in ['cleaned_uniprot_rhea.tsv','rhea_molecules.tsv']]
 reactions={};invalid=[]
 with (base/'raw/reactzyme/rhea_molecules.tsv').open() as f:
  reader=csv.DictReader(f,delimiter='\t')
  if reader.fieldnames!=['Rhea ID','substrate','product']:raise ValueError('Unexpected Rhea schema')
  for row in reader:
   rid=row['Rhea ID']
   if rid in reactions:raise ValueError('Duplicate Rhea ID')
   try:reactions[rid]=(row,molecular_identity(row['substrate']),molecular_identity(row['product']))
   except ValueError:invalid.append(row)
 records=[];rejects=[];seen=set();enzymes=set();seqs=set();rhea=set();annotation_matches=0
 scorepath=base/'source-metadata/reactzyme-uniprot-200.tsv';scores={}
 if scorepath.exists():
  with scorepath.open() as f:
   for s in csv.DictReader(f,delimiter='\t'):scores[s['Entry']]=s
 with (base/'raw/reactzyme/cleaned_uniprot_rhea.tsv').open() as f:
  reader=csv.DictReader(f,delimiter='\t')
  if reader.fieldnames!=['Entry','EC number','Rhea ID','Date of creation','Sequence']:raise ValueError('Unexpected enzyme schema')
  for source_index,row in enumerate(reader):
   seq=row['Sequence'];entry=row['Entry'];sequence_valid=bool(re.fullmatch('[ACDEFGHIKLMNPQRSTVWYBXZJUO]+',seq))
   for rid in row['Rhea ID'].split(';'):
    key=(entry,rid,seq)
    if key in seen:rejects.append(dict(source_row=source_index,entry=entry,rhea_id=rid,reason='duplicate exact enzyme/reaction/sequence'));continue
    seen.add(key)
    if not sequence_valid or rid not in reactions:rejects.append(dict(source_row=source_index,entry=entry,rhea_id=rid,reason='invalid sequence or unavailable/invalid molecular reaction'));continue
    reaction,sub,prod=reactions[rid];scorevalue,verified=annotation_quality(scores.get(entry,{}),seq)
    annotation_matches+=int(verified)
    sh=hashlib.sha256(seq.encode()).hexdigest();enzymes.add(entry);seqs.add(sh);rhea.add(rid)
    records.append(dict(record_id=f'ReactZyme2024:{entry}:{rid}',entry=entry,sequence=seq,sequence_sha256=sh,ec_number_source=row['EC number'],rhea_id=rid,date_of_creation_source=row['Date of creation'],substrate_smiles_source=reaction['substrate'],product_smiles_source=reaction['product'],substrate_canonical_smiles=sub['canonical_isomeric_smiles'],product_canonical_smiles=prod['canonical_isomeric_smiles'],substrate_molecule_group=sub['molecule_group'],product_molecule_group=prod['molecule_group'],annotation_score_current=scorevalue,annotation_score_and_sequence_verified=verified,source_row=source_index,source_row_json=json.dumps(row,sort_keys=True),split='unassigned_connected_groups',eligible_task_candidate='enzyme-reaction mapping' if verified else 'research only; annotation>=4 verification pending',training_ready=False,pair_evidence='SwissProt/Rhea annotated catalysis; not a measured kinetic rate, binding affinity, or transition-state geometry',missing_modalities_json=json.dumps(['kinetics','physical_conditions','transition_state_geometry','protein_3d_structure','cell_expression','morphology']),sequence_group='sequence:'+sh,reaction_group='rhea:'+rid))
 write_table(out/'records.parquet',records);write_json(out/'rejected-pairs.json',rejects);write_json(out/'invalid-rhea-molecules.json',invalid)
 manifest=dict(schema_version=1,seed=20261031,release='Zenodo:11494913; SwissProt/Rhea2024-01-08',license='MIT (original deposit); UniProt/Rhea attribution retained',sources=sources,retained_pairs=len(records),rejected_pairs=len(rejects),unique_enzyme_accessions=len(enzymes),unique_sequence_hashes=len(seqs),unique_rhea_reactions=len(rhea),valid_molecular_reactions=len(reactions),invalid_molecular_reactions=len(invalid),annotation_score_sequence_verified_pairs=annotation_matches,training_ready=False,split_policy='No final splits. Group transitively by protein identity/homology, reaction and participating chemical identities; preserve original split archives as independent strategies, never merge their train/test partitions.',outstanding_gates=['full annotation>=4 release/sequence verification','connected reaction/chemical/protein homology leakage groups','official split strategy alignment','shared model adapter/masks'],artifact_sha256=digest(out/'records.parquet'))
 write_json(out/'manifest.json',manifest);print('reactzyme',json.dumps({k:manifest[k] for k in ['retained_pairs','rejected_pairs','unique_enzyme_accessions','unique_rhea_reactions','annotation_score_sequence_verified_pairs']}),flush=True)

def biogrid(base):
 out=fresh_output(base/'curated/biogrid-chemicals-v1');src=provenance(base,'raw/biogrid/BIOGRID-CHEMICALS-5.0.262.chemtab.zip')
 rows=[]
 with zipfile.ZipFile(src['path']) as z:
  names=[x for x in z.namelist() if not x.endswith('/')]
  if len(names)!=1:raise ValueError('Unexpected chemical archive members')
  import io
  reader=csv.DictReader(io.TextIOWrapper(z.open(names[0]),encoding='utf-8-sig'),delimiter='\t');headers=reader.fieldnames
  for i,row in enumerate(reader):
   if None in row or any(v is None for v in row.values()):raise ValueError('Malformed chemical interaction row')
   rows.append(dict(record_id='BioGRID5.0.262:chemical:'+str(i),source_row=i,source_row_json=json.dumps(row,sort_keys=True),evidence_type='source-annotated chemical/gene interaction; preserve exact experimental system and action',split='unassigned_publication_gene_chemical_groups',training_ready=False,missing_context_json='["affinity_unless_source_measured","paired_expression","morphology","full_protein_sequence"]'))
 write_table(out/'records.parquet',rows)
 manifest=dict(schema_version=1,release='5.0.262',license='MIT; original authors attribution required',source=src,headers=headers,available_rows=len(rows),retained_rows=len(rows),training_ready=False,outstanding_gates=['gene/chemical identifiers and mechanisms adapter','publication/chemical/gene overlap grouping','shared model task integration'],output_sha256=digest(out/'records.parquet'))
 write_json(out/'manifest.json',manifest);print('biogrid',len(rows),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--source',choices=['properties','sciplex','reactzyme','biogrid'],required=True);a=p.parse_args();globals()[a.source](a.base)
if __name__=='__main__':main()
