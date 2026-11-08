import importlib.util,json,hashlib
import pytest
from pathlib import Path
spec=importlib.util.spec_from_file_location('corpus',next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())/'scripts'/'data_intake_corpus.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def test_transitive_grouping_and_order_invariance():
 records=[('a',['sequence:p','accession:a']),('b',['accession:a','reaction:r']),('c',['reaction:r']),('d',['sequence:q'])]
 a=m.connected_groups(records);b=m.connected_groups(reversed(records))
 assert a==b and a['a']==a['b']==a['c'] and a['d']!=a['a']

def test_heldout_collision_excludes_all_correlated_train_records():
 rows=[dict(record_id='train',split='train',groups=['molecule:a','scaffold:s']),dict(record_id='held',split='test',groups=['molecule:b','scaffold:s']),dict(record_id='clean',split='train',groups=['molecule:c'])]
 kept,rejected=m.exclude_heldout_collisions(rows)
 assert [x['record_id'] for x in kept]==['held','clean']
 assert rejected[0]['record_id']=='train' and rejected[0]['conflicting_groups']==['scaffold:s']

def test_target_and_audit_mutation_cannot_change_predictor_bytes():
 a=m.CorpusTask('experimental_hydration',{'smiles':'CCO'},{'expt':-5.0},{'calc':-6.0},('physical_assay_temperature',))
 b=m.CorpusTask('experimental_hydration',{'smiles':'CCO'},{'expt':999.0},{'calc':123.0},('physical_assay_temperature',))
 assert a.input_bytes()==b.input_bytes()
 assert m.CorpusTask.from_bytes(a.canonical_bytes()).canonical_bytes()==a.canonical_bytes()
 assert 'calc' not in json.loads(a.input_bytes())

def test_overlap_roles_and_nonfinite_supervision_rejected():
 with pytest.raises(ValueError):m.CorpusTask('bad',{'target':1},{'target':2},{},())
 with pytest.raises(ValueError):m.CorpusTask('bad',{}, {'x':float('nan')},{},())

def test_scaffold_group_split_deterministic_and_nonempty_over_large_set():
 a=[m.group_split('scaffold:'+str(i),seed=20261031) for i in range(1000)]
 assert a==[m.group_split('scaffold:'+str(i),seed=20261031) for i in range(1000)]
 assert set(a)=={'train','validation','test'}

def test_uniprot_export_integer_valued_decimal_annotation_scores():
 assert m.annotation_score("5.0")==5
 assert m.annotation_score("4")==4
 for value in ["4.1","nan","inf","3.0","6",""]:
  with pytest.raises(ValueError):m.annotation_score(value)

def test_unpaired_rosetta_profile_has_no_invented_intervention():
 assert m.source_link("null") is None
 assert m.source_link("[]") is None
 assert m.source_link('["study","clone","BRDN1","U2OS"]')==("study","clone","BRDN1","U2OS")
 with pytest.raises(ValueError):m.source_link('"BRDN1"')

import pyarrow as pa
import pyarrow.parquet as pq

def test_reader_filters_actual_row_split_adapter_and_source_and_preserves_bytes(tmp_path):
 rows=[]
 for name,split,supported in [('custom','train',False),('native','train',True),('held','test',True)]:
  payload=(name+'canonical-bytes').encode();rows.append(dict(record_id=name,split=split,source_family='source',model_adapter_supported=supported,native_schema='mora3.ArrayTask/v1' if supported else 'mora3.CorpusTask/v1',native_payload=payload,native_payload_sha256=hashlib.sha256(payload).hexdigest()))
 path=tmp_path/'shard.parquet';pq.write_table(pa.Table.from_pylist(rows),path)
 manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'shards':[{'path':'shard.parquet','rows':3,'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]}))
 out=list(m.read_corpus_tasks(manifest,root=tmp_path,split='train',supported_only=True))
 assert [x['record_id'] for x in out]==['native'] and out[0]['native_payload']==b'nativecanonical-bytes'
 assert len(list(m.read_corpus_tasks(manifest,root=tmp_path,split='train',supported_only=False)))==2
 assert not list(m.read_corpus_tasks(manifest,root=tmp_path,split='train',source='absent'))

def test_reader_rejects_shard_tampering_before_yield(tmp_path):
 p=tmp_path/'bad.parquet';p.write_bytes(b'bad')
 f=tmp_path/'manifest.json';f.write_text(json.dumps({'shards':[{'path':'bad.parquet','rows':1,'bytes':3,'sha256':'0'*64}]}))
 with pytest.raises(ValueError,match='integrity'):list(m.read_corpus_tasks(f,root=tmp_path))

def test_atom_maps_and_explicit_hydrogen_do_not_hide_exact_chemical_overlap():
 a=m.chemical_identity('[C:1]([H:2])([H:3])([H:4])[O:5][H:6]')
 b=m.chemical_identity('CO')
 assert a['molecule_group']==b['molecule_group'] and a['scaffold_group']==b['scaffold_group']

def test_isomeric_identity_preserved_while_stereoisomers_share_leakage_group():
 a=m.chemical_identity('N[C@H](C)C(=O)O');b=m.chemical_identity('N[C@@H](C)C(=O)O')
 assert a['canonical_isomeric_smiles']!=b['canonical_isomeric_smiles']
 assert a['molecule_group']==b['molecule_group'] and a['scaffold_group']==b['scaffold_group']

def atoms():return [dict(type_symbol='C',label_atom_id='C'+str(i),label_comp_id='DA',label_asym_id='A',label_seq_id='1',label_alt_id='.',x_angstrom=float(i),y_angstrom=1.,z_angstrom=2.) for i in range(3)]
def test_masked_source_coordinate_cannot_reach_condition_or_source_proxy():
 a=atoms();t=m.masked_coordinate_task(a,[1]);a[1]['x_angstrom']=999.;a[1]['y_angstrom']=-3.;other=m.masked_coordinate_task(a,[1]);assert t.input_bytes()==other.input_bytes();assert t.canonical_bytes()!=other.canonical_bytes();d=json.loads(t.canonical_bytes());assert d['conditions']['positions_angstrom'][1] is None and d['targets']['masked_positions_angstrom'][0] is None

def test_coordinate_mask_is_valid_known_partial_atom_axis():
 for indices in [[],[1,1],[3],[0,1,2]]:
  with pytest.raises(ValueError):m.masked_coordinate_task(atoms(),indices)

def function_row():return {'Entry':'P00001','Sequence':'ACD','Organism (ID)':'9606','Protein names':'fixture','Function [CC]':'Function {ECO:0000269|PubMed:1}.','Catalytic activity':'','Gene Ontology IDs':'GO:0000001; GO:0000002;','GeneID':''}
def test_function_target_and_citations_are_not_predictor_proxies():
 a=function_row();t=m.protein_function_task(a);a['Function [CC]']='mutatedannotation';a['Protein names']='mutatedname';u=m.protein_function_task(a);assert t.input_bytes()==u.input_bytes();assert t.canonical_bytes()!=u.canonical_bytes();assert set(json.loads(t.input_bytes())['conditions'])=={'amino_acid_sequence','organism_taxon'}
def test_gene_xref_missingness_and_duplicates_do_not_block_other_tasks():
 assert m.gene_reference_ids(' 1; 2; 2; ')==('1','2')
 assert m.gene_reference_ids('')==()
 with pytest.raises(ValueError):m.gene_reference_ids('invalid')
 assert m.protein_function_task(function_row())
def test_missing_functional_supervision_is_not_fabricated_from_name():
 r=function_row();r['Function [CC]']='';r['Gene Ontology IDs']=''
 with pytest.raises(ValueError,match='functional annotation'):m.protein_function_task(r)

def test_integer_gene_csv_display_and_missing_sentinel_are_not_new_gene_ids():
 assert m.integer_gene_id('658.0')=='658'
 assert m.integer_gene_id('658')=='658'
 assert m.integer_gene_id('-666') is None
 assert m.integer_gene_id('') is None
 for value in ['658.1','BRD-K123','1e3']:
  with pytest.raises(ValueError):m.integer_gene_id(value)
def test_source_gene_fields_reconciled_without_gene_symbol_guess():
 r=m.rosetta_gene_reference({'Metadata_NCBIGeneID':'658.0','NCBIGeneID':'658','Symbol':'untrusted-symbol'})
 assert r['gene_id']=='658' and r['status']=='source_numeric_reference'
 r=m.rosetta_gene_reference({'Metadata_NCBIGeneID':'658','NCBIGeneID':'659'})
 assert r['gene_id'] is None and r['status']=='conflicting_source_gene_fields'
 r=m.rosetta_gene_reference({'Symbol':'GENE'})
 assert r['gene_id'] is None and r['status']=='missing_source_gene_reference'

def test_heterogeneous_audit_fields_survive_parquet_schema_inference():
 rows=[{'source':'BioGRID','gene':'1'},{'source':'Rosetta','gene':'2','source_fields_json':'{"Metadata_GeneID":"2.0"}','normalization_status':'source_numeric_reference'}]
 table=m.audit_table(rows);out=table.to_pylist();assert out[0]['source_fields_json'] is None;assert out[1]['source_fields_json']=='{"Metadata_GeneID":"2.0"}';assert out[1]['normalization_status']=='source_numeric_reference'

def fixture(tmp_path):
 raw=b'canonical-source-payload';task='task';row=dict(task_id=task,record_id='image',split='train',source_family='source',model_adapter_supported=True,native_schema='mora3.ArrayTask/v1',native_payload=raw,native_payload_sha256=hashlib.sha256(raw).hexdigest(),group_ids_json='["image_pixels:encoded-file"]');data=tmp_path/'data.parquet';pq.write_table(pa.Table.from_pylist([row]),data);groups=tmp_path/'groups.parquet';pq.write_table(pa.Table.from_pylist([dict(task_id=task,effective_groups_json='["decoded_pixels:actual","gene:source-reference"]')]),groups);entry=dict(path='groups.parquet',bytes=groups.stat().st_size,sha256=hashlib.sha256(groups.read_bytes()).hexdigest(),rows=1);manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(shards=[dict(path='data.parquet',rows=1,bytes=data.stat().st_size,sha256=hashlib.sha256(data.read_bytes()).hexdigest())],effective_group_lookup={'data.parquet':entry})));return manifest,groups,raw

def test_canonical_payload_preserved_with_authoritative_effective_groups(tmp_path):
 f,g,raw=fixture(tmp_path);row=next(m.read_corpus_tasks(f,root=tmp_path));assert row['native_payload']==raw;assert json.loads(row['group_ids_json'])==['decoded_pixels:actual','gene:source-reference'];assert row['source_group_ids_json']=='["image_pixels:encoded-file"]';assert row['group_metadata_status']=='manifest_verified_effective_groups'

def test_group_sidecar_integrity_checked_before_any_task_yield(tmp_path):
 f,g,raw=fixture(tmp_path);g.write_bytes(b'tampered')
 with pytest.raises(ValueError,match='group.*integrity'):next(m.read_corpus_tasks(f,root=tmp_path))

def test_declared_effective_group_index_must_cover_consumed_shard(tmp_path):
    payload=b'native bytes';p=tmp_path/'s.parquet'
    pq.write_table(pa.Table.from_pylist([dict(task_id='t',record_id='r',split='train',source_family='s',native_schema='mora3.ArrayTask/v1',model_adapter_supported=True,native_payload=payload,native_payload_sha256=hashlib.sha256(payload).hexdigest())]),p)
    f=tmp_path/'m.json';f.write_text(json.dumps(dict(shards=[dict(path='s.parquet',rows=1,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())],effective_group_lookup={})))
    with pytest.raises(ValueError,match='missing effective group metadata'):
        list(m.read_corpus_tasks(f,root=tmp_path))
