import pathlib,json,hashlib,time,sys,copy,collections
import pyarrow as pa
import pyarrow.parquet as pq
R=pathlib.Path('/home/iska/Documents/amelie/bio/Mora-3');sys.path.insert(0,str(R/'scripts'))
from data_intake_sbml import masked_pathway_task
from data_intake_corpus import CorpusTask,audit_table
B=R/'outputs/data-intake/20261031-pathway-semantics-v2';C=B/'curated-tasks-v1';C.mkdir(exist_ok=False);(C/'shards/train').mkdir(parents=True);audit=json.loads((B/'semantic-audit.json').read_text());source=R/audit['native_graphs_path'];assert hashlib.sha256(source.read_bytes()).hexdigest()==audit['native_graphs_sha256'];start=time.perf_counter();rows=[];shards=[];counts=collections.Counter();sizes=[];decisions=[];seen=set();source_reactions=set();context_reactions=collections.Counter();payload_bytes=0;target_mutations=0

def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def rel(p):return str(pathlib.Path(p).relative_to(R))
def flush():
 global rows,payload_bytes
 if not rows:return
 p=C/'shards/train'/f'{len(shards):05d}.parquet';pq.write_table(audit_table(rows),p,compression='zstd');back=pq.read_table(p).to_pylist();assert back==audit_table(rows).to_pylist()
 for row in back:
  raw=row['native_payload'];assert hashlib.sha256(raw).hexdigest()==row['native_payload_sha256'];assert CorpusTask.from_bytes(raw).canonical_bytes()==raw
 shards.append(dict(path=rel(p),source_family='Reactome98-static-pathway-prior',split='train',rows=len(rows),bytes=p.stat().st_size,sha256=sha(p)));rows=[];payload_bytes=0
with source.open() as f:
 for source_row,line in enumerate(f):
  graph=json.loads(line);N=len(graph['reactions']);assert N>0;selected=sorted(sorted(range(N),key=lambda i:hashlib.sha256(('20261031:'+','.join(graph['reactions'][i]['source_human_reaction_identity'] or [graph['reactions'][i]['attributes']['id']])).encode()).digest())[:max(1,N//10)]);maskset=set(selected);groups=[]
  for reaction in graph['reactions']:
   for key in reaction['source_human_reaction_identity']:groups.append('reactome_reaction:'+key);source_reactions.add(key);context_reactions[key]+=1
  node_groups=[]
  for entity in graph['species'].values():
   for key in entity['source_human_reactome_identity']:node_groups.append('reactome_entity:'+key)
   for qualifier,urls in entity['qualifiers'].items():
    for url in urls:
     if '/uniprot:' in url:
      accession=url.split('/uniprot:')[-1]
      if qualifier=='is':node_groups.append('source_uniprot_reference:'+accession)
      elif qualifier=='hasPart':node_groups.append('source_component_uniprot_reference:'+accession)
  groups=sorted(set(groups+node_groups));ref=dict(source_native_graph_file=rel(source),source_native_graph_sha256=audit['native_graphs_sha256'],source_native_graph_row=source_row,source_member=graph['source_member'],source_member_sha256=graph['source_member_sha256'],source_release='Reactome98',source_model_attributes=graph['model_attributes'],mask_seed=20261031,selected_reaction_indices=selected,mask_recipe='smallestsourceIDhashes;10percentreactions,min1;participantdisplayorderindependentlypermuted',native_graph_semantics='curatedstaticannotation;nosourcekineticlaws;hasPart/homology_not_exactidentity',chemical_holdout_mapping='Chebi/molecularstate-to-structure cross-source mapping unresolved;finalholdout-sensitive optimization not admitted',source_quality='fullreferences_positiveprovidedstoichiometry;unknownstoichiometrynotimputed')
  task=masked_pathway_task(graph,selected,audit=ref);raw=task.canonical_bytes();d=json.loads(raw);semantic=hashlib.sha256(json.dumps(dict(conditions=d['conditions'],targets=d['targets']),sort_keys=True).encode()).hexdigest();record_id=graph['source_member'];duplicate=semantic in seen;decisions.append(dict(record_id=record_id,source_row=source_row,admitted=not duplicate,cause='exact_condition_target_graph_duplicate' if duplicate else 'source_valid_static_annotation_task',species=len(graph['species']),reactions=N,participant_edges=sum(len(r['participants']) for r in graph['reactions']),masked_reactions=len(selected)))
  if duplicate:counts['exact_graph_task_duplicates']+=1;continue
  seen.add(semantic);changed=copy.deepcopy(graph)
  for i,reaction in enumerate(changed['reactions']):
   reaction['attributes']['name']='MUTATED_TEST_ONLY_EXCLUDED';reaction['notes']=['MUTATED_TEST_ONLY_EXCLUDED'];reaction['qualifiers']={'isDescribedBy':['MUTATED_TEST_ONLY_EXCLUDED']}
   if i in maskset:
    for p in reaction['participants']:
     p['role']='product' if p['role']!='product' else 'reactant';p['stoichiometry_source']='42';p['source_attributes']['sboTerm']='MUTATED_TEST_ONLY_ROLE_PROXY'
  assert masked_pathway_task(changed,selected,audit={'mutated_only':True}).input_bytes()==task.input_bytes();target_mutations+=1;assert all(d['conditions']['reactions'][i]['participants'][j]['role'] is None for i in selected for j in range(len(d['conditions']['reactions'][i]['participants'])));h=hashlib.sha256(raw).hexdigest();rows.append(dict(task_id=hashlib.sha256(('Reactome98:'+record_id+':static_role_mask').encode()+raw).hexdigest(),record_id=record_id,source_family='Reactome98-static-pathway-prior',source_revision='Reactome98',source_sha256=audit['source']['sha256'],task_name='source_static_pathway_masked_participant_roles',native_schema='mora3.CorpusTask/v1',native_payload=raw,native_payload_sha256=h,split='train',split_policy='source_valid_static_annotation_train_only;finalholdout-sensitive_optimization_gated',group_ids_json=json.dumps(groups),evidence_class='source_curated_static_pathway_annotation_not_experimental_response',admitted_for_task=True,model_adapter_supported=False,final_holdout_sensitive_optimization_admitted=False,admission_receipt_json=json.dumps(dict(source_semantics='SBML3V1source_participant_roles',known_participant_incidence=True,kinetic_laws=0,chemical_identity_mapping_complete=False)),missing_context_json=json.dumps(d['missing']+['cross_source_Chebi_and_state_chemical_identity_resolution']),full_axis_values=len(graph['species'])+N+sum(len(r['participants']) for r in graph['reactions']),condition_context_json='{}'));payload_bytes+=len(raw);counts['serialized_graph_tasks']+=1;sizes.append(dict(record_id=record_id,species=len(graph['species']),reactions=N,participant_edges=sum(len(r['participants']) for r in graph['reactions']),masked_reactions=len(selected),payload_bytes=len(raw)))
  if len(rows)>=64 or payload_bytes>=16*2**20:flush()
  if (source_row+1)%500==0:print(json.dumps(dict(source_models=source_row+1,serialized=counts['serialized_graph_tasks'],duplicates=counts['exact_graph_task_duplicates'],seconds=time.perf_counter()-start)),flush=True)
flush();assert source_row+1==2889
pq.write_table(audit_table(decisions),C/'admission-decisions.parquet',compression='zstd');pq.write_table(audit_table(sizes),C/'context-sizes.parquet',compression='zstd')
# Independent completed-shard readback;all original model axes retained by pointer andtask arrays.
verified=0
for s in shards:
 p=R/s['path'];assert sha(p)==s['sha256'] and p.stat().st_size==s['bytes']
 for r in pq.read_table(p).to_pylist():
  raw=r['native_payload'];assert hashlib.sha256(raw).hexdigest()==r['native_payload_sha256'];t=CorpusTask.from_bytes(raw);assert t.canonical_bytes()==raw;d=json.loads(raw);assert len(d['conditions']['species'])>0;assert len(d['conditions']['reactions'])==len(d['targets']['masked_reaction_participants']);assert not r['model_adapter_supported'] and not r['final_holdout_sensitive_optimization_admitted'];verified+=1
manifest=dict(schema='mora3.Reactome_static_graph_admission.v1',source_semantic_audit=rel(B/'semantic-audit.json'),source_semantic_audit_sha256=sha(B/'semantic-audit.json'),source=audit['source'],raw_available_pathway_models=2889,source_quality_pass_pathway_models=2889,source_species_occurrences=160266,source_reaction_occurrences=75233,unique_source_reaction_identities=16187,source_semantic_variants=0,source_kinetic_laws=0,task_admitted_static_annotation_graphs=counts['serialized_graph_tasks'],serialized_tasks=verified,exact_condition_target_graph_duplicates=counts['exact_graph_task_duplicates'],count_unit='whole static annotation graph reconstruction task;not independentreactions/experiments',shards=shards,total_shard_bytes=sum(s['bytes'] for s in shards),all_actual_hidden_role_stoichiometry_SBO_text_mutation_checks=target_mutations,full_species_reaction_participant_axes=True,max_species=max(s['species'] for s in sizes),max_reactions=max(s['reactions'] for s in sizes),max_participant_edges=max(s['participant_edges'] for s in sizes),max_context_entities_plus_edges=max(s['species']+s['reactions']+s['participant_edges'] for s in sizes),model_adapter_supported=False,final_holdout_sensitive_optimization_admitted=False,final_holdout_gate='cross-sourceChebi/state-to-chemicalidentity mapping unresolved;provisionalevaluationnotfinalunseen',source_identity_qualifier_policy='onlyownannotations;is/hasPart/isHomologTo/hasVersion retained withroles,neverflattened',biological_response_experimental_validation=False,physical_rates_or_fluxes=False,split='train_only_static_annotation_candidate',license='CC0-1.0',standards_primary_urls=['https://sbml.org/specifications/sbml-level-3/version-1/core/sbml-level-3-version-1-core.pdf','https://sbml.org/documents/elaborations/miriam_annotation_syntax/','https://sbml.org/jsbml/files/doc/api/1.5/org/sbml/jsbml/CVTerm.Qualifier.html'],optimizer_steps=0,seconds=time.perf_counter()-start)
with (C/'manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
print('CURATED_PATHWAYS_COMPLETE',json.dumps({k:manifest[k] for k in ['serialized_tasks','exact_condition_target_graph_duplicates','total_shard_bytes','max_context_entities_plus_edges','seconds']}),flush=True)
