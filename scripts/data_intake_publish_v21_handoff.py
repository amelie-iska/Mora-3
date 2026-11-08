"""Publish only a verified whole-source PubChem union, keeping model use separate."""
from pathlib import Path
import collections, datetime, hashlib, json, shutil, sys
R=Path('/home/iska/Documents/amelie/bio/Mora-3')
sys.path.insert(0,str(R/'scripts'))
from data_intake_full_pubchem_guards import immutable_json
from data_intake_full_pubchem_guards_v2 import BuildGuard
from data_intake_PubChem_publication_counts import verify_union_counts
B=R/'outputs/data-intake/20261101-PubChem-native-corpus-v1'
F=B/'materialized/independent-final-v1'
O=B/'model-handoff-v1'
M=F/'master-corpus-manifest-v21-PubChem-whole-native-train-v1.json'
OLD=R/'outputs/data-intake/20261101-PubChem-corpus-pilot-v1/model-handoff-v2'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def ref(p):return dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p))
def load(p):return json.loads(p.read_text())
def check(r):
    p=Path(r['path']);p=p if p.is_absolute() else R/p
    if ref(p)!=r:raise ValueError('immutable receipt/reference mismatch: '+str(p))
    return p

def main():
    plan=load(B/'build-plan.json')
    guard=BuildGuard(B,census_cap=2*2**30,total_cap=12*2**30,deadline_epoch=plan['deadline_epoch'])
    guard.check('materialized',reserve=16*2**20)
    for receipt in (B/'phase-launches').glob('*.json'):
        job=load(receipt)
        for pid in (job.get('pid'),job.get('python_worker_pid')):
            p=Path('/proc')/str(pid)/'cmdline'
            if p.exists() and b'data_intake_full_pubchem_' in p.read_bytes():
                raise RuntimeError('whole-build phase still live; wait for normal completion')
    m=load(M);proof=load(F/'receipt.json');pub=load(F/'publication-receipt.json')
    if pub['master']!=ref(M) or pub['proof']!=ref(F/'receipt.json'):
        raise ValueError('final publication not bound to verified master')
    prior=load(check(plan['master']));c=load(B/'census/manifest.json');t=load(B/'materialized/manifest.json')
    counts=verify_union_counts(m,prior,c,proof)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    baseline_readiness=load(OLD/'reader-readiness.json')
    if baseline_readiness['master']!=plan['master']:raise ValueError('baseline reader counts must bind exact prior master')
    by_source=collections.Counter({f['source_family']:f['serialized_tasks'] for f in baseline_readiness['families']})
    splits={f['source_family']:collections.Counter(f['split_counts']) for f in baseline_readiness['families']}
    if sum(by_source.values())!=prior['serialized_tasks']:raise ValueError('baseline reader counts must reconcile prior master')
    for sh in t['shards']:
        by_source[sh['source_family']]+=sh['rows'];splits[sh['source_family']][sh['split']]+=sh['rows']
    if sum(by_source.values())!=m['serialized_tasks']:raise ValueError('family view counts do not reconcile')
    O.mkdir()
    evidence=dict(schema='mora3.PubChemWholeCorpusHandoffEvidence/v1',master=ref(M),
        whole_source_census=ref(B/'census/manifest.json'),independent_task_proof=ref(F/'receipt.json'),
        source_raw_occurrences=9999960,source_train_occurrences=8999964,source_validation_audit_only_occurrences=999996,
        source_quality_by_native_split=c['source_counts'],admitted_counts=counts,
        all_PubChem_native_units=proof['serialized_PubChem_native_units'],
        new_component_full_native_axes=proof['new_component_full_native_axes'],
        preserved_pilot_axes_evidence=proof['preserved_pilot_axes_evidence'],
        native_units_not_model_tokens=True,actual_model_input_scored_target_padding_packed_tokens=None,
        actual_optimizer_exposure=None,full_final_training_dataset_complete=False,
        no_new_heldout_or_independent_experiment=True,full_source_native_feature_refs_retained=True,
        original_GroupSELFIES_equivalence=False,model_adapter_support_scope='historical data-only flags; model owner must independently register and verify current native bridge')
    immutable_json(O/'axis-units-and-evidence-receipt.json',evidence)
    rd=load(OLD/'reader-readiness.json')
    rd.update(created_utc=now,master=ref(M),supersedes_readiness=ref(OLD/'reader-readiness.json'),master_full_union_verification=ref(F/'receipt.json'))
    rd['counts'].update(tasks=m['serialized_tasks'],shards=len(m['shards']),task_data_bytes=m['total_shard_bytes'],split_counts=m['split_counts'],schema_counts=m['native_schema_counts'],historical_generic_adapter_support_counts=m['adapter_support_counts'])
    for family in rd['families']:
        name=family['source_family']
        if name in by_source:
            family['serialized_tasks']=by_source[name]
            family['split_counts']=dict(splits[name])
        if name.startswith('PubChem-'):
            family.update(distinct_record_count_and_unit_from_master=m['serialized_distinct_record_counts_by_source'][name],
                evidence=ref(F/'receipt.json'),axes_and_native_units=ref(O/'axis-units-and-evidence-receipt.json'),
                current_native_data_readiness='complete source-train data-only envelope and full semantic proof; native model bridge/objective/exposure require model-owner evidence',
                count_unit='reference graph identities; codec two directions are computed representations of the same graphs',
                model_adapter_supported=False)
    if sum(f['serialized_tasks'] for f in rd['families'])!=m['serialized_tasks']:
        raise ValueError('reader table family counts do not match master')
    rd['PubChem_whole_source_train_scope']=evidence
    rd['full_training_usage_or_throughput_measured_by_data_worker']=False
    rd['actual_whole_model_token_totals']=None
    rd['weights']=None
    h=load(OLD/'executable-model-handoff.json')
    h.update(created_utc=now,master=ref(M),supersedes=ref(OLD/'executable-model-handoff.json'),
             PubChem_original_native_independent_verification=ref(F/'receipt.json'),
             PubChem_admitted_axes_and_evidence=ref(O/'axis-units-and-evidence-receipt.json'),
             current_master_read_recipe='Use the current master above with read_corpus_tasks(..., supported_only=False) and hash-bound effective groups. Historical probes remain pinned to their original masters and do not prove new pool exposure.',
             current_full_dataset_complete=False,actual_whole_model_tokens=None,actual_optimizer_exposure=None)
    for task in h['tasks']:
        if task.get('id')=='CORPUS-PUBCHEM-NATIVE-CODEC-01':
            task.update(expected_master_sha256=sha(M),
                inspect_command='/home/iska/miniconda3/envs/tokengt/bin/python scripts/data_intake_corpus.py --manifest '+str(M.relative_to(R))+' --source PubChem-standardSELFIES-codec --include-unsupported --limit 2',
                admitted_counts=counts,axes_and_native_units=ref(O/'axis-units-and-evidence-receipt.json'),
                source_range='all 8,999,964 native train occurrences audited; graph admission exclusions and preserved pilot applied; 999,996 native validation occurrences audit-only',
                model_adapter_supported=False)
    registry=B/'materialized/reference-graph-context-registry-v1/manifest.json'
    if registry.exists():
        registry_data=load(registry)
        if registry_data['master']!=ref(M):raise ValueError('reference context registry must bind current master')
        h['existing_view_reference_graph_context_scheduler_registry']=ref(registry)
        for task in h['tasks']:
            if task.get('id')=='CORPUS-REFERENCE-GRAPH-CONTEXT-SCHEDULER-01':
                task.update(registry=ref(registry),reference_graph_nodes=registry_data['exact_reference_graphs'],source_context_records=registry_data['distinct_original_source_context_records'],cross_view_attention_allowed=False,reference_hash_is_model_input=False,hidden_structure_target_not_visible_to_another_view=True,no_new_tasks_or_experiments=True)
        rd['current_reference_context_registry']=ref(registry)
    h['COUNT-TOKENS-01']=dict(native_units=ref(O/'axis-units-and-evidence-receipt.json'),input_tokens=None,scored_target_tokens=None,padding_tokens=None,packed_tokens=None,required='model-owned pinned native reader/tokenizer/context/packing and measured per-family input/target counters; actual sampled optimizer exposure separate')
    policy=ref(R/'planning/execution-policy-20261101T100536Z-user-storage-training-v1.json')
    h.update(latest_USER_execution_policy=policy,full_dataset_readiness_signature=None,substantive_training_authorized=False,pre_ready_test_limits=dict(preferred_max_optimizer_updates=32,preferred_max_runtime_seconds=120,tiny_model_explicit_slice=True))
    rd.update(latest_USER_execution_policy=policy,full_dataset_readiness_signature=None,substantive_training_authorized=False)
    immutable_json(O/'reader-readiness.json',rd)
    immutable_json(O/'executable-model-handoff.json',h)
    text=(f'Verified data-only master `{M.relative_to(R)}`; SHA256 `{sha(M)}`.\n\n'
          f'{m["serialized_tasks"]:,} task views / {len(m["shards"]):,} shards / {m["total_shard_bytes"]:,} task bytes. '
          f'Train {m["split_counts"]["train"]:,}, validation {m["split_counts"]["validation"]:,}, test {m["split_counts"]["test"]:,}.\n\n'
          f'PubChem contributes {counts["reference_graphs"]:,} unique admitted reference graphs and {counts["codec_graphs"]:,} eligible codec graphs, '
          f'{counts["full_PubChem_views"]:,} task views in total; the prior {counts["preserved_pilot_views"]:,} pilot views are retained once. '
          f'{counts["reference_graphs_without_codec"]:,} valid SMILES graphs retain their reference prior without a SELFIES target. '
          f'The new component adds {counts["new_views"]:,} train-only task views, zero new independent experiments and zero new heldouts.\n\n'
          'All source rows, graph admission masks, selected source references and new task condition/target payloads were independently replayed. '
          'Native source validation is audit-only; existing chemical/scaffold exclusions, pilot bytes and prior allocations are preserved. '
          'Exact-identity checks do not establish global homology or new-study blindness. Whole-dataset completion and final mixture weights remain open.\n\n'
          'CORPUS-PUBCHEM-NATIVE-CODEC-01 is model-owned: use domain-only conditions for the native SMILES prior, explicit native SMILES for forward standard-SELFIES codec, '
          'and explicit standard SELFIES for reverse canonical-SMILES codec. IDs, hashes, QC, target lengths, hidden representations and audit fields are excluded from predictors. '
          'Keep full charge/isotope/stereo/component semantics and native axes. Source encodings are computed reference representations, not assay, response, 3D or cellular observations. '
          'Historical generic false flags do not determine whether a later native model reader exists; register and test current native support at a model safe boundary.\n\n'
          'The native-unit receipt counts characters, SELFIES symbols and graph atom/bond slots separately, including pilot data once. These are not Mora-3, TokenGT, ConvexTok or decoder token totals. '
          'COUNT-TOKENS-01 requires the model owner to bind reader/tokenizer/context/packing revisions and record input versus scored-target/padding tokens and actual sampling exposure. '
          'No model code, checkpoint or weights were changed by this batch.\n')
    (O/'HANDOFF.md').open('x').write(text)
    immutable_json(O/'family-counts.json',dict(master=ref(M),count_unit='task_views; not disjoint biological observations',family_task_views=dict(by_source),family_split_counts={k:dict(v) for k,v in splits.items()},family_source_record_counts=m['serialized_distinct_record_counts_by_source'],raw_available_vs_admitted_source_ledger=m['source_ledger'],model_token_totals=None))
    guard.check('materialized')
    p=R/'planning/data_intake_progress.json';x=load(p)
    x.update(updated_utc=now,status='verified_v21_full_PubChem_native_train_union',running_jobs=[],
        last_completed_job=dict(kind='finite_full_PubChem_final_proof',status='completed_verified',receipt=ref(F/'receipt.json')),
        current_owned_task='next_verified_LINCS_original_source_admission_grouping_context_review; no duplicate acquisition',
        active_agent_task='verified_v21_next_paired_profile_source_review',
        final_corpus_manifest=str(M.relative_to(R)),current_corpus_manifest=str(M.relative_to(R)),current_master_corpus_manifest=str(M.relative_to(R)),
        current_corpus_manifest_sha256=sha(M),current_corpus_counts=dict(serialized_tasks=m['serialized_tasks'],shards=len(m['shards']),data_bytes=m['total_shard_bytes'],split_counts=m['split_counts']),
        serialized_task_counts=dict(by_source),corpus_split_counts=m['split_counts'],latest_union_verification=ref(F/'receipt.json'),v21_corpus=ref(M),
        current_handoff=str((O/'executable-model-handoff.json').relative_to(R)),executable_model_handoff=str((O/'executable-model-handoff.json').relative_to(R)),model_reader_readiness=str((O/'reader-readiness.json').relative_to(R)),
        final_full_training_dataset_complete=False,
        PubChem_whole_native_corpus=dict(manifest=ref(M),component=ref(B/'materialized/manifest.json'),source_original_rows=9999960,native_train_rows=8999964,native_validation_audit_rows=999996,counts=counts,source_quality_by_split=c['source_counts'],native_units=ref(O/'axis-units-and-evidence-receipt.json'),actual_model_tokens=None,model_adapter_supported=False),
        next_job='Consume only verified immutable LINCS-Pilot1 handoff; source rights, exact chemical/component/scaffold allocation, profile QC/context and reader gates before a new corpus version',
        next_actions=['Model owner: current native PubChem reader/codec full-axis target-exclusion and bounded safe joint test; mixture and actual use recorded separately','Review source-worker LINCS full-source handoff; no same-cell or matched-replicate claim from dose identity','Keep all 114 inventory entries mapped and preserve remaining protein-homology, SciPlex gene/dose and broad-evaluation gates'])
    p.write_text(json.dumps(x,indent=2)+'\n')
    p=R/'configs/dataset_acquisition_queue.json';q=load(p)
    if len(q['entries'])!=114:raise ValueError('inventory coverage unexpectedly changed')
    q.update(updated_utc=now,current_corpus_manifest=str(M.relative_to(R)),current_versioned_model_handoff=str((O/'executable-model-handoff.json').relative_to(R)))
    q['native_consumer'].update(manifest=str(M.relative_to(R)),readiness_table=str((O/'reader-readiness.json').relative_to(R)),executable_handoff=str((O/'executable-model-handoff.json').relative_to(R)))
    for e in q['entries']:
        if e['inventory_id']=='INV-002':
            e.update(resolution_status='licensed standard-SELFIES replacement full native-train corpus independently verified in v21; original GroupSELFIES rights/grammar unresolved',
                     next_action='Model-owned CORPUS-PUBCHEM-NATIVE-CODEC-01 native integration, measured joint training exposure and provisional mixture; no extra chemical/source context inferred',
                     corpus_build=dict(master=ref(M),component=ref(B/'materialized/manifest.json'),census=ref(B/'census/manifest.json'),source_original_rows_audited=9999960,unique_isomeric_graphs_admitted=counts['reference_graphs'],codec_graphs=counts['codec_graphs'],serialized_task_views=counts['full_PubChem_views'],valid_SMILES_unsupported_SELFIES_admitted=counts['reference_graphs_without_codec'],source_native_splits='native validation audit-only; no new heldouts',model_adapter_supported=False,source_data_license='Apache-2.0',source_revision='f83219601635a0a80fc99c13a9ca37f99ef34f0a'),
                     active_full_build=dict(status='completed_verified',live=False,proof=ref(F/'receipt.json'),plan=str((B/'build-plan.json').relative_to(R)),new_views=counts['new_views'],full_new_tasks_admitted=True,model_adapter_supported=False))
            e['evidence_paths'].append(str((F/'receipt.json').relative_to(R)))
    p.write_text(json.dumps(q,indent=2)+'\n')
    usage=guard.check('materialized')
    immutable_json(O/'publication-checkpoint-receipt.json',dict(master=ref(M),handoff=ref(O/'executable-model-handoff.json'),readiness=ref(O/'reader-readiness.json'),all_new_build_phase_usage=usage,cap_bytes=12*2**30,actual_live_data_jobs=[],actual_optimizer_exposure=None,actual_model_tokens=None))
    guard.check('materialized')
    if shutil.disk_usage(R).free < 200*2**30:raise RuntimeError('free-space reserve violated')
    print(json.dumps(dict(master=ref(M),handoff=ref(O/'executable-model-handoff.json'),counts=counts,usage=usage),indent=2))

if __name__=='__main__':main()
