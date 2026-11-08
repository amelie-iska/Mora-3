"""Conservative audited PubChem union counts, separate from model exposure."""

def verify_union_counts(master, prior, census, proof):
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    require(proof.get('status') == 'verified', 'complete independently verified task proof required')
    require(proof['old_new_task_ID_intersections'] == 0, 'old/new task identities must be disjoint')
    n = census['admitted_unique_reference_graphs']
    k = census['graphs_with_eligible_standardSELFIES']
    p = census['new_prior_graphs_to_serialize']
    c = census['new_codec_graphs_to_serialize']
    require(n == p + census['pilot_prior_graphs_reused'], 'reference graph census must preserve pilot once')
    require(k == c + census['pilot_codec_graphs_reused'] and k <= n, 'codec graph census must preserve eligible pilot once')
    require(master['serialized_distinct_record_counts_by_source']['PubChem-native-SMILES-reference-prior'] == n, 'master reference graph count must equal full census')
    require(master['serialized_distinct_record_counts_by_source']['PubChem-standardSELFIES-codec'] == k, 'master codec graph count must equal full census')
    new = p + 2*c
    require(master['serialized_tasks'] == prior['serialized_tasks'] + new, 'union tasks must add only new task views')
    require(proof['all_new_task_views_source_condition_target_and_native_axes_replayed'] == new, 'all new views require semantic proof')
    require(all(master['split_counts'].get(s,0) == prior['split_counts'].get(s,0) for s in ('validation','test')), 'heldout allocations cannot be invented')
    require(master['split_counts']['train'] == prior['split_counts']['train'] + new, 'new task views must be source-train only')
    u = proof['serialized_PubChem_native_units']
    require(u.get('units_are_not_neural_tokens') is True and all(v is None for v in u['actual_model_token_totals'].values()), 'native units must not claim neural token counts')
    require(u['unique_reference_graphs']['graphs'] == n and u['unique_codec_graphs']['graphs'] == k, 'native-unit graph counts must equal source census')
    for direction,count in [('reference_prior',n),('smiles_to_selfies',k),('selfies_to_smiles',k)]:
        require(u['per_view_direction'][direction]['views'] == count, 'native-unit direction counts must match admitted views')
    return dict(reference_graphs=n, codec_graphs=k, reference_graphs_without_codec=n-k,
                full_PubChem_views=n+2*k, new_views=new,
                preserved_pilot_views=census['pilot_prior_graphs_reused']+2*census['pilot_codec_graphs_reused'],
                new_independent_experiments=0, actual_model_token_totals=None, actual_optimizer_exposure=None)
