"""Reference-graph indexing is audit metadata, never an implicit assay/input join."""
def link_scope(*,exact,component,split,field_role):
    admitted=exact is True and component=='whole_source_graph' and split=='train'
    return dict(admitted_reference_graph_link=admitted,
                molecule_is_explicit_condition=bool(admitted and field_role.startswith('conditions.')),
                same_assayed_material_or_cell=False,
                cross_view_attention_allowed=False,
                reference_hash_is_model_input=False,
                dose_time_or_replicate_pair=False,
                source_target_may_condition_another_view=False,
                new_task_views=0,new_independent_observations=0,
                scope='literal reference graph only; each original view retains its own explicit condition/target masks')
