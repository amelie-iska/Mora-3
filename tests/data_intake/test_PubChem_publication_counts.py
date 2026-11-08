import copy, sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path('/home/iska/Documents/amelie/bio/Mora-3/scripts')))
from data_intake_PubChem_publication_counts import verify_union_counts


def fixture():
    census = dict(admitted_unique_reference_graphs=3, graphs_with_eligible_standardSELFIES=2,
                  new_prior_graphs_to_serialize=2, new_codec_graphs_to_serialize=1,
                  pilot_prior_graphs_reused=1, pilot_codec_graphs_reused=1)
    old = dict(serialized_tasks=103, split_counts=dict(train=83, validation=10, test=10))
    master = dict(serialized_tasks=107, split_counts=dict(train=87, validation=10, test=10),
                  serialized_distinct_record_counts_by_source={'PubChem-native-SMILES-reference-prior':3,
                                                               'PubChem-standardSELFIES-codec':2})
    proof = dict(status='verified', all_new_task_views_source_condition_target_and_native_axes_replayed=4,
                 old_new_task_ID_intersections=0,
                 serialized_PubChem_native_units=dict(units_are_not_neural_tokens=True,
                      unique_reference_graphs=dict(graphs=3), unique_codec_graphs=dict(graphs=2),
                      per_view_direction={'reference_prior':dict(views=3), 'smiles_to_selfies':dict(views=2), 'selfies_to_smiles':dict(views=2)},
                      actual_model_token_totals=dict(input=None,scored_target=None,padding=None,packed=None)))
    return master, old, census, proof


def test_counts_preserve_missing_codec_prior_and_pilot_once():
    result=verify_union_counts(*fixture())
    assert result['full_PubChem_views']==7 and result['new_views']==4
    assert result['reference_graphs_without_codec']==1
    assert result['new_independent_experiments']==0
    assert result['actual_model_token_totals'] is None


def test_counts_reject_duplicate_pilot_admission():
    m,o,c,p=fixture(); m['serialized_distinct_record_counts_by_source']['PubChem-native-SMILES-reference-prior']+=1
    with pytest.raises(ValueError,match='reference graph'):
        verify_union_counts(m,o,c,p)


def test_counts_reject_new_heldout_allocation():
    m,o,c,p=fixture();m['split_counts']['test']+=1;m['split_counts']['train']-=1
    with pytest.raises(ValueError,match='heldout'):
        verify_union_counts(m,o,c,p)


def test_counts_reject_native_units_mislabeled_as_neural_tokens():
    m,o,c,p=fixture();p['serialized_PubChem_native_units']['actual_model_token_totals']['input']=100
    with pytest.raises(ValueError,match='token'):
        verify_union_counts(m,o,c,p)


def test_counts_reject_incomplete_proof():
    m,o,c,p=fixture();p['status']='serialized_pending_proof'
    with pytest.raises(ValueError,match='verified'):
        verify_union_counts(m,o,c,p)
