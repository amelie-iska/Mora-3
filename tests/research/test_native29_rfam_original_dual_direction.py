"""Rfam dual-direction original-source qualification: synthetic fixtures are test-only."""
import copy
import pytest
from mora3.research.rfam_original_dual_direction_census import validate_dual_record_pair, audit_original_rfam_pairs

def pair():
    base={'schema':'mora3.CorpusTask/v1','audit':{'source_release':'Rfam15.1','source_evidence':'model_assigned','excluded_predictors':[],'negative_labels_available':False},'missing':['independent_experiment']}
    pred=base|{'task':'Rfam_source_family_prediction_prior','conditions':{'source_native_sequence':'ACGU'},'targets':{'source_Rfam_family_ids':['RF00001'],'source_clan_ids':['CL0001'],'family_label_policy':'source_reference'}}
    gen=base|{'task':'Rfam_source_family_generation_prior','conditions':{'desired_source_Rfam_family_ids':['RF00001'],'desired_source_clan_ids':['CL0001']},'targets':{'source_native_sequence':'ACGU'}}
    return pred,gen

def test_valid_dual_directions_are_one_parent_not_two_observations():
    p,g=pair()
    assert validate_dual_record_pair(p,g)['source_native_sequence_length']==4
    assert validate_dual_record_pair(p,g)['same_parent_observation'] is True

@pytest.mark.parametrize('damage',('sequence','family','clan','role','evidence'))
def test_reject_directional_role_or_source_mismatch(damage):
    p,g=pair();p=copy.deepcopy(p);g=copy.deepcopy(g)
    if damage=='sequence':g['targets']['source_native_sequence']='ACGA'
    elif damage=='family':g['conditions']['desired_source_Rfam_family_ids']=['RF99999']
    elif damage=='clan':g['conditions']['desired_source_clan_ids']=['CL9999']
    elif damage=='role':g['task']='Rfam_source_family_prediction_prior'
    else:g['audit']['source_evidence']='fabricated_experiment'
    with pytest.raises(ValueError):validate_dual_record_pair(p,g)

def test_original_complete_72_shard_two_direction_census():
    from pathlib import Path
    result=audit_original_rfam_pairs(Path.cwd())
    assert result['original_shards']==72
    assert result['source_rows']==36846
    assert result['parent_observations']==18423
    assert result['prediction_rows']==result['generation_rows']==18423
    assert result['all_parent_pairs_role_equal'] is True
    assert result['model_adapter_ready'] is False
    assert result['independent_experimental_labels'] is False
    assert result['training_admission'] is False
    assert result['optimizer_updates']==0
