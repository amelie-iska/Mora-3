"""Genuine original 12-assay Tox21 train-route replay; no counted training."""
from pathlib import Path
import json

import pytest

from mora3.research.common_source_tox21_complete import (
    reference_original_tox21, audit_original_tox21_complete_train,
)
from mora3.data.tox21 import ASSAYS, FORWARD, INVERSE

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')

@pytest.fixture(scope='module')
def complete():
    return audit_original_tox21_complete_train(ROOT)

def test_full_original_forward_inverse_train_census(complete):
    assert complete['source_train_shards']==51
    assert complete['original_train_task_rows']==12892
    assert complete['independent_mirrored_original_pairs']==6446
    assert complete['original_train_role_counts']=={FORWARD:6446,INVERSE:6446}
    assert complete['candidate_common_raw_native_equal']
    assert complete['independent_raw_role_mask_target_equal']
    assert complete['both_snapshot_iterators_exhausted']
    assert complete['total_qualified_train_source_records']==12892
    assert complete['source_pair_mismatch_count']==0
    assert complete['complete_original_train_source_readback']
    assert not complete['training_admission']
    assert not complete['heldout_target_payload_decoded']
    assert complete['optimizer_updates']==complete['actual_campaign_trials']==0
    assert len(complete['pairs'])==6446

def test_original_binary_mask_and_inverse_token_scope(complete):
    axis=complete['assay_axis']
    assert tuple(axis)==ASSAYS
    counts=complete['call_state_counts']
    assert sum(counts.values())==6446*len(ASSAYS)
    assert counts['observed_positive']>0
    assert counts['observed_negative']>0
    assert counts['missing_unknown']>0
    assert complete['forward_observed_target_positions']==counts['observed_positive']+counts['observed_negative']
    assert complete['inverse_source_target_tokens']>=6446
    assert complete['complete_global_group_split_admitted'] is False

def test_literal_independent_source_parser_rejects_axis_swap_false_zero_and_duplicates():
    raw=dict(schema='mora3.CorpusTask/v1',task=FORWARD,
             conditions={'canonical_isomeric_smiles':'CO','known_assay_axis':list(ASSAYS)},
             targets={'source_call_values':[1,None]*6,'source_call_mask':[True,False]*6},
             audit={'source_protocol_role':'source_training','source_missing_not_negative':True,
                    'physical_dose':None,'exposure_time':None,'source_ref':{'native_jsonl_sha256':'a'*64}},
             missing=['perassayphysicaldose_exposuretime_cellline'])
    assert reference_original_tox21(json.dumps(raw).encode(),FORWARD)['observed_positive']==6
    raw['targets']['source_call_values'][1]=0
    with pytest.raises(ValueError):reference_original_tox21(json.dumps(raw).encode(),FORWARD)
    raw['targets']['source_call_values'][1]=None
    raw['conditions']['known_assay_axis'].reverse()
    with pytest.raises(ValueError):reference_original_tox21(json.dumps(raw).encode(),FORWARD)
    raw['conditions']['known_assay_axis'].reverse()
    with pytest.raises(ValueError):
        reference_original_tox21(json.dumps(raw).replace('"schema":','"schema":"tamper","schema":').encode(),FORWARD)
