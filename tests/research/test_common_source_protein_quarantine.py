"""Actual protected protein-prior source: refuse to erase one homology quarantine."""
from dataclasses import replace
from pathlib import Path
import copy
import pytest
from mora3.model.source_assembly_inputs import load_actual_source_assets
from mora3.research.primary_native_routes import finite_view_policy_sha256
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
from mora3.research.primary_source_factory_preflight import document
from mora3.research.common_source_protein_replay import (
    reconstruct_protein_quarantine_views, audit_common_source_protein_two_windows,
)

ROOT=Path('/home/iska/Documents/amelie/bio/Mora-3')
BASE='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
CANDIDATE='4f144c456c46a393e678d640cc96c99e4d4b7af793174b50b49971088fb8f227'
COMMON='2888d9f11038f70f9b25a049c8a006d87ca9918de6e94adcb71ec5ee1c9fd34b'
QUARANTINE=('outputs/data-intake/20261031-final-corpus-build-v1/multisource-v3/shards/multisource/UniProt/train/00000.parquet',24)


def setup():
    context=load_inactive_common_source_rebind(ROOT)
    artifact=document(ROOT,context.evidence)
    row,=[r for r in artifact['route_bindings']if r['task_key']=='native-prior:protein']
    assets=load_actual_source_assets(ROOT,source_view_policy='effective-study-v2')
    return assets.native_view,context,row


def test_genuine_quarantined_protein_row_is_bound_not_substituted():
    base,context,row=setup()
    original_effective_filter=base.effective_filter
    original_exclusions=base._ordinal_exclusions
    result=reconstruct_protein_quarantine_views(ROOT,base_view=base,context=context,row=row)
    assert result.original_source_sha256=='9eb37df985160d2d7db5d49e458a96a737e35ae2ea092ac57d9c0e46a8c27326'
    assert result.original_payload_sha256=='53437fa2c97550f993fdefd6e29f47f96afbeaf3855d77775931d2d5e2fb687c'
    assert result.original_role=='train' and result.original_native_task=='protein_sequence_modeling'
    assert result.source_original_row_ordinal==24
    assert result.quarantined_coordinates==frozenset((QUARANTINE,))
    assert finite_view_policy_sha256(base)==BASE
    assert base.effective_filter is original_effective_filter and base._ordinal_exclusions is original_exclusions
    assert QUARANTINE not in base._ordinal_exclusions
    assert result.candidate._ordinal_exclusions==result.common._ordinal_exclusions==base._ordinal_exclusions|result.quarantined_coordinates
    assert len(result.common._ordinal_exclusions)==len(base._ordinal_exclusions)+1==228
    assert finite_view_policy_sha256(result.candidate)==CANDIDATE
    assert finite_view_policy_sha256(result.common)==COMMON
    assert result.candidate.manifest is result.common.manifest is base.manifest
    assert result.candidate.descriptors is result.common.descriptors is base.descriptors
    assert result.original_policy_sha256=='b463ee189a3a39e916bf9ea46c3684496e498f81864ea9e914bfb8f3ff3ecd00'
    assert result.training_admission is False


@pytest.mark.parametrize('damage',('policy-digest','wrong-common','wrong-candidate','missing-route','task-switch'))
def test_private_policy_or_source_lineage_rebinding_refuses(damage):
    base,context,row=setup();changed=copy.deepcopy(row)
    if damage=='policy-digest':changed['common_view_policy_sha256']='c3812ecfcfea8a01f81769bf5ef6fb3880e04a0ac054a6349e9272db14765393'
    elif damage=='wrong-common':changed['common']['adapter_sha256']='0'*64
    elif damage=='wrong-candidate':changed['candidate']['initial_cursor_sha256']='0'*64
    elif damage=='missing-route':changed['task_key']='native-prior:dna'
    else:changed['common']['task_key']='native-prior:dna'
    with pytest.raises(ValueError):reconstruct_protein_quarantine_views(ROOT,base_view=base,context=context,row=changed)


def test_actual_original_two_protein_windows_preserve_quarantine_and_restart():
    result=audit_common_source_protein_two_windows(ROOT)
    assert result['task_key']=='native-prior:protein'
    assert result['actual_native_rows']==result['windows_replayed']==2
    assert result['native_source_view_policy_sha256']==COMMON
    assert result['protein_original_homology_quarantine_applied'] is True
    assert result['original_quarantined_row_ordinal']==24
    assert result['original_rows_and_payloads_identical'] is True
    assert result['resumed_second_window_exact_match'] is True
    assert result['both_rebound_route_cursors_conserved'] is True
    assert result['independent_target_count_conserved'] is True
    assert result['original_source_corpus_exhausted'] is None
    assert result['training_admission'] is False
    assert result['optimizer_updates']==result['actual_campaign_runs']==0


def test_no_protein_from_unquarantined_base_or_wrong_scan_limits():
    with pytest.raises(ValueError):audit_common_source_protein_two_windows(ROOT,max_windows=1)
    with pytest.raises(ValueError):audit_common_source_protein_two_windows(ROOT,task_key='native-prior:dna')


def test_exact_original_homology_row_is_actually_skipped_by_both_native_iterators():
    from mora3.research.native_prior_campaign import NativePriorCampaignAdapter
    from mora3.research.primary_native_routes import finite_view_policy_sha256
    from mora3.model.source_assembly_inputs import load_actual_source_assets
    base,context,row=setup()
    assets=load_actual_source_assets(ROOT,source_view_policy='effective-study-v2')
    proof=reconstruct_protein_quarantine_views(ROOT,base_view=base,context=context,row=row)
    shard,ordinal=next(iter(proof.quarantined_coordinates))
    selections={}
    for label,view in (('unquarantined_reference',base),('candidate',proof.candidate),('common',proof.common)):
        binding=NativePriorCampaignAdapter(view,assets.vocabulary,context.common_snapshot.sha256,
                                           'protein',4096,128).train_binding(batch_size=1)
        route=binding.route
        index=route._paths.index(shard)
        cursor=route._cursor(index,ordinal)
        iterator=route.iter_from(cursor)
        try:
            window=next(iterator)
        finally:
            iterator.close()
        view.require_native_rows(window.batch,split='train')
        selections[label]=window.batch.source_row_ordinals
        assert window.batch.source_shard['path']==shard
        assert len(window.batch.source_row_ordinals)==1
    assert selections=={'unquarantined_reference':(24,), 'candidate':(25,), 'common':(25,)}
    assert (shard,24) in proof.candidate._ordinal_exclusions
    assert (shard,24) in proof.common._ordinal_exclusions
    assert (shard,24) not in base._ordinal_exclusions
    assert finite_view_policy_sha256(proof.common)==COMMON
