import sys
from pathlib import Path
sys.path.insert(0,str(Path('/home/iska/Documents/amelie/bio/Mora-3/scripts')))
from data_intake_lincs_reference_mapping import exact_perturbation_map


def test_repeated_graph_keeps_all_original_evidence():
    rows=[dict(pert_id='BRD-K1-001',graph_sha='a',source_row=1),dict(pert_id='BRD-K1-001',graph_sha='a',source_row=2)]
    result=exact_perturbation_map(rows)
    assert result['BRD-K1-001']['eligible_reference_graph_sha']=='a'
    assert result['BRD-K1-001']['source_rows']==[1,2]


def test_conflicting_reference_graphs_never_resolve_by_first_or_last():
    result=exact_perturbation_map([dict(pert_id='X',graph_sha='a',source_row=1),dict(pert_id='X',graph_sha='b',source_row=2)])
    assert result['X']['eligible_reference_graph_sha'] is None
    assert result['X']['reason']=='multiple_literal_reference_graphs'


def test_full_source_ids_never_merge_by_prefix_case_or_whitespace():
    result=exact_perturbation_map([dict(pert_id='BRD-K1-001',graph_sha='a',source_row=1),dict(pert_id='brd-k1-001',graph_sha='b',source_row=2)])
    assert len(result)==2
    assert 'BRD-K1' not in result and ' BRD-K1-001' not in result


def test_missing_source_graph_is_not_a_negative_or_inferred_structure():
    result=exact_perturbation_map([dict(pert_id='X',graph_sha=None,source_row=1)])
    assert result['X']['eligible_reference_graph_sha'] is None
    assert result['X']['reason']=='no_eligible_source_graph'


def test_one_unresolved_source_graph_blocks_crossassay_structure_propagation():
    result=exact_perturbation_map([dict(pert_id='X',graph_sha='a',source_row=1),dict(pert_id='X',graph_sha=None,source_row=2)])
    assert result['X']['eligible_reference_graph_sha'] is None
    assert result['X']['reason']=='unresolved_source_graphs'
