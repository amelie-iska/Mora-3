import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path('/home/iska/Documents/amelie/bio/Mora-3/scripts')))
from data_intake_reference_graph_context_scope import link_scope


def test_literal_reference_match_is_not_same_cell_or_material():
    p=link_scope(exact=True,component='whole_source_graph',split='train',field_role='conditions.molecular_smiles')
    assert p['admitted_reference_graph_link']
    assert p['molecule_is_explicit_condition']
    assert p['same_assayed_material_or_cell'] is False
    assert p['cross_view_attention_allowed'] is False
    assert p['reference_hash_is_model_input'] is False


@pytest.mark.parametrize('exact,component,split',[(False,'whole_source_graph','train'),(True,'component:0','train'),(True,'whole_source_graph','test')])
def test_nonliteral_component_or_heldout_never_becomes_training_context_link(exact,component,split):
    assert not link_scope(exact=exact,component=component,split=split,field_role='conditions.smiles')['admitted_reference_graph_link']


def test_structure_target_can_only_index_an_audit_link_never_predictor_condition():
    p=link_scope(exact=True,component='whole_source_graph',split='train',field_role='targets.molecular_smiles')
    assert p['admitted_reference_graph_link']
    assert not p['molecule_is_explicit_condition']
    assert p['cross_view_attention_allowed'] is False
