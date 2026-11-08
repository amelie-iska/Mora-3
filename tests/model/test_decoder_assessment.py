import pytest

def test_counts_keep_empty_invalid_and_unterminated_distinct():
    from mora3.model.decoder_assessment import assess_molecules
    result = assess_molecules({'smiles': ('CCO', '', None, 'bad', 'OCC'),
                               'terminated': (True, True, False, True, True)},
                              ('CCO', 'C', 'N', 'O', 'CCO'))
    assert result['terminated_count'] == 4
    assert result['valid_nonempty_count'] == 2
    assert result['exact_isomeric_graph_recovery_count'] == 2
    assert result['unique_valid_isomeric_graphs'] == 1
    assert result['valid_unique_fraction'] == .5

def test_stereochemistry_and_denominators():
    from mora3.model.decoder_assessment import assess_molecules
    r = assess_molecules({'smiles': ('F[C@H](Cl)Br', 'CC'), 'terminated': (True, True)},
                         ('F[C@@H](Cl)Br', 'CC'))
    assert r['exact_isomeric_graph_recovery_count'] == 1
    assert r['recovery_fraction_all_conditions'] == .5

@pytest.mark.parametrize('generation,targets', [
    ({'smiles': (), 'terminated': ()}, ()),
    ({'smiles': ('C',), 'terminated': (False,)}, ('C',)),
    ({'smiles': (None,), 'terminated': (True,)}, ('C',)),
    ({'smiles': ('C',), 'terminated': (True,)}, ('invalid',)),
])
def test_malformed_assessment_rejected(generation, targets):
    from mora3.model.decoder_assessment import assess_molecules
    with pytest.raises(ValueError): assess_molecules(generation, targets)
