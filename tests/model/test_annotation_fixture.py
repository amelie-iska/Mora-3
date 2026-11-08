"""Fixture acceleration must retain the complete native registry contract."""
import pytest
from dataclasses import FrozenInstanceError
from test_annotation_axes import REAL, reader


def test_prepared_registry_is_exact_complete_and_detached():
    from annotation_fixture import fixture_registry
    original = reader(REAL)
    prepared = fixture_registry()
    assert prepared.specification_json == original.specification_json
    assert prepared.categories == original.categories
    for category in original.categories:
        assert prepared.axis(category) == original.axis(category)
        assert prepared.rows(category) == original.rows(category)
    assert prepared.binding() == original.binding()
    changed = prepared.binding(); changed['version'] = -1
    assert prepared.binding() == original.binding()
    changed = prepared.rows('GO'); changed[0]['axis_id'] = 'changed'
    assert prepared.rows('GO') == original.rows('GO')
    with pytest.raises(FrozenInstanceError): prepared.specification_json = '{}'
    assert fixture_registry() is prepared
