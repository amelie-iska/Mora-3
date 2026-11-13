"""Test-only native owner/route boundary, not permission to train source data."""
from pathlib import Path
import pytest

from test_full_owner import actual_core_assets, cases, _construct
from mora3.research.rfam_campaign_routes import RfamCampaignAdapter
from mora3.research.rfam_goa_registry import GOABudget, load_original_rfam_goa_assets


@pytest.fixture(scope="module")
def original_rfam_view():
    # Read only the existing authenticated source view; do not edit its admission.
    return load_original_rfam_goa_assets(Path.cwd(), goa_budget=GOABudget(2, 1))[0]


def first_prepared(view, direction, snapshot):
    adapter = RfamCampaignAdapter(view, direction, snapshot, 1)
    binding = adapter.train_binding()
    iterator = binding.route.iter_from(binding.route.initial_cursor)
    try:
        window = next(iterator)
    finally:
        iterator.close()
    return adapter.prepare(window)


@pytest.mark.parametrize("direction", ("prediction", "generation"))
def test_foreign_source_owner_refuses_before_admission_call(
        original_rfam_view, actual_core_assets, tmp_path, monkeypatch, direction):
    owner = _construct(tmp_path, actual_core_assets)  # synthetic width-eight owner
    prepared = first_prepared(original_rfam_view, direction, "b" * 64)
    assert owner.get_extra_state()["source_snapshot_sha256"] == "a" * 64
    calls = []
    def admission_must_not_be_reached(*args, **kwargs):
        calls.append("admission")
        raise RuntimeError("source mismatch reached training-admission boundary")
    monkeypatch.setattr(owner, "require_training_admission", admission_must_not_be_reached)
    with pytest.raises(ValueError, match="source snapshot"):
        prepared.evaluate(owner)
    assert calls == []


@pytest.mark.parametrize("direction", ("prediction", "generation"))
def test_matching_snapshot_still_requires_real_owner_admission(
        original_rfam_view, actual_core_assets, tmp_path, direction):
    owner = _construct(tmp_path, actual_core_assets)
    prepared = first_prepared(original_rfam_view, direction, "a" * 64)
    with pytest.raises(ValueError, match="not full corpus"):
        prepared.evaluate(owner)


@pytest.mark.parametrize("direction", ("prediction", "generation"))
def test_prepared_target_rejects_non_owner(original_rfam_view, direction):
    prepared = first_prepared(original_rfam_view, direction, "a" * 64)
    with pytest.raises(ValueError, match="registered source-bound owner"):
        prepared.evaluate(object())
