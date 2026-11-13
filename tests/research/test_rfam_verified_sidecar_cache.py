"""Actual original Rfam group-sidecar caching and fail-closed identity tests."""
from pathlib import Path

import pytest

from mora3.data.finite_union_inverse_literal_filter import load_inverse_literal_filtered_native_view
from mora3.research.rfam_goa_registry import FILTER
from mora3.research.rfam_campaign_routes import RfamCampaignAdapter
from mora3.research.primary_campaign_source_rebind import load_inactive_common_source_rebind
import mora3.research.rfam_campaign_routes as native


@pytest.mark.parametrize("direction", ("prediction", "generation"))
def test_original_rfam_two_consecutive_windows_read_verified_group_once(monkeypatch, direction):
    root = Path.cwd().resolve()
    view = load_inverse_literal_filtered_native_view(FILTER[0], FILTER[1], root=root)
    snapshot = load_inactive_common_source_rebind(root).common_snapshot.sha256
    adapter = RfamCampaignAdapter(view, direction, snapshot, 1)
    binding = adapter.train_binding()
    it = iter(binding.route.iter_from(binding.route.initial_cursor))
    first, second = next(it), next(it)
    assert first.batch.source_shard["path"] == second.batch.source_shard["path"]
    side_path = adapter._manifest["effective_group_lookup"][first.batch.source_shard["path"]]["path"]
    original = native.read_exact
    reads = []

    def verified_read(ref, *, root):
        if ref["path"] == side_path:
            reads.append(ref["sha256"])
        return original(ref, root=root)

    monkeypatch.setattr(native, "read_exact", verified_read)
    for window in (first, second):
        result = binding.prepare(window)
        assert result.unavailable_reason is None
        assert callable(result.evaluate)
        assert result.observed_positions >= 1
    assert len(reads) == 1, "Immutable original group sidecar must be verified once per shard"


@pytest.mark.parametrize("direction", ("prediction", "generation"))
def test_rfam_cached_group_refuses_changed_file_identity(monkeypatch, direction):
    root = Path.cwd().resolve()
    view = load_inverse_literal_filtered_native_view(FILTER[0], FILTER[1], root=root)
    snapshot = load_inactive_common_source_rebind(root).common_snapshot.sha256
    adapter = RfamCampaignAdapter(view, direction, snapshot, 1)
    binding = adapter.train_binding()
    window = next(iter(binding.route.iter_from(binding.route.initial_cursor)))
    assert binding.prepare(window).unavailable_reason is None
    original = native._sidecar_file_identity
    monkeypatch.setattr(native, "_sidecar_file_identity",
        lambda p: original(p) + ("changed_identity_simulated",))
    with pytest.raises(ValueError, match="changed"):
        binding.prepare(window)
