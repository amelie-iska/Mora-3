"""Metadata-only split audits must not initialize unrelated target readers."""
from pathlib import Path
import pytest
import mora3.research.cross_source_chemical_roles as module


def test_exact_pinned_metadata_view_without_full_registry(monkeypatch,tmp_path):
    calls=[];sentinel=object()
    def view(path,pin,*,root):calls.append((path,pin,root));return sentinel
    monkeypatch.setattr(module,'load_study_filtered_native_view',view,raising=False)
    monkeypatch.setattr(module,'finite_view_policy_sha256',lambda v:module._GROUP_VIEW_POLICY,raising=False)
    monkeypatch.setattr(module,'load_assets',lambda *a:pytest.fail('Unrelated target-bearing registry must not initialize'),raising=False)
    assert module._load_group_metadata_view(tmp_path)is sentinel
    path,pin=module.CURRENT_VIEW_PINS['effective_source_view']
    assert calls==[(tmp_path/path,pin,tmp_path)]


def test_changed_metadata_view_policy_refuses(monkeypatch,tmp_path):
    monkeypatch.setattr(module,'load_study_filtered_native_view',lambda *a,**kw:object(),raising=False)
    monkeypatch.setattr(module,'finite_view_policy_sha256',lambda v:'0'*64,raising=False)
    with pytest.raises(ValueError,match='policy'):module._load_group_metadata_view(tmp_path)
