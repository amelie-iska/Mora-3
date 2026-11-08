"""Synthetic boundary tests; no source qualification or scientific training."""
from pathlib import Path
import sys
import pytest
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from data_sources_native import reaction_condition

@pytest.mark.parametrize('release', [None, 'RGD1.figshare.21066901.v6', 'RGD1.figshare.21066901.v9', 'unknown'])
@pytest.mark.parametrize('flags', [{}, {'task_admitted': True, 'loss_mask': 1, 'qualified': True}])
def test_unqualified_energy_conditioning_refuses_despite_legacy_flags(release, flags):
    record = dict(source_native={'reactant': 'SYNTHETIC_R', 'product': 'SYNTHETIC_P'}, source_release=release, **flags)
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        reaction_condition(record)

def test_explicit_audit_preserves_target_exclusion():
    a = {'source_native': {'reactant': 'SYNTHETIC_R', 'product': 'SYNTHETIC_P', 'DE_F': 3}}
    b = {'source_native': {**a['source_native'], 'DE_F': 999}, 'task_admitted': True}
    assert reaction_condition(a, audit_only=True) == reaction_condition(b, audit_only=True) == {'reactant': 'SYNTHETIC_R', 'product': 'SYNTHETIC_P'}

@pytest.mark.parametrize('value', [1, 'true', None, [], {}])
def test_truthy_audit_flag_cannot_bypass_gate(value):
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        reaction_condition({}, audit_only=value)

def test_policy_integrity_failure_closes_even_audit(monkeypatch, tmp_path):
    import data_sources_native as native
    policy = tmp_path / 'policy.json'
    policy.write_text('{}')
    monkeypatch.setattr(native, 'RGD1_POLICY_PATH', policy, raising=False)
    with pytest.raises(ValueError, match='RGD1.*policy integrity'):
        reaction_condition({'source_native': {'reactant': 'R', 'product': 'P'}}, audit_only=True)
