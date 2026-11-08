"""Corpus admission audits must read legacy envelopes without upgrading their support."""
import importlib.util
from pathlib import Path
import pytest
ROOT = Path('/home/iska/Documents/amelie/bio/Mora-3')
spec = importlib.util.spec_from_file_location('audit_metadata', ROOT/'scripts/data_intake_corpus_audit_metadata.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_legacy_sciplex_missing_field_retains_original_support():
    assert m.adapter_support({'native_schema':'mora3.ObservationalSciPlexTask/v1'}) is True

def test_unknown_generic_support_is_not_promoted():
    assert m.adapter_support({'native_schema':'mora3.CorpusTask/v1'}) is False
    assert m.adapter_support({'native_schema':'unregistered.Future/v1'}) is False

def test_explicit_gate_overrides_legacy_and_must_be_boolean():
    assert m.adapter_support({'native_schema':'mora3.ObservationalSciPlexTask/v1','model_adapter_supported':False}) is False
    with pytest.raises(ValueError,match='boolean'):
        m.adapter_support({'native_schema':'mora3.CorpusTask/v1','model_adapter_supported':'false'})
