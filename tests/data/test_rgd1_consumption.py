"""Synthetic source-policy boundaries; no scientific data admission."""
import json
import pytest
from mora3.data import rgd1_consumption as gate

@pytest.mark.parametrize('purpose', ['conditioning','objective','evaluation'])
@pytest.mark.parametrize('release', [None,'v6','v9','unknown'])
def test_legacy_flags_and_claimed_releases_cannot_grant_use(purpose, release):
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        gate.require_corpus_consumption({'source_family':'RGD1-computed-energy','source_revision':release,
            'task_admitted':True,'model_adapter_supported':True,'loss_mask':0}, purpose=purpose)

@pytest.mark.parametrize('field,value', [('task_name','RGD1_forward_energy'),('source_record_id','RGD1:v6:1'),('source_release','RGD1.figshare.21066901.v9')])
def test_alternate_identity_fields_remain_closed(field,value):
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        gate.require_corpus_consumption({field:value})

def test_native_envelope_audit_identity_is_checked():
    raw=json.dumps(dict(task='energy',conditions={},targets={'energy':1},audit={'source_family':'RGD1'},missing=[])).encode()
    with pytest.raises(ValueError, match='RGD1.*quarantine'):
        gate.require_corpus_consumption({'native_payload':raw})

def test_unrelated_source_is_unchanged():
    row={'source_family':'Rosetta','native_payload':b'opaque array bytes'}
    assert gate.require_corpus_consumption(row) is None

@pytest.mark.parametrize('audit', [True,1,'true'])
def test_no_audit_bypass_in_model_consumption(audit):
    with pytest.raises(ValueError,match='RGD1.*quarantine'):
        gate.require_corpus_consumption({'source_family':'RGD1','audit_only':audit})

def test_policy_reverified_each_call(monkeypatch,tmp_path):
    policy=tmp_path/'policy.json';policy.write_bytes(gate.POLICY_PATH.read_bytes())
    monkeypatch.setattr(gate,'POLICY_PATH',policy)
    with pytest.raises(ValueError,match='RGD1.*quarantine'):
        gate.require_corpus_consumption({'source_family':'RGD1'})
    policy.write_bytes(b'{}')
    with pytest.raises(ValueError,match='policy integrity'):
        gate.require_corpus_consumption({'source_family':'RGD1'})

def test_unknown_purpose_cannot_grant_use():
    with pytest.raises(ValueError):gate.require_corpus_consumption({'source_family':'RGD1'},purpose='audit')
