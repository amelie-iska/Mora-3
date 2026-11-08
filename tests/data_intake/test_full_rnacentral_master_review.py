from pathlib import Path
import sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_union_master import validate_union_approval,task_digest

def expected():return dict(plan_sha256='p',component_manifest_sha256='m',source_task_proof_sha256='q',clock_sha256='t',dependency_refs=[{'path':'g','sha256':'h','bytes':1}])
def approved():return dict(approved=True,phase='union-master',scope='whole-native-RNAcentral-master-union-only',**expected())

def test_master_publication_has_separate_complete_source_task_proof_review():
 validate_union_approval(approved(),expected())

@pytest.mark.parametrize('field,value',[('approved',False),('phase','census'),('scope','pilot'),('component_manifest_sha256','x'),('source_task_proof_sha256','x'),('clock_sha256','x'),('dependency_refs',[])])
def test_source_only_or_changed_proof_does_not_promote_a_master(field,value):
 a=approved();a[field]=value
 with pytest.raises(ValueError):validate_union_approval(a,expected())


def test_only_literal_canonical_task_hashes_can_match_binary_index():
 text='a1'*32
 assert task_digest(text)==bytes.fromhex(text)
 assert task_digest(text.upper()) is None
 assert task_digest('external:source_id') is None
 assert task_digest(' '*64) is None
