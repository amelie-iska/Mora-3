from pathlib import Path
import sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_verify_source_tasks import validate_source_task_proof_approval

def expected():return dict(plan_sha256='p',census_manifest_sha256='c',component_manifest_sha256='m',clock_sha256='t',script_sha256='s',dependency_refs=[{'path':'native-proof-helper','sha256':'f','bytes':1}])
def approval():return dict(approved=True,phase='verify-source-tasks',scope='whole-native-RNAcentral-source-task-proof-only',**expected())

def test_full_source_and_task_proof_requires_its_own_exact_review():
 validate_source_task_proof_approval(approval(),expected())

@pytest.mark.parametrize('field,value',[('approved',False),('phase','census'),('scope','pilot'),('component_manifest_sha256','wrong'),('census_manifest_sha256','wrong'),('clock_sha256','wrong'),('script_sha256','wrong'),('dependency_refs',[])])
def test_another_phase_partial_scope_or_changed_component_cannot_authorize_proof(field,value):
 a=approval();a[field]=value
 with pytest.raises(ValueError):validate_source_task_proof_approval(a,expected())
