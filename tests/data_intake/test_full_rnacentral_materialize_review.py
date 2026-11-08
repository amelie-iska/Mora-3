from pathlib import Path
import sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_materialize import validate_materialization_approval

def kwargs():return dict(plan='p',census='c',clock='t',script='s',resource='r',dependencies=[{'path':'f','sha256':'d','bytes':1}])
def approved():return dict(approved=True,phase='materialize',scope='whole-native-RNAcentral-materialization-only',plan_sha256='p',census_manifest_sha256='c',clock_sha256='t',script_sha256='s',resource_coordination_sha256='r',dependency_refs=kwargs()['dependencies'])

def test_materialization_has_separate_exact_phase_and_dependency_review():
 validate_materialization_approval(approved(),**kwargs())

@pytest.mark.parametrize('field,value',[('approved',False),('phase','census'),('scope','pilot'),('plan_sha256','x'),('census_manifest_sha256','x'),('clock_sha256','x'),('script_sha256','x'),('resource_coordination_sha256','x'),('dependency_refs',[])])
def test_census_approval_old_source_clock_or_changed_helper_cannot_authorize_writer(field,value):
 a=approved();a[field]=value
 with pytest.raises(ValueError):validate_materialization_approval(a,**kwargs())
