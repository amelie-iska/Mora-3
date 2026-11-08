from pathlib import Path
import sys
import pytest
sys.path.insert(0,'/home/iska/Documents/amelie/bio/Mora-3/scripts')
from data_intake_full_rnacentral_launch import validate_approval

def approval():return dict(approved=True,phase='census',scope='whole-native-RNAcentral-census-only',plan_sha256='p',launcher_sha256='l',resource_coordination_sha256='r',whole_source_occurrences=61026468)

def test_exact_reviewed_complete_source_scope_is_required():
 validate_approval(approval(),'p','l','r')

@pytest.mark.parametrize('field,value',[('approved',False),('phase','materialize'),('scope','pilot'),('plan_sha256','different'),('launcher_sha256','different'),('resource_coordination_sha256','different'),('whole_source_occurrences',500000)])
def test_missing_wrong_version_or_unreviewed_phase_cannot_launch(field,value):
 a=approval();a[field]=value
 with pytest.raises(ValueError):validate_approval(a,'p','l','r')
