"""Metadata eligibility never promotes a provenance declaration to authority."""
from dataclasses import replace
import importlib
import pytest
from test_current_frame_request_binding import fixture, bind
from mora3.data import geometry_frame_policy as p


def candidate():
    c=fixture()
    c['current']=replace(c['current'],source=replace(c['current'].source,
        source_id='Transition1x:v4:reaction',origin=p.Origin.UNQUALIFIED_SOURCE,
        group_sha256='c'*64,heldout_sha256='d'*64))
    b=bind(c)
    return dict(binding=b,request_sha256=b.request_sha256,
        atom_axis_sha256=b.native_axis_sha256,source_identity_sha256='b'*64,
        group_sha256='c'*64,heldout_sha256='d'*64,context_sha256='e'*64,
        expected_context_sha256='e'*64,current_xyz_sha256='f'*64,
        provenance='independent_noise',parent_target_sha256=None,
        recipe_sha256='1'*64,seed=33801,native_split='val',expected_native_split='val',use='evaluation',
        source_role='independent_noise')


def preflight(c):
    m=importlib.import_module('mora3.data.native_current_preflight')
    return m.preflight_native_current(**c)


def test_candidate_retains_binding_but_all_authorities_stay_closed():
    c=candidate();r=preflight(c)
    assert r.binding is c['binding'] and r.context_sha256=='e'*64
    assert r.status=='unqualified_metadata_candidate'
    for name in ('require_source_use','require_model_use','require_training_use'):
        with pytest.raises(ValueError):getattr(r,name)()


@pytest.mark.parametrize('field,value',[
    ('request_sha256','0'*64),('atom_axis_sha256','0'*64),
    ('source_identity_sha256','0'*64),('group_sha256','0'*64),
    ('heldout_sha256','0'*64),('context_sha256','0'*64),
    ('current_xyz_sha256','not-a-hash'),('recipe_sha256',None),
    ('seed',True),('seed',-1),('native_split','test'),('use','training'),
    ('provenance','gold_noising'),('parent_target_sha256','2'*64),
    ('source_role','reactant'),('source_role','product'),
    ('source_role','transition_state'),('provenance','source_observed'),
])
def test_unbound_target_or_heldout_training_claims_refused(field,value):
    c=candidate();c[field]=value
    with pytest.raises(ValueError):preflight(c)


def test_stale_or_synthetic_binding_refused():
    c=candidate();object.__setattr__(c['binding'],'xyz_available',(False,True,True))
    with pytest.raises(ValueError):preflight(c)
    c=candidate();c['binding']=bind(fixture())
    with pytest.raises(ValueError):preflight(c)


def test_native_train_split_requires_separate_admission():
    c=candidate();c.update(native_split='train',expected_native_split='train',use='training')
    with pytest.raises(ValueError,match='admission'):preflight(c)
