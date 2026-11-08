"""Desired assertions stay separate from source observations and segmentation targets."""
from dataclasses import replace
import pytest
from test_color_semantic_inference import fixture
from mora3.model import color_semantic_inference as api


def build(**changes):
    image,_=fixture()
    args=dict(image=image,expected_image_sha256=api.image_artifact_digest(image),
              request_id='research-request-1',origin='user-declared',
              values=(('tissue','desired tissue'),('organism','desired organism')))
    args.update(changes)
    return api.desired_color_context_request(**args)


def test_request_is_canonical_and_does_not_rewrite_source():
    image,_=fixture();before=api.image_artifact_digest(image)
    a=build();b=build(values=tuple(reversed(a.values)))
    assert a==b and a.canonical_bytes()==b.canonical_bytes()
    assert a.evidence_role=='desired' and a.training_admission is False
    assert a.image_sha256==before
    assert api.image_artifact_digest(image)==before
    assert b.canonical_bytes()!=build(request_id='other').canonical_bytes()
    assert a.canonical_bytes()!=build(origin='protocol-request').canonical_bytes()


@pytest.mark.parametrize('values',[(('mask','x'),),(('nuclear_area','3'),),
    (('response','x'),),(('tissue',None),),(('tissue',3),),(('tissue',' '),),
    (('tissue','x'),('tissue','y')),(),{'tissue':'x'},(('tissue','x','extra'),)])
def test_invalid_or_target_bearing_requests_refuse(values):
    with pytest.raises(ValueError):build(values=values)


@pytest.mark.parametrize('field,value',[('request_id',''),('origin',' '),
    ('expected_image_sha256','a'*64)])
def test_request_requires_origin_identity_and_exact_source_binding(field,value):
    with pytest.raises(ValueError):build(**{field:value})


def test_changed_source_parent_refuses_stale_request_binding():
    image,_=fixture()
    with pytest.raises(ValueError):build(image=replace(image,parent_id='different'))
