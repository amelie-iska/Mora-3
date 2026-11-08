"""Contract tests only: fixtures are not biological training coverage."""
from dataclasses import replace
import json
import pytest
from mora3.data.cooperative_segmentation import SegmentationArtifact, SegmentationRequest


def image(**changes):
    fields = dict(artifact_id='image1', kind='image', evidence='source_observed',
                  role='condition', source_id='source:v1', parent_id='field1',
                  context=(('assay', 'a1'),), asset='source/image.tif',
                  representation='yx-v1', checkpoint='generator:v1', channels=('phase',), spatial_shape=(32, 32))
    fields.update(changes)
    return SegmentationArtifact(**fields)


def request(artifact=None, **changes):
    fields = dict(artifacts=(image() if artifact is None else artifact,),
                  supported_channels=(('phase',),), supported_objects=('whole_cell',),
                  object_type='whole_cell', head_checkpoint='native:checkpoint1')
    fields.update(changes)
    return SegmentationRequest(**fields)


@pytest.mark.parametrize('evidence,role', [('source_observed','condition'),
    ('model_predicted','condition'), ('model_generated','condition'), ('desired','desired')])
def test_distinct_image_origins_route_and_survive_snapshot(evidence, role):
    r=request(image(evidence=evidence, role=role))
    assert r.route() == (('image1', 'native_segmentation'),)
    restored=SegmentationRequest.from_bytes(r.canonical_bytes())
    assert restored == r
    assert restored.artifacts[0].evidence == evidence
    assert restored.artifacts[0].role == role


def test_profile_is_unavailable_and_target_image_is_not_a_condition():
    p=image(kind='profile', channels=(), spatial_shape=())
    assert request(p).route() == (('image1','unavailable:no_image'),)
    r=request(image(role='target'))
    assert r.route() == (('image1','unavailable:held_out_target'),)
    assert r.condition_ids() == ()


@pytest.mark.parametrize('changes,reason', [
    ({'supported_channels':(('fluorescence',),)},'unsupported_channels'),
    ({'head_checkpoint':None},'missing_native_head'),
    ({'object_type':'nucleus'},'unsupported_object')])
def test_missing_capability_is_explicit(changes, reason):
    assert request(**changes).route() == (('image1', 'unavailable:'+reason),)


def test_source_algorithmic_mask_is_not_manual_and_has_typed_image_link():
    mask=image(artifact_id='mask1', kind='mask', evidence='source_algorithmic_mask',
               role='target', links=(('mask_of','image1'),), channels=())
    r=request(artifacts=(image(),mask))
    assert r.condition_ids() == ('image1',)
    assert r.supervision_ids() == ()
    assert r.supervision_ids(allow_source_algorithmic=True) == ('mask1',)
    assert SegmentationRequest.from_bytes(r.canonical_bytes()).artifacts[1] == mask


def test_generated_masks_never_become_observed_supervision():
    mask=image(artifact_id='m', kind='mask', evidence='predicted_mask', role='target',
               channels=(), links=(('mask_of','image1'),), checkpoint='segmenter:v1')
    assert request(artifacts=(image(evidence='model_generated'),mask)).supervision_ids() == ()


@pytest.mark.parametrize('changes', [
    {'evidence':'desired','role':'condition'}, {'evidence':'model_generated','role':'desired'},
    {'kind':'mask','evidence':'manual_annotation','channels':()},
    {'spatial_shape':(0,32)}, {'pixel_size_um':-1}, {'context':(('assay','a'),('assay','b'))},
    {'evidence':'model_generated','checkpoint':None}, {'seed':True}])
def test_invalid_provenance_rejected(changes):
    base=dict(checkpoint='generator:v1')
    base.update(changes)
    with pytest.raises(ValueError): image(**base)


@pytest.mark.parametrize('change', ['context','parent','shape','dangling','duplicate'])
def test_cross_context_or_unaligned_links_fail(change):
    mask=image(artifact_id='m',kind='mask',evidence='manual_annotation',role='target',
               channels=(),links=(('mask_of','image1'),))
    if change=='context': mask=replace(mask,context=(('assay','other'),))
    if change=='parent': mask=replace(mask,parent_id='other')
    if change=='shape': mask=replace(mask,spatial_shape=(31,32))
    if change=='dangling': mask=replace(mask,links=(('mask_of','absent'),))
    if change=='duplicate': mask=replace(mask,artifact_id='image1')
    with pytest.raises(ValueError): request(artifacts=(image(),mask))


def test_snapshot_rejects_unknown_fields_duplicate_keys_and_mutated_roles():
    raw=request().canonical_bytes()
    d=json.loads(raw); d['extra']=1
    with pytest.raises(ValueError): SegmentationRequest.from_bytes(json.dumps(d))
    with pytest.raises(ValueError): SegmentationRequest.from_bytes(b'{"version":1,"version":1}')
    d=json.loads(raw); d['artifacts'][0]['role']='desired'
    with pytest.raises(ValueError): SegmentationRequest.from_bytes(json.dumps(d))


def test_unverified_source_annotation_never_promoted_to_manual_supervision():
    mask=image(artifact_id='m',kind='mask',evidence='source_annotation_unverified',
               role='target',channels=(),links=(('mask_of','image1'),))
    r=request(artifacts=(image(),mask))
    assert r.supervision_ids(allow_source_algorithmic=True) == ()
    assert SegmentationRequest.from_bytes(r.canonical_bytes()) == r
