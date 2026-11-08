"""Discrete native semantic foreground area; no instance or biological validation.

Caller quality acceptance is a declaration, never estimated calibration. This
assessment consumes a prediction, not a source target or a training admission.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import re
import numpy as np
from mora3.data.cooperative_segmentation import SegmentationArtifact
from .color_semantic_inference import image_artifact_digest


@dataclass(frozen=True)
class SemanticAreaConstraint:
    image_sha256: str
    request_id: str
    origin: str
    unit: str
    lower: float | None
    upper: float | None

    def __post_init__(self):
        if not isinstance(self.image_sha256,str) or re.fullmatch('[0-9a-f]{64}',self.image_sha256) is None:
            raise ValueError('exact image artifact digest required')
        if any(not isinstance(v,str) or not v.strip() for v in (self.request_id,self.origin)):
            raise ValueError('explicit desired request identity and origin required')
        if self.unit not in ('px2','um2'):
            raise ValueError('semantic area unit required')
        if self.lower is None and self.upper is None:
            raise ValueError('at least one bound required')
        for value in (self.lower,self.upper):
            if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0):
                raise ValueError('finite nonnegative bounds required')
        if self.lower is not None and self.upper is not None and self.lower>self.upper:
            raise ValueError('ordered inclusive bounds required')


def assess_semantic_area(mask,artifact,image,constraint,*,quality_accepted):
    """Assess inclusive bounds on a native hard semantic foreground prediction.

    No object count/mean object area is derivable from semantic foreground. No
    differentiation through hard thresholding, physical-scale inference, or
    default quality acceptance is supplied. Mask digest binds the assessed bytes;
    this receipt does not authenticate saved prediction files or model execution.
    """
    if not isinstance(constraint,SemanticAreaConstraint) or constraint.image_sha256!=image_artifact_digest(image):
        raise ValueError('desired image binding mismatch')
    if quality_accepted is not None and type(quality_accepted) is not bool:
        raise ValueError('explicit boolean or unknown quality required')
    if (not isinstance(artifact,SegmentationArtifact)
        or (artifact.kind,artifact.evidence,artifact.role)!=('mask','predicted_mask','evaluation')
        or artifact.links!=(('mask_of',image.artifact_id),)
        or any(getattr(artifact,k)!=getattr(image,k) for k in
               ('source_id','parent_id','context','spatial_shape','reference_frame','pixel_size_um'))
        or not artifact.representation.startswith('nuclear-semantic-')
        or not artifact.representation.endswith('logit-ge0-v1')
        or not artifact.checkpoint.startswith('native-state-sha256:')
        or re.fullmatch('[0-9a-f]{64}',artifact.checkpoint[len('native-state-sha256:'):]) is None
        or 'semantic_not_instance' not in artifact.quality_flags):
        raise ValueError('bound native semantic prediction required')
    if not isinstance(mask,np.ndarray) or mask.dtype!=np.bool_ or mask.shape!=image.spatial_shape:
        raise ValueError('aligned native boolean semantic mask required')
    request=asdict(constraint)
    result=dict(status='evaluated',value=None,satisfied=None,metric='foreground_area',unit=constraint.unit,
        desired=request,request_sha256=hashlib.sha256(json.dumps(request,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),prediction=asdict(artifact),
        quality_accepted=quality_accepted,instance_count=None,biological_validation=False,
        evidence_scope='uncalibrated_predicted_segmentation_only',differentiable=False,training_admission=False)
    reason=('quality_unassessed' if quality_accepted is None else
            'quality_not_accepted' if not quality_accepted else
            'missing_scale' if constraint.unit=='um2' and image.pixel_size_um is None else None)
    if reason:
        result['status']='unevaluated:'+reason
        return result
    value=int(np.count_nonzero(mask))
    if constraint.unit=='um2':value*=image.pixel_size_um**2
    if not math.isfinite(value):
        result['status']='unevaluated:nonfinite_measurement'
        return result
    result.update(value=value,satisfied=(constraint.lower is None or value>=constraint.lower)
                  and (constraint.upper is None or value<=constraint.upper))
    return result


def infer_and_assess_semantic_area(model, image, pixels, constraint, *,
        quality_accepted, expected_pixel_sha256, expected_model_sha256,
        mask_asset, seed, context_fields=None, expected_image_sha256=None,
        desired_context=None):
    """Execute native inference and assess exactly its returned semantic mask.

    Area bounds are assessment requests, never encoder inputs. Execution hashes
    bind the supplied pixels/state; they authenticate neither source rights nor
    training history. Quality acceptance remains an explicit caller declaration.
    No target mask, external prediction receipt or object count is accepted.
    """
    from .color_semantic_inference import infer_nuclear_semantic
    if (not isinstance(constraint, SemanticAreaConstraint)
            or constraint.image_sha256 != image_artifact_digest(image)):
        raise ValueError('desired image binding mismatch')
    if quality_accepted is not None and type(quality_accepted) is not bool:
        raise ValueError('explicit boolean or unknown quality required')
    prediction = infer_nuclear_semantic(model, image, pixels,
        expected_pixel_sha256=expected_pixel_sha256,
        expected_model_sha256=expected_model_sha256, mask_asset=mask_asset,
        seed=seed, context_fields=context_fields,
        expected_image_sha256=expected_image_sha256, desired_context=desired_context)
    assessment = assess_semantic_area(prediction['mask'], prediction['artifact'],
        image, constraint, quality_accepted=quality_accepted)
    assessment['execution'] = dict(policy='executed-native-color-area-v1',
        pixel_sha256=expected_pixel_sha256, model_sha256=expected_model_sha256,
        image_artifact_sha256=constraint.image_sha256, seed=seed,
        conditioning=prediction.get('conditioning'))
    return dict(prediction, area_assessment=assessment)
