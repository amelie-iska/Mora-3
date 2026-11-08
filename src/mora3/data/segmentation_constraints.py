"""CSEG-05a discrete geometry feasibility, without biological validation.

Bounds are inclusive and units are explicit. This nondifferentiable bridge uses
native instance measurements, never source target masks. A caller's quality flag
is a declaration, not a calibrated segmentation-quality estimate. No ranking,
therapeutic reward, physical scale inference or training admission is supplied.
"""
from dataclasses import dataclass
import math
from .cooperative_segmentation import SegmentationArtifact
from .segmentation_measurements import measure_predicted_instances


@dataclass(frozen=True)
class GeometryConstraint:
    desired: SegmentationArtifact
    object_type: str
    metric: str
    unit: str
    lower: float | None
    upper: float | None

    def __post_init__(self):
        a=self.desired
        if not isinstance(a,SegmentationArtifact) or (a.kind,a.role,a.evidence)!=('objects','desired','desired'):
            raise ValueError('explicit desired objects required')
        if not isinstance(self.object_type,str) or not self.object_type.strip():
            raise ValueError('object type required')
        units={'object_count':{'count'},'total_area':{'px2','um2'},'mean_area':{'px2','um2'}}
        if self.metric not in units or self.unit not in units[self.metric]:
            raise ValueError('unsupported metric/unit')
        if self.lower is None and self.upper is None:
            raise ValueError('at least one bound required')
        for b in (self.lower,self.upper):
            if b is not None and (type(b) not in (int,float) or not math.isfinite(b) or b<0):
                raise ValueError('finite nonnegative bounds required')
        if self.lower is not None and self.upper is not None and self.lower>self.upper:
            raise ValueError('ordered bounds required')


def assess_geometry(labels,image,constraint,*,object_type,checkpoint,mask_asset,quality_accepted=True):
    """Assess a declared object class in an exactly bound image/context/frame.

    Returns unevaluated (never passed) for missing scale, rejected quality or
    undefined empty-set mean. Even a passed constraint is model-derived evidence.
    No tensor gradients flow through hard instance decoding/measurement.
    """
    if not isinstance(constraint,GeometryConstraint) or not isinstance(image,SegmentationArtifact):
        raise ValueError('validated constraint and image required')
    if type(quality_accepted) is not bool:
        raise ValueError('explicit boolean quality acceptance required')
    if object_type!=constraint.object_type:
        raise ValueError('object type mismatch')
    a=constraint.desired
    if a.links!=(('objects_of',image.artifact_id),) or any(getattr(a,k)!=getattr(image,k) for k in
            ('source_id','parent_id','context','spatial_shape','reference_frame','pixel_size_um')):
        raise ValueError('desired/image binding mismatch')
    measurement=measure_predicted_instances(labels,image,checkpoint=checkpoint,mask_asset=mask_asset)
    result=dict(status='evaluated',value=None,satisfied=None,unit=constraint.unit,
                metric=constraint.metric,desired=constraint,measurement=measurement,
                biological_validation=False,differentiable=False,
                evidence_scope='model_consistency_only' if image.evidence in ('model_generated','model_predicted')
                else 'predicted_segmentation_only')
    reason=None
    objects=measurement['objects']
    if not quality_accepted: reason='quality_not_accepted'
    elif constraint.unit=='um2' and image.pixel_size_um is None: reason='missing_scale'
    elif constraint.metric=='mean_area' and not objects: reason='no_objects'
    if reason:
        result['status']='unevaluated:'+reason
        return result
    if constraint.metric=='object_count': value=len(objects)
    else:
        key='area_um2' if constraint.unit=='um2' else 'area_px'
        value=sum(o[key] for o in objects)
        if constraint.metric=='mean_area': value/=len(objects)
    if not math.isfinite(value):
        result['status']='unevaluated:nonfinite_measurement'
        return result
    result.update(value=value,satisfied=(constraint.lower is None or value>=constraint.lower)
                  and (constraint.upper is None or value<=constraint.upper))
    return result
