"""Bound independent nuclear-slot geometry constraints, never biological truth.

Total area sums independent planes, counting overlap multiplicity. Pairwise overlap
sums unordered slot-pair intersections (a triple overlap counts three times).
No union, deduplication, source instance matching or calibrated quality is implied.
Hard geometry is nondifferentiable. No weights, pixels or truth masks are loaded.
"""
import copy
from dataclasses import dataclass
import math
from .segmentation_constraints import GeometryConstraint
from .segmentation_shared import project_bound_slot_geometry


@dataclass(frozen=True)
class SlotGeometryConstraint(GeometryConstraint):
    def __post_init__(self):
        # Reuse the established desired-object and bound validation, mapping only
        # the additional pairwise metric onto the same area unit contract.
        GeometryConstraint(self.desired,self.object_type,
            'total_area' if self.metric=='pairwise_overlap_area' else self.metric,
            self.unit,self.lower,self.upper)
        if self.object_type != 'nucleus':
            raise ValueError('native nuclear slot constraint required')


def assess_bound_slot_geometry(image,prediction,constraint,*,max_mask_bytes,
                               quality_accepted=None):
    """Replay inference bindings before assessing inclusive desired bounds.

    Quality is unknown by default; explicit True is only a caller declaration,
    not measured validation. Existing bound shared projection authenticates the
    internal metadata/mask/geometry consistency, not asset or weight provenance.
    """
    if not isinstance(constraint,SlotGeometryConstraint):
        raise ValueError('validated nuclear slot constraint required')
    if quality_accepted is not None and type(quality_accepted) is not bool:
        raise ValueError('explicit boolean or unknown quality acceptance required')
    desired=constraint.desired
    if desired.links!=(('objects_of',image.artifact_id),) or any(
            getattr(desired,k)!=getattr(image,k) for k in
            ('source_id','parent_id','context','spatial_shape','reference_frame','pixel_size_um')):
        raise ValueError('desired/image binding mismatch')
    from mora3.model.color_semantic_inference import image_condition_graph
    provenance=image_condition_graph(image,
        pixel_sha256=prediction['binding']['pixel_sha256'])[0].provenance
    project_bound_slot_geometry(image,prediction,provenance=provenance,
        requested_targets=('geometry_assessment',),max_mask_bytes=max_mask_bytes)
    measurement=copy.deepcopy(prediction['measurements'])
    result=dict(status='evaluated',value=None,satisfied=None,unit=constraint.unit,
        metric=constraint.metric,desired=constraint,measurement=measurement,
        binding=copy.deepcopy(prediction['binding']),mask_sha256=prediction['mask_sha256'],
        quality_accepted=quality_accepted,biological_validation=False,
        training_admission=False,differentiable=False,
        evidence_scope='model_consistency_only' if image.evidence in
        ('model_generated','model_predicted') else 'predicted_segmentation_only')
    objects=measurement['objects']
    reason=None
    if quality_accepted is None:reason='quality_unevaluated'
    elif quality_accepted is False:reason='quality_not_accepted'
    elif constraint.unit=='um2' and image.pixel_size_um is None:reason='missing_scale'
    elif constraint.metric=='mean_area' and not objects:reason='no_objects'
    if reason:
        result['status']='unevaluated:'+reason
        return result
    if constraint.metric=='object_count':value=len(objects)
    elif constraint.metric=='pairwise_overlap_area':
        value=sum(edge['overlap_px'] for edge in measurement['edges'])
        if constraint.unit=='um2':value*=image.pixel_size_um**2
    else:
        key='area_um2' if constraint.unit=='um2' else 'area_px'
        value=sum(obj[key] for obj in objects)
        if constraint.metric=='mean_area':value/=len(objects)
    if not math.isfinite(value):
        result['status']='unevaluated:nonfinite_measurement'
        return result
    result.update(value=value,satisfied=(constraint.lower is None or value>=constraint.lower)
        and (constraint.upper is None or value<=constraint.upper))
    return result
