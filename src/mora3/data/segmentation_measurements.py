"""Discrete predicted-instance geometry, in the original image pixel frame.

Area counts pixels; centroid averages pixel indices; bounding boxes are half-open;
perimeter counts exposed four-neighbour pixel sides including image boundaries.
Edges mean shared pixel boundary only, never biological interaction. No truth
mask or expression/perturbation payload is consumed. This CPU measurement bridge
is nondifferentiable and does not itself execute or validate a segmentation head.
"""
from dataclasses import replace
import numpy as np
import torch
from .cooperative_segmentation import SegmentationArtifact


def measure_predicted_instances(labels, image, *, checkpoint, mask_asset):
    """Measure a nonnegative int64 instance map; zero denotes background.

    Only a condition/desired image can contribute derived conditioning geometry.
    The caller must bind checkpoint and mask asset to its actual inference output;
    these identifiers are provenance declarations, not checkpoint verification.
    """
    if not isinstance(image, SegmentationArtifact) or image.kind!='image':
        raise ValueError('image artifact required')
    if image.role not in ('condition','desired'):
        raise ValueError('condition or desired image required; hidden targets excluded')
    if (not isinstance(labels,torch.Tensor) or labels.dtype!=torch.long
            or tuple(labels.shape)!=image.spatial_shape or (labels<0).any()):
        raise ValueError('aligned nonnegative int64 instance map required')
    for value in (checkpoint,mask_asset):
        if not isinstance(value,str) or not value.strip():
            raise ValueError('checkpoint and immutable prediction asset required')
    a=labels.detach().cpu().numpy()
    objects=[]
    for obj in np.unique(a):
        if obj==0: continue
        y,x=np.nonzero(a==obj)
        # Padded image accounts for exposed image borders, without wrapping.
        m=np.pad(a==obj,1)
        center=m[1:-1,1:-1]
        perimeter=sum(int((center & ~other).sum()) for other in
                      (m[:-2,1:-1],m[2:,1:-1],m[1:-1,:-2],m[1:-1,2:]))
        area=len(y)
        objects.append(dict(object_id=int(obj),area_px=area,
            centroid_yx=[float(y.mean()),float(x.mean())],
            bbox_yxyx=[int(y.min()),int(x.min()),int(y.max())+1,int(x.max())+1],
            perimeter_px=perimeter,area_um2=None if image.pixel_size_um is None
            else area*image.pixel_size_um**2))
    counts={}
    for left,right in ((a[:,:-1],a[:,1:]),(a[:-1,:],a[1:,:])):
        valid=(left>0)&(right>0)&(left!=right)
        pairs=np.sort(np.stack((left[valid],right[valid]),axis=1),axis=1)
        unique,n=np.unique(pairs,axis=0,return_counts=True)
        for pair,count in zip(unique,n):
            key=tuple(int(i) for i in pair)
            counts[key]=counts.get(key,0)+int(count)
    flags=('model_consistency_only',) if image.evidence in ('model_generated','model_predicted') else ()
    artifact=replace(image,artifact_id=image.artifact_id+':predicted_objects',kind='objects',
        evidence='model_predicted',role='condition',asset=mask_asset,
        representation='pixel-object-geometry-v1',channels=(),
        links=(('objects_of',image.artifact_id),),checkpoint=checkpoint,
        quality_flags=image.quality_flags+flags,uncertainty='uncalibrated')
    return dict(artifact=artifact,objects=objects,edges=[dict(source=u,target=v,
        relation='pixel_boundary_adjacency',shared_boundary_px=n) for (u,v),n in sorted(counts.items())],
        image_evidence=image.evidence,biological_validation=False)


def measure_predicted_mask_set(masks, image, *, checkpoint, mask_asset, max_mask_bytes,
                               overlap_method="reference"):
    """Measure independent boolean slot planes without resolving overlaps.

    Slot index + 1 is a prediction-local ID, never source instance identity.
    Empty slots are reported separately; duplicate nonempty planes remain distinct.
    Edges count shared membership, not contact or biological interaction. Pairwise
    work is O(slots**2 * pixels); this CPU reference makes no efficiency claim.
    Opt-in bbox_exact intersects half-open object bounds before counting shared
    pixels; it preserves every plane and exact result. Worst-case work remains
    quadratic. The byte policy bounds the input copy only, not total peak memory. Metadata
    identifiers are caller declarations; this function does not execute inference
    or authenticate weights/assets. Source/desired truth masks are not inputs.
    """
    if type(overlap_method) is not str or overlap_method not in ("reference", "bbox_exact"):
        raise ValueError("overlap method must be reference or bbox_exact")
    if (not isinstance(image, SegmentationArtifact) or image.kind != 'image'
            or image.role != 'condition' or image.evidence not in
            ('source_observed', 'model_predicted', 'model_generated')
            or image.reference_frame != 'source_pixel_yx'):
        raise ValueError('visible condition image in source pixel frame required')
    if (not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool
            or masks.ndim != 3 or not 1 <= masks.shape[0] <= 512
            or tuple(masks.shape[1:]) != image.spatial_shape):
        raise ValueError('aligned boolean set of 1..512 prediction slots required')
    if (type(max_mask_bytes) is not int or max_mask_bytes < 1
            or masks.numel() * masks.element_size() > max_mask_bytes):
        raise ValueError('mask input-copy byte policy exceeded')
    if (not isinstance(checkpoint, str) or not checkpoint.strip()
            or not isinstance(mask_asset, str) or not mask_asset.strip()
            or mask_asset == image.asset):
        raise ValueError('checkpoint and new immutable prediction asset required')
    planes = masks.detach().cpu().numpy()
    objects, empty, active = [], [], []
    for index, plane in enumerate(planes):
        slot_id = index + 1
        if not plane.any():
            empty.append(slot_id)
            continue
        # Reuse the established pixel geometry equation for ONE independent
        # plane at a time. No union/argmax or label-map conversion across slots.
        measured = measure_predicted_instances(torch.from_numpy(plane.astype(np.int64)),
            image, checkpoint=checkpoint, mask_asset=mask_asset)
        objects.append(dict(measured['objects'][0], object_id=slot_id))
        active.append(index)
    # Bounds come from complete planes above; half-open intersection cannot
    # omit a shared pixel. Keep the reference default and identical edge order.
    bounds = {obj["object_id"] - 1: obj["bbox_yxyx"] for obj in objects}
    edges = []
    for position, left in enumerate(active):
        for right in active[position + 1:]:
            if overlap_method == "bbox_exact":
                a, b = bounds[left], bounds[right]
                y0, x0 = max(a[0], b[0]), max(a[1], b[1])
                y1, x1 = min(a[2], b[2]), min(a[3], b[3])
                if y0 >= y1 or x0 >= x1:
                    continue
                overlap = int(np.count_nonzero(
                    planes[left, y0:y1, x0:x1] & planes[right, y0:y1, x0:x1]))
            else:
                overlap = int(np.count_nonzero(planes[left] & planes[right]))
            if overlap:
                edges.append(dict(source=left + 1, target=right + 1,
                    relation='pixel_membership_overlap', overlap_px=overlap))
    flags = (('model_consistency_only',) if image.evidence in
             ('model_generated', 'model_predicted') else ())
    artifact = replace(image, artifact_id=image.artifact_id + ':predicted_slot_objects',
        kind='objects', evidence='model_predicted', role='condition', asset=mask_asset,
        representation='overlapping-slot-object-geometry-v1', channels=(),
        links=(('objects_of', image.artifact_id),), checkpoint=checkpoint,
        quality_flags=image.quality_flags + flags, uncertainty='uncalibrated')
    return dict(artifact=artifact, objects=objects, edges=edges,
        empty_slot_ids=empty, slots=len(planes), image_evidence=image.evidence,
        biological_validation=False)
