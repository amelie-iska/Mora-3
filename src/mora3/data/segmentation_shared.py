"""Native discrete geometry -> shared TokenGT conditions, without source joins.

Pixel measurements are recomputed to bind saved geometry to labels and the visible
image. Checkpoint identity is checked against the request; callers must obtain
predictions through NativeRasterSegmentation (this is not a checkpoint verifier).
Target placeholders are unavailable, never pseudo supervision. This adapter adds
no response observations, image pixels, cross-study links or training admission.
"""
from dataclasses import asdict
import hashlib
import json
from .cooperative_segmentation import SegmentationRequest
from .segmentation_measurements import measure_predicted_instances
from .schema import GraphRecord, Node, Edge, Measurement
from .provenance import LeakageKey
from .tasks import ConditionalTask


def project_native_geometry(request, predictions, *, provenance, requested_targets):
    """Build visible image-context/object nodes and directed spatial edges.

    Geometry remains in the source pixel frame; boundary adjacency is represented
    in both directions, containment in image->object direction. All available
    fields are conditions. Requested targets have no source-supported values here.
    Source/parent/checkpoint/asset identities stay in audit metadata, not features.
    """
    if not isinstance(request, SegmentationRequest):
        raise ValueError('validated request required')
    targets=tuple(requested_targets)
    if (not targets or len(set(targets))!=len(targets) or any(
            not isinstance(k,str) or not k.strip() or k in ('image_context','predicted_geometry') for k in targets)):
        raise ValueError('distinct nonempty requested target names required')
    nodes=[]; edges=[]; roles=[]; missing={}; artifacts=[]; keys=[]
    visible=[a for a in request.artifacts if a.kind=='image' and a.role in ('condition','desired')]
    if not visible: raise ValueError('visible image context required')
    routes=dict(request.route())

    def add_node(kind, values):
        idx=len(nodes)
        measurements=[]
        for name,value,units,status in values:
            measurements.append(Measurement(name,value,units,status if value is not None else 'missing',value is not None))
            roles.append((('node',idx,name),'condition' if value is not None else 'unused'))
        nodes.append(Node(kind,tuple(measurements)))
        return idx

    def add_edge(source,target,kind,values=()):
        idx=len(edges)
        ms=tuple(Measurement(k,v,u,'predicted',True) for k,v,u in values)
        edges.append(Edge(source,target,kind,True,ms))
        roles.extend((('edge',idx,m.name),'condition') for m in ms)

    geometry_present=False
    for image in visible:
        status='observed' if image.evidence=='source_observed' else 'predicted'
        context=tuple(image.context)
        missing[image.artifact_id]=tuple(k for k,v in context if v is None)
        values=[('height',image.spatial_shape[0],'pixel',status),('width',image.spatial_shape[1],'pixel',status),
                ('pixel_size_um',image.pixel_size_um,'um/pixel',status),('image_evidence',image.evidence,None,status)]
        for k,v in context:
            values.extend(((f'context:{k}',v,None,status),(f'context_available:{k}',v is not None,None,status)))
        image_idx=add_node('microscopy_image_context',values)
        artifacts.append(asdict(image))
        keys.append(LeakageKey(image.source_id+':parent-v1',image.parent_id))
        # Never resolve hidden target payloads, even if supplied in predictions.
        if routes[image.artifact_id]!='native_segmentation': continue
        result=predictions.get(image.artifact_id)
        if not isinstance(result,dict) or result.get('status')!='predicted':
            raise ValueError('native prediction required for routed image')
        m=result['measurement']; artifact=m['artifact']
        if artifact.checkpoint!=request.head_checkpoint:
            raise ValueError('prediction checkpoint mismatch')
        expected=measure_predicted_instances(result['labels'],image,checkpoint=request.head_checkpoint,mask_asset=artifact.asset)
        if m!=expected: raise ValueError('prediction geometry/provenance mismatch')
        artifacts.append(asdict(artifact)); geometry_present=True
        mapping={}
        for obj in m['objects']:
            values=[(k,obj[k],u,'predicted') for k,u in (('area_px','pixel^2'),('perimeter_px','pixel'),('area_um2','um^2'))]
            values.extend((k,v,'pixel','predicted') for k,v in zip(('centroid_y','centroid_x'),obj['centroid_yx']))
            values.extend((k,v,'pixel','predicted') for k,v in zip(('bbox_y_min','bbox_x_min','bbox_y_max','bbox_x_max'),obj['bbox_yxyx']))
            idx=add_node('predicted_whole_cell_geometry',values)
            mapping[obj['object_id']]=idx
            add_edge(image_idx,idx,'contains_predicted_object')
        for edge in m['edges']:
            u,v=mapping[edge['source']],mapping[edge['target']]
            for left,right in ((u,v),(v,u)):
                add_edge(left,right,'pixel_boundary_adjacency',(('shared_boundary_px',edge['shared_boundary_px'],'pixel'),))
    # Fixed unavailable placeholders cannot expose hidden target topology/values.
    first=nodes[0]
    placeholders=tuple(Measurement('requested:'+k,None,None,'missing',False) for k in targets)
    nodes[0]=Node(first.kind,first.measurements+placeholders)
    roles.extend((('node',0,m.name),'target') for m in placeholders)
    identity=hashlib.sha256(json.dumps([asdict(a) for a in visible],sort_keys=True).encode()).hexdigest()
    graph=GraphRecord('native-segmentation-conditions-v1:'+identity,tuple(nodes),tuple(edges),tuple(provenance),tuple(keys))
    return dict(input_graph=graph,task=ConditionalTask(tuple(roles),topology_known=True),
        modality_masks=dict(image_context=True,predicted_geometry=geometry_present,**{k:False for k in targets}),
        missing_context=missing,artifacts=artifacts,biological_validation=False)


def project_bound_slot_geometry(image, prediction, *, provenance,
                                requested_targets, max_mask_bytes,
                                overlap_method="reference"):
    """Bound independent nuclear slots -> shared conditions, never supervision.

    Replays masks/geometry and image-only inference context hashes. Does not
    authenticate a checkpoint/file or establish source rights, training quality,
    biological identity or cross-modal pairing. Hard masks detach gradients.
    Empty slots and local slot IDs are audit metadata, never model features.
    """
    if type(overlap_method) is not str or overlap_method not in ("reference", "bbox_exact"):
        raise ValueError("overlap method must be reference or bbox_exact")
    import re
    import numpy as np
    import torch
    from dataclasses import replace
    from .segmentation_measurements import measure_predicted_mask_set
    from mora3.model.color_semantic_inference import (
        image_artifact_digest, image_condition_graph)
    targets = tuple(requested_targets)
    if (not targets or any(not isinstance(k, str) or not k.strip() or
            k in ('image_context', 'predicted_geometry') for k in targets)
            or len(set(targets)) != len(targets)):
        raise ValueError('distinct nonempty requested target names required')
    if (not isinstance(prediction, dict)
            or prediction.get('threshold') != 'independent-logit-ge0-v1'
            or prediction.get('quality_accepted') is not None
            or prediction.get('training_admission') is not False
            or prediction.get('biological_validation') is not False):
        raise ValueError('bound unevaluated native instance prediction required')
    binding = prediction['binding']
    if image_artifact_digest(image) != binding['image_sha256']:
        raise ValueError('image binding mismatch')
    for key in ('model_sha256', 'pixel_sha256'):
        if not isinstance(binding[key], str) or not re.fullmatch('[0-9a-f]{64}', binding[key]):
            raise ValueError('explicit inference digests required')
    base, base_task = image_condition_graph(image, pixel_sha256=binding['pixel_sha256'])
    for key, value in (('graph_sha256', base), ('task_sha256', base_task)):
        if hashlib.sha256(value.canonical_bytes()).hexdigest() != binding[key]:
            raise ValueError('inference context binding mismatch')
    masks = prediction['masks']
    if (not isinstance(masks, np.ndarray) or masks.dtype != np.bool_
            or not masks.flags.c_contiguous):
        raise ValueError('contiguous boolean prediction planes required')
    if (type(max_mask_bytes) is not int or max_mask_bytes < 1
            or masks.nbytes > max_mask_bytes):
        raise ValueError('mask input-copy byte policy exceeded')
    mask_sha = hashlib.sha256(masks.tobytes()).hexdigest()
    if mask_sha != prediction['mask_sha256']:
        raise ValueError('mask binding mismatch')
    artifact = prediction['measurements']['artifact']
    if type(artifact.seed) is not int or artifact.seed < 0:
        raise ValueError('explicit inference seed required')
    expected = measure_predicted_mask_set(torch.from_numpy(masks.copy()), image,
        checkpoint='native-state-sha256:' + binding['model_sha256'],
        mask_asset=artifact.asset, max_mask_bytes=max_mask_bytes,
        overlap_method=overlap_method)
    expected['artifact'] = replace(expected['artifact'], seed=artifact.seed,
        artifact_id=image.artifact_id+':predicted_slot_objects:'+mask_sha,
        quality_flags=expected['artifact'].quality_flags+('quality_unevaluated',))
    if prediction['measurements'] != expected:
        raise ValueError('prediction geometry/provenance mismatch')
    nodes = [Node(base.nodes[0].kind, tuple(m for m in base.nodes[0].measurements
                                          if m.name != 'nuclear_mask'))]
    edges = []
    mapping = {}
    for obj in expected['objects']:
        fields = [(k, obj[k], u) for k, u in (
            ('area_px','pixel^2'), ('perimeter_px','pixel'), ('area_um2','um^2'))]
        fields += [(k,v,'pixel') for k,v in zip(
            ('centroid_y','centroid_x'),obj['centroid_yx'])]
        fields += [(k,v,'pixel') for k,v in zip(
            ('bbox_y_min','bbox_x_min','bbox_y_max','bbox_x_max'),obj['bbox_yxyx'])]
        idx = len(nodes); mapping[obj['object_id']] = idx
        nodes.append(Node('predicted_nuclear_slot_geometry',tuple(
            Measurement(k,v,u,'predicted' if v is not None else 'missing',v is not None)
            for k,v,u in fields)))
        edges.append(Edge(0,idx,'contains_predicted_object',True))
    for edge in expected['edges']:
        for a,b in ((edge['source'],edge['target']), (edge['target'],edge['source'])):
            edges.append(Edge(mapping[a],mapping[b],'pixel_membership_overlap',True,
                (Measurement('overlap_px',edge['overlap_px'],'pixel^2','predicted',True),)))
    nodes[0] = Node(nodes[0].kind,nodes[0].measurements+tuple(
        Measurement('requested:'+k,None,None,'missing',False) for k in targets))
    graph = GraphRecord('bound-slot-conditions-v1:'+binding['image_sha256']+':'+mask_sha,
        tuple(nodes),tuple(edges),tuple(provenance),base.leakage_keys)
    roles = tuple(((kind,i,m.name),'target' if m.name.startswith('requested:')
        else 'condition' if m.available else 'unused')
        for kind,entities in (('node',graph.nodes),('edge',graph.edges))
        for i,entity in enumerate(entities) for m in entity.measurements)
    task = ConditionalTask(roles,topology_known=True)
    task.input_fields(graph)
    return dict(input_graph=graph,task=task,slot_node_indices=mapping,
        empty_slot_ids=list(expected['empty_slot_ids']),
        modality_masks=dict(image_context=True,predicted_geometry=True,
                            **{k:False for k in targets}),
        missing_context={image.artifact_id:tuple(k for k,v in image.context if v is None)},
        artifacts=[asdict(image),asdict(artifact)],biological_validation=False,
        training_admission=False,quality_accepted=None)
