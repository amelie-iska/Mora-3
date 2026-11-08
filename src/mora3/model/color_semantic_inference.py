"""Source-observed color nuclear semantic inference; no target-mask interface.

Copies the supplied model for evaluation so caller modes, gradients and buffers
stay intact. State digest binds the actual weights/config, not a claimed training
history. No model quality, instance identity or training admission is inferred.
"""
import copy
from dataclasses import asdict, replace
import hashlib
import json
import re
import numpy as np
import torch
from mora3.data.cooperative_segmentation import SegmentationArtifact
from .color_input import color_tensor
from .color_semantic import ColorNuclearSemantic
from .color_conditioning import ConditionedColorSemantic
from mora3.data.schema import GraphRecord, Node, Measurement
from mora3.data.provenance import Provenance, LeakageKey
from mora3.data.tasks import ConditionalTask


def model_digest(model):
    """Canonical actual tensor/config digest; device-independent, dtype-sensitive."""
    if not isinstance(model,(ColorNuclearSemantic,ConditionedColorSemantic)):
        raise ValueError('native color nuclear model required')
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        h.update(json.dumps(name).encode())
        if torch.is_tensor(value):
            h.update(json.dumps([str(value.dtype),list(value.shape)]).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:h.update(json.dumps(value,sort_keys=True).encode())
    return h.hexdigest()


def _require_source_image(image):
    if (not isinstance(image,SegmentationArtifact) or image.kind!='image'
        or image.evidence!='source_observed' or image.role!='condition'
        or image.channels not in (('red','green','blue'),('red','green','blue','alpha'))
        or image.reference_frame!='source_pixel_yx'
        or image.representation!='source-yx-no-transform-v1'
        or dict(image.context).get('object_type')!='nucleus'):
        raise ValueError('source-observed aligned nuclear RGB condition required')


def image_condition_graph(image, *, pixel_sha256):
    """Versioned image-only conditions; no user graph, mask or cross-source join.

    Dimensions/physical scale come exclusively from the validated image artifact.
    Identities and pixel hash are audit-only. Source terms remain unspecified in
    this pixel-level receipt; it is not an archive/license/admission verifier.
    Other biological context and all object geometry remain unsupported here.
    """
    _require_source_image(image)
    if not isinstance(pixel_sha256,str) or re.fullmatch('[0-9a-f]{64}',pixel_sha256) is None:
        raise ValueError('explicit pixel sha256 required')
    fields=(Measurement('height',image.spatial_shape[0],'pixel','observed',True),
            Measurement('width',image.spatial_shape[1],'pixel','observed',True),
            Measurement('pixel_size_um',image.pixel_size_um,'um/pixel',
                        'missing' if image.pixel_size_um is None else 'observed',
                        image.pixel_size_um is not None),
            Measurement('nuclear_mask',None,None,'missing',False))
    graph=GraphRecord('color-image-only-v1:'+image.artifact_id,
        (Node('microscopy_image_context',fields),),(),
        (Provenance(image.source_id,image.representation,image.artifact_id,
                    pixel_sha256,'unspecified-no-admission'),),
        (LeakageKey(image.source_id+':parent-v1',image.parent_id),))
    task=ConditionalTask(tuple((('node',0,m.name),
        'target' if m.name=='nuclear_mask' else 'condition' if m.available else 'unused')
        for m in fields),topology_known=True)
    return graph,task


# Closed source-context vocabulary: no output geometry, response, or IDs become
# features. These are source assertions, not independently verified biology.
_SOURCE_CONTEXT_FIELDS = frozenset(('organism', 'tissue', 'cell_type', 'stain'))


def image_artifact_digest(image):
    """Bind complete source artifact, including parent/context; not authentication."""
    _require_source_image(image)
    return hashlib.sha256(json.dumps(asdict(image), sort_keys=True,
        separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def selected_image_condition_graph(image, *, pixel_sha256, context_fields,
                                   expected_image_sha256):
    """Select only exact source-image context; absent values remain unused.

    No caller values, graph joins, desired outcomes, or target/geometry proxies
    are accepted. Context keys are a closed vocabulary. The caller must retain
    the source adapter's rights/lineage evidence; hashes do not grant admission.
    """
    if image_artifact_digest(image) != expected_image_sha256:
        raise ValueError('image artifact binding mismatch')
    if (not isinstance(context_fields, (tuple, list)) or not context_fields
        or any(not isinstance(k, str) or k not in _SOURCE_CONTEXT_FIELDS for k in context_fields)
        or len(set(context_fields)) != len(context_fields)):
        raise ValueError('explicit unique supported source context fields required')
    graph, task = image_condition_graph(image, pixel_sha256=pixel_sha256)
    source = dict(image.context)
    additions = tuple(Measurement('source_context:'+key, source.get(key), None,
        'observed' if source.get(key) is not None else 'missing',
        source.get(key) is not None) for key in sorted(context_fields))
    graph = replace(graph, record_id='color-selected-context-v1:'+image.artifact_id,
        nodes=(Node('microscopy_image_context', graph.nodes[0].measurements+additions),))
    task = replace(task, roles=task.roles+tuple((('node',0,m.name),
        'condition' if m.available else 'unused') for m in additions))
    task.input_fields(graph)
    return graph, task


def desired_image_condition_graph(image, *, pixel_sha256, desired_context,
                                  context_fields=None):
    """Explicit desired task roles; observed source assertions stay separate.

    Measurement status describes the observed caller request text, not an
    observed biological outcome. The task's desired role and field namespace
    carry that distinction into the encoder. Identity/origin stay audit-only.
    """
    from .color_desired_context import DesiredColorContext
    if not isinstance(desired_context, DesiredColorContext):
        raise ValueError('typed desired context required')
    if desired_context.image_sha256 != image_artifact_digest(image):
        raise ValueError('desired image artifact binding mismatch')
    if context_fields is None:
        graph, task = image_condition_graph(image, pixel_sha256=pixel_sha256)
    else:
        graph, task = selected_image_condition_graph(image,
            pixel_sha256=pixel_sha256, context_fields=context_fields,
            expected_image_sha256=desired_context.image_sha256)
    additions = tuple(Measurement('desired_context:'+key, value, None,
                                  'observed', True)
                      for key, value in desired_context.values)
    graph = replace(graph, record_id='color-explicit-desired-context-v1:'+image.artifact_id,
        nodes=(Node('microscopy_image_context', graph.nodes[0].measurements+additions),))
    task = replace(task, roles=task.roles+tuple((('node',0,m.name),'desired')
                                               for m in additions))
    task.input_fields(graph)
    return graph, task


def infer_nuclear_semantic(model,image,pixels,*,expected_pixel_sha256,
                           expected_model_sha256,mask_asset,seed,
                           context_fields=None,expected_image_sha256=None,
                           desired_context=None):
    """Predict threshold-zero boolean nuclei mask in the unchanged source frame.

    Observed pixels may carry an explicitly bound caller desired request.
    Desired text is not measured biology. Generated/predicted image
    adapters and object measurements remain separate, unsupported directions.
    Caller persists the returned mask to the declared new asset.
    """
    _require_source_image(image)
    if (not isinstance(mask_asset,str) or not mask_asset.strip() or mask_asset==image.asset
        or type(seed) is not int or seed<0):
        raise ValueError('new prediction asset and explicit seed required')
    # Validate bounded native pixels before copying/evaluating any model.
    if not isinstance(model,(ColorNuclearSemantic,ConditionedColorSemantic)):
        raise ValueError('native color nuclear model required')
    native=model.color if isinstance(model,ConditionedColorSemantic) else model
    color_tensor(pixels,alpha_policy='require-opaque',input_policy=native.input_policy)
    if (pixels.shape[:2]!=image.spatial_shape or pixels.shape[2]!=len(image.channels)
        or hashlib.sha256(pixels.tobytes()).hexdigest()!=expected_pixel_sha256):
        raise ValueError('image shape/channel/pixel binding mismatch')
    digest=model_digest(model)
    if digest!=expected_model_sha256:raise ValueError('model state binding mismatch')
    conditioned=isinstance(model,ConditionedColorSemantic)
    binding=None
    selected=context_fields is not None
    if desired_context is not None and not conditioned:
        raise ValueError('desired context requires conditioned native model')
    if selected and not conditioned:
        raise ValueError('selected context requires conditioned native model')
    if desired_context is not None and expected_image_sha256 != image_artifact_digest(image):
        raise ValueError('desired image artifact binding mismatch')
    if not selected and desired_context is None and expected_image_sha256 is not None:
        raise ValueError('artifact binding requires explicit context selection')
    if conditioned:
        if desired_context is not None:
            graph,task=desired_image_condition_graph(image,
                pixel_sha256=expected_pixel_sha256,desired_context=desired_context,
                context_fields=context_fields)
        elif selected:
            graph,task=selected_image_condition_graph(image,
                pixel_sha256=expected_pixel_sha256,context_fields=context_fields,
                expected_image_sha256=expected_image_sha256)
        else:
            graph,task=image_condition_graph(image,pixel_sha256=expected_pixel_sha256)
        binding=dict(policy='source-observed-image-only-v1',
                     graph_sha256=hashlib.sha256(graph.canonical_bytes()).hexdigest(),
                     task_sha256=hashlib.sha256(task.canonical_bytes()).hexdigest(),
                     pixel_sha256=expected_pixel_sha256,training_admission=False,
                     biological_pairing=False)
    if selected:
        binding.update(policy='source-observed-selected-context-v1',
            image_artifact_sha256=expected_image_sha256,
            context_fields=tuple(sorted(context_fields)),
            context_evidence='source_asserted_not_independently_verified')
    if desired_context is not None:
        binding.update(policy='source-image-explicit-desired-context-v1',
            image_artifact_sha256=expected_image_sha256,
            desired_request_sha256=hashlib.sha256(desired_context.canonical_bytes()).hexdigest(),
            desired_request=json.loads(desired_context.canonical_bytes()),
            desired_evidence='caller_declared_not_observed_biology')
    evaluator=copy.deepcopy(model).eval()
    with torch.inference_mode():
        logits=evaluator(pixels,graph,task) if conditioned else evaluator(pixels)
        if not torch.isfinite(logits).all():raise ValueError('nonfinite native prediction')
        mask=(logits>=0).cpu().numpy().copy()
    mask.setflags(write=False)
    artifact=SegmentationArtifact(
        artifact_id=image.artifact_id+':nuclear-semantic:'+hashlib.sha256(mask.tobytes()).hexdigest(),
        kind='mask',evidence='predicted_mask',role='evaluation',source_id=image.source_id,
        parent_id=image.parent_id,context=image.context,asset=mask_asset,
        representation=('nuclear-semantic-explicit-desired-context-logit-ge0-v1' if desired_context is not None else
                        'nuclear-semantic-selected-context-logit-ge0-v1' if selected else
                        'nuclear-semantic-image-only-conditioned-logit-ge0-v1' if conditioned
                        else 'nuclear-semantic-logit-ge0-v1'),spatial_shape=image.spatial_shape,
        links=(('mask_of',image.artifact_id),),reference_frame=image.reference_frame,
        pixel_size_um=image.pixel_size_um,checkpoint='native-state-sha256:'+digest,seed=seed,
        uncertainty='uncalibrated',quality_flags=('quality_unevaluated','semantic_not_instance'))
    result=dict(status='predicted',mask=mask,artifact=artifact,instance_count=None,
                quality_accepted=None,biological_validation=False,
                foreground_area_px2=int(np.count_nonzero(mask)))

    if binding is not None:result['conditioning']=binding
    return result


def desired_color_context_request(image, *, expected_image_sha256, request_id,
                                  origin, values):
    """Bind a desired request to the exact observed image without modifying it.

    Origin is a caller declaration, not authenticated evidence. Desired values
    are not source annotations, target measurements or approved model inputs.
    Inference attachment requires a separate explicit versioned interface.
    """
    from .color_desired_context import DesiredColorContext
    if image_artifact_digest(image) != expected_image_sha256:
        raise ValueError('image artifact binding mismatch')
    return DesiredColorContext(expected_image_sha256, request_id, origin, values)
