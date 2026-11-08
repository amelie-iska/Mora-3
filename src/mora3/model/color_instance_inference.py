"""Bound source-observed native instance inference; no truth mask inputs.

Returned independent boolean planes must be persisted by the caller at mask_asset
(e.g. NumPy save with allow_pickle=False), retaining shape/dtype/raw-byte digest.
The asset is a declaration, not a verified file. Hashes bind state and metadata,
not training history, source rights or biological validity. No calibrated count.
"""
import copy
import hashlib
import json
import torch
from mora3.data.segmentation_measurements import measure_predicted_mask_set
from .color_conditioning import ConditionedColorInstance
from .color_semantic_inference import (
    _require_source_image, image_artifact_digest, image_condition_graph)
from .color_input import color_tensor


def instance_model_digest(model):
    """Device-independent digest of exact native conditioned state/configuration."""
    if type(model) is not ConditionedColorInstance:
        raise ValueError('conditioned native color instance model required')
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(json.dumps(name).encode())
        if torch.is_tensor(value):
            h.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    return h.hexdigest()


def infer_nuclear_instances(model, image, pixels, *, expected_pixel_sha256,
        expected_model_sha256, expected_image_sha256, mask_asset, seed,
        max_mask_bytes, overlap_method="reference"):
    """Threshold logits >=0 independently; retain overlaps, duplicates and empties.

    Image-only shared context excludes source masks and target-derived geometry.
    Deep-copy evaluation preserves caller modes, buffers and gradients. Byte gates
    bound output/boolean copy, not total peak memory; pair geometry is quadratic.
    Seed records provenance only; deterministic evaluation consumes no random seed.
    """
    if type(overlap_method) is not str or overlap_method not in ("reference", "bbox_exact"):
        raise ValueError("overlap method must be reference or bbox_exact")
    _require_source_image(image)
    if image_artifact_digest(image) != expected_image_sha256:
        raise ValueError('image artifact binding mismatch')
    if (not isinstance(mask_asset, str) or not mask_asset.strip()
            or mask_asset == image.asset or type(seed) is not int or seed < 0):
        raise ValueError('new prediction asset and explicit seed required')
    if type(model) is not ConditionedColorInstance:
        raise ValueError('conditioned native color instance model required')
    head = model.color.head
    count = head.slots * image.spatial_shape[0] * image.spatial_shape[1]
    if type(max_mask_bytes) is not int or max_mask_bytes < count:
        raise ValueError('mask input-copy byte policy exceeded')
    if count * head.projection.weight.element_size() > head.max_output_bytes:
        raise ValueError('instance output-byte policy exceeded')
    color_tensor(pixels, alpha_policy=model.color.alpha_policy,
                 input_policy=model.color.input_policy)
    if (pixels.shape[:2] != image.spatial_shape
            or pixels.shape[2] != len(image.channels)
            or hashlib.sha256(pixels.tobytes()).hexdigest() != expected_pixel_sha256):
        raise ValueError('image shape/channel/pixel binding mismatch')
    digest = instance_model_digest(model)
    if digest != expected_model_sha256:
        raise ValueError('model state binding mismatch')
    graph, task = image_condition_graph(image, pixel_sha256=expected_pixel_sha256)
    evaluator = copy.deepcopy(model).eval()
    with torch.inference_mode():
        logits = evaluator(pixels, graph, task)
        if not torch.isfinite(logits).all():
            raise ValueError('nonfinite native prediction')
        masks = (logits >= 0).cpu().numpy().copy()
    mask_sha = hashlib.sha256(masks.tobytes()).hexdigest()
    measurements = measure_predicted_mask_set(torch.from_numpy(masks), image,
        checkpoint='native-state-sha256:' + digest, mask_asset=mask_asset,
        max_mask_bytes=max_mask_bytes, overlap_method=overlap_method)
    from dataclasses import replace
    measurements['artifact'] = replace(measurements['artifact'], seed=seed,
        artifact_id=image.artifact_id+':predicted_slot_objects:'+mask_sha,
        quality_flags=measurements['artifact'].quality_flags+('quality_unevaluated',))
    masks.setflags(write=False)
    return dict(masks=masks, mask_sha256=mask_sha, measurements=measurements,
        threshold='independent-logit-ge0-v1', quality_accepted=None,
        training_admission=False, biological_validation=False,
        binding=dict(image_sha256=expected_image_sha256,
            pixel_sha256=expected_pixel_sha256, model_sha256=digest,
            graph_sha256=hashlib.sha256(graph.canonical_bytes()).hexdigest(),
            task_sha256=hashlib.sha256(task.canonical_bytes()).hexdigest()))
