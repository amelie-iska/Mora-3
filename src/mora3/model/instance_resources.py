"""Allocation-free storage equations for the current mask-set BCE reference.

Counts named tensors, not allocator peaks, encoder/autograd memory, SciPy
workspace or runtime. FP64 matching uses matrix products, never Q*N*H*W.
An explicit extended policy accounts for up to 512 slots; defaults stay at 64.
"""


def instance_resource_preflight(*, instances, slots, height, width,
                                element_bytes, max_output_bytes,
                                slot_policy='bounded-max64-v1', pixel_block_size=None):
    if slot_policy not in ('bounded-max64-v1', 'extended-max512-v2'):
        raise ValueError('unknown slot policy')
    values = (instances, slots, height, width, element_bytes, max_output_bytes)
    if (any(type(v) is not int for v in values) or instances < 0
            or min(slots, height, width, max_output_bytes) < 1
            or element_bytes not in (2, 4, 8)):
        raise ValueError('integer counts/extents/budget and floating element size required')
    if pixel_block_size is not None and (type(pixel_block_size) is not int or pixel_block_size < 1):
        raise ValueError('pixel block size must be a positive integer')
    pixels = height * width
    output = slots * pixels * element_bytes
    refusals = []
    if slots > (512 if slot_policy == 'extended-max512-v2' else 64):
        refusals.append('slot_policy')
    if instances > slots:
        refusals.append('instance_capacity')
    if max(height, width) > 1000:
        refusals.append('extent_policy')
    if output > max_output_bytes:
        refusals.append('output_budget')
    report = dict(version='ordered-instance-storage-preflight-v1', instances=instances,
        slots=slots, height=height, width=width, element_bytes=element_bytes,
        max_output_bytes=max_output_bytes, logits_bytes=output,
        target_bool_bytes=instances * pixels,
        matching_logits_fp64_bytes=slots * pixels * 8,
        matching_targets_fp64_bytes=instances * pixels * 8,
        matching_cost_fp64_bytes=slots * slots * 8,
        assigned_target_bytes=output, pairwise_pixel_workspace_bytes=0,
        naive_pairwise_pixel_fp64_bytes=slots * instances * pixels * 8,
        refusals=tuple(refusals), current_reference_supported=not refusals,
        peak_memory_estimate=False, training_admission=False)

    if slot_policy == 'extended-max512-v2':
        report.update(version='ordered-instance-storage-preflight-v2', slot_policy=slot_policy)
    if pixel_block_size is not None:
        # Named extents only: loop temporaries, BCE/autograd, encoder, allocator
        # and CPU assignment workspaces prevent interpreting this as a peak.
        block = min(pixel_block_size, pixels)
        report.update(version='ordered-instance-blocked-storage-preflight-v1',
            slot_policy=slot_policy, pixel_block_size=pixel_block_size,
            matching_logits_fp64_bytes=slots * block * 8,
            matching_targets_fp64_bytes=instances * block * 8,
            matching_softplus_fp64_bytes=slots * block * 8)
    return report
