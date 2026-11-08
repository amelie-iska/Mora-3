"""Explicit scalar supervision dispatch for existing conditional graph tasks.

This layer neither admits sources nor joins records. A binding is the exact
record/task snapshot used by the caller's prediction path, not proof of rights
or independent proof of target-free inference. Existing input adapters enforce
that boundary. No automatic mixture weights or invented absent labels.
"""
import torch
from mora3.data.tasks import ConditionalTask, field_ref


def scalar_objectives(record, task, predictions, *, selected, binding):
    """Return per-target squared error and their mean, or None when absent.

    Numeric observed scalar targets only, in exactly matching native units.
    Selection is explicit (scheduler-owned); absent/unselected predictions are
    never read. Labels are immutable source scalars with no gradient. Gradients
    flow only through selected predictions and their upstream shared features.
    Discrete geometry remains conditioning, not differentiable supervision.
    """
    if not isinstance(task, ConditionalTask):
        raise ValueError('validated conditional task required')
    if binding != (record.canonical_bytes(), task.canonical_bytes()):
        raise ValueError('prediction record/task/context binding mismatch')
    fields = task._fields(record)
    refs = tuple(field_ref(r) for r in selected)
    roles = dict(task.roles)
    if len(set(refs)) != len(refs) or any(roles.get(r) != 'target' for r in refs):
        raise ValueError('distinct requested target subset required')
    terms = {}; missing = []
    for ref in refs:
        label = fields[ref]
        if not label.available:
            missing.append(ref)
            continue
        if label.status != 'observed' or type(label.value) not in (int, float):
            raise ValueError('observed numeric scalar supervision required')
        try:
            value, units = predictions[ref]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('prediction required for selected available target') from exc
        if (not isinstance(value, torch.Tensor) or value.ndim != 0
                or not value.is_floating_point() or not torch.isfinite(value)
                or units != label.units):
            raise ValueError('finite scalar prediction in exact target units required')
        terms[ref] = (value - value.new_tensor(label.value)).square()
    if terms:
        values = tuple(terms.values())
        if len({(x.device, x.dtype) for x in values}) != 1:
            raise ValueError('objective terms must share device and dtype')
        total = torch.stack(values).mean()
    else:
        total = None
    return dict(terms=terms, missing=tuple(missing), total=total)
