"""Read-only desired log1p-count interval assessment of native sampled responses.

Inclusive bounds apply to explicit requested uniquely named genes only.
Repeated native names retain every axis coordinate; explicit zero-based ordinal
and exact-name requests disambiguate them. Unrequested and unknown
constraints are never zeros or successes. This measures model/desired consistency,
not observed accuracy, a causal effect, or a calibrated therapeutic reward.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import torch
from .response_sampling import ResponseContext


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class DesiredExpression:
    context: ResponseContext
    intervals: tuple

    def __post_init__(self):
        if (type(self.context) is not ResponseContext
                or type(self.intervals) is not tuple or not self.intervals):
            raise ValueError('ordered source axis and immutable requested intervals required')
        seen = set()
        for entry in self.intervals:
            if type(entry) is not tuple or len(entry) != 3:
                raise ValueError('interval must be an immutable gene/lower/upper triple')
            gene, lo, hi = entry
            if not isinstance(gene, str) or self.context.axis_ids.count(gene) != 1 or gene in seen:
                raise ValueError('unique unambiguous requested source gene required')
            seen.add(gene)
            if lo is None and hi is None:
                continue
            if (any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                    for x in (lo, hi)) or lo > hi):
                raise ValueError('finite nonnegative inclusive log1p-count bounds or explicit unknown required')


@dataclass(frozen=True)
class DesiredExpressionOrdinal:
    """Immutable (zero-based source ordinal, exact gene name, lower, upper)."""
    context: ResponseContext
    intervals: tuple

    def __post_init__(self):
        if (type(self.context) is not ResponseContext
                or type(self.intervals) is not tuple or not self.intervals):
            raise ValueError('ordered source axis and immutable requested intervals required')
        seen = set()
        for entry in self.intervals:
            if type(entry) is not tuple or len(entry) != 4:
                raise ValueError('ordinal/name/lower/upper tuple required')
            ordinal, gene, lo, hi = entry
            if (type(ordinal) is not int or not 0 <= ordinal < len(self.context.axis_ids)
                    or type(gene) is not str or self.context.axis_ids[ordinal] != gene
                    or ordinal in seen):
                raise ValueError('unique source ordinal and matching exact gene name required')
            seen.add(ordinal)
            if lo is None and hi is None:
                continue
            if (any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                    for x in (lo, hi)) or lo > hi):
                raise ValueError('finite nonnegative inclusive log1p-count bounds or explicit unknown required')


@torch.no_grad()
def assess_desired_expression(sample, requests):
    """Assess detached FP64 predictions per record, without pooling contexts.

    Sampler configuration binding is checked, not a signature of model weights
    or predictions. The report separately hashes the actual prediction bytes.
    Callers own authenticating the model/source/checkpoint and admission. No
    measured targets or model calls are accepted by this interface.
    """
    requests = tuple(requests)
    if not requests or any(type(d) not in (DesiredExpression, DesiredExpressionOrdinal) for d in requests):
        raise ValueError('explicit desired requests required')
    try:
        cfg = sample['configuration']
        p = sample['log_expression']
        axis = requests[0].context.axis_ids
        if (sample['binding'] != _digest(cfg)
                or cfg['version'] != 1 or cfg['model']['space'] != 'log1p-counts'
                or cfg['model']['native']['transform'] != 'log1p-counts'
                or tuple(cfg['model']['native']['axis_ids']) != axis
                or cfg['solver'] != 'fixed-euler'
                or cfg['negative_policy'] not in ('preserve', 'clamp-zero')
                or _digest(cfg['contexts']) != _digest([asdict(d.context) for d in requests])
                or any(d.context.axis_ids != axis for d in requests)
                or not isinstance(p, torch.Tensor) or not p.is_floating_point()
                or p.ndim != 2 or tuple(p.shape) != (len(requests),len(axis))
                or not torch.isfinite(p).all()):
            raise ValueError('sample context, source, axis, space or prediction mismatch')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError('native bound sample required') from exc
    # CPU FP64 gives explicit reduction/rounding semantics across device/dtype.
    prediction = p.detach().to(device='cpu', dtype=torch.float64).contiguous()
    indices = {gene:i for i,gene in enumerate(axis)}
    rows = []
    for row, desired in zip(prediction, requests):
        coordinates = (desired.intervals if type(desired) is DesiredExpressionOrdinal else
                       tuple((indices[g], g, lo, hi) for g, lo, hi in desired.intervals))
        available_coordinates = [(i, g, lo, hi) for i,g,lo,hi in coordinates if lo is not None]
        available = [(g,lo,hi) for _,g,lo,hi in available_coordinates]
        values = [float(row[i]) for i,_,_,_ in available_coordinates]
        violations = [max(lo-v, 0., v-hi) for v,(_,lo,hi) in zip(values, available)]
        try:
            penalty = math.fsum(v*v for v in violations)/len(violations) if violations else None
        except OverflowError as exc:
            raise ValueError('nonfinite interval penalty') from exc
        if penalty is not None and not math.isfinite(penalty):
            raise ValueError('nonfinite interval penalty')
        rows.append(dict(gene_ids=[g for g,_,_ in available],
            squared_violation_mean=penalty, satisfied_count=sum(v==0 for v in violations),
            assessed_count=len(available),
            unavailable_requested_count=len(desired.intervals)-len(available),
            unrequested_gene_count=len(axis)-len(desired.intervals)))
        if type(desired) is DesiredExpressionOrdinal:
            rows[-1]['axis_ordinals'] = [i for i,_,_,_ in available_coordinates]
    return dict(version=1, space='log1p-counts', bounds='inclusive',
        evidence_role='model_prediction_desired_consistency', desired_role='user_requested',
        observed_label_count=0, sample_binding=sample['binding'],
        request_sha256=_digest([asdict(d) for d in requests]),
        prediction_sha256=hashlib.sha256(prediction.numpy().tobytes()).hexdigest(),
        reduction='per-record-available-genes-cpu-float64', per_record=rows,
        full_dataset_ready=False, training_admission=False)
