"""Lossless residue assertions from source-validated entry prediction views.

Only exact canonical bounds support coordinate supervision. Isoform/fuzzy targets
remain scoped entry annotations. Unannotated positions are never negatives.
This module admits neither sequence generation nor experimental truth labels.
"""
from dataclasses import dataclass
import hashlib
import json
from .uniprot_annotations import AnnotationPredictionView, _canonical


@dataclass(frozen=True)
class ResidueTarget:
    feature_type: str | None
    source_pointer: str
    source_bytes: bytes
    coordinate_status: str
    interval0: tuple[int, int] | None
    endpoint_indices0: tuple[int, int] | None

    def source(self):
        return json.loads(self.source_bytes)


def residue_targets(view):
    """Retain complete features/evidence while projecting safe 0-based geometry.

    UniProt inclusive 1-based intervals become half-open intervals; bonds retain
    two endpoints, never the intervening residues. No isoform alignment inferred.
    """
    if not isinstance(view, AnnotationPredictionView):
        raise TypeError('source-validated prediction view required')
    rows = []
    for i, f in enumerate(view.targets()['annotation_packet'].get('features', [])):
        loc = f.get('location', {})
        start, end = loc.get('start', {}), loc.get('end', {})
        a, b = start.get('value'), end.get('value')
        status = 'exact_canonical'
        if loc.get('sequence') not in (None, view.accession):
            status = 'unmapped_sequence_scope'
        elif (type(a) is not int or type(b) is not int or
              start.get('modifier') != 'EXACT' or end.get('modifier') != 'EXACT'):
            status = 'fuzzy_or_unknown'
        elif not 1 <= a <= b <= len(view.sequence):
            status = 'outside_canonical'
        elif (f.get('alternativeSequence', {}).get('originalSequence') is not None and
              f['alternativeSequence']['originalSequence'] != view.sequence[a-1:b]):
            status = 'reference_sequence_mismatch'
        interval = endpoints = None
        if status == 'exact_canonical':
            if f.get('type') in ('Disulfide bond', 'Cross-link'):
                endpoints = (a-1, b-1)
            else:
                interval = (a-1, b)
        rows.append(ResidueTarget(f.get('type'), f'/features/{i}', _canonical(f),
                                  status, interval, endpoints))
    return tuple(rows)


@dataclass(frozen=True)
class ResidueBatch:
    feature_axis: tuple[str, ...]
    observed: tuple[tuple[bool, ...], ...]
    input_bytes: bytes
    target_bytes: bytes
    binding_sha256: str
    missing_is_negative: bool = False

    def inputs(self):
        return tuple(json.loads(self.input_bytes))

    def targets(self):
        return json.loads(self.target_bytes)


def residue_batch(views, *, feature_axis, assertions=None):
    """Bind source/provenance, sequence, ordered axis and complete scoped targets.

    Observed flags indicate assertion availability only, not per-residue negative
    labels or canonical eligibility. Caller caches must equal fresh projections.
    """
    views, axis = tuple(views), tuple(feature_axis)
    if (not views or not axis or any(not isinstance(x, str) or not x for x in axis)
            or len(set(axis)) != len(axis)):
        raise ValueError('nonempty views and unique feature axis required')
    projected = tuple(residue_targets(v) for v in views)
    if assertions is not None and tuple(tuple(x) for x in assertions) != projected:
        raise ValueError('residue assertions must be exact source-bound projections')
    targets, observed, bindings = [], [], []
    for v, rows in zip(views, projected):
        kinds = {r.feature_type for r in rows}
        if not kinds <= set(axis):
            raise ValueError('feature axis omits source targets')
        observed.append(tuple(k in kinds for k in axis))
        targets.append([dict(feature_type=r.feature_type, source_pointer=r.source_pointer,
            source=r.source(), coordinate_status=r.coordinate_status,
            interval0=r.interval0, endpoint_indices0=r.endpoint_indices0) for r in rows])
        bindings.append(dict(accession=v.accession, sequence=v.sequence,
            targets_sha256=hashlib.sha256(v.target_bytes).hexdigest(),
            provenance_sha256=hashlib.sha256(v.provenance_bytes).hexdigest()))
    target_bytes = _canonical(targets)
    digest = hashlib.sha256(_canonical(dict(schema='mora3.residue_batch.v1',
        axis=axis, views=bindings, targets_sha256=hashlib.sha256(target_bytes).hexdigest()))).hexdigest()
    return ResidueBatch(axis, tuple(observed), _canonical([v.inputs() for v in views]),
                        target_bytes, digest)
