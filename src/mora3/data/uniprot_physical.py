"""Scoped source-observed physical targets; no numeric or generation admission.

Consumes source-validated AnnotationPredictionView, never unvalidated corpus rows.
Units present in JSON are explicit; absent units remain unknown. Whole parent
comments survive projection so isoform and processed-form context is not lost.
"""
from dataclasses import dataclass
import json
from .uniprot_annotations import AnnotationPredictionView, _canonical


@dataclass(frozen=True)
class PhysicalTarget:
    parameter_type: str
    source_pointer: str
    source_bytes: bytes
    parent_bytes: bytes
    value: object
    unit: str | None
    molecule_scope: str | None
    method: str | None = None
    error: object = None
    numeric_objective_eligible: bool = False
    assay_pH: None = None
    assay_temperature: None = None
    converted_to_kcat: bool = False

    def source(self):
        return json.loads(self.source_bytes)

    def parent(self):
        return json.loads(self.parent_bytes)

    @property
    def unit_status(self):
        return 'explicit_source_unit' if self.unit else 'unknown'

    @property
    def canonical_property_supervision(self):
        # Scope compatibility only; this does not confer numeric admission.
        return self.molecule_scope is None

    def numeric_target(self):
        raise ValueError('physical numeric objective not admitted: unit/context review required')


def physical_targets(view):
    """Lossless assertion and parent projections, retaining missing-not-negative.

    No sibling pH/temperature text is interpreted as an assay condition. Vmax
    stays maximumVelocity; no enzyme concentration or kcat is inferred. Provider
    computed sequence descriptors remain in the original view, not assay targets.
    """
    if not isinstance(view, AnnotationPredictionView):
        raise TypeError('source-validated prediction view required')
    rows = []
    for i, parent in enumerate(view.targets()['annotation_packet'].get('comments', [])):
        pointer = f'/comments/{i}'
        scope = parent.get('molecule')
        def add(kind, source, path, value=None, unit=None, method=None, error=None):
            rows.append(PhysicalTarget(kind, path, _canonical(source), _canonical(parent),
                                       value, unit or None, scope, method, error))
        if parent.get('commentType') == 'MASS SPECTROMETRY':
            add('mass_spectrometry_source_mass', parent, pointer, parent.get('molWeight'),
                parent.get('unit'), parent.get('method'), parent.get('molWeightError'))
        elif parent.get('commentType') == 'BIOPHYSICOCHEMICAL PROPERTIES':
            for key, source in parent.items():
                if key in ('commentType', 'molecule'):
                    continue
                if key == 'kineticParameters':
                    for array, kind, field in (
                        ('michaelisConstants', 'MichaelisConstant', 'constant'),
                        ('maximumVelocities', 'maximumVelocity', 'velocity')):
                        for j, assertion in enumerate(source.get(array, [])):
                            add(kind, assertion, f'{pointer}/{key}/{array}/{j}',
                                assertion.get(field), assertion.get('unit'))
                    add('kineticParameters_container', source, f'{pointer}/{key}')
                else:
                    add(key, source, f'{pointer}/{key}')
    return tuple(rows)


@dataclass(frozen=True)
class PhysicalBatch:
    """Structured prediction targets, separate from sequence-only model inputs.

    Observation means a source assertion exists, never a negative or numeric
    admission. Scoped assertions remain entry-annotation targets, not canonical
    protein properties. Bytes prevent callers mutating a checkpoint binding.
    """
    parameter_axis: tuple[str, ...]
    observed: tuple[tuple[bool, ...], ...]
    input_bytes: bytes
    target_bytes: bytes
    binding_sha256: str

    def inputs(self):
        return tuple(json.loads(self.input_bytes))

    def targets(self):
        return json.loads(self.target_bytes)


def physical_batch(views, *, parameter_axis, assertions=None):
    """Bind a bounded ordered batch to complete source views and exact parents.

    Optional cached assertions are checked against fresh source projections;
    dataclass construction or replacement alone cannot confer applicability.
    No truncation, unit conversion, cross-entry or canonical/form reassignment.
    """
    import hashlib
    from dataclasses import asdict
    views = tuple(views)
    axis = tuple(parameter_axis)
    if not views:
        raise ValueError('nonempty physical batch required')
    if not axis or any(not isinstance(x, str) or not x for x in axis) or len(set(axis)) != len(axis):
        raise ValueError('nonempty unique parameter axis required')
    projected = tuple(physical_targets(v) for v in views)
    if assertions is not None and tuple(tuple(r) for r in assertions) != projected:
        raise ValueError('physical assertions must be exact source-bound projections')
    targets, observed, bindings = [], [], []
    for v, rows in zip(views, projected):
        kinds = {r.parameter_type for r in rows}
        if not kinds <= set(axis):
            raise ValueError('parameter axis omits source targets')
        observed.append(tuple(k in kinds for k in axis))
        payload = []
        for r in rows:
            d = asdict(r)
            d.pop('source_bytes'); d.pop('parent_bytes')
            d.update(source=r.source(), parent=r.parent(), unit_status=r.unit_status,
                     canonical_property_supervision=r.canonical_property_supervision,
                     source_sha256=hashlib.sha256(r.source_bytes).hexdigest(),
                     parent_sha256=hashlib.sha256(r.parent_bytes).hexdigest())
            payload.append(d)
        targets.append(payload)
        bindings.append(dict(accession=v.accession, sequence=v.sequence,
                             targets_sha256=hashlib.sha256(v.target_bytes).hexdigest(),
                             provenance_sha256=hashlib.sha256(v.provenance_bytes).hexdigest()))
    inputs = _canonical([v.inputs() for v in views])
    target_bytes = _canonical(targets)
    digest = hashlib.sha256(_canonical(dict(schema='mora3.physical_batch.v1',
        axis=axis, observed=observed, views=bindings,
        targets_sha256=hashlib.sha256(target_bytes).hexdigest()))).hexdigest()
    return PhysicalBatch(axis, tuple(observed), inputs, target_bytes, digest)
