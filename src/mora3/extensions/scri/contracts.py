"""Native cell-state records, explicit pairing, and leakage-safe partitions.

No gene alignment, unit conversion, imputation, normalization, cross-study join,
or source admission is implicit. A source identifier or digest is an audit
coordinate, never a predictive feature or proof of biological validity.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import re


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def text(value, field):
    if type(value) is not str or not value.strip():
        raise ValueError(f'{field} must be nonempty text')
    value.encode('utf-8')


def digest(value, field):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError(f'{field} must be a SHA256 digest')


def number(value, field):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{field} must be a finite native number, not a boolean')


@dataclass(frozen=True)
class SourceEvidence:
    dataset_id: str
    revision: str
    artifact_sha256: str
    source_locator: str
    license_id: str
    evidence_kind: str
    study_id: str
    sample_id: str
    donor_id: str | None
    experiment_id: str
    lineage_id: str | None

    def __post_init__(self):
        for name in ('dataset_id','revision','source_locator','license_id',
                     'study_id','sample_id','experiment_id'):
            text(getattr(self,name),name)
        digest(self.artifact_sha256,'artifact_sha256')
        if self.evidence_kind not in ('experimental','simulation','model_prediction','synthetic_fixture'):
            raise ValueError('explicit experimental/simulation/prediction/fixture evidence required')
        for name in ('donor_id','lineage_id'):
            if getattr(self,name) is not None:
                text(getattr(self,name),name)


@dataclass(frozen=True)
class NativeValue:
    modality: str
    feature_id: str
    value: float | int | str | None
    unit: str
    observed: bool

    def __post_init__(self):
        for name in ('modality','feature_id','unit'):
            text(getattr(self,name),name)
        if type(self.observed) is not bool:
            raise ValueError('observed must be boolean')
        if self.observed != (self.value is not None):
            raise ValueError('missing measurements must remain None, never fabricated zero')
        if self.value is not None:
            if type(self.value) is str:
                text(self.value,'native category')
                if self.unit != 'category':
                    raise ValueError('string values require explicit category units; numeric units need numbers')
            else:
                number(self.value,'native value')
                if self.unit == 'count' and (self.value < 0 or int(self.value) != self.value):
                    raise ValueError('native count must be a nonnegative integer')


@dataclass(frozen=True)
class StateRecord:
    record_id: str
    source: SourceEvidence
    organism: str
    cell_type: str | None
    time_value: float | int | None
    time_unit: str | None
    time_reference: str | None
    measurements: tuple[NativeValue, ...]
    coordinates: tuple[float, ...] | None
    coordinate_unit: str | None
    coordinate_frame: str | None
    intervention_ids: tuple[str, ...]
    official_split: str | None

    def __post_init__(self):
        validate_record(self)

    def wire(self):
        return asdict(self)

    @classmethod
    def from_mapping(cls, data):
        if type(data) is not dict:
            raise ValueError('record must be a JSON object')
        d=dict(data)
        try:
            if (type(d['source']) is not dict or type(d['measurements']) is not list
                    or type(d['intervention_ids']) is not list
                    or d['coordinates'] is not None and type(d['coordinates']) is not list):
                raise ValueError('native JSON arrays and source object required; no implicit coercion')
            d['source']=SourceEvidence(**d['source'])
            d['measurements']=tuple(NativeValue(**x) for x in d['measurements'])
            d['intervention_ids']=tuple(d['intervention_ids'])
            d['coordinates']=None if d['coordinates'] is None else tuple(d['coordinates'])
            return cls(**d)
        except (TypeError,KeyError) as exc:
            raise ValueError('complete exact native record fields required') from exc


def validate_record(row):
    if type(row) is not StateRecord or type(row.source) is not SourceEvidence:
        raise ValueError('typed StateRecord and SourceEvidence required')
    row.source.__post_init__()
    for name in ('record_id','organism'):
        text(getattr(row,name),name)
    if row.cell_type is not None:
        text(row.cell_type,'cell_type')
    if (row.time_value is None) != (row.time_unit is None) or (row.time_value is None) != (row.time_reference is None):
        raise ValueError('time value, native unit and event reference must be present together')
    if row.time_value is not None:
        number(row.time_value,'time_value')
        text(row.time_unit,'time_unit');text(row.time_reference,'time_reference')
    if (type(row.measurements) is not tuple or not row.measurements
            or any(type(x) is not NativeValue for x in row.measurements)):
        raise ValueError('complete ordered native feature tuple required')
    for x in row.measurements:
        x.__post_init__()
    if len({(x.modality,x.feature_id) for x in row.measurements}) != len(row.measurements):
        raise ValueError('duplicate native modality/feature identity')
    if row.coordinates is None:
        if row.coordinate_unit is not None or row.coordinate_frame is not None:
            raise ValueError('missing spatial coordinates must remain explicit')
    else:
        if type(row.coordinates) is not tuple or len(row.coordinates) not in (2,3):
            raise ValueError('native 2D or 3D coordinates required')
        for v in row.coordinates:
            number(v,'coordinate')
        text(row.coordinate_unit,'coordinate_unit');text(row.coordinate_frame,'coordinate_frame')
    if type(row.intervention_ids) is not tuple or len(set(row.intervention_ids)) != len(row.intervention_ids):
        raise ValueError('explicit unique native intervention tuple required')
    for value in row.intervention_ids:
        text(value,'intervention_id')
    if row.official_split not in (None,'train','validation','test'):
        raise ValueError('unknown official split')
    return row


@dataclass(frozen=True)
class TransitionPair:
    baseline: StateRecord
    outcome: StateRecord
    pairing_kind: str
    pairing_evidence_sha256: str

    def __post_init__(self):
        a=validate_record(self.baseline);b=validate_record(self.outcome)
        digest(self.pairing_evidence_sha256,'pairing evidence')
        if self.pairing_kind not in ('same_cell','matched_population'):
            raise ValueError('unpaired atlases cannot supply paired transition supervision')
        if a.source.evidence_kind != b.source.evidence_kind:
            raise ValueError('experimental, simulated and predicted evidence cannot be paired implicitly')
        if (a.source.study_id,a.source.experiment_id,a.organism) != (b.source.study_id,b.source.experiment_id,b.organism):
            raise ValueError('source study, experiment and organism must match; no implicit cross-study join')
        if a.source.donor_id is None or a.source.donor_id != b.source.donor_id:
            raise ValueError('matched, known donor is required for a transition pair')
        if self.pairing_kind == 'same_cell' and (
                a.source.lineage_id is None or a.source.lineage_id != b.source.lineage_id):
            raise ValueError('same-cell longitudinal pairing requires explicit shared lineage evidence')
        if (a.time_value is None or b.time_value is None
                or (a.time_unit,a.time_reference) != (b.time_unit,b.time_reference)
                or b.time_value <= a.time_value):
            raise ValueError('strictly ordered times in one native unit/reference required')
        if a.official_split is not None and b.official_split is not None and a.official_split != b.official_split:
            raise ValueError('official split conflict within transition pair')


def compile_transition(pair):
    """Build target-free conditions plus separate native targets and audit.

Matched populations remain population-level conditional evidence; this never
asserts that a destructively assayed baseline cell became the outcome cell.
"""
    if type(pair) is not TransitionPair:
        raise ValueError('typed paired source evidence required')
    pair.__post_init__();a,b=pair.baseline,pair.outcome
    axis=lambda r: tuple((x.modality,x.feature_id,x.unit) for x in r.measurements)
    if axis(a) != axis(b):
        raise ValueError('exact full native feature axis/order/units required; align explicitly upstream')
    condition=dict(
        organism=a.organism,baseline_cell_type=a.cell_type,
        baseline=[asdict(x) for x in a.measurements],
        time_delta=b.time_value-a.time_value,time_unit=a.time_unit,
        time_reference=a.time_reference,interventions=list(b.intervention_ids),
        coordinates=a.coordinates,coordinate_unit=a.coordinate_unit,
        coordinate_frame=a.coordinate_frame,pairing_kind=pair.pairing_kind,
    )
    target=dict(axis=[list(x) for x in axis(b)],
                values=[x.value for x in b.measurements],
                observed=[x.observed for x in b.measurements],
                evidence_kind=b.source.evidence_kind,causal_claim=False)
    audit=dict(baseline_id=a.record_id,outcome_id=b.record_id,
               baseline_source=asdict(a.source),outcome_source=asdict(b.source),
               pairing_evidence_sha256=pair.pairing_evidence_sha256,
               original_value_verification='source-adapter-and-owner-admission-required',
               training_admitted=False)
    return dict(schema='Mora3.SCRITransitionTask/v1',condition=condition,target=target,audit=audit,
                condition_sha256=sha256(canonical(condition)).hexdigest())


_AXES={'study':'study_id','donor':'donor_id','sample':'sample_id',
       'experiment':'experiment_id','lineage':'lineage_id'}


def _keys(row, axes):
    keys=[]
    for axis in axes:
        if axis == 'intervention':
            if not row.intervention_ids:
                # Controls are one explicit shared intervention group. A cohort
                # requiring shared controls must be designed before partitioning.
                keys.append(('intervention','__explicit_control__'))
            else:
                keys.extend(('intervention',v) for v in row.intervention_ids)
        else:
            value=getattr(row.source,_AXES[axis])
            if value is None:
                raise ValueError(f'unknown {axis} cannot certify a {axis}-held-out split')
            # Dataset scope avoids accidentally joining unrelated local donor IDs.
            # Cross-dataset donor equivalence requires an explicit external link.
            scope='' if axis=='study' else row.source.study_id+'::'
            keys.append((axis,scope+value))
    # Repeated use of the exact raw observation is always a leakage edge.
    keys.append(('source_observation',row.source.artifact_sha256+'::'+row.source.source_locator))
    return tuple(keys)


def leakage_components(records, *, holdout_axes=('study','donor'), links=()):
    rows=tuple(records)
    if (not rows or type(holdout_axes) is not tuple or not holdout_axes
            or len(set(holdout_axes)) != len(holdout_axes)
            or not set(holdout_axes) <= set(_AXES)|{'intervention'}):
        raise ValueError('nonempty rows and explicit supported holdout axes required')
    by_id={validate_record(r).record_id:r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError('duplicate source record IDs')
    parent={key:key for key in by_id}
    def find(x):
        while parent[x] != x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a != b: parent[max(a,b)]=min(a,b)
    first={}
    for row in rows:
        for key in _keys(row,holdout_axes):
            if key in first: union(row.record_id,first[key])
            else: first[key]=row.record_id
    for link in links:
        if type(link) not in (tuple,list) or len(link)!=2 or any(x not in by_id for x in link):
            raise ValueError('cross-source links must name two present audited record IDs')
        union(*link)
    groups={}
    for key in sorted(by_id): groups.setdefault(find(key),[]).append(key)
    return tuple(sorted(tuple(v) for v in groups.values()))


def split_components(records, *, holdout_axes=('study','donor'), seed=20261108,
                     fractions=(0.8,0.1,0.1), links=()):
    rows=tuple(records)
    if type(seed) is not int or seed<0 or type(fractions) is not tuple or len(fractions)!=3:
        raise ValueError('explicit nonnegative seed and three split fractions required')
    for f in fractions: number(f,'split fraction')
    if any(f<=0 for f in fractions) or not math.isclose(sum(fractions),1.0,rel_tol=0,abs_tol=1e-12):
        raise ValueError('positive split fractions must sum to one')
    by_id={r.record_id:r for r in rows};result={}
    for group in leakage_components(rows,holdout_axes=holdout_axes,links=links):
        declared={by_id[k].official_split for k in group}-{None}
        if len(declared)>1: raise ValueError('official split conflict in a connected leakage component')
        value=int.from_bytes(sha256(canonical([seed,group])).digest()[:8],'big')/2**64
        chosen=next(iter(declared)) if declared else (
            'train' if value<fractions[0] else 'validation' if value<sum(fractions[:2]) else 'test')
        result.update({key:chosen for key in group})
    return dict(sorted(result.items()))


def validate_splits(records, assignments, *, holdout_axes=('study','donor'), links=()):
    rows=tuple(records)
    if (type(assignments) is not dict or set(assignments)!={r.record_id for r in rows}
            or not set(assignments.values())<={'train','validation','test'}):
        raise ValueError('exact complete source-to-split assignment required')
    for group in leakage_components(rows,holdout_axes=holdout_axes,links=links):
        if len({assignments[x] for x in group})!=1:
            raise ValueError('leakage component crosses split boundaries')
    for row in rows:
        if row.official_split is not None and row.official_split!=assignments[row.record_id]:
            raise ValueError('official split changed')
    return True
