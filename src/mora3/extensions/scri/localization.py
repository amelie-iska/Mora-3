"""Explicit spatial constraints and measured off-region outcome accounting.

A computational region restricts proposed actions. It is not evidence that a
drug, field, cell product, or biological effect stays inside that region.
"""
from dataclasses import dataclass

from .contracts import StateRecord,validate_record,text,number


@dataclass(frozen=True)
class Region:
    frame: str
    unit: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self):
        text(self.frame,'spatial frame');text(self.unit,'spatial unit')
        if (type(self.lower) is not tuple or type(self.upper) is not tuple
                or len(self.lower) not in (2,3) or len(self.lower)!=len(self.upper)):
            raise ValueError('matching native 2D or 3D region bounds required')
        for lo,hi in zip(self.lower,self.upper,strict=True):
            number(lo,'lower bound');number(hi,'upper bound')
            if lo>=hi: raise ValueError('region bounds must have positive extent')

    def contains(self,row):
        validate_record(row)
        if row.coordinates is None:
            raise ValueError('missing coordinates cannot establish spatial eligibility')
        if (row.coordinate_frame,row.coordinate_unit,len(row.coordinates)) != (self.frame,self.unit,len(self.lower)):
            raise ValueError('native coordinate frame, dimension and units must match; no implicit conversion')
        return all(lo<=v<=hi for lo,v,hi in zip(self.lower,row.coordinates,self.upper,strict=True))


@dataclass(frozen=True)
class ActionProposal:
    cell_id: str
    action_id: str

    def __post_init__(self):
        text(self.cell_id,'cell_id');text(self.action_id,'action_id')


def constrain_actions(records,region,proposals,*,admissible_actions,max_actions):
    """Reject the complete candidate if any proposed action violates its scope."""
    rows=tuple(records);proposals=tuple(proposals)
    if type(region) is not Region or type(max_actions) is not int or max_actions<1:
        raise ValueError('typed region and positive finite action budget required')
    region.__post_init__()
    if type(admissible_actions) is not tuple or not admissible_actions or len(set(admissible_actions))!=len(admissible_actions):
        raise ValueError('explicit finite unique admissible action vocabulary required')
    for action in admissible_actions: text(action,'admissible action')
    by_id={validate_record(r).record_id:r for r in rows}
    if not rows or len(by_id)!=len(rows): raise ValueError('unique nonempty native cell population required')
    if not proposals or len(proposals)>max_actions: raise ValueError('complete action proposal exceeds finite budget or is empty')
    if any(type(p) is not ActionProposal for p in proposals) or len(set(proposals))!=len(proposals):
        raise ValueError('unique typed action proposals required')
    for p in proposals:
        if p.cell_id not in by_id or p.action_id not in admissible_actions:
            raise ValueError('unknown cell or unsupported intervention action')
        if not region.contains(by_id[p.cell_id]):
            raise ValueError('off-region action rejected; candidate not silently clipped')
    return dict(status='spatial_action_constraints_satisfied',actions=len(proposals),
                physical_localization_validated=False,biological_effect_localization_validated=False)


def localization_metrics(baseline,outcome,region,*,modality,feature_id):
    """Absolute measured changes inside/outside one region, with native units.

    Repeated IDs must describe the same observed lineage and donor. Independent
    cell samples need a separately defined population statistic, not this API.
    """
    if type(region) is not Region:
        raise ValueError('typed native region required')
    region.__post_init__();text(modality,'modality');text(feature_id,'feature_id')
    baseline=tuple(baseline);outcome=tuple(outcome)
    before={r.record_id:validate_record(r) for r in baseline}
    after={r.record_id:validate_record(r) for r in outcome}
    if len(before)!=len(baseline) or len(after)!=len(outcome):
        raise ValueError('duplicate longitudinal cell IDs cannot be silently collapsed')
    if not before or set(before)!=set(after):
        raise ValueError('same complete longitudinal cell IDs required')
    if len({r.source.evidence_kind for r in (*baseline,*outcome)})!=1:
        raise ValueError('experimental, simulation and prediction evidence cannot be pooled')
    inside=[];outside=[];missing_inside=0;missing_outside=0;unit=None;cohort=None
    for key,a in before.items():
        b=after[key];location=region.contains(a)
        if (a.source.lineage_id is None or a.source.lineage_id!=b.source.lineage_id
                or a.source.donor_id is None or a.source.donor_id!=b.source.donor_id
                or a.source.study_id!=b.source.study_id or a.source.experiment_id!=b.source.experiment_id
                or a.organism!=b.organism
                or a.source.evidence_kind!=b.source.evidence_kind
                or a.time_value is None or b.time_value is None or b.time_value<=a.time_value
                or (a.time_unit,a.time_reference)!=(b.time_unit,b.time_reference)):
            raise ValueError('explicit same-lineage/donor/experiment/time evidence required')
        context=(a.source.study_id,a.source.experiment_id,a.source.donor_id,
                 a.source.sample_id,b.source.sample_id,a.organism,
                 a.coordinate_frame,a.coordinate_unit,
                 a.time_value,b.time_value,a.time_unit,a.time_reference,
                 a.intervention_ids,b.intervention_ids)
        if cohort is None:cohort=context
        elif context!=cohort:
            raise ValueError('one shared cohort context, specimen, interval and intervention is required; stratify separately')
        x=next((v for v in a.measurements if (v.modality,v.feature_id)==(modality,feature_id)),None)
        y=next((v for v in b.measurements if (v.modality,v.feature_id)==(modality,feature_id)),None)
        if x is None or y is None or not x.observed or not y.observed:
            if location: missing_inside+=1
            else: missing_outside+=1
            continue
        if x.unit!=y.unit or unit is not None and unit!=x.unit:
            raise ValueError('incompatible native outcome units')
        unit=x.unit;number(x.value,'baseline outcome');number(y.value,'follow-up outcome')
        (inside if location else outside).append(abs(y.value-x.value))
    numerator=sum(outside);denominator=sum(inside)+numerator
    # A zero denominator is undefined, not perfect localization. No observed
    # outside cells is missing evidence, not a measured zero off-target effect.
    fraction=numerator/denominator if outside and inside and denominator>0 else None
    return dict(native_unit=unit,inside_observed=len(inside),outside_observed=len(outside),
        inside_missing=missing_inside,outside_missing=missing_outside,
        inside_absolute_change=sum(inside) if inside else None,
        outside_absolute_change=sum(outside) if outside else None,
        outside_fraction_of_measured_absolute_change=fraction,
        region_membership_reference='baseline_coordinates',
        evidence_kinds=sorted({r.source.evidence_kind for r in before.values()}),
        clinical_or_causal_validation=False)
