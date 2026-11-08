"""Canonical local experiment evidence, with a caller-owned offline W&B adapter.

No SDK import, login, credential access, network, or run initialization occurs.
These batch export helpers do not authorize training or certify dataset quality.
Resume segments retain their original optimizer steps; event_index is a distinct
monotonic export index so W&B cannot silently collapse equal resumed steps.
"""
from dataclasses import asdict, dataclass, fields
import json
import math
import os
from pathlib import Path
import re


_SHA = re.compile(r'[0-9a-f]{64}\Z')
_SCHEMA = 'mora3.metric-record.v1'


def _text(value):
    return type(value) is str and bool(value.strip())


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


@dataclass(frozen=True)
class MetricRecord:
    run_id: str
    trial_id: str
    segment_id: str
    seed: int
    step: int
    x_axis: str
    name: str
    value: float | None
    unit: str
    direction: str
    task: str
    split: str
    source_sha256: str
    evidence_sha256: str
    scope: str
    numerator: float | None = None
    denominator: float | None = None
    missing_reason: str | None = None

    def __post_init__(self):
        for key in ('run_id','trial_id','segment_id','x_axis','name','unit','task','split'):
            if not _text(getattr(self,key)):
                raise ValueError(f'{key} requires a nonempty explicit string')
        for key in ('seed','step'):
            value=getattr(self,key)
            if type(value) is not int or value < 0:
                raise ValueError(f'{key} requires a nonnegative integer')
        if self.direction not in ('minimize','maximize','none'):
            raise ValueError('explicit metric direction required')
        if self.scope not in ('implementation_diagnostic','train','validation','heldout','resource'):
            raise ValueError('explicit evidence scope required')
        for key in ('source_sha256','evidence_sha256'):
            value=getattr(self,key)
            if type(value) is not str or _SHA.fullmatch(value) is None:
                raise ValueError(f'{key} requires an exact lowercase SHA256')
        if self.value is None:
            if not _text(self.missing_reason):
                raise ValueError('missing values require an explicit reason')
        elif not _finite(self.value) or self.missing_reason is not None:
            raise ValueError('observed values must be finite and cannot have a missing reason')
        if (self.numerator is None) != (self.denominator is None):
            raise ValueError('native numerator and denominator must appear together')
        if self.denominator is not None:
            if (not _finite(self.numerator) or not _finite(self.denominator)
                    or self.denominator < 0 or (self.denominator == 0 and self.value is not None)):
                raise ValueError('invalid native counts or observed zero-denominator ratio')


def _checked(records):
    rows=tuple(records)
    keys=set(); registry={}; run_bindings={}; last_steps={}
    for row in rows:
        if type(row) is not MetricRecord:
            raise ValueError('exact MetricRecord required')
        # Revalidate even if a frozen instance was forged through object.__setattr__.
        row.__post_init__()
        binding=(row.trial_id,row.seed)
        if run_bindings.setdefault(row.run_id,binding) != binding:
            raise ValueError('run trial/seed binding drift')
        series=(row.run_id,row.name,row.task,row.split)
        meaning=(row.unit,row.direction,row.x_axis,row.scope)
        if registry.setdefault(series,meaning) != meaning:
            raise ValueError('metric unit/direction/axis/scope drift')
        key=(row.run_id,row.segment_id,row.step,row.name,row.task,row.split,row.source_sha256)
        if key in keys:
            raise ValueError('duplicate metric event; never overwrite a resumed step')
        keys.add(key)
        ordered=(row.run_id,row.segment_id,row.x_axis)
        if row.step < last_steps.get(ordered,-1):
            raise ValueError('steps regress within a run segment/axis')
        last_steps[ordered]=row.step
    return rows


def _wire(row,index):
    return dict(schema=_SCHEMA,event_index=index,**asdict(row))


def write_history(path,records):
    """Write a validated finite export batch to a new file; refuse replacement.

    Every scalar/missing/censored record is retained. Callers keep raw training
    logs separately. A filesystem error preserves any partial new output and is
    raised; there is no automatic retry, truncation, or deletion of that output.
    """
    rows=_checked(records)
    with Path(path).open('x',encoding='utf-8') as handle:
        for index,row in enumerate(rows):
            handle.write(json.dumps(_wire(row,index),sort_keys=True,separators=(',',':'),
                                    allow_nan=False)+'\n')
        handle.flush();os.fsync(handle.fileno())
    return len(rows)


def _unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result


def read_history(path):
    rows=[]
    expected={field.name for field in fields(MetricRecord)}|{'schema','event_index'}
    with Path(path).open(encoding='utf-8') as handle:
        for index,line in enumerate(handle):
            try:
                wire=json.loads(line,object_pairs_hook=_unique_object)
                if (type(wire) is not dict or set(wire) != expected
                        or wire.pop('schema') != _SCHEMA
                        or type(wire['event_index']) is not int or wire.pop('event_index') != index):
                    raise ValueError('metric history schema/event order drift')
                rows.append(MetricRecord(**wire))
            except (TypeError,json.JSONDecodeError) as error:
                raise ValueError(f'invalid metric record at event {index}') from error
    return _checked(rows)


def emit_wandb_offline(records,run):
    """Emit all canonical records to an already-created, caller-owned offline run.

    Uses W&B's explicit step only for the export index; native run/segment/step,
    provenance, units, counts and missing reasons remain in every record. This
    deliberately does not create a dashboard/quality aggregate from unlike units.
    SDK exceptions propagate and raw canonical files remain the source of truth.
    The caller must use a fresh offline run for each exported history batch.
    """
    rows=_checked(records)
    if getattr(getattr(run,'settings',None),'mode',None) != 'offline':
        raise ValueError('an existing explicitly offline W&B run is required')
    if type(getattr(run,'step',None)) is not int or run.step != 0:
        raise ValueError('a fresh offline run at export step zero is required')
    for index,row in enumerate(rows):
        run.log({'mora3/event_index':index,'mora3/record':_wire(row,index)},step=index)
    return len(rows)

