"""Fresh append-only canonical metric capture; forensic inspection never resumes.

One fsync per nonempty batch; no past-history rescans during append. Retained
identity sets grow linearly in event count. A successful append acknowledges
that batch's fsync, not a model checkpoint or resumable optimizer transaction.
After process death, complete lines are observed forensic evidence only: neither
batch completeness nor the last acknowledged fsync can be inferred from bytes.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path

from mora3.research.metric_records import MetricRecord, _checked, _wire, _unique_object


def _encode(row,index):
    return (json.dumps(_wire(row,index),sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


class DurableMetricJournal:
    """Single existing worker only. Existing files are never reopened for append."""
    def __init__(self,path):
        path=Path(path)
        self._fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_APPEND,0o600)
        self._failed=False
        self._count=0
        self._keys=set();self._registry={};self._bindings={};self._steps={}
        try:
            os.fsync(self._fd)
            parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
            try: os.fsync(parent)
            finally: os.close(parent)
        except BaseException:
            self.close()
            raise

    def append_batch(self,records):
        if self._failed or self._fd is None:
            raise ValueError('Closed or failed journal cannot append or resume')
        try:
            rows=_checked(records)
            for row in rows:
                binding=(row.trial_id,row.seed)
                if self._bindings.setdefault(row.run_id,binding)!=binding:
                    raise ValueError('run trial/seed binding drift')
                series=(row.run_id,row.name,row.task,row.split)
                meaning=(row.unit,row.direction,row.x_axis,row.scope)
                if self._registry.setdefault(series,meaning)!=meaning:
                    raise ValueError('metric unit/direction/axis/scope drift')
                key=(row.run_id,row.segment_id,row.step,row.name,row.task,row.split,row.source_sha256)
                if key in self._keys: raise ValueError('duplicate metric event')
                self._keys.add(key)
                ordered=(row.run_id,row.segment_id,row.x_axis)
                if row.step<self._steps.get(ordered,-1): raise ValueError('steps regress within run segment/axis')
                self._steps[ordered]=row.step
            if not rows: return self._count
            payload=memoryview(b''.join(_encode(row,self._count+i) for i,row in enumerate(rows)))
            while payload:
                written=os.write(self._fd,payload)
                if written<=0: raise OSError('Journal write made no progress')
                payload=payload[written:]
            os.fsync(self._fd)
            self._count+=len(rows)
            return self._count
        except BaseException:
            self._failed=True
            raise

    def close(self):
        if self._fd is not None:
            fd=self._fd;self._fd=None
            os.close(fd)

    def __enter__(self): return self
    def __exit__(self,*args): self.close()


@dataclass(frozen=True)
class ForensicMetricPrefix:
    metrics: tuple[MetricRecord,...]
    torn_tail_bytes: int
    torn_tail_sha256: str | None
    resume_authorized: bool=False


def read_forensic_prefix(path):
    """Read canonical complete lines; report, never truncate, an incomplete tail.

    Caller must stop the sole writer before inspection. Malformed or noncanonical
    complete lines and cross-record semantic errors fail closed rather than being
    mistaken for an interrupted suffix. This is not terminal selection evidence.
    """
    rows=[];tail=b''
    with Path(path).open('rb') as stream:
        for index,line in enumerate(stream):
            if not line.endswith(b'\n'):
                tail=line
                break
            try:
                wire=json.loads(line,object_pairs_hook=_unique_object)
                if type(wire) is not dict or type(wire.get('event_index')) is not int or wire['event_index']!=index:
                    raise ValueError('Metric journal event order differs')
                schema=wire.pop('schema');wire.pop('event_index')
                row=MetricRecord(**wire)
                if _encode(row,index)!=line or schema!=_wire(row,index)['schema']:
                    raise ValueError('Canonical metric journal wire required')
                rows.append(row)
            except (KeyError,TypeError,UnicodeError) as error:
                raise ValueError('Invalid complete journal event') from error
    return ForensicMetricPrefix(_checked(rows),len(tail),sha256(tail).hexdigest() if tail else None)
