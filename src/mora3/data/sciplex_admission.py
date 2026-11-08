"""Explicit limited admission; never changes the source inspection quarantine.

Receipts are provenance assertions, not authentication. Their issuer must verify
saved audit/source hashes and bind only independently audited task snapshots.
No physical-dose, causal, raw-barcode or held-out eligibility is granted.
"""
from dataclasses import dataclass
import hashlib
import re
from .connected_sciplex import SciPlexTask, _canonical, _load

POLICY = 'author-export-observational-train-only-v1'


@dataclass(frozen=True)
class ObservationalSciPlexTask:
    task: SciPlexTask
    receipt_bytes: bytes

    def __post_init__(self):
        if not isinstance(self.task, SciPlexTask) or not isinstance(self.receipt_bytes, bytes):
            raise ValueError('immutable source task and receipt required')
        try:
            r = _load(self.receipt_bytes)
            expected = dict(version=1, policy=POLICY, split='train_only',
                leakage_group='study:GSE139944:GSM4150377',
                audit_sha256=r['audit_sha256'], source_sha256=self.task.source_sha256,
                task_sha256=hashlib.sha256(self.task.canonical_bytes()).hexdigest())
            if (type(r['version']) is not int or r != expected or
                not isinstance(r['audit_sha256'],str) or
                not re.fullmatch('[0-9a-f]{64}',r['audit_sha256'])):
                raise ValueError('incompatible task admission receipt')
            object.__setattr__(self,'receipt_bytes',_canonical(r))
        except (KeyError,TypeError,UnicodeError) as exc:
            raise ValueError('malformed task admission receipt') from exc

    @classmethod
    def from_receipt(cls, task, receipt):
        return cls(task, _canonical(receipt))

    def project(self, *, device='cpu', dtype=None, for_training=False):
        # Preserve source task masks and values exactly. The wrapper's receipt
        # grants only this named observational policy, not source-wide readiness.
        kwargs = dict(device=device)
        if dtype is not None:
            kwargs['dtype'] = dtype
        return self.task.project(**kwargs)

    def input_metadata(self):
        return dict(self.task.input_metadata(),policy=POLICY,split='train_only')

    def canonical_bytes(self):
        return _canonical(dict(version=1,task=_load(self.task.canonical_bytes()),
                               receipt=_load(self.receipt_bytes)))

    @classmethod
    def from_bytes(cls, raw):
        try:
            d=_load(raw)
            if set(d)!={'version','task','receipt'} or type(d['version']) is not int or d['version']!=1:
                raise ValueError('unsupported admission snapshot')
            return cls.from_receipt(SciPlexTask.from_bytes(_canonical(d['task'])),d['receipt'])
        except (KeyError,TypeError,UnicodeError) as exc:
            raise ValueError('malformed admission snapshot') from exc

    def assert_resume_matches(self, other):
        if not isinstance(other,ObservationalSciPlexTask) or self.canonical_bytes()!=other.canonical_bytes():
            raise ValueError('resume admission/task mismatch')
