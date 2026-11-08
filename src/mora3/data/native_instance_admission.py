"""Immutable native ordered-instance receipt validation, without execution admission.

Corpus owner must supply and independently authenticate the expected receipt,
master and task digests. Hashes establish byte identity, not scientific rights or
lineage truth. This first boundary never enables training or replaces old tasks.
"""
from dataclasses import dataclass
import hashlib
import json
import re
from .public_image_masks import PublicImageMasks
from .public_image_admission import _unique


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


@dataclass(frozen=True)
class NativeInstanceReceipt:
    source: PublicImageMasks
    receipt_bytes: bytes
    receipt_sha256: str
    expected_master_sha256: str
    expected_task_sha256: str

    def __post_init__(self):
        if type(self.source) is not PublicImageMasks:
            raise ValueError('validated native source required')
        if type(self.receipt_bytes) is not bytes or not 0<len(self.receipt_bytes)<=1024*1024:
            raise ValueError('bounded immutable receipt bytes required')
        if not all(_sha(x) for x in (self.receipt_sha256,self.expected_master_sha256,self.expected_task_sha256)):
            raise ValueError('explicit pinned receipt/master/task digests required')
        if hashlib.sha256(self.receipt_bytes).hexdigest()!=self.receipt_sha256:
            raise ValueError('receipt checksum mismatch')
        try:
            d=json.loads(self.receipt_bytes,object_pairs_hook=_unique)
        except (ValueError,UnicodeError) as exc:
            raise ValueError('malformed native receipt') from exc
        s=self.source
        expected=dict(schema='mora3.native_instance_receipt.v1',issuer='dataset_construction',
            native_binding_sha256=s.binding_sha256,image_sha256=s.image_sha256,
            ordered_mask_sha256=list(s.mask_sha256),source_id=s.image.source_id,
            parent_id=s.image.parent_id,source_split=s.split,
            master_sha256=self.expected_master_sha256,task_sha256=self.expected_task_sha256,
            effective_split='train',objective='ordered-nuclear-instance-set-BCE-v1',
            roles={'image':'condition','ordered_masks':'target'},training_eligible=True)
        extra={'rights_evidence_sha256','lineage_evidence_sha256','effective_group'}
        if type(d) is not dict or set(d)!=set(expected)|extra:
            raise ValueError('exact native receipt fields required')
        if any(d[k]!=v for k,v in expected.items()) or d['training_eligible'] is not True:
            raise ValueError('native receipt source/master/task/roles mismatch')
        if not all(_sha(d[k]) for k in ('rights_evidence_sha256','lineage_evidence_sha256')):
            raise ValueError('pinned corpus rights and lineage evidence required')
        if type(d['effective_group']) is not str or not d['effective_group'].strip():
            raise ValueError('explicit effective leakage group required')
        if s.split!='train' or s.encoding!='ordered_binary_instances':
            raise ValueError('held-out or incompatible native source cannot train')
        if dict(s.image.context).get('object_type')!='nucleus':
            raise ValueError('native nuclear object scope required')

    @property
    def source_binding_sha256(self):
        return self.source.binding_sha256

    def require_training(self):
        raise ValueError('receipt validation alone is not native execution admission')


@dataclass(frozen=True)
class NativeInstanceEvidence:
    """Authenticated evidence identities; master/task membership remains unresolved."""
    receipt_sha256: str
    rights_sha256: str
    lineage_sha256: str

    def require_training(self):
        raise ValueError('evidence identity is not corpus membership or execution admission')


def resolve_native_instance_evidence(receipt, *, rights_bytes, lineage_bytes):
    """Resolve bounded corpus evidence supplied by the caller without filesystem access.

    This verifies exact assertions and binding, not the legal truth of source terms.
    No master/task membership, source eligibility or execution grant is inferred.
    """
    if type(receipt) is not NativeInstanceReceipt:
        raise ValueError('validated native receipt required')
    p=json.loads(receipt.receipt_bytes,object_pairs_hook=_unique)
    common=dict(issuer='dataset_construction',native_binding_sha256=receipt.source.binding_sha256,
                source_id=receipt.source.image.source_id,parent_id=receipt.source.image.parent_id)
    rights=dict(common,schema='mora3.native_instance_rights.v1',training_permitted=True)
    lineage=dict(common,schema='mora3.native_instance_lineage.v1',
        image_sha256=receipt.source.image_sha256,ordered_mask_sha256=list(receipt.source.mask_sha256),
        effective_group=p['effective_group'],effective_split=p['effective_split'],
        cross_split_conflict=False,parent_audit_complete=True)
    for name,raw,expected,extra in (
        ('rights',rights_bytes,rights,{'source_terms','source_revision'}),
        ('lineage',lineage_bytes,lineage,set())):
        if type(raw) is not bytes or not 0<len(raw)<=1024*1024:
            raise ValueError('bounded immutable evidence bytes required')
        if hashlib.sha256(raw).hexdigest()!=p[name+'_evidence_sha256']:
            raise ValueError('evidence checksum mismatch')
        try:
            d=json.loads(raw,object_pairs_hook=_unique)
        except (ValueError,UnicodeError) as exc:
            raise ValueError('malformed native evidence') from exc
        if type(d) is not dict or set(d)!=set(expected)|extra:
            raise ValueError('exact native evidence fields required')
        if any(type(d[k]) is not type(v) or d[k]!=v for k,v in expected.items()):
            raise ValueError('native evidence assertion or binding mismatch')
        if any(type(d[k]) is not str or not d[k].strip() for k in extra):
            raise ValueError('explicit source terms and revision required')
    return NativeInstanceEvidence(receipt.receipt_sha256,p['rights_evidence_sha256'],p['lineage_evidence_sha256'])


@dataclass(frozen=True)
class NativeInstanceMembership:
    """Prospective pinned membership identity, never independent admission proof."""
    receipt_sha256: str
    master_sha256: str
    task_sha256: str
    group_sha256: str
    task_id: str
    effective_groups: tuple

    def require_training(self):
        raise ValueError('membership identity is not independent execution admission')


def resolve_native_instance_membership(receipt, *, rights_bytes, lineage_bytes,
        master_bytes, task_bytes, group_bytes, current_master_sha256):
    """Authenticate one bounded prospective native task shard and aligned groups.

    The receipt task digest denotes the complete Parquet shard, not a row hash.
    A separately supplied current-master digest must match the receipt. This is
    a caller-pinned snapshot, not automatic discovery of the latest corpus. The
    new native row schema below has no actual corpus-issued artifacts yet.
    Caller supplies bytes: no path access, whole-corpus scan or eligibility edit.
    Cross-source leakage and scientific admission remain independent gates.
    """
    import io
    import pyarrow.parquet as pq
    resolve_native_instance_evidence(receipt,rights_bytes=rights_bytes,lineage_bytes=lineage_bytes)
    if not _sha(current_master_sha256) or current_master_sha256!=receipt.expected_master_sha256:
        raise ValueError('independent current master pin mismatch')

    def checked(raw, sha, cap):
        if type(raw) is not bytes or not 0<len(raw)<=cap or hashlib.sha256(raw).hexdigest()!=sha:
            raise ValueError('bounded immutable membership bytes/checksum mismatch')

    checked(master_bytes,current_master_sha256,128*1024**2)
    try:
        master=json.loads(master_bytes,object_pairs_hook=_unique)
    except (ValueError,UnicodeError) as exc:
        raise ValueError('malformed master') from exc
    if type(master) is not dict or master.get('schema')!='mora3.master_corpus_manifest.v6':
        raise ValueError('explicit supported current master required')
    shards=master.get('shards');lookup=master.get('effective_group_lookup')
    if type(shards) is not list or type(lookup) is not dict or any(type(s) is not dict for s in shards):
        raise ValueError('master shards and group lookup required')
    matches=[s for s in shards if s.get('sha256')==receipt.expected_task_sha256]
    if len(matches)!=1:
        raise ValueError('unique complete task shard membership required')
    shard=matches[0];path=shard.get('path')
    if type(path) is not str or not path or sum(s.get('path')==path for s in shards)!=1:
        raise ValueError('unique task shard path required')
    if shard.get('source_family')!='MoNuSeg' or shard.get('split')!='train':
        raise ValueError('native source family/train membership required')
    side=lookup.get(path)
    if type(side) is not dict:
        raise ValueError('current master effective group sidecar missing')

    def rows(raw,pin):
        if not _sha(pin.get('sha256')):
            raise ValueError('pinned complete shard required')
        checked(raw,pin['sha256'],16*1024**2)
        if type(pin.get('bytes')) is not int or pin['bytes']!=len(raw):
            raise ValueError('complete shard byte count mismatch')
        count=pin.get('rows')
        if type(count) is not int or not 1<=count<=256:
            raise ValueError('bounded shard row count 1..256 required')
        try:
            pf=pq.ParquetFile(io.BytesIO(raw))
            # Check decoded size before materializing caller-provided Parquet.
            meta=pf.metadata
            if meta.num_rows!=count or sum(meta.row_group(i).total_byte_size for i in range(meta.num_row_groups))>64*1024**2:
                raise ValueError('complete shard rows/decoded budget mismatch')
            return pf.read().to_pylist()
        except Exception as exc:
            raise ValueError('invalid bounded membership Parquet') from exc

    tasks=rows(task_bytes,shard);groups=rows(group_bytes,side)
    if len(tasks)!=len(groups):
        raise ValueError('task/group count alignment mismatch')
    p=json.loads(receipt.receipt_bytes,object_pairs_hook=_unique)
    seen=set();selected=[]
    for row,group in zip(tasks,groups):
        tid=row.get('task_id')
        if type(tid) is not str or not tid or tid in seen or group.get('task_id')!=tid:
            raise ValueError('distinct aligned task/group identities required')
        seen.add(tid)
        if row.get('split')!='train' or group.get('effective_split')!='train':
            raise ValueError('held-out effective allocation refuses')
        try:
            gs=json.loads(group['effective_groups_json'],object_pairs_hook=_unique)
        except (KeyError,TypeError,ValueError,UnicodeError) as exc:
            raise ValueError('explicit effective group identities required') from exc
        if type(gs) is not list or not gs or any(type(g) is not str or not g.strip() for g in gs) or len(set(gs))!=len(gs):
            raise ValueError('distinct nonempty effective groups required')
        if row.get('native_binding_sha256')==receipt.source.binding_sha256:
            selected.append((row,gs))
    if len(selected)!=1:
        raise ValueError('unique complete source-bound native task required')
    row,gs=selected[0]
    expected=dict(source_family='MoNuSeg',native_schema='mora3.native_ordered_instance_task.v1',
        native_binding_sha256=receipt.source.binding_sha256,source_id=receipt.source.image.source_id,
        parent_id=receipt.source.image.parent_id,image_sha256=receipt.source.image_sha256,
        task_name=p['objective'],split='train',admitted_for_task=True)
    if any(type(row.get(k)) is not type(v) or row.get(k)!=v for k,v in expected.items()):
        raise ValueError('native task identity/objective/admission mismatch')
    try:
        masks=json.loads(row['ordered_mask_sha256_json'],object_pairs_hook=_unique)
        roles=json.loads(row['roles_json'],object_pairs_hook=_unique)
    except (KeyError,TypeError,ValueError,UnicodeError) as exc:
        raise ValueError('complete native masks/roles required') from exc
    if masks!=list(receipt.source.mask_sha256) or roles!=p['roles'] or gs!=[p['effective_group']]:
        raise ValueError('complete ordered masks/target roles/effective group mismatch')
    return NativeInstanceMembership(receipt.receipt_sha256,current_master_sha256,
        receipt.expected_task_sha256,side['sha256'],row['task_id'],tuple(gs))
