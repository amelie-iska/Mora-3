"""Opt-in process-local LRU of admitted immutable tensor preparation only.

No learned features or autograd state. Complete task+receipt bytes bind targets
separately from inputs; callers receive clones to prevent cache poisoning.
Byte budget covers retained tensor storage AND canonical key bytes. Temporary
projection/return clones and Python allocator overhead are outside this budget.
Not serialized: checkpoints remain reference-compatible and rebuild cold.
"""
from collections import OrderedDict
import json
import torch
from mora3.data.sciplex_admission import ObservationalSciPlexTask


def _clone(value):
    if isinstance(value,torch.Tensor):return value.detach().clone()
    if isinstance(value,dict):return {k:_clone(v) for k,v in value.items()}
    raise ValueError('tensor dictionaries required')


def _size(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    return sum(_size(v) for v in value.values())


class SciPlexPreparationCache:
    def __init__(self, *, max_bytes):
        if type(max_bytes) is not int or max_bytes<=0:raise ValueError('positive byte budget required')
        self.max_bytes=max_bytes
        self._entries=OrderedDict()
        self.bytes_used=0
        self.hits=0
        self.misses=0

    def clear(self):
        self._entries.clear()
        self.bytes_used=0

    def project(self, task, *, axis_ids, device='cpu', dtype=torch.float64, for_training=True):
        if not isinstance(task,ObservationalSciPlexTask):raise ValueError('observational admission required')
        raw=json.loads(task.task.sidecar_bytes)
        where='targets' if raw['task']=='response' else 'conditions'
        if tuple(raw[where]['expression']['axis_ids'])!=tuple(axis_ids):raise ValueError('gene order mismatch')
        device=torch.device(device)
        if device.type=='cuda' and device.index is None:device=torch.device('cuda',torch.cuda.current_device())
        # Exact immutable bytes avoid digest collision and bind admission/content.
        key=(task.canonical_bytes(),tuple(axis_ids),str(device),dtype,bool(for_training))
        if key in self._entries:
            self.hits+=1
            value,size=self._entries.pop(key)
            self._entries[key]=(value,size)
            return _clone(value)
        self.misses+=1
        value=task.project(device=device,dtype=dtype,for_training=for_training)
        size=_size(value)+len(key[0])+sum(len(x.encode()) for x in axis_ids)
        if size<=self.max_bytes:
            while self.bytes_used+size>self.max_bytes:
                _,(_,old_size)=self._entries.popitem(last=False)
                self.bytes_used-=old_size
            self._entries[key]=(_clone(value),size)
            self.bytes_used+=size
        return value


class PreparedSciPlexTask(ObservationalSciPlexTask):
    """Opt-in fixed-device projection, bound to an immutable admitted snapshot.

    Construction validates the complete receipt and ordered axis once. Tensor
    returns are cloned; learned features and RNG state are never retained.
    Rebuild after resume/device changes. Canonical checkpoint bytes are unchanged.
    Retained private tensors must not be mutated by callers.
    """
    def __setattr__(self, name, value):
        from dataclasses import FrozenInstanceError
        raise FrozenInstanceError('prepared task is immutable')

    def __delattr__(self, name):
        from dataclasses import FrozenInstanceError
        raise FrozenInstanceError('prepared task is immutable')

    @classmethod
    def from_admitted(cls, task, *, axis_ids, device='cpu', dtype=torch.float64):
        if not isinstance(task, ObservationalSciPlexTask):
            raise ValueError('observational admission required')
        result = cls(task.task, task.receipt_bytes)
        raw = json.loads(task.task.sidecar_bytes)
        direction = raw['task']
        where = 'targets' if direction == 'response' else 'conditions'
        axis = tuple(raw[where]['expression']['axis_ids'])
        if axis != tuple(axis_ids):
            raise ValueError('gene order mismatch')
        value = ObservationalSciPlexTask.project(result, device=device, dtype=dtype)
        tensor = next(iter(value['inputs'].values()))
        object.__setattr__(result, 'prepared_axis_ids', axis)
        object.__setattr__(result, 'prepared_direction', direction)
        object.__setattr__(result, '_prepared_device', tensor.device)
        object.__setattr__(result, '_prepared_dtype', dtype)
        object.__setattr__(result, '_prepared_projection', value)
        return result

    def project(self, *, device=None, dtype=None, for_training=False):
        device = self._prepared_device if device is None else torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        dtype = self._prepared_dtype if dtype is None else dtype
        if device != self._prepared_device or dtype != self._prepared_dtype:
            return ObservationalSciPlexTask.project(self, device=device, dtype=dtype,
                                                    for_training=for_training)
        return _clone(self._prepared_projection)
