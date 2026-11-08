"""Bounded immutable CPU preparation; no learned tensors or automatic adoption.

Bindings cover full views, ordered vocabulary/axis and applicability. Call validate
at the owner boundary before reusing preparation with any external batch.
"""
from dataclasses import dataclass
import torch
from torch.nn import functional as F
from mora3.data.uniprot_residue import residue_batch
from .annotation_trainer import _digest


def _binding(views, tokenizer, feature_axis, applicable):
    batch = residue_batch(tuple(views), feature_axis=tuple(feature_axis))
    if len(applicable) != len(views) or any(type(x) is not bool for x in applicable):
        raise ValueError('Boolean applicability required')
    return _digest(dict(batch=batch.binding_sha256,
        vocabulary=[bytes(x).hex() for x in tokenizer.vocabulary],
        applicable=tuple(applicable), version=1))


@dataclass(frozen=True)
class PreparedResidues:
    binding: str
    feature_axis: tuple
    token_ids: tuple
    residue_indices: tuple
    positive_indices: tuple

    def validate(self, views, tokenizer, *, feature_axis, applicable):
        if self.binding != _binding(tuple(views), tokenizer, tuple(feature_axis), tuple(applicable)):
            raise ValueError('residue preparation binding mismatch')

    def align(self, row, token_features):
        if token_features.ndim != 2 or len(token_features) != len(self.token_ids[row]):
            raise ValueError('one feature row per complete token required')
        return token_features[torch.tensor(self.residue_indices[row],
                                          dtype=torch.long, device=token_features.device)]

    def loss(self, logits):
        logits = tuple(logits)
        if len(logits) != len(self.token_ids):
            raise ValueError('one logit matrix per view required')
        selected=[]; zeros=[]
        for z, indices, positives in zip(logits, self.residue_indices, self.positive_indices):
            if (z.shape != (len(indices), len(self.feature_axis)) or not z.is_floating_point()
                    or z.device != logits[0].device or z.dtype != logits[0].dtype):
                raise ValueError('residue dimensions, dtype and device must agree')
            zeros.append(z.reshape(-1)[:0].sum())
            if positives:
                p,j=zip(*positives)
                selected.append(z[list(p),list(j)])
        zero=torch.stack(zeros).sum()
        return F.softplus(-torch.cat(selected)).mean()+zero if selected else zero


def prepare_residues(views, tokenizer, *, feature_axis, applicable, max_residues):
    views,axis,active=tuple(views),tuple(feature_axis),tuple(applicable)
    if type(max_residues) is not int or max_residues < 1 or sum(len(v.sequence) for v in views)>max_residues:
        raise ValueError('residue preparation budget exceeded')
    binding=_binding(views,tokenizer,axis,active)
    batch=residue_batch(views,feature_axis=axis)
    tokens=[]; expansions=[]; positives=[]
    for v, rows, enabled in zip(views,batch.targets(),active):
        if not v.sequence or any(not 'A'<=c<='Z' for c in v.sequence):
            raise ValueError('uppercase ASCII sequence required')
        ids=tuple(tokenizer.encode(v.sequence.encode('ascii')))
        if tokenizer.decode(ids)!=v.sequence.encode('ascii'):
            raise ValueError('sequence token roundtrip failed')
        tokens.append(ids)
        expansions.append(tuple(i for i,t in enumerate(ids) for _ in tokenizer.vocabulary[t]))
        cells=set()
        if enabled:
            for row in rows:
                if row['coordinate_status']!='exact_canonical':continue
                j=axis.index(row['feature_type'])
                span=row['interval0']
                indices=range(*span) if span is not None else row['endpoint_indices0'] or ()
                cells.update((p,j) for p in indices)
        positives.append(tuple(sorted(cells)))
    return PreparedResidues(binding,axis,tuple(tokens),tuple(expansions),tuple(positives))


class ResiduePreparationCache:
    """Explicit owner of bounded immutable CPU preparations; no learned state.

    LRU retention is limited by both entries and complete residue count. Requests
    exceeding the total budget fail before eviction. Keys bind full source views,
    targets, vocabulary, feature axis and applicability, never Python identity.
    Caller owns the cache lifetime; it is not model/checkpoint state or admission.
    Residue count bounds payload cardinality, not exact Python allocator bytes.
    """
    def __init__(self, *, max_entries, max_residues):
        from collections import OrderedDict
        if any(type(x) is not int or x < 1 for x in (max_entries,max_residues)):
            raise ValueError('positive cache entry and residue budgets required')
        self.max_entries,self.max_residues=max_entries,max_residues
        self._items=OrderedDict()

    @property
    def entries(self):
        return len(self._items)

    @property
    def residues(self):
        return sum(n for _,n in self._items.values())

    def get(self, views, tokenizer, *, feature_axis, applicable):
        views,axis,active=tuple(views),tuple(feature_axis),tuple(applicable)
        n=sum(len(v.sequence) for v in views)
        if n > self.max_residues:
            raise ValueError('residue preparation budget exceeded')
        key=_binding(views,tokenizer,axis,active)
        if key in self._items:
            self._items.move_to_end(key)
            return self._items[key][0]
        prepared=prepare_residues(views,tokenizer,feature_axis=axis,
                                 applicable=active,max_residues=self.max_residues)
        while self.entries >= self.max_entries or self.residues+n > self.max_residues:
            self._items.popitem(last=False)
        self._items[key]=(prepared,n)
        return prepared
