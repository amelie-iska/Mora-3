"""Source-bound four-head forest supervision, without issuing training admission.

A known skeleton and a frozen coordinate/encoder export are explicit task inputs.
Flow targets, decoder reconstruction states, and scored candidates have distinct
roles. This composition supplies neither unknown topology nor source qualification.
"""
from dataclasses import asdict, dataclass, fields, is_dataclass
from hashlib import sha256
import math
import torch
from .forest_flow import (NativeForestFlowHead, _sha, _wire,
    forest_flow_matching_loss)
from .forest_token_decoder import NativeForestTokenDecoder, forest_token_loss, _tensor_sha
from .forest_quality import NativeForestQualityHead, ForestQualityLabels, forest_quality_loss
from .forest_analogy import NativeForestAnalogyHead, ForestAnalogyItem, forest_analogy_loss

from mora3.data.schema import GraphRecord
from mora3.data.tasks import ConditionalTask
from mora3.reasoning.flow_forest_refinement import ForestState

from .forest_topology import NativeForestTopologyHead,forest_topology_loss
from mora3.reasoning.forest_topology import canonical_forest_actions

OBJECTIVES = ('flow', 'tokens', 'quality', 'analogy')


@dataclass(frozen=True)
class ForestTrainingHeads:
    flow: NativeForestFlowHead
    decoder: NativeForestTokenDecoder
    quality: NativeForestQualityHead
    analogy: NativeForestAnalogyHead
    topology: NativeForestTopologyHead | None = None

    def binding(self):
        expected = (NativeForestFlowHead, NativeForestTokenDecoder,
                    NativeForestQualityHead, NativeForestAnalogyHead)
        values = (self.flow, self.decoder, self.quality, self.analogy)
        if any(type(x) is not t for x, t in zip(values, expected)):
            raise ValueError('Four exact native forest heads required')
        if any(x.field is not self.flow for x in values[1:]):
            raise ValueError('All forest objectives must share the registered field')
        if self.topology is not None:
            if type(self.topology)is not NativeForestTopologyHead or self.topology.field is not self.flow:
                raise ValueError('Optional topology must share the exact native forest field')
            values += (self.topology,)
        ref = self.flow.readout.weight
        for head in values:
            for tensor in tuple(head.parameters()) + tuple(head.buffers()):
                if tensor.device != ref.device or tensor.is_floating_point() and tensor.dtype != ref.dtype:
                    raise ValueError('All native forest operators must share exact dtype and device')
        names=('flow','decoder','quality','analogy')+(()if self.topology is None else('topology',))
        return dict(schema='Mora3.NativeForestTrainingHeads/v1'if self.topology is None else'Mora3.NativeForestTrainingHeads/v2',
                    heads={name:value.get_extra_state()for name,value in zip(names,values)})

    @property
    def sha256(self):
        return sha256(_wire(self.binding())).hexdigest()


@dataclass(frozen=True)
class ForestTrainingSpec:
    source_snapshot_sha256: str
    coordinate_sha256: str
    encoder_sha256: str
    group_policy_sha256: str
    skeleton_policy_sha256: str
    heads_sha256: str
    weights: tuple[float, float, float, float]
    max_tensor_elements: int = 8_000_000
    topology_weight: float = 0.

    def __post_init__(self):
        for name in ('source_snapshot_sha256', 'coordinate_sha256', 'encoder_sha256',
                     'group_policy_sha256', 'skeleton_policy_sha256', 'heads_sha256'):
            _sha(getattr(self, name))
        if (type(self.topology_weight)not in(int,float)or not math.isfinite(self.topology_weight)or self.topology_weight<0):
            raise ValueError('Finite nonnegative topology weight required')
        if (type(self.weights) is not tuple or len(self.weights) != 4
                or any(type(x) not in (int, float) or not math.isfinite(x) or x < 0
                       for x in self.weights)
                or not 0 < sum(self.weights)+self.topology_weight < float('inf')
                or type(self.max_tensor_elements) is not int
                or not 1 <= self.max_tensor_elements <= 16_777_216):
            raise ValueError('Exact finite nonnegative four-objective weights and capacity required')

    @property
    def sha256(self):
        self.__post_init__()
        values=asdict(self)
        if not self.topology_weight:values.pop('topology_weight')
        return sha256(_wire(dict(schema='Mora3.ForestTrainingObjective/v2'if self.topology_weight else'Mora3.ForestTrainingObjective/v1', **values,
            reduction='fixed-weight-sum-of-active-head-means-no-renormalization',
            topology=self.skeleton_policy,
            decoder_input='separate-detached-source-encoder-export',
            quality_input='separate-detached-scored-candidate'))).hexdigest()

    @property
    def skeleton_policy(self):
        return ('source-declared-flow-skeleton-plus-prefix-topology-NLL-v1'if self.topology_weight
                else'source-declared-fixed-skeleton')


def _texts(values):
    return (type(values) is tuple and bool(values)
            and all(type(x) is str and 0 < len(x) <= 4096 for x in values)
            and len(set(values)) == len(values))


@dataclass(frozen=True)
class ForestBatchIdentity:
    source_snapshot_sha256: str
    coordinate_sha256: str
    encoder_sha256: str
    group_policy_sha256: str
    skeleton_policy_sha256: str
    source_revision: str
    record_ids: tuple[str, ...]
    original_ordinals: tuple[int, ...]
    groups: tuple[tuple[str, ...], ...]
    split: str
    evidence_kind: str
    export_sha256: str

    def __post_init__(self):
        for name in ('source_snapshot_sha256', 'coordinate_sha256', 'encoder_sha256',
                     'group_policy_sha256', 'skeleton_policy_sha256', 'export_sha256'):
            _sha(getattr(self, name))
        if (type(self.source_revision) is not str or not 0 < len(self.source_revision) <= 4096
                or not _texts(self.record_ids) or len(self.record_ids) > 256
                or type(self.original_ordinals) is not tuple
                or len(self.original_ordinals) != len(self.record_ids)
                or any(type(x) is not int or x < 0 for x in self.original_ordinals)
                or len(set(self.original_ordinals)) != len(self.original_ordinals)
                or type(self.groups) is not tuple or len(self.groups) != len(self.record_ids)
                or any(not _texts(x) for x in self.groups)
                or self.split not in ('train', 'validation', 'test')
                or self.evidence_kind not in ('observed', 'computed', 'simulated',
                                             'model_generated', 'synthetic_fixture')):
            raise ValueError('Exact original source identities, groups, role and evidence kind required')


@dataclass(frozen=True)
class FlowSupervision:
    values: torch.Tensor
    available: torch.Tensor


@dataclass(frozen=True)
class TokenSupervision:
    states: torch.Tensor
    values: torch.Tensor
    available: torch.Tensor
    alignment_sha256: str


@dataclass(frozen=True)
class QualitySupervision:
    states: torch.Tensor
    labels: ForestQualityLabels


@dataclass(frozen=True)
class AnalogySupervision:
    queries: tuple[ForestAnalogyItem, ...]
    memories: tuple[ForestAnalogyItem, ...]
    positive: torch.Tensor
    known: torch.Tensor
    evidence_sha256: str


@dataclass(frozen=True)
class ForestTopologySupervision:
    root_count: int
    forests: tuple[tuple[int, ...], ...]
    construction_policy_sha256: str
    evidence_sha256: str


@dataclass(frozen=True)
class ForestTrainingBatch:
    identity: ForestBatchIdentity
    record: GraphRecord
    task: ConditionalTask
    template: ForestState
    flow: FlowSupervision | None = None
    tokens: TokenSupervision | None = None
    quality: QualitySupervision | None = None
    analogy: AnalogySupervision | None = None
    topology: ForestTopologySupervision | None = None


def _tensor(value, shape, dtype, device, *, detached=False, finite=False):
    if (type(value) is not torch.Tensor or value.is_meta or value.layout != torch.strided
            or tuple(value.shape) != tuple(shape) or value.dtype != dtype or value.device != device
            or detached and value.requires_grad or finite and not torch.isfinite(value).all()):
        raise ValueError('Exact complete native tensor axes, dtype, device and role required')


def _mask(value, shape, device):
    _tensor(value, shape, torch.bool, device, detached=True)


def _tensor_elements(obj):
    if type(obj) is torch.Tensor:
        return obj.numel()
    if is_dataclass(obj):
        return sum(_tensor_elements(getattr(obj, f.name)) for f in fields(obj))
    if type(obj) in (tuple, list):
        return sum(_tensor_elements(x) for x in obj)
    if type(obj) is dict:
        return sum(_tensor_elements(x) for x in obj.values())
    return 0


def preflight_forest_batch(heads, spec, batch):
    """Validate ALL enabled objectives without a forward call or random draw.

    These are executable data contracts, not proof of upstream rights or global
    group closure. Production calls must additionally pass full-owner admission.
    """
    if (type(heads) is not ForestTrainingHeads or type(spec) is not ForestTrainingSpec
            or type(batch) is not ForestTrainingBatch or type(batch.identity) is not ForestBatchIdentity):
        raise ValueError('Exact forest heads, frozen objective and source batch required')
    spec.__post_init__(); batch.identity.__post_init__()
    if heads.sha256 != spec.heads_sha256:
        raise ValueError('Frozen native forest head configuration changed')
    for name in ('source_snapshot_sha256', 'coordinate_sha256', 'encoder_sha256',
                 'group_policy_sha256', 'skeleton_policy_sha256'):
        if getattr(batch.identity, name) != getattr(spec, name):
            raise ValueError('Forest batch source/coordinate/encoder/group/skeleton policy mismatch')
    if (batch.identity.split != 'train'
            or heads.flow.spec.source_snapshot_sha256 != spec.source_snapshot_sha256
            or heads.flow.spec.coordinate_sha256 != spec.coordinate_sha256
            or heads.decoder.source_codebook.encoder_sha256 != spec.encoder_sha256
            or heads.analogy.spec.group_policy_sha256 != spec.group_policy_sha256):
        raise ValueError('Train-only source-aligned forest objective required')
    from mora3.reasoning.flow_forest_refinement import ForestState
    if (type(batch.template) is not ForestState or batch.template.skeleton_origin != 'declared'
            or batch.template.states.requires_grad):
        raise ValueError('Detached source-declared known skeleton required; unknown topology is separate')
    if _tensor_elements(batch) > spec.max_tensor_elements:
        raise ValueError('Complete forest supervision exceeds declared tensor capacity')
    b = len(batch.identity.record_ids); n = len(batch.template.parents)
    ref = heads.flow.readout.weight; shape = (b, n, heads.flow.spec.dimension)
    probe = batch.template.states.detach()[None].expand(b, -1, -1)
    heads.flow._validate(batch.record, batch.task, probe, probe.new_zeros(b), batch.template)
    counts = dict.fromkeys(OBJECTIVES, 0)
    if batch.topology is not None and type(batch.topology)is not ForestTopologySupervision:
        raise ValueError('Exact source topology supervision required')
    if spec.topology_weight:
        if type(heads.topology)is not NativeForestTopologyHead:
            raise ValueError('Topology objective requires its registered native policy')
        counts['topology']=0
        if batch.topology is not None:
            x=batch.topology;_sha(x.construction_policy_sha256);_sha(x.evidence_sha256)
            if (x.construction_policy_sha256!=heads.topology.spec.construction_policy_sha256
                    or type(x.root_count)is not int
                    or x.root_count!=sum(p==-1 for p in batch.template.parents)
                    or type(x.forests)is not tuple or len(x.forests)!=b
                    or not 1<=b<=heads.topology.spec.max_batch_size):
                raise ValueError('Complete original topology batch and given root policy required')
            traces=tuple(canonical_forest_actions(heads.topology.spec.bounds,p,root_count=x.root_count)for p in x.forests)
            count=sum(len(z.actions)for z in traces)
            if count>heads.topology.spec.max_total_decisions:
                raise ValueError('Complete topology labels exceed decision budget')
            for trace in traces:
                for i in range(len(trace.actions)):
                    heads.topology.preflight(batch.record,batch.task,root_count=x.root_count,actions=trace.actions[:i])
            counts['topology']=count
    for name, kind in zip(OBJECTIVES, (FlowSupervision, TokenSupervision, QualitySupervision, AnalogySupervision)):
        value = getattr(batch, name)
        if value is not None and type(value) is not kind:
            raise ValueError('Exact typed per-objective supervision required')
    if spec.weights[0] and batch.flow is not None:
        x = batch.flow
        _tensor(x.values, shape, ref.dtype, ref.device); _mask(x.available, shape, ref.device)
        if x.available[:, list(batch.template.clamped)].any() or not torch.isfinite(x.values.detach()[x.available]).all():
            raise ValueError('Available free-node flow targets must be finite, never clamped conditions')
        counts['flow'] = int(x.available.sum())
    if spec.weights[1] and batch.tokens is not None:
        x = batch.tokens; _sha(x.alignment_sha256)
        _tensor(x.states, shape, ref.dtype, ref.device, detached=True, finite=True)
        tshape = (b, n, heads.decoder.spec.max_slots)
        _tensor(x.values, tshape, torch.long, ref.device); _mask(x.available, tshape, ref.device)
        if x.available[:, list(batch.template.clamped)].any():
            raise ValueError('Clamped token payloads cannot be training labels')
        allowed = set(heads.decoder.allowed_token_ids)
        for row, mask in zip(x.values.reshape(-1, tshape[-1]), x.available.reshape(-1, tshape[-1])):
            count = int(mask.sum())
            if not count: continue
            ids = row[:count].tolist()
            if (not mask[:count].all() or mask[count:].any()
                    or any(k not in allowed for k in ids) or ids[-1] != heads.decoder.stop_token
                    or heads.decoder.stop_token in ids[:-1]):
                raise ValueError('Complete contiguous token target with exactly one final stop required')
        counts['tokens'] = int(x.available.sum())
    if spec.weights[2] and batch.quality is not None:
        x = batch.quality; _tensor(x.states, shape, ref.dtype, ref.device, detached=True, finite=True)
        labels = x.labels
        if type(labels) is not ForestQualityLabels:
            raise ValueError('Exact source-bound criterion labels required')
        labels.__post_init__()
        if (labels.split != 'train' or labels.source_snapshot_sha256 != spec.source_snapshot_sha256
                or labels.label_policy_sha256 != heads.quality.spec.label_policy_sha256):
            raise ValueError('Same-source train-only quality-label policy required')
        q = len(heads.quality.spec.criteria)
        for values, mask, dims in ((labels.whole_values, labels.whole_available, (b, q)),
                                  (labels.node_values, labels.node_available, (b, n, q))):
            _tensor(values, dims, ref.dtype, ref.device); _mask(mask, dims, ref.device)
            known = values.detach()[mask]
            if not torch.isfinite(known).all() or ((known < 0) | (known > 1)).any():
                raise ValueError('Known criterion targets must be finite in [0,1]')
        if labels.node_available[:, list(batch.template.clamped)].any():
            raise ValueError('Clamped conditions cannot be quality targets')
        counts['quality'] = int(labels.whole_available.sum() + labels.node_available.sum())
    if spec.weights[3] and batch.analogy is not None:
        x = batch.analogy; _sha(x.evidence_sha256)
        if (type(x.queries) is not tuple or type(x.memories) is not tuple
                or not 1 <= len(x.queries) <= heads.analogy.spec.max_queries
                or not 1 <= len(x.memories) <= heads.analogy.spec.max_memory):
            raise ValueError('Bounded complete analogy query/memory tuples required')
        dims = (len(x.queries), len(x.memories))
        _mask(x.positive, dims, ref.device); _mask(x.known, dims, ref.device)
        for items, memory in ((x.queries, False), (x.memories, True)):
            if len({i.identity for i in items if type(i) is ForestAnalogyItem}) != len(items):
                raise ValueError('Unique typed analogy identities required')
            for item in items:
                heads.analogy._check_item(item, memory=memory, training=True)
                if memory:
                    f = item.forest
                    if f.states.requires_grad:
                        raise ValueError('Detached source-export memory required')
                    heads.flow._validate(item.record, item.task, f.states[None], f.states.new_ones(1), f)
                else:
                    # Validate context capacity without calling encode_context.
                    from .forest_flow import _condition_wire
                    raw = _condition_wire(item.record, item.task, spec.source_snapshot_sha256)
                    g = heads.flow.geometry; nh = g._source_backbone_reference[0].configuration()['heads']
                    if (len(raw) > heads.flow.spec.max_condition_bytes or len(item.record.nodes) > g.config['id_width']
                            or nh * (1 + len(item.record.nodes) + len(item.record.edges))**2 > heads.flow.spec.max_attention_elements):
                        raise ValueError('Complete analogy query exceeds native context capacity')
        disjoint = torch.tensor([[not(set(q.groups) & set(m.groups)) for m in x.memories]
                                for q in x.queries], dtype=torch.bool, device=ref.device)
        eligible = x.known & disjoint
        if ((x.positive & ~eligible).any() or not x.positive.any(1).all()
                or not (eligible & ~x.positive).any(1).all()):
            raise ValueError('Explicit disjoint positive and supported negative pairs required')
        counts['analogy'] = int(eligible.sum())
    return counts


def forest_joint_loss(heads, spec, batch, *, noise=None, time=None):
    """Fixed weighted sum of original per-head means; no optimizer or RNG calls."""
    counts = preflight_forest_batch(heads, spec, batch)
    if counts['flow']:
        if noise is None or time is None:
            raise ValueError('Caller-owned Gaussian noise and uniform solver times required')
        heads.flow._validate(batch.record, batch.task, noise, time, batch.template)
        if noise.requires_grad or time.requires_grad or noise.shape != batch.flow.values.shape:
            raise ValueError('Detached complete noise and solver-time tensors required')
    losses = {}; r, t, f = batch.record, batch.task, batch.template
    if counts['flow']:
        x = batch.flow
        losses['flow'] = forest_flow_matching_loss(heads.flow, r, t, f, x.values, noise, time, x.available)['loss']
    if counts['tokens']:
        x = batch.tokens
        losses['tokens'] = forest_token_loss(heads.decoder, r, t, x.states, f, x.values, x.available)['loss']
    if counts['quality']:
        x = batch.quality
        losses['quality'] = forest_quality_loss(heads.quality, r, t, x.states, f, x.labels)['loss']
    if counts['analogy']:
        x = batch.analogy
        losses['analogy'] = forest_analogy_loss(heads.analogy, x.queries, x.memories, x.positive, x.known)['loss']
    if counts.get('topology',0):
        x=batch.topology
        losses['topology']=forest_topology_loss(heads.topology,r,t,x.forests,
            root_count=x.root_count,split=batch.identity.split)['loss']
    loss = sum((spec.topology_weight if k=='topology'else spec.weights[OBJECTIVES.index(k)]) * v
               for k,v in losses.items()) if losses else None
    if loss is not None and (loss.ndim != 0 or not torch.isfinite(loss)):
        raise ValueError('Finite scalar combined native forest objective required')
    return dict(loss=loss, component_losses=losses, counts=counts,
                objective_sha256=spec.sha256, training_admission=False)


def _identity(value):
    if type(value) is torch.Tensor:
        return dict(tensor_sha256=_tensor_sha(value), shape=list(value.shape), dtype=str(value.dtype))
    if is_dataclass(value):
        return dict(type=type(value).__name__, fields={f.name: _identity(getattr(value, f.name)) for f in fields(value)
            if not(type(value)is ForestTrainingBatch and f.name=='topology'and value.topology is None)})
    if type(value) in (tuple, list):
        return dict(type=type(value).__name__, values=[_identity(x) for x in value])
    if type(value) is dict:
        return {k: _identity(v) for k, v in value.items()}
    if type(value) is bytes:
        return dict(bytes_sha256=sha256(value).hexdigest(), length=len(value))
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError('Unsupported object in source forest commitment')


def forest_batch_sha256(batch):
    """Content commitment only; hashes do not supply source rights or split proof."""
    if type(batch) is not ForestTrainingBatch:
        raise ValueError('Exact complete forest batch required')
    return sha256(_wire(_identity(batch))).hexdigest()
