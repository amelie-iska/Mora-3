"""Source-observed entry-level GO prediction on shared backbone latents.

Unknown labels are not negatives. KB GO is incomplete; positive-only supervision
can collapse and is not a calibrated closed-world classifier. Full assertions,
source evidence, aspects and form scopes remain in the immutable reader targets.
No ontology propagation or cross-entry label transfer is performed here.
"""
import re
import torch
from torch import nn
from torch.nn import functional as F
from mora3.data.uniprot_annotations import AnnotationPredictionView


# Immutable syntax only; never cache validation results or source coordinates.
_GO_SYNTAX = re.compile(r'GO:\d{7}')


def _axis(axis_ids):
    axis = tuple(axis_ids)
    if (not axis or len(set(axis)) != len(axis)
            or any(not isinstance(x, str) or _GO_SYNTAX.fullmatch(x) is None for x in axis)):
        raise ValueError('nonempty unique ordered GO axis required')
    return axis


def go_targets(views, axis_ids, *, device='cpu', dtype=torch.float32, exclude_nd=False):
    """Project present source assertions only, excluding isoform-scoped references.

    exclude_nd opts into the new ND exclusion; historical checkpoint callers
    retain their original policy until a versioned objective binds the change.
    Explicit NOT may be in either source qualifier field. Conflicting positive/
    negative assertions fail closed rather than choosing one. Unknown assertions
    outside the requested head axis remain losslessly available on each view.
    """
    axis = _axis(axis_ids)
    if not views or any(not isinstance(v, AnnotationPredictionView) for v in views):
        raise ValueError('nonempty prediction view batch required')
    index = {x: i for i, x in enumerate(axis)}
    y = torch.zeros(len(views), len(axis), device=device, dtype=dtype)
    mask = torch.zeros_like(y, dtype=torch.bool)
    for row, view in enumerate(views):
        signs = {}
        for ref in view.targets()['annotation_packet'].get('uniProtKBCrossReferences', []):
            term = ref.get('id')
            if ref.get('database') != 'GO' or ref.get('isoformId') or term not in index:
                continue
            evidence = [p.get('value') for p in ref.get('properties', [])
                        if p.get('key') == 'GoEvidenceType']
            if exclude_nd and any(isinstance(e, str) and e.split(':', 1)[0] == 'ND' for e in evidence):
                continue
            qualifiers = [p.get('value', '') for p in ref.get('properties', [])
                          if p.get('key') in ('Qualifier', 'GoAnnotationQualifier')]
            if any(not isinstance(q, str) for q in qualifiers):
                raise ValueError('invalid source qualifier')
            positive = not any('NOT' in q.split('|') for q in qualifiers)
            if term in signs and signs[term] != positive:
                raise ValueError('conflicting source GO polarity')
            signs[term] = positive
        for term, positive in signs.items():
            y[row, index[term]] = float(positive)
            mask[row, index[term]] = True
    return y, mask


def observed_go_loss(logits, targets, observed):
    """Mean BCE over observed assertions; zero connected loss if none observed.

    Index before BCE so unknown NaN targets never enter arithmetic or gradients.
    """
    if (logits.ndim != 2 or logits.shape != targets.shape or logits.shape != observed.shape
            or observed.dtype != torch.bool or targets.device != logits.device
            or observed.device != logits.device or not logits.is_floating_point()
            or not targets.is_floating_point()):
        raise ValueError('matching logits/targets and Boolean observation mask required')
    z, y = logits[observed], targets[observed]
    if not torch.isfinite(z).all() or not torch.isfinite(y).all() or ((y != 0) & (y != 1)).any():
        raise ValueError('finite logits and binary observed labels required')
    if z.numel() == 0:
        return z.sum()
    return F.binary_cross_entropy_with_logits(z, y)


class GOAnnotationHead(nn.Module):
    """Head usable on existing TokenGT graph readout; ordered label axis is bound."""
    def __init__(self, width, axis_ids):
        super().__init__()
        self.axis_ids = _axis(axis_ids)
        self.output = nn.Linear(width, len(self.axis_ids))

    def forward(self, latent):
        return self.output(latent)

    def get_extra_state(self):
        return dict(version=1, axis_ids=self.axis_ids, missing_is_negative=False,
                    objective='observed-entry-GO-BCE-v1')

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('GO axis/objective checkpoint mismatch')


def annotation_graph_task(view):
    """Bridge an admitted prediction view to shared payload/task interfaces.

    One known protein entity; complete lossless targets are an excluded JSON
    measurement, never graph structure or encoder metadata. This does not admit
    new records, infer family splits, or authorize annotation-conditioned generation.
    Caller must obtain views from the source-validating prediction reader.
    """
    import hashlib
    from mora3.data.schema import GraphRecord, Node, Measurement
    from mora3.data.provenance import Provenance, LeakageKey
    from mora3.data.tasks import ConditionalTask
    if not isinstance(view, AnnotationPredictionView):
        raise ValueError('prediction view required')
    p = view.provenance()
    if p.get('split') != 'train' or p.get('license') != 'CC-BY-4.0':
        raise ValueError('train-only rights scope required')
    ref = p['source_ref']
    record = GraphRecord('uniprot:' + view.accession, (Node('protein', (
        Measurement('sequence', view.inputs()['canonical_sequence'], None, 'observed', True),
        Measurement('annotations', view.target_bytes.decode('utf-8'), None, 'observed', True))),), (),
        (Provenance('UniProtKB', ref['source_release'], view.accession,
                    ref['raw_file_sha256'], p['license']),),
        (LeakageKey('sequence:protein:sha256:v1', hashlib.sha256(view.sequence.encode()).hexdigest()),))
    task = ConditionalTask(((('node', 0, 'sequence'), 'condition'),
                            (('node', 0, 'annotations'), 'target')), topology_known=True)
    return record, task


# GO source short codes; ND is explicitly a no-biological-data placeholder.
_GOA_EVIDENCE = frozenset(('EXP','IDA','IPI','IMP','IGI','IEP','HTP','HDA',
    'HMP','HGI','HEP','IBA','IBD','IKR','IRD','ISS','ISO','ISA','ISM','IGC',
    'RCA','TAS','NAS','IC','IEA'))


def goa_targets(assertion_rows, axis_ids, *, subjects, device='cpu', dtype=torch.float32):
    """Conservative entry/form-specific projection of raw QuickGO assertions.

    This pure target projector does not grant corpus admission or authenticate
    subjects. Callers must bind each subject to its exact source sequence and
    leakage group. No canonical/form transfer, ontology propagation, ECO-to-GO
    inference or experimental-status inference is performed. Context extensions
    require a separate conditional objective; they are excluded here. Original
    assertions remain caller-owned and unmodified, including excluded evidence.
    """
    axis = _axis(axis_ids)
    if (not assertion_rows or len(assertion_rows) != len(subjects)
            or any(not isinstance(s, str) or not s.startswith('UniProtKB:')
                   or not s[len('UniProtKB:'):] for s in subjects)):
        raise ValueError('nonempty rows and exact UniProtKB subjects required')
    index = {term: i for i, term in enumerate(axis)}
    y = torch.zeros(len(assertion_rows), len(axis), device=device, dtype=dtype)
    if not y.is_floating_point():
        raise ValueError('floating target dtype required')
    mask = torch.zeros_like(y, dtype=torch.bool)
    for row, (assertions, subject) in enumerate(zip(assertion_rows, subjects)):
        signs = {}
        for a in assertions:
            term = a.get('goId')
            if (a.get('geneProductId') != subject or term not in index
                    or a.get('goEvidence') not in _GOA_EVIDENCE
                    or a.get('extensions')):
                continue
            qualifier = a.get('qualifier')
            if not isinstance(qualifier, str) or not qualifier:
                continue
            parts = qualifier.split('|')
            if any(not part for part in parts):
                raise ValueError('invalid source GOA qualifier')
            relations = set(parts) - {'NOT'}
            allowed = {'enables','contributes_to','involved_in','acts_upstream_of',
                       'acts_upstream_of_or_within','acts_upstream_of_positive_effect',
                       'acts_upstream_of_negative_effect',
                       'acts_upstream_of_or_within_positive_effect',
                       'acts_upstream_of_or_within_negative_effect',
                       'located_in','is_active_in','colocalizes_with','part_of'}
            if not relations or not relations <= allowed:
                continue
            positive = 'NOT' not in parts
            if term in signs and signs[term] != positive:
                raise ValueError('conflicting source GOA polarity')
            signs[term] = positive
        for term, positive in signs.items():
            y[row, index[term]] = float(positive)
            mask[row, index[term]] = True
    return y, mask


class NativeGOAHead(GOAnnotationHead):
    """Opt-in full-axis GOA objective on shared sequence-only latents.

    Bind a finite reader-verified source pool, including quarantine decisions.
    Hashes authenticate later use against that pool, not arbitrary caller-created
    dataclasses. Construct only after the source/group/sequence/axis reader gates.
    No historical GO head or checkpoint semantics are changed.
    """
    policy = 'exact-subject-coarse-GOA-observed-BCE-v1'

    @staticmethod
    def _receipt(view):
        import hashlib
        import json
        from mora3.data.goa_packets import ApplicableGOAPacket
        if not isinstance(view, ApplicableGOAPacket):
            raise ValueError('native applicable GOA packet required')
        p = view.packet.provenance()
        objective = p.get('goa_objective', {})
        if (objective.get('policy') != NativeGOAHead.policy
                or objective.get('status') != view.status
                or view.status not in ('applicable', 'quarantined_polarity_conflict')
                or p.get('sequence_source_verified') is not True
                or not p.get('authoritative_groups') or not p.get('groups')
                or p.get('generation_eligible') is not False
                or objective.get('unknown_is_negative') is not False
                or len(view.labels) != len(view.axis_ids)
                or len(view.observed) != len(view.axis_ids)
                or any(type(x) is not bool for x in view.observed)
                or any(x not in (0., 1.) for x in view.labels)
                or (view.status != 'applicable' and any(view.observed))):
            raise ValueError('native GOA policy/source/mask mismatch')
        h = hashlib.sha256()
        for raw in (view.packet.condition_bytes, view.packet.target_bytes,
                    view.packet.provenance_bytes,
                    json.dumps([view.axis_ids,view.labels,view.observed,view.status],
                               separators=(',', ':')).encode()):
            h.update(len(raw).to_bytes(8, 'big')); h.update(raw)
        return dict(packet_sha256=h.hexdigest(), subject=p['subject'],
                    status=view.status, objective=objective)

    def __init__(self, width, packets, *, parameterization_policy='legacy',base_width=None,output_mult=1.0):
        from mora3.data.goa_streamed import StreamedGOASourcePool
        streamed=type(packets)is StreamedGOASourcePool
        if streamed:
            if parameterization_policy!='mup-goa-v1':
                raise ValueError('streamed native source requires explicit primary GOA parameterization')
            axis=packets.axis_ids
            streamed_binding=packets.head_binding()
            receipts=[]
        else:
            if not packets:
                raise ValueError('nonempty verified native GOA pool required')
            axis = packets[0].axis_ids
            if any(v.axis_ids != axis for v in packets):
                raise ValueError('exact ordered native GOA axis required')
            receipts = [self._receipt(v) for v in packets]
        import math
        import copy
        if (parameterization_policy not in ('legacy','mup-goa-v1')
                or (parameterization_policy=='legacy' and (base_width is not None or output_mult!=1))
                or (parameterization_policy!='legacy' and (type(width) is not int or type(base_width) is not int
                    or base_width<1 or width<base_width or type(output_mult) not in (int,float)
                    or not math.isfinite(output_mult) or output_mult<=0))):
            raise ValueError('explicit compatible native GOA readout parameterization required')
        super().__init__(width, axis)
        self._native_binding = dict(version=1, policy=self.policy, axis_ids=axis,
                                    packets=receipts, generation_eligible=False)
        self._packet_hashes = frozenset(r['packet_sha256'] for r in receipts)
        self._streamed_owned_binding=None
        if streamed:
            self._native_binding=dict(version=3,policy=self.policy,axis_ids=axis,
                streamed_pool=copy.deepcopy(streamed_binding),generation_eligible=False)
            self._streamed_owned_binding=copy.deepcopy(streamed_binding)
        self.parameterization_policy=parameterization_policy
        self._goa_mup=None
        self._goa_owned_mup=None
        if parameterization_policy!='legacy':
            with torch.no_grad():
                ratio=math.sqrt(width/base_width)
                self.output.weight.mul_(ratio);self.output.bias.mul_(ratio)
            self._goa_mup=dict(schema='native-GOA-readout-parameterization-v1',policy=parameterization_policy,
                width=width,base_width=base_width,output_mult=float(output_mult),
                readout='native-fresh-times-sqrt-width-over-base-input-divided-by-width-over-base')
            self._goa_owned_mup=copy.deepcopy(self._goa_mup)
            self._native_binding.update(version=3 if streamed else 2,mup=copy.deepcopy(self._goa_mup))

    def _validate_goa_parameterization(self):
        from .mixed_payload_projection import exact_wire
        spec=self._goa_mup
        if (spec is None or self.parameterization_policy!='mup-goa-v1'
                or not exact_wire(spec,self._goa_owned_mup)
                or not exact_wire(self._native_binding.get('mup'),self._goa_owned_mup)
                or type(self._native_binding.get('version')) is not int or self._native_binding['version'] not in(2,3)
                or (self._native_binding['version']==3 and(
                    type(self._streamed_owned_binding)is not dict
                    or self._streamed_owned_binding.get('schema')!='Mora3.StreamedCanonicalGOASourcePool/v1'
                    or not exact_wire(self._native_binding.get('streamed_pool'),self._streamed_owned_binding)))
                or (self._native_binding['version']==2 and self._streamed_owned_binding is not None)
                or self._native_binding['policy']!=self.policy or self._native_binding['axis_ids']!=self.axis_ids
                or type(self.output) is not nn.Linear or self.output._modules or self.output._buffers
                or (self.output.in_features,self.output.out_features)!=(spec['width'],len(self.axis_ids))
                or tuple(self.output.weight.shape)!=(len(self.axis_ids),spec['width'])
                or tuple(self.output.bias.shape)!=(len(self.axis_ids),)):
            raise ValueError('native GOA source/axis/readout configuration drift')

    def forward(self,latent):
        if self._goa_mup is not None:
            self._validate_goa_parameterization();spec=self._goa_mup
            latent=latent*(spec['output_mult']/(spec['width']/spec['base_width']))
        return self.output(latent)

    def get_extra_state(self):
        import copy
        if self._goa_mup is not None:self._validate_goa_parameterization()
        return copy.deepcopy(self._native_binding)

    def set_extra_state(self, state):
        from .mixed_payload_projection import exact_wire
        mismatch=(not exact_wire(state,self.get_extra_state()) if self._goa_mup is not None
                  else state!=self.get_extra_state())
        if mismatch:
            raise ValueError('native GOA source/axis/policy checkpoint mismatch')

    def require_streamed_batch(self,batch):
        from mora3.data.goa_streamed import StreamedGOABatch,StreamedGOASourcePool
        self.get_extra_state()
        if (self._native_binding['version']!=3 or type(batch)is not StreamedGOABatch
                or type(batch.pool)is not StreamedGOASourcePool):
            raise ValueError('exact current streamed GOA native source batch required')
        return batch.pool.require_batch(batch,binding=self._streamed_owned_binding)

    def objective(self, logits, packets):
        if self._native_binding['version']==3:
            packets=self.require_streamed_batch(packets)
        if (not packets or logits.shape != (len(packets), len(self.axis_ids))
                or any(v.axis_ids != self.axis_ids for v in packets)):
            raise ValueError('native GOA batch/axis mismatch')
        for v in packets:
            receipt=self._receipt(v)
            if self._native_binding['version']!=3 and receipt['packet_sha256'] not in self._packet_hashes:
                raise ValueError('packet outside bound native GOA pool')
        y = logits.new_tensor([v.labels for v in packets])
        mask = torch.tensor([v.observed for v in packets],
                            device=logits.device, dtype=torch.bool)
        return observed_go_loss(logits, y, mask)


def goa_graph_task(view):
    """Known single protein, sequence condition and excluded complete GOA target.

    Audit-only provenance/groups never become predictor fields. No cellular
    joins or annotation-conditioned generation are implied by this bridge.
    """
    import hashlib
    from mora3.data.schema import GraphRecord, Node, Measurement
    from mora3.data.provenance import Provenance, LeakageKey
    from mora3.data.tasks import ConditionalTask
    NativeGOAHead._receipt(view)
    packet = view.packet
    p = packet.provenance()
    sequence = packet.inputs()['protein_sequence']
    record = GraphRecord('goa:' + p['subject'], (Node('protein', (
        Measurement('sequence', sequence, None, 'observed', True),
        Measurement('annotations', packet.target_bytes.decode('utf-8'),
                    None, 'observed', True))),), (),
        (Provenance('GOA', 'source-bound-packet', p['subject'],
                    hashlib.sha256(packet.provenance_bytes).hexdigest(), 'CC-BY-4.0'),),
        (LeakageKey('sequence:protein:sha256:v1',
                    hashlib.sha256(sequence.encode()).hexdigest()),))
    task = ConditionalTask(((('node', 0, 'sequence'), 'condition'),
                            (('node', 0, 'annotations'), 'target')), topology_known=True)
    return record, task
