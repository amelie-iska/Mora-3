"""Field-specific reconstruction of observed full-record UniProt source values."""

import copy

from torch import nn

from mora3.data.admet import canonical
from mora3.data.provenance import Provenance
from mora3.data.schema import GraphRecord, Measurement, Node
from mora3.data.uniprot_full_record import FullRecordUniProtAxes, FullRecordUniProtPacket, FullRecordUniProtSource
from .mixed_payload_projection import exact_wire
from .mu_optimizer import _storage_signature
from .parameterization import parameter_ledger
from .pubchem_decoder import PubChemDecoder
from .sequence_mup import sequence_mup_parameter_roles
from .source_byte_objective import source_byte_objective, validate_byte_targets
from .source_byte_bounded_adjoint import source_byte_bounded_adjoint, validate_bounded_adjoint_decoder


_KIND = 'known-uniprot-full-record-query'
_SEQUENCE = 'canonical_amino_acid_sequence'
_FIELD = 'requested_source_field'
_PROVENANCE = (Provenance('uniprot-full-record-known-input', 'UniProt2026_03', 'input', '0'*64, 'source-audit-external'),)


def full_record_stream_bound(axes):
    def overhead(name):
        return 2+len(canonical(dict(name=name,units=None,status='observed',value_type='str')))
    return (len(canonical(dict(kind=_KIND,directed=None))) + overhead(_SEQUENCE) + axes.max_sequence_bytes
            + overhead(_FIELD) + max(len(field.name.encode('utf-8')) for field in axes.fields))


def _graph(condition, field):
    measurements = (
        Measurement(_SEQUENCE,condition.canonical_sequence,None,'observed',True),
        Measurement(_FIELD,field,None,'observed',True),
    )
    return GraphRecord('uniprot-full-record-known-input',(Node(_KIND,measurements),),(),_PROVENANCE)


class SharedUniProtFullRecordModel(nn.Module):
    """Use the exact shared byte decoder for one requested observed field.

    The reversible field codec is bound to the finite native field name, root
    type and byte extent. Ordered list occurrences and supplied nested values
    are retained. Separate provider-computed numeric properties never enter
    this field-byte loss or the known canonical-sequence condition.
    """

    def __init__(self, decoder, axes, *, max_batch_size=16, max_total_target_bytes=8*1024*1024):
        super().__init__()
        if type(decoder) is not PubChemDecoder or type(axes) is not FullRecordUniProtAxes:
            raise ValueError('Exact native shared byte decoder and full-record UniProt axes required')
        axes.binding()
        if (
            decoder.get_extra_state()['version'] != 2 or decoder.shared.payload.max_tokens < full_record_stream_bound(axes)
            or type(max_batch_size) is not int or max_batch_size < 1
            or type(max_total_target_bytes) is not int
            or max_total_target_bytes < max(field.max_wire_bytes for field in axes.fields)
        ):
            raise ValueError('Complete UniProt condition and largest full source field resource bounds required')
        self.decoder = decoder
        self._owner = (decoder,)
        self.axes = axes
        self.max_batch_size = max_batch_size
        self.max_total_target_bytes = max_total_target_bytes
        self._configuration = self._local()

    def _local(self):
        return dict(
            schema='Mora3.SharedUniProtFullRecordModel/v1', source=self.axes.binding(),
            max_batch_size=self.max_batch_size, max_total_target_bytes=self.max_total_target_bytes,
            max_complete_condition_stream_upper_bound=full_record_stream_bound(self.axes),
            condition='complete source canonical AA sequence and explicitly requested native source field only',
            target='one observed typed original annotation field, reversible complete wire and EOS',
            objective='mean source byte-token negative log likelihood over observed records for this field',
            unavailable='absent source field has no prediction/loss; all unavailable returns explicit disposition',
            computed_properties='separately typed source values; excluded from this lexical field objective',
            training_admission=False,
        )

    def get_extra_state(self):
        if (
            self.decoder is not self._owner[0] or set(self._modules) != {'decoder'}
            or self._parameters or self._buffers or not exact_wire(self._local(),self._configuration)
        ):
            raise ValueError('Exact UniProt full-record owner or field axis configuration drift')
        return dict(**copy.deepcopy(self._configuration),decoder=self.decoder.get_extra_state())

    def set_extra_state(self, state):
        if not exact_wire(state,self.get_extra_state()):
            raise ValueError('Exact full-record UniProt checkpoint source and field configuration required')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        import torch
        for key,value in self.state_dict().items():
            if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                raise ValueError('Native UniProt full-record checkpoint source/configuration mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _context(self, conditions, field):
        shared = self.decoder.shared
        ids = shared.sequence_identifiers()
        fields = {('node',0,_SEQUENCE),('node',0,_FIELD)}
        graphs = [shared.payload(_graph(condition,field),input_fields=fields,identifiers=ids)
                  for condition in conditions]
        return shared.backbone(graphs)[0][:,0]

    def _reference_field_objective(self, packets, field, *, chunk_size, recompute, backward_policy='checkpoint-v1'):
        self.get_extra_state()
        axis = self.axes.field(field)
        packets = tuple(packets)
        if (
            not 1 <= len(packets) <= self.max_batch_size
            or any(type(packet) is not FullRecordUniProtPacket for packet in packets)
            or type(chunk_size) is not int or chunk_size < 1 or type(recompute) is not bool
            or type(backward_policy) is not str or backward_policy not in ('checkpoint-v1','bounded-chunk-adjoint-v1')
            or backward_policy=='bounded-chunk-adjoint-v1' and not recompute
        ):
            raise ValueError('Complete bounded native UniProt records and explicit field objective policy required')
        known, targets = [], []
        for packet in packets:
            self.axes.require_condition(packet.condition)
            raw = packet.target(field)
            if raw is not None:
                axis.decode(raw)
                known.append(packet.condition)
                targets.append(raw)
        counts = dict(observed=len(known),missing=len(packets)-len(known),
                      exact_target_UTF8_bytes=sum(map(len,targets)),native_output=field)
        if not known:
            return dict(loss=None,missing_reason='source_field_unavailable',**counts)
        targets = validate_byte_targets(targets,max_batch_size=self.max_batch_size,
            max_total_target_bytes=self.max_total_target_bytes,chunk_size=chunk_size,recompute=recompute)
        if backward_policy=='bounded-chunk-adjoint-v1':
            validate_bounded_adjoint_decoder(self.decoder)
        hidden = self._context(tuple(known),field)
        args=dict(chunk_size=chunk_size,max_batch_size=self.max_batch_size,max_total_target_bytes=self.max_total_target_bytes)
        if backward_policy=='bounded-chunk-adjoint-v1':
            loss = source_byte_bounded_adjoint(self.decoder,hidden,targets,**args)
        else:
            loss = source_byte_objective(self.decoder,hidden,targets,recompute=recompute,**args)
        return dict(loss=loss,missing_reason=None,**counts)

    def field_objective(self, source, batch, field, *, split='train', chunk_size=1024, recompute=True, backward_policy='checkpoint-v1'):
        if type(source) is not FullRecordUniProtSource or not exact_wire(source.axes.binding(),self.axes.binding()):
            raise ValueError('Exact current native full-record UniProt source required')
        return self._reference_field_objective(source.packets(batch,split=split,field=field),field,
                                                chunk_size=chunk_size,recompute=recompute,backward_policy=backward_policy)


def uniprot_full_record_parameter_roles(model, *, identifier_axis):
    if type(model) is not SharedUniProtFullRecordModel:
        raise ValueError('Exact native full-record UniProt shared owner required')
    model.get_extra_state()
    roles = {'decoder.'+name:role for name,role in
             sequence_mup_parameter_roles(model.decoder,identifier_axis=identifier_axis).items()}
    _storage_signature(model)
    parameter_ledger(model,roles)
    return roles
