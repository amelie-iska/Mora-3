"""Opt-in canonical GOA interface sharing the native cooperative encoder.

The existing frozen ConvexTok vocabulary is reused without fitting GOA targets.
This is a new configuration, not a migration of byte-payload GOA checkpoints.
Source admission belongs to the packet reader; independent pools imply no joins.
"""
from .nucleotide_cooperative import RfamCooperativeModel
from .uniprot_go import NativeGOAHead, goa_graph_task


class GOACooperativeModel(RfamCooperativeModel):
    def __init__(self, *args, goa_packets, **kwargs):
        super().__init__(*args, **kwargs)
        from mora3.data.goa_streamed import StreamedGOASourcePool
        pool=goa_packets if type(goa_packets)is StreamedGOASourcePool else tuple(goa_packets)
        self.goa = NativeGOAHead(self.annotation.backbone.feature.out_features,
                                 pool,**(
            dict(parameterization_policy='mup-goa-v1',base_width=self.annotation._configuration['mup']['base_width'],
                output_mult=self.annotation._configuration['mup']['output_mult'])
            if self.parameterization_policy!='legacy' else {}))
        self.configuration = dict(self.configuration,
            architecture='unpaired-native-full-annotation-rfam-GOA-interface-v1',
            goa=self.goa.get_extra_state(),
            goa_encoder='existing-frozen-ConvexTok-positional-shared-TokenGT-v1',
            goa_payload=self.annotation.payload.get_extra_state(),
            goa_role='canonical-sequence-prediction-only-no-generation')

    def goa_logits(self, packets):
        packets = (self.goa.require_streamed_batch(packets)
            if self.goa.get_extra_state()['version']==3 else tuple(packets))
        if not packets:
            raise ValueError('nonempty native GOA batch required')
        graphs = []
        for packet in packets:
            record, task = goa_graph_task(packet)
            graphs.append(self.annotation.payload.forward_task(record, task=task,
                identifiers=self.annotation.sequence_identifiers()))
        return self.goa(self.annotation.backbone(graphs)[0][:, 0])

    def goa_objective(self, packets):
        if self.goa.get_extra_state()['version']!=3:packets=tuple(packets)
        return self.goa.objective(self.goa_logits(packets), packets)


from .nucleotide_cooperative import RfamCooperativeTrainer
from .annotation_trainer import _digest


class GOACooperativeTrainer(RfamCooperativeTrainer):
    """Opt-in thirteenth source objective; empty GOA pools receive no updates.

    Exact reader-issued pool/axis/ontology/applicability receipts bind the
    schedule. Quarantine remains masked by NativeGOAHead, never negative labels.
    Historical twelve-objective trainers and checkpoint bindings are unchanged.
    """
    def __init__(self, model, batch, pools, *, goa_packets, goa_batch_size, **kwargs):
        if not isinstance(model, GOACooperativeModel):
            raise ValueError('native GOA cooperative model required')
        if type(goa_batch_size) is not int or goa_batch_size <= 0:
            raise ValueError('positive integer GOA batch size required')
        packets = tuple(goa_packets)
        for packet in packets:
            if (packet.axis_ids != model.goa.axis_ids or
                    model.goa._receipt(packet)['packet_sha256'] not in model.goa._packet_hashes):
                raise ValueError('GOA packet outside bound native pool/axis')
        super().__init__(model, batch, pools, **kwargs)
        self.goa_packets = packets
        self.goa_batch_size = goa_batch_size
        if packets:
            self.kinds += ('goa',)
        self._goa_policy = self._goa_configuration()
        self.binding = _digest(dict(version=1, rfam_binding=self.binding,
            goa_policy=self._goa_policy, goa_batch_size=goa_batch_size,
            schedule=self.kinds, policy='equal-active-objective-cyclic-GOA-v1'))

    def _goa_configuration(self):
        payload = self.model.annotation.payload
        return dict(head=self.model.goa.get_extra_state(),
            pool=[self.model.goa._receipt(v) for v in self.goa_packets],
            payload=payload.get_extra_state(), max_tokens=payload.max_tokens,
            pooling=payload.pooling, configuration=self.model.get_extra_state())

    def _check_policy(self):
        super()._check_policy()
        if self._goa_configuration() != self._goa_policy:
            raise ValueError('GOA cooperative source/payload policy mismatch')

    def additional_objective(self, kind):
        if kind != 'goa':
            return super().additional_objective(kind)
        if not self.goa_packets:
            raise ValueError('absent GOA objective')
        size = min(self.goa_batch_size, len(self.goa_packets))
        start = (self.steps // len(self.kinds)) * size
        packets = tuple(self.goa_packets[(start+i) % len(self.goa_packets)]
                        for i in range(size))
        return self.model.goa_objective(packets), len(packets)
