"""Exact original-role bindings for guarded UniProt function and inverse sequence."""
from dataclasses import dataclass
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import FiniteUnionNativeView
from mora3.data.uniprot_native_annotations import (
    UNIPROT_FORWARD_TARGET_FIELDS, UNIPROT_NATIVE_FORWARD_TASK, UNIPROT_NATIVE_INVERSE_TASK,
    uniprot_native_forward_rows, uniprot_native_inverse_rows,
)
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.uniprot_native_objectives import SharedUniProtNativeAnnotationModel
from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets,
    finite_view_policy_sha256,
)

_FILTER = '2ac36e62d203f85a5db8f8a17d28bfabe0db2eba94f069f095262d3fa5ad5aba'
_VIEW_POLICY = '11b7bb1121567ec3098c044c67a9254b954242f37e9f43a4002655c87c29c046'
_SCHEMA = 'mora3.CorpusTask/v1'
_SHA = re.compile(r'[a-f0-9]{64}\Z')


@dataclass(frozen=True)
class UniProtNativeCampaignAdapter:
    view: FiniteUnionNativeView
    source_snapshot_sha256: str
    direction: str
    codec: PubChemByteCodec
    output_field: str | None = None
    batch_size: int = 1
    chunk_size: int = 1024

    def __post_init__(self):
        if (type(self.source_snapshot_sha256) is not str or not _SHA.fullmatch(self.source_snapshot_sha256)
                or self.direction not in ('forward','inverse')
                or (self.direction=='forward' and self.output_field not in UNIPROT_FORWARD_TARGET_FIELDS)
                or (self.direction=='inverse' and self.output_field is not None)
                or type(self.batch_size) is not int or self.batch_size<1
                or type(self.chunk_size) is not int or self.chunk_size<1
                or type(self.codec) is not PubChemByteCodec
                or finite_view_policy_sha256(self.view)!=_VIEW_POLICY):
            raise ValueError('Exact guarded UniProt source, direction, requested field and byte codec required')
        object.__setattr__(self,'_policy',canonical(self._binding()))

    @property
    def family(self):
        return 'UniProt-full-record-annotations' if self.direction=='inverse' else 'UniProt-function-annotations'

    @property
    def source_task(self):
        return UNIPROT_NATIVE_INVERSE_TASK if self.direction=='inverse' else UNIPROT_NATIVE_FORWARD_TASK

    @property
    def task_key(self):
        return self.source_task if self.direction=='inverse' else f'{self.source_task}:{self.output_field}'

    def _binding(self):
        return dict(schema='Mora3.UniProtNativeCampaignAdapter/v1',
            source_snapshot_sha256=self.source_snapshot_sha256,
            view_policy_sha256=finite_view_policy_sha256(self.view),
            source_family=self.family,source_task=self.source_task,task_key=self.task_key,
            native_schema=_SCHEMA,split='train',direction=self.direction,field=self.output_field,
            codec=self.codec.configuration(),batch_size=self.batch_size,chunk_size=self.chunk_size,
            objective='complete native byte-token plus EOS mean loss on one observed source field')

    def executable_binding(self):
        binding = self._binding()
        if canonical(binding)!=self._policy:
            raise ValueError('Native UniProt source, filter or byte codec binding changed')
        return binding

    def train_binding(self):
        self.executable_binding()
        paths=[]
        revisions=set()
        for descriptor in self.view.descriptors:
            matches=tuple(r for r in descriptor['source_task_schema_counts']
                if (r['source_family'],r['task_name'],r['native_schema'],r['split'],r['eligible_view_disposition'])
                ==(self.family,self.source_task,_SCHEMA,'train',0))
            if matches:
                paths.append(descriptor['source_shard']['path'])
                revisions.update(r['source_revision'] for r in matches)
        route=FiniteNativeTrainRoute(self.view,task_key=self.task_key,
            source_snapshot_sha256=self.source_snapshot_sha256,native_view_policy_sha256=_VIEW_POLICY,
            source_family=self.family,task_name=self.source_task,native_schema=_SCHEMA,
            source_revisions=tuple(sorted(revisions)),source_paths=tuple(sorted(paths)),
            batch_size=self.batch_size)
        return NativeRouteBinding(route,self.prepare,
            'native byte-token negative log likelihood in nats','observed byte tokens and EOS')

    def prepare(self,window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact finite original UniProt window required')
        batch=window.batch
        if self.direction=='inverse':
            rows=uniprot_native_inverse_rows(self.view,batch,split='train')
            targets=tuple(row.canonical_sequence.encode('ascii') for row in rows)
        else:
            rows=uniprot_native_forward_rows(self.view,batch,split='train')
            targets=tuple(raw for row in rows if (raw:=row.target_bytes(self.output_field)) is not None)
        if not targets:
            return PreparedNativeTargets(0,'original requested source field unavailable in this window',None)
        count=sum(len(self.codec.tokenizer.encode(raw))+1 for raw in targets)
        size=sum(map(len,targets))

        def evaluate(model):
            if type(model) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            model.require_training_admission()
            branch=model.uniprot_native_annotation
            if (type(branch) is not SharedUniProtNativeAnnotationModel
                    or canonical(branch.decoder.codec.configuration())!=canonical(self.codec.configuration())):
                raise ValueError('Exact native UniProt branch and fitted decoder codec required')
            if self.direction=='inverse':
                result=branch.inverse_objective(self.view,batch,split='train',chunk_size=self.chunk_size,recompute=True)
            else:
                result=branch.forward_objective(self.view,batch,self.output_field,split='train',
                                                chunk_size=self.chunk_size,recompute=True)
            if (result['observed'],result['missing'],result['exact_target_UTF8_bytes'])!=(len(targets),len(rows)-len(targets),size):
                raise ValueError('Native UniProt objective source, availability or complete byte count changed')
            return result['loss']

        return PreparedNativeTargets(count,None,evaluate)
