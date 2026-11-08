"""Guarded train-only campaign routes over seven native sequence priors."""

from dataclasses import dataclass, field
import re

from mora3.data.admet import canonical, digest
from mora3.data.source_native_priors import NativePriorView, _ROUTES, source_native_prior_views
from mora3.model.annotation_vocabulary import AnnotationVocabulary
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.native_sequence_priors import NativeSequencePriorModel
from mora3.model.nucleotide_decoder import NucleotideCodec
from mora3.model.pubchem_decoder import PubChemByteCodec
from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets, finite_view_policy_sha256,
)


_SHA256 = re.compile(r'[a-f0-9]{64}\Z')
_SCHEMA = 'mora3.CorpusTask/v1'
_SOURCE_ROUTES = {detail[0]:(family,task,detail[3]) for (family,task),detail in _ROUTES.items()}
_NUCLEOTIDES = frozenset(('dna','rfam_rna','rnacentral_rna'))


def _prior_target_count(records,route,codec):
    if not records or any(type(record) is not NativePriorView or record.route!=route for record in records):
        raise ValueError('One exact original native prior route required')
    if route in _NUCLEOTIDES:
        symbols = NucleotideCodec()
        return sum(len(symbols.encode(record.target_bytes.decode('utf-8')))-1 for record in records)
    return sum(len(codec.encode(record.target_bytes))-1 for record in records)


@dataclass(frozen=True)
class NativePriorCampaignAdapter:
    view: object
    vocabulary: AnnotationVocabulary
    source_snapshot_sha256: str
    route: str
    max_total_target_bytes: int
    chunk_size: int = 1024
    _codec: PubChemByteCodec = field(init=False,repr=False,compare=False)
    _identity: bytes = field(init=False,repr=False)

    def __post_init__(self):
        if (
            type(self.vocabulary) is not AnnotationVocabulary or type(self.route) is not str or self.route not in _SOURCE_ROUTES
            or type(self.source_snapshot_sha256) is not str or not _SHA256.fullmatch(self.source_snapshot_sha256)
            or any(type(n) is not int or n<1 for n in (self.max_total_target_bytes,self.chunk_size))
        ):
            raise ValueError('Exact source, route, fitted codec and complete native prior byte budget required')
        object.__setattr__(self,'_codec',PubChemByteCodec(self.vocabulary.tokenizer()))
        object.__setattr__(self,'_identity',canonical(self._document()))

    @property
    def task_key(self):
        return 'native-prior:'+self.route

    @property
    def loss_unit(self):
        return 'full native symbol CE in nats' if self.route in _NUCLEOTIDES else 'full native byte-token CE in nats'

    @property
    def target_unit(self):
        return ('complete native nucleotide symbols and EOS' if self.route in _NUCLEOTIDES
                else 'native shortest-path payload tokens and EOS')

    def _document(self):
        family,task,revision = _SOURCE_ROUTES[self.route]
        return dict(schema='Mora3.NativePriorPrimaryCampaignAdapter/v1',
            source_snapshot_sha256=self.source_snapshot_sha256,native_view_policy_sha256=finite_view_policy_sha256(self.view),
            source_family=family,native_task=task,native_schema=_SCHEMA,source_revision=revision,route=self.route,
            role='train',validation='no original validation or test role for this source direction',
            payload_vocabulary_sha256=digest(self.vocabulary.specification_json.encode('utf-8')),
            byte_codec=self._codec.configuration(),nucleotide_symbols=NucleotideCodec.symbols,
            max_total_target_bytes=self.max_total_target_bytes,chunk_size=self.chunk_size,train_recompute=True,
            loss_unit=self.loss_unit,target_unit=self.target_unit)

    def executable_binding(self):
        scope = self._document()
        if canonical(scope)!=self._identity:
            raise ValueError('Exact native prior source snapshot/codec/resource binding changed')
        return scope

    def train_binding(self,*,batch_size):
        scope = self.executable_binding()
        family,task,revision = _SOURCE_ROUTES[self.route]
        paths = tuple(sorted(d['source_shard']['path'] for d in self.view.descriptors if any(
            (row['source_family'],row['task_name'],row['native_schema'],row['source_revision'],
             row['split'],row['eligible_view_disposition'])==(family,task,_SCHEMA,revision,'train',0)
            for row in d['source_task_schema_counts'])))
        route = FiniteNativeTrainRoute(self.view,task_key=self.task_key,
            source_snapshot_sha256=self.source_snapshot_sha256,native_view_policy_sha256=scope['native_view_policy_sha256'],
            source_family=family,task_name=task,native_schema=_SCHEMA,source_revisions=(revision,),
            source_paths=paths,batch_size=batch_size)
        return NativeRouteBinding(route,self.prepare,self.loss_unit,self.target_unit)

    def prepare(self,window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow or window.batch.original_split!='train':
            raise ValueError('Exact original native prior train window required')
        records = source_native_prior_views(self.view,window.batch,split='train')
        if sum(len(record.target_bytes) for record in records)>self.max_total_target_bytes:
            raise ValueError('Complete original native prior targets exceed the declared byte budget')
        count = _prior_target_count(records,self.route,self._codec)

        def evaluate(model):
            if (type(model) is not SourceBoundOwner
                    or model.get_extra_state()['source_snapshot_sha256']!=self.source_snapshot_sha256
                    or type(model.native_priors) is not NativeSequencePriorModel):
                raise ValueError('Exact registered native sequence prior owner and snapshot required')
            model.require_training_admission()
            branch = model.native_priors
            if (self.max_total_target_bytes>branch.max_total_target_bytes
                    or canonical(branch.decoder.codec.configuration())!=canonical(self._codec.configuration())
                    or branch.nucleotide.codec.symbols!=NucleotideCodec.symbols
                    or branch.decoder.shared.payload.get_extra_state()['vocabulary']!=self.vocabulary.specification_json):
                raise ValueError('Exact native sequence prior owner codec/resource policy required')
            return branch.objective(self.view,window.batch,split='train',chunk_size=self.chunk_size,recompute=True)

        return PreparedNativeTargets(count,None,evaluate)
