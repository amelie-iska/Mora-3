"""Original-role native UniProt full-record field campaign binding."""

from dataclasses import dataclass
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import FiniteUnionNativeView
from mora3.data.uniprot_full_record import FAMILY, SCHEMA, TASK, FullRecordUniProtSource
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.uniprot_full_record_model import SharedUniProtFullRecordModel

from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets,
    finite_view_policy_sha256,
)


@dataclass(frozen=True)
class UniProtFullRecordCampaignAdapter:
    view: FiniteUnionNativeView
    source: FullRecordUniProtSource
    codec: PubChemByteCodec
    field: str
    source_snapshot_sha256: str
    chunk_size: int = 1024

    def __post_init__(self):
        if (type(self.source) is not FullRecordUniProtSource
                or type(self.codec) is not PubChemByteCodec
                or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch('[0-9a-f]{64}', self.source_snapshot_sha256)
                or type(self.field) is not str or type(self.chunk_size) is not int
                or self.chunk_size < 1):
            raise ValueError('Exact native UniProt full-record source, field and byte codec required')
        self.source.axes.field(self.field)
        object.__setattr__(self, '_policy', canonical(self._binding()))

    @property
    def task_key(self):
        return f'{TASK}:{self.field}'

    def _binding(self):
        axes = self.source.axes
        if axes.view_policy_sha256 != finite_view_policy_sha256(self.view):
            raise ValueError('Exact retained UniProt full-record view policy required')
        return dict(
            schema='Mora3.UniProtFullRecordCampaignAdapter/v1', task_key=self.task_key,
            source_family=FAMILY, source_task=TASK, native_schema=SCHEMA, split='train',
            axes=axes.binding(), paths=self.source.source_paths, field=self.field,
            source_snapshot_sha256=self.source_snapshot_sha256,
            codec=self.codec.configuration(), batch_size=1, chunk_size=self.chunk_size,
            backward_policy='bounded-chunk-adjoint-v1',
        )

    def executable_binding(self):
        binding = self._binding()
        if canonical(binding) != self._policy:
            raise ValueError('Native UniProt full-record axes, source paths or codec changed')
        return binding

    def train_binding(self):
        self.executable_binding()
        view = self.view
        paths = self.source.source_paths
        path_set = frozenset(paths)
        revisions = tuple(sorted({
            row['source_revision']
            for descriptor in view.descriptors if descriptor['source_shard']['path'] in path_set
            for row in descriptor['source_task_schema_counts']
            if (row['source_family'], row['task_name'], row['native_schema'], row['split'],
                row['eligible_view_disposition']) == (FAMILY, TASK, SCHEMA, 'train', 0)
        }))
        route = FiniteNativeTrainRoute(
            view, task_key=self.task_key, source_snapshot_sha256=self.source_snapshot_sha256,
            native_view_policy_sha256=self.source.axes.view_policy_sha256,
            source_family=FAMILY, task_name=TASK, native_schema=SCHEMA,
            source_revisions=revisions, source_paths=paths, batch_size=1,
        )
        return NativeRouteBinding(
            route, self.prepare, 'native byte-token negative log likelihood in nats',
            'observed byte tokens and EOS',
        )

    def prepare(self, window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow or window.original_source_rows != 1:
            raise ValueError('Exact single original UniProt full-record window required')
        batch = window.batch
        packet, = self.source.packets(batch, split='train', field=self.field)
        raw = packet.target(self.field)
        if raw is None:
            return PreparedNativeTargets(0, 'original requested source field unavailable', None)
        count = len(self.codec.encode(raw)) - 1

        def evaluate(model):
            if type(model) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            model.require_training_admission()
            branch = model.uniprot_full_record
            if (type(branch) is not SharedUniProtFullRecordModel
                    or canonical(branch.decoder.codec.configuration()) != canonical(self.codec.configuration())):
                raise ValueError('Exact native UniProt full-record branch and fitted decoder codec required')
            result = branch.field_objective(
                self.source, batch, self.field, split='train', chunk_size=self.chunk_size,
                recompute=True, backward_policy='bounded-chunk-adjoint-v1',
            )
            if ((result['observed'], result['missing'], result['exact_target_UTF8_bytes'],
                 result['native_output']) != (1, 0, len(raw), self.field)):
                raise ValueError('Native UniProt full-record objective changed source availability or wire')
            return result['loss']

        return PreparedNativeTargets(count, None, evaluate)
