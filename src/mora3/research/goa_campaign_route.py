"""Native original-role GOA sign supervision over the current streamed pool."""

from dataclasses import dataclass
import re
from threading import RLock

from mora3.data.admet import canonical
from mora3.data.goa_streamed import GOAFormOutsideCanonicalScope, StreamedGOASourcePool
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.lincs_cooperative import LincsCooperativeModel

from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets,
    finite_view_policy_sha256,
)


FAMILY = 'UniProt-GOA-assertion-packets'
TASK = 'source_GOA_assertion_packet_prediction'
SCHEMA = 'mora3.CorpusTask/v1'
REVISION = 'GOAannotation2026-07-28;exactcurrentKBsequence2026_03'


@dataclass(frozen=True)
class GOACampaignAdapter:
    pool: StreamedGOASourcePool
    source_snapshot_sha256: str

    def __post_init__(self):
        if (type(self.pool) is not StreamedGOASourcePool
                or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch('[0-9a-f]{64}', self.source_snapshot_sha256)
                or self.pool.head_binding().get('native_filter_policy') != 'combined-study-overlay-v2'):
            raise ValueError('Exact current original-role GOA source pool and snapshot required')
        object.__setattr__(self, '_lock', RLock())
        object.__setattr__(self, '_policy', canonical(self._binding()))

    def _binding(self):
        return dict(
            schema='Mora3.GOACampaignAdapter/v1', source_snapshot_sha256=self.source_snapshot_sha256,
            pool=self.pool.head_binding(), view_policy_sha256=finite_view_policy_sha256(self.pool.native_view),
            task_key=TASK, family=FAMILY, native_schema=SCHEMA, revision=REVISION,
            split='train', batch_size=1,
            objective='BCE with logits over observed signs only; conflicts and unknowns masked',
        )

    def executable_binding(self):
        binding = self._binding()
        if canonical(binding) != self._policy:
            raise ValueError('Current streamed GOA source pool or axis changed')
        return binding

    def train_binding(self):
        binding = self.executable_binding()
        route = FiniteNativeTrainRoute(
            self.pool.native_view, task_key=TASK, source_snapshot_sha256=self.source_snapshot_sha256,
            native_view_policy_sha256=binding['view_policy_sha256'], source_family=FAMILY,
            task_name=TASK, native_schema=SCHEMA, source_revisions=(REVISION,),
            source_paths=self.pool.source_paths, batch_size=1,
        )
        return NativeRouteBinding(
            route, self.prepare, 'native observed sign binary cross-entropy in nats',
            'observed GOA assertion signs',
        )

    def prepare(self, window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow or window.original_source_rows != 1:
            raise ValueError('Exact single original GOA train route window required')
        try:
            with self._lock:
                batch = self.pool.bind_native_rows(window.batch, split='train')
        except GOAFormOutsideCanonicalScope:
            return PreparedNativeTargets(0, 'original GOA form requires a separate source policy', None)
        packet, = batch.packets
        observed = sum(packet.observed)
        if observed == 0:
            return PreparedNativeTargets(0, 'original GOA packet has no observed signs', None)

        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            model = owner.core.parent.parent
            if type(model) is not LincsCooperativeModel:
                raise ValueError('Exact registered cooperative GOA owner required')
            with self._lock:
                return model.goa_objective(batch)

        return PreparedNativeTargets(observed, None, evaluate)
