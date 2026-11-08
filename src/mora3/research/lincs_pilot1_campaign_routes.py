"""Exact original LINCS-Pilot1 conditional profile campaign bindings."""

from dataclasses import dataclass
from hashlib import sha256
import json
import re

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import FiniteUnionNativeView
from mora3.data.lincs_native import read_lincs_native_views
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.lincs_inverse import SharedLincsInverseModel
from mora3.model.lincs_model import SharedLincsModel
from mora3.model.pubchem_decoder import PubChemByteCodec
from mora3.model.source_assembly_inputs import PINS

from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets,
    finite_view_policy_sha256,
)


PREFIX = 'Rosetta-LINCS-Pilot1-molecule-profile-'
SCHEMA = 'mora3.CorpusTask/v1'
MODALITIES = ('CellPainting', 'L1000')
DIRECTIONS = ('molecule_to_profile', 'profile_to_molecule')
COMPONENT_PATH, COMPONENT_SHA = PINS['lincs_manifest']


@dataclass(frozen=True)
class LINCSCampaignAdapter:
    view: FiniteUnionNativeView
    codec: PubChemByteCodec
    modality: str
    direction: str
    source_snapshot_sha256: str
    chunk_size: int = 1024

    def __post_init__(self):
        if (type(self.view) is not FiniteUnionNativeView or type(self.codec) is not PubChemByteCodec
                or type(self.modality) is not str or self.modality not in MODALITIES
                or type(self.direction) is not str or self.direction not in DIRECTIONS
                or type(self.chunk_size) is not int or self.chunk_size < 1
                or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch('[0-9a-f]{64}', self.source_snapshot_sha256)):
            raise ValueError('Exact current LINCS source, byte codec, assay and direction required')
        raw = (self.view.root / COMPONENT_PATH).read_bytes()
        if sha256(raw).hexdigest() != COMPONENT_SHA:
            raise ValueError('Exact qualified LINCS component manifest required')
        component = json.loads(raw)
        shards = {item['path']: item for item in component['shards']}
        relevant = {
            item['source_shard']['path']: item for item in self.view.descriptors
            if any(row['source_family'].startswith(PREFIX) for row in item['source_task_schema_counts'])
        }
        if (len(shards) != len(component['shards']) or set(relevant) != set(shards)
                or any(path in shards for path, _ in self.view._ordinal_exclusions)
                or any(
                    any(item['source_shard'][key] != shards[path][key] for key in ('sha256', 'bytes', 'rows'))
                    or any(item['original_group_sidecar'][key] != component['effective_group_lookup'][path][key]
                           for key in ('sha256', 'bytes', 'rows'))
                    for path, item in relevant.items()
                )):
            raise ValueError('Current LINCS source coordinates or authoritative groups differ from qualified originals')
        paths, revisions = [], set()
        for path, item in relevant.items():
            for row in item['source_task_schema_counts']:
                if (row['source_family'], row['task_name'], row['native_schema'], row['split'],
                    row['eligible_view_disposition']) == (self.family, self.direction, SCHEMA, 'train', 0):
                    paths.append(path)
                    revisions.add(row['source_revision'])
        object.__setattr__(self, '_paths', tuple(sorted(set(paths))))
        object.__setattr__(self, '_path_set', frozenset(paths))
        object.__setattr__(self, '_revisions', tuple(sorted(revisions)))
        object.__setattr__(self, '_policy', canonical(self._binding()))

    @property
    def family(self):
        return f'{PREFIX}{self.modality}'

    @property
    def task_key(self):
        return f'{self.family}:{self.direction}'

    def _binding(self):
        return dict(
            schema='Mora3.LINCSCampaignAdapter/v1', source_snapshot_sha256=self.source_snapshot_sha256,
            view_policy_sha256=finite_view_policy_sha256(self.view), qualified_component_sha256=COMPONENT_SHA,
            task_key=self.task_key, family=self.family, task=self.direction, native_schema=SCHEMA,
            split='train', paths=self._paths, revisions=self._revisions, batch_size=1,
            codec=self.codec.configuration(), chunk_size=self.chunk_size,
            source_dose='qualified literal export coordinate; unconfirmed physical units; no conversion',
        )

    def executable_binding(self):
        doc = self._binding()
        if canonical(doc) != self._policy:
            raise ValueError('Exact original LINCS source, assay or codec changed')
        return doc

    def train_binding(self):
        doc = self.executable_binding()
        route = FiniteNativeTrainRoute(
            self.view, task_key=self.task_key, source_snapshot_sha256=self.source_snapshot_sha256,
            native_view_policy_sha256=doc['view_policy_sha256'], source_family=self.family,
            task_name=self.direction, native_schema=SCHEMA, source_revisions=self._revisions,
            source_paths=self._paths, batch_size=1,
        )
        if self.direction == 'molecule_to_profile':
            return NativeRouteBinding(route, self.prepare, 'native observed assay-coordinate MSE',
                                      'observed source profile positions')
        return NativeRouteBinding(route, self.prepare, 'native byte-token negative log likelihood in nats',
                                  'native SMILES byte tokens and EOS')

    def prepare(self, window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow or window.original_source_rows != 1:
            raise ValueError('Exact single original LINCS train window required')
        batch = window.batch
        self.view.require_native_rows(batch, split='train')
        path = batch.source_shard['path']
        row, = batch.native_rows.to_pylist()
        if (path not in self._path_set or (row['source_family'], row['task_name'], row['native_schema'], row['split'])
                != (self.family, self.direction, SCHEMA, 'train')):
            raise ValueError('Exact selected original LINCS directional task required')
        packet, = read_lincs_native_views(
            COMPONENT_PATH, root=self.view.root, expected_manifest_sha256=COMPONENT_SHA,
            shard_paths=(path,), limit=1, task_ids=(row['task_id'],),
        )
        proof = packet.provenance()
        if (proof['record_id'] != row['record_id'] or proof['qualified_payload_sha256'] != row['native_payload_sha256']
                or proof['corpus_binding']['shard_sha256'] != batch.source_shard['sha256']
                or proof['consumed_file_bytes_verified'] is not True
                or (packet.direction, packet.modality) != (self.direction, self.modality)):
            raise ValueError('Current LINCS original row differs from the verified qualified source task')
        forward = self.direction == 'molecule_to_profile'
        count = (sum(packet.native_profile().observed_mask) if forward else
                 len(self.codec.encode(packet.targets()['canonical_isomeric_smiles'].encode('utf-8'))) - 1)

        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            model = owner.core.parent.parent.lincs
            if type(model) is not SharedLincsModel or model.axes[self.modality] != packet.axis:
                raise ValueError('Exact native owner-bound LINCS model and source axis required')
            if forward:
                return model.objective(model((packet,)), (packet,))
            inverse = owner.lincs_inverse
            if (type(inverse) is not SharedLincsInverseModel or inverse.lincs is not model
                    or canonical(inverse.decoder.codec.configuration()) != canonical(self.codec.configuration())):
                raise ValueError('Exact owner-bound LINCS inverse model and fitted byte codec required')
            return inverse.objective((packet,), chunk_size=self.chunk_size, recompute=True)

        return PreparedNativeTargets(count, None, evaluate)
