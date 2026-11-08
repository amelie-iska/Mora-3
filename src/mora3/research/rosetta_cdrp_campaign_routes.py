"""Original CDRP profile/structure objectives on source-fit axes and codec."""

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import re

import pyarrow.parquet as pq

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import FiniteUnionNativeView, read_exact
from mora3.data.molecule_profile import molecule_profile_view
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.molecule_profile import SharedMoleculeProfileModel
from mora3.model.profile_decoder import ProfileSelfiesDecoder, SelfiesCodec
from mora3.model.source_assembly_inputs import PINS

from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets,
    finite_view_policy_sha256,
)


SCHEMA = 'mora3.CorpusTask/v1'
AXIS_COUNTS = {'CellPainting': 1783, 'L1000': 977}


@dataclass(frozen=True)
class RosettaCDRPCampaignAdapter:
    view: FiniteUnionNativeView
    fit_view: FiniteUnionNativeView
    modality: str
    direction: str
    source_snapshot_sha256: str

    def __post_init__(self):
        if (type(self.modality) is not str or self.modality not in AXIS_COUNTS
                or self.direction not in ('molecule_to_profile', 'profile_to_molecule')
                or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch('[0-9a-f]{64}', self.source_snapshot_sha256)
                or type(self.view) is not FiniteUnionNativeView
                or type(self.fit_view) is not FiniteUnionNativeView
                or self.view.root != self.fit_view.root
                or canonical(self.view.manifest) != canonical(self.fit_view.manifest)):
            raise ValueError('Exact original CDRP source views, assay, direction and snapshot required')
        group_path, group_pin = PINS['rosetta_groups']
        codec_path, codec_pin = PINS['rosetta_codec']
        group_raw = (self.view.root / group_path).read_bytes()
        codec_raw = (self.view.root / codec_path).read_bytes()
        if sha256(group_raw).hexdigest() != group_pin or sha256(codec_raw).hexdigest() != codec_pin:
            raise ValueError('Exact retained CDRP group and train-fit codec files required')
        group_doc, codec_doc = json.loads(group_raw), json.loads(codec_raw)
        filter_ref = group_doc['source_effective_view']
        filter_raw = (self.view.root / filter_ref['path']).read_bytes()
        if (sha256(filter_raw).hexdigest() != filter_ref['sha256']
                or json.loads(filter_raw) != self.fit_view.effective_filter
                or group_doc['encoding_failures'] or codec_doc['axis_fit_split'] != 'all-retained-Rosetta-CDRP-inverse-train-targets-only'
                or codec_doc['heldout_symbols_added'] or any(codec_doc['heldout_unknown_symbols'].values())):
            raise ValueError('Exact complete source-train codec fit and original filter required')
        paths = {}
        revisions = set()
        for item in self.view.descriptors:
            for row in item['source_task_schema_counts']:
                if (row['source_family'], row['task_name'], row['native_schema'], row['split'],
                    row['eligible_view_disposition']) == (self.family, self.direction, SCHEMA, 'train', 0):
                    paths[item['source_shard']['path']] = item['original_group_sidecar']
                    revisions.add(row['source_revision'])
        if (not paths
                or any(self.view._sealed_by_path[path] != self.fit_view._sealed_by_path.get(path) for path in paths)
                or {p for p in self.view._ordinal_exclusions if p[0] in paths}
                != {p for p in self.fit_view._ordinal_exclusions if p[0] in paths}):
            raise ValueError('Current original CDRP source differs from the sealed source codec fit')
        object.__setattr__(self, '_paths', paths)
        object.__setattr__(self, '_revisions', tuple(sorted(revisions)))
        object.__setattr__(self, '_codec', SelfiesCodec.from_ordered_source_symbols(codec_doc['symbols']))
        object.__setattr__(self, '_max_tokens', codec_doc['max_tokens_including_BOS_EOS'])
        object.__setattr__(self, '_policy', canonical(self._binding()))

    @property
    def family(self):
        return f'Rosetta-CDRP-molecule-profile-{self.modality}'

    @property
    def task_key(self):
        return f'{self.family}:{self.direction}'

    def _binding(self):
        return dict(
            schema='Mora3.RosettaCDRPCampaignAdapter/v1', family=self.family, task=self.direction,
            task_key=self.task_key, native_schema=SCHEMA, split='train', batch_size=1,
            source_snapshot_sha256=self.source_snapshot_sha256,
            view_policy_sha256=finite_view_policy_sha256(self.view),
            fit_view_policy_sha256=finite_view_policy_sha256(self.fit_view),
            source_group_fit_sha256=PINS['rosetta_groups'][1], codec_sha256=PINS['rosetta_codec'][1],
            max_complete_target_tokens=self._max_tokens, symbols=self._codec.symbols,
            source_paths=tuple(sorted(self._paths)), source_revisions=self._revisions,
        )

    def executable_binding(self):
        doc = self._binding()
        if canonical(doc) != self._policy:
            raise ValueError('Exact original CDRP assay, source or codec policy changed')
        return doc

    def train_binding(self):
        doc = self.executable_binding()
        route = FiniteNativeTrainRoute(
            self.view, task_key=self.task_key, source_snapshot_sha256=self.source_snapshot_sha256,
            native_view_policy_sha256=doc['view_policy_sha256'], source_family=self.family,
            task_name=self.direction, native_schema=SCHEMA, source_revisions=self._revisions,
            source_paths=tuple(sorted(self._paths)), batch_size=1,
        )
        if self.direction == 'molecule_to_profile':
            return NativeRouteBinding(route,self.prepare,'native profile MSE','complete observed original assay coordinates')
        return NativeRouteBinding(route,self.prepare,'SELFIES token negative log likelihood in nats','native SELFIES tokens and EOS')

    def prepare(self, window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow or window.original_source_rows != 1:
            raise ValueError('Exact single original CDRP train window required')
        batch = window.batch
        self.view.require_native_rows(batch, split='train')
        path = batch.source_shard['path']
        side = self._paths[path]
        groups = pq.read_table(BytesIO(read_exact(side, root=self.view.root)))
        if groups.num_rows != batch.source_shard['rows']:
            raise ValueError('Complete original CDRP authoritative group sidecar required')
        ordinal, = batch.source_row_ordinals
        row, = batch.native_rows.to_pylist()
        group, = groups.slice(ordinal, 1).to_pylist()
        if row['task_id'] != group['task_id'] or row['task_name'] != self.direction or row['source_family'] != self.family:
            raise ValueError('Exact original CDRP train task and authoritative group required')
        effective_groups = json.loads(group['effective_groups_json'])
        if not set(json.loads(row['group_ids_json'])) <= set(effective_groups):
            raise ValueError('Original CDRP groups lost from authoritative sidecar')
        packet = molecule_profile_view(row, effective_groups=effective_groups)
        if len(packet.axis) != AXIS_COUNTS[self.modality]:
            raise ValueError('Complete original CDRP profile coordinate axis required')
        forward = self.direction == 'molecule_to_profile'
        count = (len(packet.axis) if forward else len(self._codec.encode(
            packet.targets()['canonical_isomeric_smiles'], max_tokens=self._max_tokens)) - 1)

        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            core = owner.core.parent.parent
            profile = core.profile
            if (type(profile) is not SharedMoleculeProfileModel
                    or tuple(profile.axes[self.modality]) != packet.axis):
                raise ValueError('Exact owner-bound CDRP profile model and original axis required')
            if forward:
                return profile.objective(profile((packet,)), (packet,))
            decoder = core.decoder
            if (type(decoder) is not ProfileSelfiesDecoder or decoder.profiles is not profile
                    or decoder.codec.symbols != self._codec.symbols or decoder.max_tokens < self._max_tokens):
                raise ValueError('Exact owner-bound CDRP inverse decoder and train-fit codec required')
            return decoder.objective((packet,))

        return PreparedNativeTargets(count, None, evaluate)
