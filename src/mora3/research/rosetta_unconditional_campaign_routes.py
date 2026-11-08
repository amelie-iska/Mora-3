"""Original target-only Rosetta profile-flow campaign route bindings."""

from dataclasses import dataclass
from hashlib import sha256
import json
import re

import torch

from mora3.data.admet import canonical
from mora3.data.finite_union_native_rows import FiniteUnionNativeView
from mora3.data.rosetta_unconditional_profiles import (
    load_rosetta_profile_axes, rosetta_profile_targets, rosetta_view_policy_sha256,
)
from mora3.model.full_owner import SourceBoundOwner
from mora3.model.rosetta_profile_flow import RosettaProfileFlow

from .primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow, PreparedNativeTargets,
    finite_view_policy_sha256,
)


FAMILIES = ('Rosetta-CellPainting', 'Rosetta-L1000')
TASK = 'unconditional_source_profile_modeling'
SCHEMA = 'mora3.ArrayTask/v1'
CENSUS = 'outputs/implementation/readiness-20261106-model-mup/rosetta-unconditional-array-census-v1/manifest.json'
CENSUS_SHA = 'd1b905c63885fd537fad23b549a3d55cebe78e3f777a8e7658c8c6ad56c17b15'


@dataclass(frozen=True)
class RosettaUnconditionalCampaignAdapter:
    current_view: FiniteUnionNativeView
    census_view: FiniteUnionNativeView
    family: str
    source_snapshot_sha256: str

    def __post_init__(self):
        if (self.family not in FAMILIES or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch('[0-9a-f]{64}', self.source_snapshot_sha256)
                or type(self.current_view) is not FiniteUnionNativeView
                or self.current_view.root != self.census_view.root
                or canonical(self.current_view.manifest) != canonical(self.census_view.manifest)):
            raise ValueError('Exact original Rosetta source views, family and campaign snapshot required')
        axes = load_rosetta_profile_axes(self.census_view.root, CENSUS, CENSUS_SHA, self.census_view)
        census = json.loads((self.census_view.root / CENSUS).read_bytes())
        paths = tuple(sorted(census['route_results'][self.family]['source_paths']))
        path_set = frozenset(paths)
        if (any(self.census_view._sealed_by_path[path] != self.current_view._sealed_by_path.get(path) for path in paths)
                or {p for p in self.census_view._ordinal_exclusions if p[0] in path_set}
                != {p for p in self.current_view._ordinal_exclusions if p[0] in path_set}):
            raise ValueError('Current Rosetta source coordinates differ from the original array census')
        object.__setattr__(self, '_axes', axes)
        object.__setattr__(self, '_paths', paths)
        object.__setattr__(self, '_policy', canonical(self._binding()))

    @property
    def task_key(self):
        return f'{self.family}:{TASK}'

    def _binding(self):
        return dict(
            schema='Mora3.RosettaUnconditionalCampaignAdapter/v1',
            source_snapshot_sha256=self.source_snapshot_sha256,
            current_view_policy_sha256=finite_view_policy_sha256(self.current_view),
            census_view_policy_sha256=rosetta_view_policy_sha256(self.census_view), source_census_sha256=CENSUS_SHA,
            axes=tuple(axis.sha256 for axis in self._axes), paths=self._paths, source_family=self.family,
            source_task=TASK, task_key=self.task_key, native_schema=SCHEMA, split='train', batch_size=1,
            noise='standard normal over all original target coordinates; private trial/window RNG',
            time='uniform interior binary24 grid; private trial/window RNG',
        )

    def executable_binding(self):
        binding = self._binding()
        if canonical(binding) != self._policy:
            raise ValueError('Original Rosetta axes, census or RNG policy changed')
        return binding

    def train_binding(self):
        binding = self.executable_binding()
        paths = frozenset(self._paths)
        revisions = tuple(sorted({
            row['source_revision']
            for item in self.current_view.descriptors if item['source_shard']['path'] in paths
            for row in item['source_task_schema_counts']
            if (row['source_family'], row['task_name'], row['native_schema'], row['split'],
                row['eligible_view_disposition']) == (self.family, TASK, SCHEMA, 'train', 0)
        }))
        route = FiniteNativeTrainRoute(
            self.current_view, task_key=self.task_key, source_snapshot_sha256=self.source_snapshot_sha256,
            native_view_policy_sha256=binding['current_view_policy_sha256'], source_family=self.family,
            task_name=TASK, native_schema=SCHEMA, source_revisions=revisions, source_paths=self._paths, batch_size=1,
        )
        return NativeRouteBinding(
            route, self.prepare, 'native assay-profile velocity MSE', 'complete source array positions',
        )

    def prepare(self, window):
        binding = self.executable_binding()
        if (type(window) is not NativeRouteWindow or window.original_source_rows != 1
                or window.batch.source_shard['path'] not in self._paths
                or window.batch.native_rows['source_family'].to_pylist() != [self.family]):
            raise ValueError('Exact single original Rosetta train window required')
        batch = window.batch
        packet, = rosetta_profile_targets(self.census_view, batch, self._axes, split='train')
        count = len(packet.values)

        def evaluate(owner):
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered source-bound owner required')
            owner.require_training_admission()
            branch = owner.rosetta_flow
            if (type(branch) is not RosettaProfileFlow or branch.axes != self._axes
                    or branch.source_census_sha256 != CENSUS_SHA
                    or branch.native_view_policy_sha256 != binding['census_view_policy_sha256']):
                raise ValueError('Exact registered Rosetta profile flow branch and census required')
            reference = branch.backbone.feature.weight
            if reference.dtype not in (torch.float32, torch.float64):
                raise ValueError('Exact FP32 or FP64 native Rosetta profile flow precision required')
            key = canonical([torch.initial_seed(), window.evidence_sha256, packet.source_coordinate[2]])
            seed = int.from_bytes(sha256(key).digest()[:8], 'big') % (2**63 - 1)
            generator = torch.Generator(device=reference.device).manual_seed(seed)
            time = torch.randint(1, 2**24, (1,), device=reference.device, generator=generator).to(reference.dtype) / 2**24
            noise = torch.randn((1, count), dtype=reference.dtype, device=reference.device, generator=generator)
            return branch.objective(self.census_view, batch, split='train', noise=noise, time=time)

        return PreparedNativeTargets(count, None, evaluate)
