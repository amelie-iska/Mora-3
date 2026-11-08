"""Staged train-only route bindings for three existing native objectives.

This module composes immutable source readers and registered full-owner branches;
it adds no labels and performs no model work during preparation.
"""
from dataclasses import dataclass

from mora3.data.admet import canonical, digest
from mora3.data.finite_union_native_rows import SelectedNativeSourceBatch
from mora3.data import string_score_current as string_data
from mora3.data import reactome_static_pathways as reactome_data
from mora3.data import biogrid_chemical_current as biogrid_data
from mora3.model.string_score_model import NativeStringScoreModel
from mora3.model.reactome_static_model import NativeReactomeStaticModel
from mora3.model.biogrid_chemical_model import NativeBioGRIDChemicalModel
from mora3.research.primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow,
    PreparedNativeTargets, finite_view_policy_sha256,
)


@dataclass(frozen=True)
class _Spec:
    key: str
    family: str
    task: str
    schema: str
    source_type: type
    branch_name: str
    branch_type: type
    objective_unit: str


_SPECS = {
    'string': _Spec('string_score_profile', string_data.FAMILY, string_data.TASK,
        string_data.SCHEMA, string_data.StringScoreCurrentSource, 'string_score',
        NativeStringScoreModel, 'mean squared error over four native score fractions'),
    'reactome': _Spec('reactome_static_participant_roles', reactome_data.FAMILY,
        reactome_data.TASK, reactome_data.SCHEMA, reactome_data.ReactomeStaticSource,
        'reactome_static', NativeReactomeStaticModel,
        'per-graph mean role cross entropy plus supplied-only log-stoichiometry MSE'),
    'biogrid_chemical': _Spec('biogrid_chemical_annotation', biogrid_data.FAMILY,
        biogrid_data.TASK, biogrid_data.SCHEMA, biogrid_data.BioGRIDChemicalCurrentSource,
        'biogrid_chemical', NativeBioGRIDChemicalModel,
        'mean cross entropy over original joint action/type occurrences'),
}


def _target_count(spec, packets):
    if spec.key == 'string_score_profile':
        # Four native score coordinates, including literal zero scores.
        return sum(len(packet.scores) for packet in packets)
    if spec.key == 'reactome_static_participant_roles':
        # Preserve role slots and separately expose measured quantity coordinates.
        return sum(len(packet.targets) + sum(value is not None for _, value in packet.targets)
                   for packet in packets)
    return len(packets)


def _route(source, current_view, *, snapshot_sha256, batch_size, spec):
    if type(source) is not spec.source_type:
        raise ValueError('Exact corresponding native current source object required')
    policy = finite_view_policy_sha256(current_view)
    source_view = source._view
    if (finite_view_policy_sha256(source_view) != source.axes.native_view_policy_sha256
            or source_view.root != current_view.root
            or canonical(source_view.manifest) != canonical(current_view.manifest)):
        raise ValueError('Source adapter and route must share the exact immutable finite native view')
    paths = set(source.source_paths)
    if (any(source_view._sealed_by_path.get(path)!=current_view._sealed_by_path.get(path) for path in paths)
            or {row for row in source_view._ordinal_exclusions if row[0] in paths}
               != {row for row in current_view._ordinal_exclusions if row[0] in paths}):
        raise ValueError('Later finite view changes this original native source scope')
    selected_paths, revisions = [], set()
    for descriptor in current_view.descriptors:
        path = descriptor['source_shard']['path']
        if path not in paths:
            continue
        matches = {row['source_revision'] for row in descriptor['source_task_schema_counts']
            if (row['source_family'], row['task_name'], row['native_schema'], row['split'],
                row['eligible_view_disposition']) == (spec.family, spec.task, spec.schema, 'train', 0)}
        if matches:
            selected_paths.append(path)
            revisions.update(matches)
    if set(selected_paths) != paths or not revisions:
        raise ValueError('Every adapter source shard must bind to retained original train task rows')
    route = FiniteNativeTrainRoute(current_view, task_key=spec.key,
        source_snapshot_sha256=snapshot_sha256, native_view_policy_sha256=policy,
        source_family=spec.family, task_name=spec.task, native_schema=spec.schema,
        source_revisions=tuple(sorted(revisions)), source_paths=tuple(sorted(selected_paths)),
        batch_size=batch_size)
    return route


@dataclass(frozen=True)
class NativeSourceRouteAdapter:
    """Prepare one source route against the exact shared owner branch."""
    source: object
    current_view: object
    snapshot_sha256: str
    batch_size: int
    kind: str

    def __post_init__(self):
        if self.kind not in _SPECS or type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('One of the three exact native objectives and positive raw batch size required')
        spec = _SPECS[self.kind]
        route = _route(self.source, self.current_view, snapshot_sha256=self.snapshot_sha256,
                       batch_size=self.batch_size, spec=spec)
        object.__setattr__(self, '_spec', spec)
        object.__setattr__(self, '_route', route)
        object.__setattr__(self, '_policy', canonical(self._document()))

    def _document(self):
        return dict(schema='Mora3.NativeSourceRouteAdapter/v1', task_key=self._spec.key,
            source_axes=self.source.axes.binding(), route_policy_sha256=self._route.native_view_policy_sha256,
            source_snapshot_sha256=self.snapshot_sha256, role='train',
            source_paths=self._route._paths, objective_unit=self._spec.objective_unit,
            target_unit=('observed native score positions' if self.kind == 'string' else
                'native role and supplied stoichiometry positions' if self.kind == 'reactome' else
                'original joint annotation occurrences'))

    def executable_binding(self):
        binding=self._document()
        if canonical(binding)!=self._policy:
            raise ValueError('Exact source, route or objective axes changed')
        return binding

    def train_binding(self):
        # Validate sealed source and model configuration before returning a binding.
        self.executable_binding()
        return NativeRouteBinding(self._route, self.prepare, self._spec.objective_unit,
                                  self.executable_binding()['target_unit'])

    def prepare(self, window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact finite native train route window required')
        batch = window.batch
        if type(batch) is not SelectedNativeSourceBatch or batch.original_split != 'train':
            raise ValueError('Original complete train source batch required')
        packets = self.source.packets(batch, split='train')
        count = _target_count(self._spec, packets)
        if count < 1:
            return PreparedNativeTargets(0, 'original_native_objective_has_no_observed_positions', None)
        def evaluate(owner):
            from mora3.model.full_owner import SourceBoundOwner
            if type(owner) is not SourceBoundOwner:
                raise ValueError('Exact registered shared full-owner required')
            owner.require_training_admission()
            branch = getattr(owner, self._spec.branch_name)
            if type(branch) is not self._spec.branch_type:
                raise ValueError('Exact existing registered native objective branch missing')
            if canonical(branch.axes.binding()) != canonical(self.source.axes.binding()):
                raise ValueError('Registered branch source axes differ from native route source')
            result=branch.objective(self.source, batch, split='train')
            if self.kind=='string':
                observed=result['observed_source_score_positions']
            elif self.kind=='reactome':
                observed=result['role_positions']+result['supplied_stoichiometry_positions']
            else:
                observed=result['original_occurrences']
            if observed!=count:
                raise ValueError('Registered objective and original observed position count differ')
            return result['loss']
        return PreparedNativeTargets(count, None, evaluate)


def bind_string(source, current_view, *, snapshot_sha256, batch_size):
    return NativeSourceRouteAdapter(source, current_view, snapshot_sha256, batch_size, 'string').train_binding()


def bind_reactome(source, current_view, *, snapshot_sha256, batch_size):
    return NativeSourceRouteAdapter(source, current_view, snapshot_sha256, batch_size, 'reactome').train_binding()


def bind_biogrid_chemical(source, current_view, *, snapshot_sha256, batch_size):
    return NativeSourceRouteAdapter(source, current_view, snapshot_sha256, batch_size,
                                    'biogrid_chemical').train_binding()
