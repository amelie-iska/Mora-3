"""Staged train-only GPerturb sci-Plex2 NativeRouteBinding adapters.

Closures are deliberately not called here. The full SourceBoundOwner gate is
the first operation in each evaluate closure and currently refuses admission.
"""
from dataclasses import dataclass
import hashlib
import re
import json

from mora3.data.admet import canonical
from mora3.data.sciplex_admission import ObservationalSciPlexTask
from mora3.model.cooperative_training import CooperativeModel
from mora3.model.sciplex import SciPlexSharedModel
from mora3.research.primary_native_routes import (
    FiniteNativeTrainRoute, NativeRouteBinding, NativeRouteWindow,
    PreparedNativeTargets, finite_view_policy_sha256,
)

_FAMILY = 'GPerturb-sciPlex2'
_SCHEMA = 'mora3.ObservationalSciPlexTask/v1'
_DIRECTIONS = ('intervention', 'response')


def _unit(direction):
    if direction == 'response':
        return (
            'existing SciPlex response MSE in log1p(source selected-gene count layer) units',
            'observed original source-column expression positions (5000 per admitted cell)',
        )
    if direction == 'intervention':
        return (
            'existing intervention categorical cross entropy in nats plus MSE in source_scaled_coordinate units',
            'observed source intervention category and source_scaled_coordinate per admitted cell',
        )
    raise ValueError('unsupported source task direction')


def _prepare(direction, window):
    if type(window) is not NativeRouteWindow or window.batch.original_split != 'train':
        raise ValueError('exact original GPerturb train route window required')
    rows = window.batch.native_rows
    required = {
        'source_family', 'task_name', 'native_schema', 'split',
        'native_payload', 'native_payload_sha256',
    }
    if not required <= set(rows.schema.names):
        raise ValueError('complete source-native train payload fields required')
    tasks = []
    for row in rows.to_pylist():
        raw = row['native_payload']
        if (
            row['source_family'] != _FAMILY or row['task_name'] != direction
            or row['native_schema'] != _SCHEMA or row['split'] != 'train'
            or type(raw) is not bytes
            or hashlib.sha256(raw).hexdigest() != row['native_payload_sha256']
        ):
            raise ValueError('exact source direction, train role and payload digest required')
        task = ObservationalSciPlexTask.from_bytes(raw)
        if task.input_metadata()['task'] != direction:
            raise ValueError('native task direction differs from original route')
        sidecar = json.loads(task.task.sidecar_bytes)
        if sidecar['conditions']['context'] != dict(
            study='GSM4150377', cell_line='A549', exposure_hours=24,
            baseline=None, desired_response=None,
        ):
            raise ValueError('source context/missingness differs from pinned GPerturb contract')
        if direction == 'response':
            array = sidecar['targets']['expression']
            if (array['status'] != 'observed' or array['units'] != 'source selected-gene count layer'
                    or array['axis_identity'] != 'source_column_position' or len(array['values']) != 5000):
                raise ValueError('complete observed original response axis required')
            count = 5000
        else:
            intervention = sidecar['targets']['intervention']
            if (intervention['units'] != 'source_scaled_coordinate'
                    or intervention['physical_concentration'] is not None):
                raise ValueError('native source dose coordinate and unresolved physical dose required')
            count = 2
        tasks.append((task, count))
    if not tasks:
        return PreparedNativeTargets(0, 'empty original train window', None)
    objective_unit, target_unit = _unit(direction)
    observed = sum(n for _, n in tasks)

    def evaluate(owner):
        # The whole owner gate must remain first. It currently refuses by design.
        from mora3.model.full_owner import SourceBoundOwner
        if type(owner) is not SourceBoundOwner:
            raise ValueError('exact whole SourceBoundOwner required')
        owner.require_training_admission()
        native = owner.core.parent.parent.native
        if type(native) is not CooperativeModel or type(native.response) is not SciPlexSharedModel:
            raise ValueError('exact registered cooperative SciPlex owner branch required')
        if any(task.input_metadata()['task'] != direction for task, _ in tasks):
            raise ValueError('homogeneous original source task direction required')
        if any(tuple(json.loads(task.task.sidecar_bytes)[
                'targets' if direction == 'response' else 'conditions']['expression']['axis_ids'])
               != native.response.axis_ids for task, _ in tasks):
            raise ValueError('source-column-position expression axis mismatch')
        # This is the existing cooperative SciPlex objective, returning its loss tensor.
        return native.objective(direction, [task for task, _ in tasks])

    return PreparedNativeTargets(observed, None, evaluate)



@dataclass(frozen=True)
class GPerturbNativeCampaignAdapter:
    view: object
    source_snapshot_sha256: str
    direction: str
    batch_size: int

    def __post_init__(self):
        if (self.direction not in _DIRECTIONS or type(self.batch_size) is not int or self.batch_size<1
                or type(self.source_snapshot_sha256) is not str
                or not re.fullmatch(r'[a-f0-9]{64}',self.source_snapshot_sha256)):
            raise ValueError('Exact original GPerturb direction, source snapshot and finite batch size required')
        object.__setattr__(self,'_identity',canonical(self._document()))

    def _document(self):
        loss,unit=_unit(self.direction)
        return dict(schema='Mora3.GPerturbSciPlexCampaignAdapter/v1',
            source_snapshot_sha256=self.source_snapshot_sha256,view_policy_sha256=finite_view_policy_sha256(self.view),
            family=_FAMILY,direction=self.direction,native_schema=_SCHEMA,split='train',
            batch_size=self.batch_size,loss_unit=loss,target_unit=unit)

    def executable_binding(self):
        doc=self._document()
        if canonical(doc)!=self._identity:
            raise ValueError('Exact original GPerturb source binding changed')
        return doc

    def train_binding(self):
        doc=self.executable_binding()
        paths,revisions=[],set()
        for descriptor in self.view.descriptors:
            matches=tuple(r for r in descriptor['source_task_schema_counts']
                if (r['source_family'],r['task_name'],r['native_schema'],r['split'],r['eligible_view_disposition'])
                ==(_FAMILY,self.direction,_SCHEMA,'train',0))
            if matches:
                paths.append(descriptor['source_shard']['path'])
                revisions.update(r['source_revision']for r in matches)
        route=FiniteNativeTrainRoute(self.view,task_key=f'GPerturb-sciPlex2:{self.direction}',
            source_snapshot_sha256=self.source_snapshot_sha256,native_view_policy_sha256=doc['view_policy_sha256'],
            source_family=_FAMILY,task_name=self.direction,native_schema=_SCHEMA,
            source_revisions=tuple(sorted(revisions)),source_paths=tuple(sorted(paths)),batch_size=self.batch_size)
        return NativeRouteBinding(route,self.prepare,doc['loss_unit'],doc['target_unit'])

    def prepare(self,window):
        self.executable_binding()
        if type(window) is not NativeRouteWindow:
            raise ValueError('Exact original GPerturb window required')
        self.view.require_native_rows(window.batch,split='train')
        return _prepare(self.direction,window)
