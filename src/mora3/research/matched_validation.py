"""One immutable original-validation exposure shared across all proxy trials."""

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re

from mora3.data.admet import _json, canonical, digest
from mora3.data.finite_union_native_rows import SelectedNativeSourceBatch
from .mcsa_campaign_route import MCSACampaignTrainRoute
from .mup_sweep_selection import MAX_UPDATES, FrozenProxySelection, FrozenValidationTask
from .primary_native_routes import finite_view_policy_sha256


_SCHEMA = 'Mora3.MatchedNativeValidationCohort/v1'
_SHA256 = re.compile(r'[a-f0-9]{64}\Z')


def _sha(value):
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _text(value):
    return type(value) is str and bool(value.strip())


@dataclass(frozen=True)
class ValidationSourceRow:
    source_path: str
    shard_sha256: str
    original_row_ordinal: int
    task_id: str
    payload_sha256: str
    source_revision: str

    def __post_init__(self):
        if (
            not all(_text(value) for value in (self.source_path,self.task_id,self.source_revision))
            or not _sha(self.shard_sha256) or not _sha(self.payload_sha256)
            or type(self.original_row_ordinal) is not int or self.original_row_ordinal < 0
        ):
            raise ValueError('Exact original validation coordinate and provenance required')


@dataclass(frozen=True)
class MatchedValidationTask:
    task_key: str
    source_family: str
    native_task_name: str
    native_schema: str
    coordinates: tuple[ValidationSourceRow, ...]
    observed_targets: int
    initial_reference_loss: Decimal
    normalization_floor: Decimal
    loss_unit: str
    target_unit: str

    def __post_init__(self):
        if (
            not all(_text(value) for value in (self.task_key,self.source_family,self.native_task_name,self.native_schema,
                                                self.loss_unit,self.target_unit))
            or type(self.coordinates) is not tuple or not self.coordinates
            or any(type(row) is not ValidationSourceRow for row in self.coordinates)
        ):
            raise ValueError('Exact native validation task, cohort coordinates and metric units required')
        for row in self.coordinates:
            row.__post_init__()
        keys = tuple((row.source_path,row.original_row_ordinal) for row in self.coordinates)
        if keys != tuple(sorted(set(keys))):
            raise ValueError('Validation coordinates must be unique and sorted by original path/ordinal')
        self.selection_task()

    def _source_wire(self):
        return dict(
            task_key=self.task_key,source_family=self.source_family,native_task_name=self.native_task_name,
            native_schema=self.native_schema,coordinates=tuple(asdict(x) for x in self.coordinates),
            observed_targets=self.observed_targets,loss_unit=self.loss_unit,target_unit=self.target_unit,
            role='validation',
        )

    @property
    def source_sha256(self):
        return digest(canonical(self._source_wire()))

    def selection_task(self):
        return FrozenValidationTask(self.task_key,self.source_sha256,self.initial_reference_loss,
                                    self.normalization_floor,self.observed_targets)

    def binding(self):
        return dict(**self._source_wire(),initial_reference_loss=str(self.initial_reference_loss),
                    normalization_floor=str(self.normalization_floor))


@dataclass(frozen=True)
class CampaignValidationSources:
    """An exact finite original view plus its separately reviewed M-CSA arm."""
    original: object
    mcsa: MCSACampaignTrainRoute

    def __post_init__(self):
        finite_view_policy_sha256(self.original)
        if type(self.mcsa) is not MCSACampaignTrainRoute:
            raise ValueError('Exact reviewed M-CSA validation source required')


def matched_validation_source_policy_sha256(view):
    if type(view) is CampaignValidationSources:
        view.__post_init__()
        return digest(canonical(['Mora3.MixedOriginalAndMCSAValidationSources/v1',
            finite_view_policy_sha256(view.original),view.mcsa.validation_policy_sha256]))
    return finite_view_policy_sha256(view)


@dataclass(frozen=True)
class MatchedValidationCohort:
    source_snapshot_sha256: str
    view_policy_sha256: str
    same_snapshot_group_guard_sha256: str
    initial_reference_artifact_sha256: str
    comparison_update: int
    read_batch_size: int
    max_selected_arrow_bytes: int
    tasks: tuple[MatchedValidationTask, ...]
    train_only_dispositions: tuple[tuple[str, str], ...]

    def __post_init__(self):
        if (
            not all(_sha(value) for value in (self.source_snapshot_sha256,self.view_policy_sha256,
                self.same_snapshot_group_guard_sha256,self.initial_reference_artifact_sha256))
            or type(self.comparison_update) is not int or not 1 <= self.comparison_update <= MAX_UPDATES
            or type(self.read_batch_size) is not int or self.read_batch_size < 1
            or type(self.max_selected_arrow_bytes) is not int or self.max_selected_arrow_bytes < 1
            or type(self.tasks) is not tuple or not self.tasks
            or any(type(task) is not MatchedValidationTask for task in self.tasks)
            or type(self.train_only_dispositions) is not tuple
            or any(type(row) is not tuple or len(row) != 2 or any(not _text(x) for x in row)
                   for row in self.train_only_dispositions)
        ):
            raise ValueError('Exact source cohort, guard/reference identities and finite validation exposure required')
        keys = tuple(task.task_key for task in self.tasks)
        excluded = tuple(key for key,_ in self.train_only_dispositions)
        if keys != tuple(sorted(set(keys))) or excluded != tuple(sorted(set(excluded))) or set(keys)&set(excluded):
            raise ValueError('Disjoint unique ordered selection and train-only task dispositions required')
        for task in self.tasks:
            task.__post_init__()

    def wire(self):
        self.__post_init__()
        return canonical(dict(
            schema=_SCHEMA,source_snapshot_sha256=self.source_snapshot_sha256,
            view_policy_sha256=self.view_policy_sha256,
            same_snapshot_group_guard_sha256=self.same_snapshot_group_guard_sha256,
            initial_reference_artifact_sha256=self.initial_reference_artifact_sha256,
            comparison_update=self.comparison_update,read_batch_size=self.read_batch_size,
            max_selected_arrow_bytes=self.max_selected_arrow_bytes,tasks=tuple(task.binding() for task in self.tasks),
            train_only_dispositions=self.train_only_dispositions,
        ))

    @property
    def sha256(self):
        return digest(self.wire())

    @classmethod
    def load(cls, path, expected_sha256):
        raw = Path(path).read_bytes()
        if not _sha(expected_sha256) or digest(raw) != expected_sha256:
            raise ValueError('Exact pinned matched validation cohort artifact required')
        try:
            document = _json(raw)
            if document.pop('schema') != _SCHEMA:
                raise ValueError('Unknown matched validation cohort schema')
            tasks = []
            for task in document.pop('tasks'):
                if task.pop('role') != 'validation':
                    raise ValueError('Only original validation role may select transfer')
                coordinates = tuple(ValidationSourceRow(**row) for row in task.pop('coordinates'))
                for field in ('initial_reference_loss','normalization_floor'):
                    value = task[field]
                    if type(value) is not str:
                        raise ValueError('Exact decimal string in frozen validation manifest required')
                    task[field] = Decimal(value)
                tasks.append(MatchedValidationTask(**task,coordinates=coordinates))
            excluded = tuple(tuple(row) for row in document.pop('train_only_dispositions'))
            result = cls(**document,tasks=tuple(tasks),train_only_dispositions=excluded)
        except (KeyError,AttributeError,TypeError,UnicodeError,InvalidOperation) as exc:
            raise ValueError('Malformed exact matched validation cohort') from exc
        if result.wire() != raw:
            raise ValueError('Matched validation cohort is not exact canonical wire')
        return result

    def selection_contract(self, campaign_sha256):
        return FrozenProxySelection(campaign_sha256,self.sha256,self.comparison_update,
                                    tuple(task.selection_task() for task in self.tasks))

    def batches(self, view, task_key):
        """Materialize bounded, completely authenticated original validation rows."""
        import pyarrow as pa

        self.__post_init__()
        if matched_validation_source_policy_sha256(view) != self.view_policy_sha256:
            raise ValueError('Matched validation source view snapshot differs')
        matched = tuple(task for task in self.tasks if task.task_key == task_key)
        if len(matched) != 1:
            raise ValueError('Task outside exact matched validation selection axis')
        task, = matched
        if type(view) is CampaignValidationSources:
            if view.mcsa.source_snapshot_sha256!=self.source_snapshot_sha256:
                raise ValueError('Matched M-CSA and original validation source snapshot differs')
            if task_key==view.mcsa.task_key:
                return view.mcsa.matched_validation_batches(task,read_batch_size=self.read_batch_size,
                    max_selected_bytes=self.max_selected_arrow_bytes)
            view = view.original
        remaining = {(row.source_path,row.original_row_ordinal):row for row in task.coordinates}
        paths = tuple(dict.fromkeys(row.source_path for row in task.coordinates))
        batches, observed, selected_bytes = [], [], 0
        for batch in view.iter_native_task_batches(split='validation',source_paths=paths,batch_size=self.read_batch_size):
            path = batch.source_shard['path']
            selected = tuple((i,o,remaining[(path,o)]) for i,o in enumerate(batch.source_row_ordinals) if (path,o) in remaining)
            if not selected:
                continue
            indices, ordinals, coordinates = zip(*selected,strict=True)
            exact = SelectedNativeSourceBatch(batch.source_shard,ordinals,'validation',
                batch.native_rows.take(pa.array(indices,type=pa.int64())))
            view.require_native_rows(exact,split='validation')
            fields = ('source_family','task_name','native_schema','split','task_id','native_payload_sha256','source_revision')
            if not set(fields) <= set(exact.native_rows.schema.names):
                raise ValueError('Exact native validation route and provenance fields required')
            native = zip(*(exact.native_rows[key].to_pylist() for key in fields),strict=True)
            for row,coordinate in zip(native,coordinates,strict=True):
                if (
                    batch.source_shard['sha256'] != coordinate.shard_sha256
                    or row != (task.source_family,task.native_task_name,task.native_schema,'validation',
                               coordinate.task_id,coordinate.payload_sha256,coordinate.source_revision)
                ):
                    raise ValueError('Original validation role/task/release or exact source identity differs')
                observed.append(coordinate)
                del remaining[(path,coordinate.original_row_ordinal)]
            selected_bytes += exact.native_rows.nbytes
            if selected_bytes > self.max_selected_arrow_bytes:
                raise ValueError('Complete matched validation selected Arrow budget exceeded')
            batches.append(exact)
        if remaining or tuple(observed) != task.coordinates:
            raise ValueError('Every frozen original validation coordinate must be available in exact order')
        return tuple(batches)
