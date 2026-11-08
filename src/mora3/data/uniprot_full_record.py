"""Current full-record UniProt source targets with exact field-specific wires."""

from dataclasses import dataclass
import math

from .admet import _json, canonical, digest
from .finite_union_native_rows import FiniteUnionNativeView, read_exact
from .uniprot_annotations import TASK


FAMILY = 'UniProt-full-record-annotations'
SCHEMA = 'mora3.CorpusTask/v1'
_AA = frozenset('ABCDEFGHIKLMNOPQRSTUVWXYZ')
_FIELD_ROOTS = {'list': list, 'dict': dict, 'str': str}
_PROPERTY_AXIS = (
    ('sequence_length', 'residue', '/sequence/length'),
    ('sequence_mass', 'Da', '/sequence/molWeight'),
)
_COMPUTED = 'provider_computed_from_canonical_sequence'


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class FullRecordFieldAxis:
    name: str
    root_type: str
    max_wire_bytes: int
    observed_train_rows: int

    def binding(self):
        if (
            type(self.name) is not str or not self.name or not self.name.isascii() or not self.name.isalpha()
            or type(self.root_type) is not str or self.root_type not in _FIELD_ROOTS
            or type(self.max_wire_bytes) is not int or self.max_wire_bytes < 1
            or type(self.observed_train_rows) is not int or self.observed_train_rows < 1
        ):
            raise ValueError('Exact observed UniProt source field and complete wire bound required')
        return dict(name=self.name, root_type=self.root_type, max_wire_bytes=self.max_wire_bytes,
                    observed_train_rows=self.observed_train_rows)

    def encode(self, value):
        if type(value) is not _FIELD_ROOTS[self.root_type]:
            raise ValueError('Observed UniProt source field must match its literal root type')
        wire = canonical(value)
        if len(wire) > self.max_wire_bytes:
            raise ValueError('Complete UniProt source field exceeds its native wire bound')
        return wire

    def decode(self, wire):
        if type(wire) is not bytes or not 1 <= len(wire) <= self.max_wire_bytes:
            raise ValueError('Complete canonical UniProt source field wire required')
        value = _json(wire)
        if self.encode(value) != wire:
            raise ValueError('UniProt source field wire is not exactly reversible')
        return value


@dataclass(frozen=True)
class FullRecordUniProtCondition:
    canonical_sequence: str


@dataclass(frozen=True)
class FullRecordProviderProperty:
    parameter_type: str
    value: int | float
    unit: str
    source_pointer: str
    source_value_json: str


@dataclass(frozen=True)
class FullRecordUniProtPacket:
    condition: FullRecordUniProtCondition
    field_targets: tuple[tuple[str, bytes | None], ...]
    computed_properties: tuple[FullRecordProviderProperty, ...]
    source_coordinate: tuple[str, int, str]

    def target(self, field):
        for name, raw in self.field_targets:
            if name == field:
                return raw
        raise ValueError('Field outside the complete native UniProt source axis')


@dataclass(frozen=True)
class FullRecordUniProtAxes:
    census_sha256: str
    view_policy_sha256: str
    original_train_rows: int
    max_sequence_bytes: int
    fields: tuple[FullRecordFieldAxis, ...]

    def binding(self):
        if (
            not _sha(self.census_sha256) or not _sha(self.view_policy_sha256)
            or type(self.original_train_rows) is not int or self.original_train_rows < 1
            or type(self.max_sequence_bytes) is not int or self.max_sequence_bytes < 1
            or type(self.fields) is not tuple or not self.fields
            or any(type(x) is not FullRecordFieldAxis for x in self.fields)
        ):
            raise ValueError('Complete current UniProt source fields and known sequence bound required')
        names = tuple(field.name for field in self.fields)
        if names != tuple(sorted(set(names))) or any(
            field.observed_train_rows > self.original_train_rows for field in self.fields
        ):
            raise ValueError('Exact ordered UniProt source field axis and counts required')
        return dict(
            schema='Mora3.FullRecordUniProtAxes/v1', original_train_rows=self.original_train_rows,
            max_sequence_bytes=self.max_sequence_bytes, fields=tuple(field.binding() for field in self.fields),
            census_sha256=self.census_sha256, view_policy_sha256=self.view_policy_sha256,
            numeric_provider_property_axis=_PROPERTY_AXIS, source_release='UniProtKB 2026_03',
            original_role='train', wire='canonical UTF8 of one explicitly requested typed source field',
        )

    def field(self, name):
        for field in self.fields:
            if field.name == name:
                return field
        raise ValueError('Exact requested native UniProt full-record field required')

    def require_condition(self, condition):
        if type(condition) is not FullRecordUniProtCondition:
            raise ValueError('Only the original canonical sequence is a native full-record condition')
        sequence = condition.canonical_sequence
        if type(sequence) is not str or not 1 <= len(sequence) <= self.max_sequence_bytes or set(sequence) - _AA:
            raise ValueError('Complete original native canonical sequence within source bounds required')


def _properties(values, length):
    if type(values) is not list or len(values) != len(_PROPERTY_AXIS):
        raise ValueError('Separate complete native computed sequence property axis required')
    result = []
    for item, (kind, unit, pointer) in zip(values, _PROPERTY_AXIS, strict=True):
        if (
            type(item) is not dict or set(item) != {
                'parameter_type', 'value', 'unit', 'source_pointer', 'source_value_json', 'measurement_context', 'evidence_role',
            }
            or (item['parameter_type'], item['unit'], item['source_pointer']) != (kind,unit,pointer)
            or item['measurement_context'] is not None or item['evidence_role'] != _COMPUTED
        ):
            raise ValueError('Exact native provider-computed property semantics required')
        value = item['value']
        if (
            type(value) not in (int,float) or not math.isfinite(value) or value <= 0
            or canonical(value).decode('utf-8') != item['source_value_json']
            or kind == 'sequence_length' and (type(value) is not int or value != length)
        ):
            raise ValueError('Exact original computed property value and source literal required')
        result.append(FullRecordProviderProperty(kind,value,unit,pointer,item['source_value_json']))
    return tuple(result)


def _project_full_record(row, path, ordinal, axes, *, requested=None):
    try:
        if (
            (row['source_family'],row['task_name'],row['native_schema'],row['split']) != (FAMILY,TASK,SCHEMA,'train')
            or row['admitted_for_task'] is not True or row['model_adapter_supported'] is not False
        ):
            raise ValueError('Exact original full-record UniProt train source task required')
        raw = row['native_payload']
        identity = digest(raw)
        if identity != row['native_payload_sha256']:
            raise ValueError('Original full-record native UniProt payload checksum differs')
        task = _json(raw)
        if (
            set(task) != {'schema','task','conditions','targets','audit','missing'}
            or (task['schema'],task['task']) != (SCHEMA,TASK) or canonical(task) != raw
            or set(task['conditions']) != {'canonical_sequence'}
            or set(task['targets']) != {'annotation_packet','provider_computed_sequence_properties'}
        ):
            raise ValueError('Exact complete native source record, known condition and targets required')
        condition = FullRecordUniProtCondition(task['conditions']['canonical_sequence'])
        axes.require_condition(condition)
        source = task['audit']['source_ref']
        protein = task['audit']['identity']
        receipt = _json(row['admission_receipt_json'])
        sequence = condition.canonical_sequence.encode('ascii')
        ev, sv = protein['entry_version'], protein['sequence_version']
        if (
            type(ev) is not int or ev < 1 or type(sv) is not int or sv < 1
            or row['source_revision'] != f'UniProt2026_03;entryVersion{ev};sequenceVersion{sv}'
            or row['record_id'] != protein['primary_accession']
            or protein['sequence_sha256'] != digest(sequence) or protein['sequence_length'] != len(sequence)
            or source['source_release'] != '2026_03' or source['raw_file_sha256'] != row['source_sha256']
            or not _sha(row['source_sha256']) or source['entry_version'] != ev or source['sequence_version'] != sv
            or receipt['source_ref'] != source or receipt['license'] != 'CC-BY-4.0'
            or receipt['reviewed'] is not True or receipt['canonical_sequence_exact_baseline'] is not True
        ):
            raise ValueError('Exact full-record source, reviewed scope and canonical sequence provenance required')
        values = task['targets']['annotation_packet']
        fields = {field.name:field for field in axes.fields}
        if type(values) is not dict or set(values) - set(fields):
            raise ValueError('Full-record observed annotation field outside exact source scope')
        for name,value in values.items():
            if type(value) is not _FIELD_ROOTS[fields[name].root_type]:
                raise ValueError('Native source field root type differs')
        chosen = axes.fields if requested is None else (axes.field(requested),)
        targets = tuple((field.name,field.encode(values[field.name]) if field.name in values else None)
                        for field in chosen)
        properties = _properties(task['targets']['provider_computed_sequence_properties'],len(sequence))
        return FullRecordUniProtPacket(condition,targets,properties,(path,ordinal,identity))
    except (KeyError,TypeError,UnicodeError,OverflowError) as exc:
        raise ValueError('Malformed original native UniProt full annotation record') from exc


def _view_policy(view):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError('Exact current native UniProt full-record view required')
    return digest(canonical([view.manifest,getattr(view,'effective_filter',None)]))


class FullRecordUniProtSource:
    def __init__(self, view, census_path, expected_sha256):
        policy = _view_policy(view)
        path = (view.root / census_path).resolve()
        path.relative_to(view.root.resolve())
        raw = path.read_bytes()
        if not _sha(expected_sha256) or digest(raw) != expected_sha256:
            raise ValueError('Pinned current native UniProt full-record census checksum differs')
        proof = _json(raw)
        if (
            proof['schema'] != 'Mora3.CurrentFullRecordUniProtPredictionSourceCensus/v1'
            or (proof['source_family'],proof['original_task'],proof['native_schema']) != (FAMILY,TASK,SCHEMA)
            or proof['exact_condition_fields'] != ['canonical_sequence']
            or proof['all_native_payload_sha_and_role_bindings_verified'] is not True
            or proof['provider_computed_properties_remain_separate'] is not True
            or _json(read_exact(proof['effective_filter'],root=view.root)) != view.effective_filter
        ):
            raise ValueError('Exact guarded full-record source census and view required')
        maximum = proof['maximum_native_axes_and_wire_bytes']
        total = proof['total_retained_train']
        fields = []
        for name in proof['complete_annotation_packet_fields']:
            counts = proof['annotation_packet_field_types'][name]
            if type(counts) is not dict or len(counts) != 1 or next(iter(counts.values())) != proof['present_source_field_positions'][name]:
                raise ValueError('Exact single root type for every observed native field required')
            fields.append(FullRecordFieldAxis(name,next(iter(counts)),maximum[name],counts[next(iter(counts))]))
        self.axes = FullRecordUniProtAxes(expected_sha256,policy,total,maximum['complete_known_canonical_AA_bytes'],tuple(fields))
        self.axes.binding()
        original = proof['original_source_shard_counts']
        base = {}
        for descriptor in view.descriptors:
            relevant = tuple(row for row in descriptor['source_task_schema_counts']
                if (row['source_family'],row['task_name'],row['native_schema'],row['split'],row['eligible_view_disposition'])
                == (FAMILY,TASK,SCHEMA,'train',0))
            if relevant:
                base[descriptor['source_shard']['path']] = sum(row['rows'] for row in relevant)
        if (
            not original or not set(original) <= set(base) or sum(original.values()) != total
            or any(type(count) is not int or not 1 <= count <= base[path] for path,count in original.items())
            or proof['retained_original_role_counts'] != {'train':total,'validation':0,'test':0}
        ):
            raise ValueError('Complete current original full-record source coordinates required')
        self.source_paths = tuple(sorted(original))
        self._path_set = frozenset(original)
        self._view = view

    def packets(self, batch, *, split, field=None):
        if split != 'train' or _view_policy(self._view) != self.axes.view_policy_sha256:
            raise ValueError('Exact native full-record UniProt view and original train role required')
        if field is not None:
            self.axes.field(field)
        self._view.require_native_rows(batch,split=split)
        path = batch.source_shard['path']
        if path not in self._path_set:
            raise ValueError('Native UniProt source path outside complete retained full-record census')
        return tuple(_project_full_record(row,path,ordinal,self.axes,requested=field) for row,ordinal in zip(
            batch.native_rows.to_pylist(),batch.source_row_ordinals,strict=True,
        ))
