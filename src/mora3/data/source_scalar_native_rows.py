"""Project typed scalar observations from an immutable native corpus view."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import struct

from .admet import MOLECULE_CLASSES, EndpointDefinition, NativeCell, NativeEntity
from .finite_union_native_rows import FiniteUnionNativeView, SelectedNativeSourceBatch


SCOPES = {
    "FreeSolv": "experimental_hydration_free_energy",
    "ChEMBL37-AZ-experimental-logD7_4": "source_reported_experimental_logD7_4",
    "UniProt-kinetics": "curated_kinetic_annotation_value",
}


class _NumberToken(str):
    """A JSON number's original lexeme, distinct from a JSON string."""


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate native JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise ValueError("nonfinite native JSON")


def _json(raw, *, numeric_tokens=False):
    options = {"object_pairs_hook": _pairs, "parse_constant": _reject_constant}
    if numeric_tokens:
        options.update(parse_int=_NumberToken, parse_float=_NumberToken)
    return json.loads(raw, **options)


def _float64(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("finite native numeric value required")
    return float(value)


def _float_bits(value):
    return None if value is None else struct.pack(">d", value).hex()


def _float_from_decimal(literal):
    if type(literal) is not str:
        raise ValueError("native decimal string required")
    try:
        decimal = Decimal(literal)
    except InvalidOperation as exc:
        raise ValueError("invalid native decimal") from exc
    if not decimal.is_finite():
        raise ValueError("finite native decimal required")
    return _float64(float(decimal))


@dataclass(frozen=True)
class ScalarTarget:
    kind: str
    reported_value: float | None
    lower: float | None
    upper: float | None
    lower_inclusive: bool | None
    upper_inclusive: bool | None
    raw_relation: str | None
    native_numeric_lexeme: str | None
    lexeme_scope: str
    provider_lexemes_json: bytes
    uncertainty_value: float | None
    uncertainty_kind: str | None
    native_target_metadata_json: bytes

    def __post_init__(self):
        for value in (self.reported_value, self.lower, self.upper, self.uncertainty_value):
            if value is not None and (type(value) is not float or not math.isfinite(value)):
                raise ValueError("numeric coordinates require finite Float64 values or explicit None")
        shape = self.reported_value is not None, self.lower is not None, self.upper is not None
        expected = {
            "reported_point": (True, False, False),
            "reported_nominal_coordinate": (True, False, False),
            "censored_left": (False, False, True),
            "censored_right": (False, True, False),
            "interval": (False, True, True),
            "missing": (False, False, False),
            "unqualified": (False, False, False),
        }
        if expected.get(self.kind) != shape:
            raise ValueError("numeric fields differ from the explicit observation kind")
        relations = {
            "reported_point": (None, "="),
            "reported_nominal_coordinate": (None,),
            "censored_left": ("<", "<="),
            "censored_right": (">", ">="),
            "interval": ("interval",),
            "missing": (None,),
            "unqualified": (None,),
        }
        if self.raw_relation not in relations[self.kind]:
            raise ValueError("source relation differs from the explicit observation kind")
        if self.kind == "interval" and not self.lower < self.upper:
            raise ValueError("ordered finite Float64 bounds required")
        if self.kind in ("interval", "censored_left", "censored_right"):
            if type(self.lower_inclusive) is not bool or type(self.upper_inclusive) is not bool:
                raise ValueError("explicit bound inclusion flags required")
            if self.kind == "censored_left" and (self.lower_inclusive or self.upper_inclusive != (self.raw_relation == "<=")):
                raise ValueError("upper bound inclusivity differs from the source relation")
            if self.kind == "censored_right" and (self.upper_inclusive or self.lower_inclusive != (self.raw_relation == ">=")):
                raise ValueError("lower bound inclusivity differs from the source relation")
        elif self.lower_inclusive is not None or self.upper_inclusive is not None:
            raise ValueError("bound inclusion flags require explicit bounds")
        if (self.uncertainty_value is None) != (self.uncertainty_kind is None):
            raise ValueError("uncertainty requires a literal source interpretation")
        for raw in (self.provider_lexemes_json, self.native_target_metadata_json):
            if type(raw) is not bytes or type(_json(raw)) is not dict:
                raise ValueError("immutable native target metadata object required")

    @property
    def numeric_coordinate_available(self):
        return self.kind not in ("missing", "unqualified")

    @property
    def reported_value_float64_bits(self):
        return _float_bits(self.reported_value)

    @classmethod
    def from_native_cell(cls, cell, *, metadata=None):
        """Preserve an explicitly interpreted NativeCell and its raw literal."""
        if type(cell) is not NativeCell:
            raise ValueError("exact native cell required")
        source = cell.as_mapping()
        status, relation = source["status"], source["relation"]
        point = lower = upper = lower_inclusive = upper_inclusive = None
        if status == "missing":
            kind = "missing"
        elif status == "exact":
            kind = "reported_point"
            point = _float_from_decimal(source["decimal_literal"])
        elif status == "censored_bound":
            if relation in ("<", "<="):
                kind = "censored_left"
                upper = _float_from_decimal(source["decimal_literal"])
                lower_inclusive, upper_inclusive = False, relation == "<="
            else:
                kind = "censored_right"
                lower = _float_from_decimal(source["decimal_literal"])
                lower_inclusive, upper_inclusive = relation == ">=", False
        elif status == "interval_bound":
            kind = "interval"
            lower = _float_from_decimal(source["lower_decimal_literal"])
            upper = _float_from_decimal(source["upper_decimal_literal"])
            lower_inclusive, upper_inclusive = source["lower_inclusive"], source["upper_inclusive"]
        else:
            kind, relation = "unqualified", None
        return cls(
            kind, point, lower, upper, lower_inclusive, upper_inclusive, relation,
            source["decimal_literal"], "explicit-native-cell-literal",
            _wire(source), None, None, _wire({} if metadata is None else metadata),
        )


@dataclass(frozen=True)
class SourceScalarNativeRow:
    source_family: str
    task_name: str
    original_split: str
    source_coordinate_json: bytes
    condition_json: bytes
    endpoint_definition: EndpointDefinition | None
    target: ScalarTarget
    partition_key: tuple[str, str, str, str | None, str | None, str | None, str] | tuple[()]
    disposition: str
    unqualified_reasons: tuple[str, ...]
    source_audit_json: bytes

    def condition_inputs(self):
        return _json(self.condition_json)

    def audit(self):
        return _json(self.source_audit_json)

    @property
    def training_authorized(self):
        return False


@dataclass(frozen=True)
class NativeScalarBatch:
    native_source_batch: SelectedNativeSourceBatch
    scalar_rows: tuple[SourceScalarNativeRow, ...]
    original_split: str


@dataclass(frozen=True)
class _SourceObservation:
    quantity: str
    unit: str | None
    method: str | None
    scope: str | None
    coordinate: str
    entity: NativeEntity
    known_context: dict
    target_field: str
    kind: str
    value: float
    provider_lexemes: dict
    missingness: tuple[str, ...]
    uncertainty: float | None = None
    uncertainty_kind: str | None = None


def _entity(molecule_class, representation_type, representation):
    return NativeEntity.from_mapping(dict(
        molecule_class=molecule_class, representation_type=representation_type, representation=representation,
    ))


def _definition(family, task, observation):
    quantity, unit = observation.quantity, observation.unit
    key = hashlib.sha256(_wire([family, task, quantity, unit, observation.method, observation.coordinate])).hexdigest()
    native = dict(
        endpoint_key="source-scalar:" + key, source_endpoint=task,
        description="Reported native source coordinate; physical and assay missingness remains explicit",
        readout=quantity, unit=unit, unit_basis="source_literal" if unit is not None else "unresolved",
        scale=observation.coordinate, measurement_type=quantity, assay_protocol_context=observation.method,
    )
    applicability = {
        name: "applicable" if name == observation.entity.molecule_class else "unknown"
        for name in MOLECULE_CLASSES
    }
    return EndpointDefinition.from_native(
        native, objective_kind="regression", applicability=applicability, semantic_name=quantity,
    )


def _freesolv(conditions, targets, audit):
    if set(conditions) != {"smiles"} or set(targets) != {"expt", "unit"}:
        raise ValueError("FreeSolv requires the molecule and reported experimental scalar")
    label_contract = audit["label_contract"]
    if (targets["unit"] != "kcal/mol" or label_contract["default_target_fields"] != ["expt"]
            or label_contract["label_evidence"]["expt"] != "experimental"):
        raise ValueError("native FreeSolv experimental target or unit differs")
    source = audit["source_row"]
    value = _float64(targets["expt"])
    if _float_bits(_float64(source["expt"])) != _float_bits(value):
        raise ValueError("original FreeSolv experimental scalar differs")
    uncertainty = None
    uncertainty_kind = None
    if source.get("d_expt") is not None:
        uncertainty = _float64(source["d_expt"])
        uncertainty_kind = label_contract["label_evidence"]["d_expt"]
        if uncertainty_kind != "experimental_uncertainty_may_be_default":
            raise ValueError("unknown FreeSolv uncertainty provenance")
    return _SourceObservation(
        "experimental_hydration_free_energy", targets["unit"], None, None,
        "source_experimental_reported_point", _entity("small_molecule", "SMILES", conditions["smiles"]),
        {}, "expt", "reported_point", value, {},
        ("assay_method", "assay_temperature", "assay_pressure"), uncertainty, uncertainty_kind,
    )


def _logd(conditions, targets, audit):
    field = "source_reported_logD7_4"
    if set(conditions) != {"canonical_isomeric_smiles", "pH", "solvent_pair", "method"} or set(targets) != {field}:
        raise ValueError("complete native logD reporting query and target required")
    original = audit["source_activity"]
    known = {"pH": 7.4, "solvent_pair": ["octan-1-ol", "water"], "method": "shake_flask"}
    if any(conditions[key] != value for key, value in known.items()):
        raise ValueError("AZ assay conditions differ from the frozen native contract")
    expected = {
        "assay_chembl_id": "CHEMBL3301363", "assay_type": "P", "type": "LogD7.4",
        "standard_type": "LogD7.4", "standard_flag": 1, "data_validity_comment": None,
        "relation": None, "standard_relation": None, "units": None, "standard_units": None,
    }
    if (any(_wire(original[key]) != _wire(value) for key, value in expected.items())
            or audit["source_relation"] is not None or audit["source_units"] is not None):
        raise ValueError("AZ original assay, relation or unit contract differs")
    value = _float64(targets[field])
    provider = {key: original[key] for key in ("value", "standard_value")}
    if not -1.5 < value < 4.5 or any(_float_from_decimal(item) != value for item in provider.values()):
        raise ValueError("AZ nominal coordinate differs or reaches an excluded boundary")
    return _SourceObservation(
        "source_reported_nominal_logD7_4", None, known["method"], None,
        "declared_native_nominal_logD7_4_coordinate_relation_unspecified",
        _entity("small_molecule", "SMILES", conditions["canonical_isomeric_smiles"]),
        known, field, "reported_nominal_coordinate", value, provider,
        ("source_relation_unspecified", "literal_source_units_unspecified", "assay_temperature",
         "measurement_uncertainty"),
    )


def _kinetics(conditions, targets, audit):
    if set(conditions) != {"amino_acid_sequence", "measurement_type", "substrate_alias", "unit"} or set(targets) != {"value"}:
        raise ValueError("complete source protein, kinetic kind, substrate and unit required")
    quantity, unit = conditions["measurement_type"], conditions["unit"]
    original = audit["original_measurement"]
    field = {"Km": "constant", "Vmax": "velocity"}.get(quantity)
    value = _float64(targets["value"])
    if field is None or _float_bits(_float64(original[field])) != _float_bits(value):
        raise ValueError("original Km or Vmax numeric field differs")
    if type(unit) is not str or not unit or original["unit"] != unit:
        raise ValueError("exact literal original kinetic unit required")
    # The historical envelope puts the native Vmax enzyme field in a
    # column called substrate_alias. Its source meaning remains distinct.
    alias_field = "substrate" if quantity == "Km" else "enzyme"
    alias = conditions["substrate_alias"]
    if type(alias) is not str or not alias or original[alias_field] != alias:
        raise ValueError("exact literal native substrate or enzyme field required")
    # The selected native packet carries no parent construct, method, pH or
    # temperature. A new source variant needs its own reviewed projection.
    if any(key in original for key in ("molecule", "method", "pH", "temperature")):
        raise ValueError("kinetic assay context requires a separate source review")
    return _SourceObservation(
        quantity, unit, None, None, "source_curated_reported_" + quantity,
        _entity("protein", "amino_acid_sequence", conditions["amino_acid_sequence"]),
        {alias_field + "_alias": alias, "measurement_type": quantity, "literal_unit": unit},
        "value", "reported_point", value, {},
        ("assay_method", "assay_pH", "assay_temperature", "assayed_construct", "parent_comment_molecule_scope"),
    )


_PROJECTORS = {
    "FreeSolv": _freesolv,
    "ChEMBL37-AZ-experimental-logD7_4": _logd,
    "UniProt-kinetics": _kinetics,
}


def _coordinate(row, source_shard, ordinal):
    return _wire({
        "source_shard": source_shard, "absolute_original_row_ordinal": ordinal,
        "task_id": row["task_id"], "native_payload_sha256": row["native_payload_sha256"],
    })


def _project(row, source_shard, ordinal):
    raw = row["native_payload"]
    family, task = row["source_family"], row["task_name"]
    schema = row["native_schema"]
    if (type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != row["native_payload_sha256"]
            or SCOPES.get(family) != task or schema != "mora3.CorpusTask/v1"):
        raise ValueError("exact supported task and original payload SHA required")
    packet = _json(raw)
    if (set(packet) != {"schema", "task", "conditions", "targets", "audit", "missing"}
            or packet["schema"] != schema or packet["task"] != task):
        raise ValueError("complete original native scalar task envelope required")
    observation = _PROJECTORS[family](packet["conditions"], packet["targets"], packet["audit"])
    definition = _definition(family, task, observation)
    inputs = dict(task="source_scalar_prediction", native_entity=observation.entity.as_mapping(),
                  endpoint=definition.query(), known_context=observation.known_context)
    lexeme = _json(raw, numeric_tokens=True)["targets"][observation.target_field]
    if type(lexeme) is not _NumberToken:
        raise ValueError("exact numeric token in the immutable task JSON required")
    target = ScalarTarget(
        observation.kind, observation.value, None, None, None, None, None, str(lexeme),
        "exact-number-token-in-immutable-native-task-JSON;provider-spelling-not-inferred",
        _wire(observation.provider_lexemes), observation.uncertainty, observation.uncertainty_kind,
        _wire(dict(
            original_targets=packet["targets"], literal_unit=observation.unit,
            reported_coordinate=observation.coordinate, physical_missingness=observation.missingness,
            errorbar_is_not_assumed_measured_standard_deviation=True,
        )),
    )
    partition = (family, task, observation.quantity, observation.unit, observation.method,
                 observation.scope, observation.coordinate)
    audit = dict(
        original_native_task=packet,
        original_envelope_metadata={key: value for key, value in row.items() if key != "native_payload"},
        physical_missingness=observation.missingness, native_method=observation.method,
        parent_molecule_scope=observation.scope,
    )
    return SourceScalarNativeRow(
        family, task, row["split"], _coordinate(row, source_shard, ordinal), _wire(inputs), definition,
        target, partition, "qualified-native-reported-source-coordinate", (), _wire(audit),
    )


def _unqualified(row, source_shard, ordinal, reason):
    target = ScalarTarget(
        "unqualified", None, None, None, None, None, None, None,
        "unqualified-native-payload", b"{}", None, None, b"{}",
    )
    # NativeScalarBatch retains the complete typed row and original payload.
    # The coordinate identifies it even if its payload cannot be parsed.
    return SourceScalarNativeRow(
        row["source_family"], row["task_name"], row["split"], _coordinate(row, source_shard, ordinal),
        b"{}", None, target, (), "unqualified-native-source-coordinate", (reason,), b"{}",
    )


def project_native_scalar_batch(view, native_batch, *, split):
    """Authenticate complete original rows and project each to one disposition."""
    if type(view) is not FiniteUnionNativeView or type(native_batch) is not SelectedNativeSourceBatch:
        raise ValueError("immutable native view and complete source batch required")
    view.require_native_rows(native_batch, split=split)
    rows = []
    for ordinal, row in zip(native_batch.source_row_ordinals, native_batch.native_rows.to_pylist(), strict=True):
        if row["source_family"] not in SCOPES:
            raise ValueError("only explicitly selected native scalar sources allowed")
        try:
            projected = _project(row, native_batch.source_shard, ordinal)
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            projected = _unqualified(row, native_batch.source_shard, ordinal, type(exc).__name__)
        rows.append(projected)
    return NativeScalarBatch(native_batch, tuple(rows), split)


def iter_native_scalar_batches(view, *, split, batch_size=128, source_paths=None):
    if type(view) is not FiniteUnionNativeView:
        raise ValueError("immutable native view required")
    if source_paths is None:
        source_paths = tuple(
            entry["source_shard"]["path"] for entry in view.descriptors
            if entry["source_shard"].get("source_family") in SCOPES
        )
    for native in view.iter_native_task_batches(split=split, batch_size=batch_size, source_paths=source_paths):
        yield project_native_scalar_batch(view, native, split=split)


def require_native_scalar_batch(view, batch, *, split):
    if type(batch) is not NativeScalarBatch or batch.original_split != split:
        raise ValueError("exact native scalar batch and original role required")
    expected = project_native_scalar_batch(view, batch.native_source_batch, split=split)
    if expected.scalar_rows != batch.scalar_rows:
        raise ValueError("native scalar inputs, targets, dispositions or source metadata altered")
    for original, supplied in zip(expected.scalar_rows, batch.scalar_rows, strict=True):
        for field in ("reported_value", "lower", "upper", "uncertainty_value"):
            if _float_bits(getattr(original.target, field)) != _float_bits(getattr(supplied.target, field)):
                raise ValueError("native Float64 bits altered")
