"""Qualified LINCS observational native profiles and source dose coordinates.

The pinned component is verified but awaits global master inclusion. Full source
feature bits/nulls are bound via retained independently replayed original task
proofs; this reader does not reread the 475 MB raw single-row-group exports.
No physical dose converter, scaling, neural objective or training is implemented.
"""
from dataclasses import dataclass, replace
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import re

BASE = "outputs/data-intake/20261101-LINCS-corpus-v1"
COMPONENT_SHA256 = "52f5055af4e71297af793ac9ad55c6ae975654e5982341dd1f2cd9600bcfb29d"
HANDOFF = dict(path=BASE + "/model-handoff-component-v2/executable-model-handoff.json", bytes=8115,
    sha256="ce828d80d84bd58e4acedc2d3d4fd78bf378b041ab0a495e47586a4926deee9e")
SOURCE_HANDOFF = dict(path="outputs/data-sources/20261101-LINCS-Pilot1-source-v1/handoff-two-originals-full-native-verified-v1.json",
    bytes=7399, sha256="e237ccecb269b4561e00e291aaa26fc9870d13c3af805de2e77fb3622964ef17")
AUDIT = dict(path=BASE + "/metadata-admission-audit-v1/manifest.json", bytes=96095,
    sha256="147bd29451dc7713c99ff0c2d0e71da9e7212ebee032bcd7336b9d93e8776cab")
QUALIFICATION = dict(path=BASE + "/source-dose-coordinate-qualified-v2/source-dose-qualification.json", bytes=978,
    sha256="4493a10294ee7320fbcaa7e870db08afd248566393e12e389339131163ad4036")
TASK_PROOF_SHA256 = "ecbcd507bae7dc9dfd1dde328c5a9cf941b2ec2b2bb6f21e0d25d7f88309a54b"
PROJECTION_PROOF_SHA256 = "aeeab053b6d03866918b60397e87f87c71b766ecf9e624dce4c3c28104dfe126"
CANDIDATES_SHA256 = "e870dffb0f7b673cb26b741e85cf0476fd4b78b0c5ad71af23060404a4d9785d"
_PINS = {
    "CellPainting": dict(n=1783, time=48, asset="DSE-LINCS-Pilot1-CP",
        source="92d2a4e37d746ac78eb45eb68de772790eb42c981c76d19faef42dc17bb123bb",
        axis="757fd2db4cbb4a0aa8540509c7ec122c08912c2bcf2995dfb1b64fb9977db0b4",
        manifest="e983b574a43f0d3d680dad42e28debd60016fa7808247f8dabec470a9ecd2a78",
        axis_file="7d5513fea06d1d2bd2f59b0038360523a63b3d7170e088ab25ed46a11e4e559e",
        binding_scope="same-study exact full pert_id reference-only; not assay pair"),
    "L1000": dict(n=978, time=24, asset="DSE-LINCS-Pilot1-L1000",
        source="326df7e37de0ccf41a04ad1220945cf891a6f2113675fc7f72f7490ab03f7822",
        axis="d826656873901dcb8cdf6207aed31a3032e71fed2041350d2ff0f8a128ca915e",
        manifest="7116d019b8ab232a6bd77352a3b4c3fee6e38e9c36954633a82016c893ad2c61",
        axis_file="57e2c606d620747d0b0bd722412d58106c548d48639fd3b1bfaf063ba1343155",
        binding_scope="directGEsourceSMILES"),
}
DIRECTIONS = ("molecule_to_profile", "profile_to_molecule")
POLICY = "source_train_only;known_old_chemical_and_frozen_pilot_guards_excluded;no_new_heldout"
MISSING = ["raw_image_mask", "registered_single_cell_or_spatial_entities", "physical_pixel_scale_and_channels",
    "matched_cell_state", "same_cell_crossassay_pair", "calibrated_assay_concentration",
    "independent_biological_validation", "confirmed_physical_dose_unit"]


def _canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _is_sha(x):
    return type(x) is str and re.fullmatch("[0-9a-f]{64}", x) is not None


def _pairs(pairs):
    d = {}
    for k, v in pairs:
        if k in d:
            raise ValueError("duplicate JSON key")
        d[k] = v
    return d


def _json(raw):
    if type(raw) not in (str, bytes):
        raise ValueError("serialized native JSON required")
    def invalid(x):
        raise ValueError("nonfinite native JSON")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def _same(a, b, message):
    # Exact structural types, including nested bool/int and int/float aliases.
    if _canonical(a) != _canonical(b):
        raise ValueError(message)


def _axis(source_axis, modality):
    if modality not in _PINS or _sha(_canonical(source_axis)) != _PINS[modality]["axis"]:
        raise ValueError("pinned full native feature axis/schema required")
    names = source_axis["ordered_feature_names"]
    schema = {r["name"]: r["type"] for r in source_axis["native_schema"]}
    return tuple(names), tuple(schema[n] for n in names)


@dataclass(frozen=True)
class NativeProfileArray:
    axis_ids: tuple
    native_types: tuple
    values: tuple
    observed_mask: tuple
    feature_units: str


    def __post_init__(self):
        n = len(self.axis_ids)
        if (not n or any(type(x) is not tuple for x in
                (self.axis_ids, self.native_types, self.values, self.observed_mask))
                or any(len(x) != n for x in (self.native_types, self.values, self.observed_mask))
                or any(type(x) is not str or not x for x in self.axis_ids)
                or len(set(self.axis_ids)) != n or type(self.feature_units) is not str or not self.feature_units):
            raise ValueError("immutable complete native array columns required")
        for value, observed, native_type in zip(self.values, self.observed_mask, self.native_types):
            if type(observed) is not bool or observed is not (value is not None) or native_type not in ("double", "int64"):
                raise ValueError("native type/null mask required")
            if value is not None and (
                (native_type == "int64" and (type(value) is not int or not -2**63 <= value < 2**63))
                or (native_type == "double" and (type(value) is not float or not math.isfinite(value)))):
                raise ValueError("native values must preserve exact int64/float64 types")
        if not any(self.observed_mask):
            raise ValueError("native array needs an observed feature")

    @classmethod
    def from_mapping(cls, array, *, source_axis):
        modality = next((m for m, p in _PINS.items() if _sha(_canonical(source_axis)) == p["axis"]), None)
        axis, types = _axis(source_axis, modality)
        if (type(array) is not dict or set(array) != {"axis_ids", "native_types", "values", "observed_mask", "feature_units"}
                or any(type(array[k]) is not list for k in ("axis_ids", "native_types", "values", "observed_mask"))):
            raise ValueError("exact native array fields and ordered lists required")
        _same(array["axis_ids"], list(axis), "complete ordered source labels required")
        _same(array["native_types"], list(types), "exact native per-feature types required")
        if (len(array["values"]) != len(axis) or len(array["observed_mask"]) != len(axis)
                or array["feature_units"] != source_axis["feature_units"]):
            raise ValueError("full native values/masks/units required")
        for value, observed, native_type in zip(array["values"], array["observed_mask"], types):
            if type(observed) is not bool or observed is not (value is not None):
                raise ValueError("exact observed/null mask required; no imputation")
            if value is None:
                continue
            if native_type == "int64":
                if type(value) is not int or not -2**63 <= value < 2**63:
                    raise ValueError("source int64 cannot be coerced")
            elif type(value) is not float or not math.isfinite(value):
                raise ValueError("native finite float64 or None required")
        if not any(array["observed_mask"]):
            raise ValueError("source profile needs an observed feature")
        return cls(axis, types, tuple(array["values"]), tuple(array["observed_mask"]), array["feature_units"])

    def as_mapping(self):
        return dict(axis_ids=list(self.axis_ids), native_types=list(self.native_types),
            values=list(self.values), observed_mask=list(self.observed_mask), feature_units=self.feature_units)


def _dose(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("positive unchanged source export coordinate required")
    return dict(value=value, unit=None, physical_unit_confirmed=False, cross_assay_conversion_allowed=False,
        scope="assay-specific source export coordinate; not calibrated concentration")


def _time(modality):
    return dict(value=_PINS[modality]["time"], unit="hours",
        evidence="author curation assignment retained in every native row")


def lincs_input_bytes(task, *, modality, direction, source_axis):
    """Project only declared conditions; does not establish source admission.

    Hidden targets/audit/missingness are deliberately unread. Source-bound
    lincs_native_view/read_lincs_native_views must validate before inference.
    """
    try:
        return _inputs(task, modality=modality, direction=direction, source_axis=source_axis)
    except (KeyError, TypeError, UnicodeError) as exc:
        raise ValueError("malformed LINCS predictor conditions") from exc


def _inputs(task, *, modality, direction, source_axis):
    if modality not in _PINS or direction not in DIRECTIONS:
        raise ValueError("supported native modality/direction required")
    axis, types = _axis(source_axis, modality)
    c = task["conditions"]
    forward = direction == "molecule_to_profile"
    keys = {"assay_modality", "cell_line", "source_export_dose_coordinate", "source_exposure_time"} | (
        {"canonical_isomeric_smiles", "known_profile_axis", "known_native_feature_types"}
        if forward else {"source_profile_native_array"})
    if type(c) is not dict or set(c) != keys:
        raise ValueError("condition allowlist rejects hidden targets/audit/source identity")
    _same(c["assay_modality"], modality, "source assay modality mismatch")
    _same(c["cell_line"], "A549", "known source cell required")
    _same(c["source_exposure_time"], _time(modality), "source exposure time mismatch")
    _same(c["source_export_dose_coordinate"], _dose(c["source_export_dose_coordinate"]["value"]),
          "dose coordinate must have no physical unit/conversion")
    if forward:
        if type(c["canonical_isomeric_smiles"]) is not str or not c["canonical_isomeric_smiles"]:
            raise ValueError("source canonical graph required")
        _same(c["known_profile_axis"], list(axis), "fixed source feature axis required")
        _same(c["known_native_feature_types"], list(types), "fixed native types required")
    else:
        NativeProfileArray.from_mapping(c["source_profile_native_array"], source_axis=source_axis)
    return _canonical(c)


def _binding(inputs, targets, provenance):
    return _sha(_canonical(dict(schema="mora3.LincsNativeView/v1", inputs=_sha(inputs),
        targets=_sha(targets), provenance=_sha(provenance))))


@dataclass(frozen=True)
class LincsNativeView:
    direction: str
    modality: str
    axis: tuple
    input_bytes: bytes
    target_bytes: bytes
    provenance_bytes: bytes
    binding_sha256: str

    def __post_init__(self):
        if (self.direction not in DIRECTIONS or self.modality not in _PINS
                or any(type(x) is not bytes for x in (self.input_bytes, self.target_bytes, self.provenance_bytes))
                or self.binding_sha256 != _binding(self.input_bytes, self.target_bytes, self.provenance_bytes)):
            raise ValueError("immutable bound native view required")

    def inputs(self):
        return _json(self.input_bytes)

    def targets(self):
        return _json(self.target_bytes)

    def provenance(self):
        return _json(self.provenance_bytes)

    def native_profile(self):
        # Reconstruct immutable values without numeric coercion or null fill.
        a = (self.targets() if self.direction == "molecule_to_profile" else self.inputs())["source_profile_native_array"]
        return NativeProfileArray(tuple(a["axis_ids"]), tuple(a["native_types"]),
            tuple(a["values"]), tuple(a["observed_mask"]), a["feature_units"])


@dataclass(frozen=True)
class LincsNativeBatch:
    views: tuple
    direction: str
    modality: str
    axis: tuple
    profiles: tuple

    def inputs(self):
        return tuple(v.inputs() for v in self.views)

    def targets(self):
        return tuple(v.targets() for v in self.views)


def collate_lincs_native_views(views):
    """Homogeneous native arrays, preserving int64/double/None per feature."""
    views = tuple(views)
    if not views or any(not isinstance(v, LincsNativeView) for v in views):
        raise ValueError("nonempty source-bound native views required")
    first = views[0]
    if (any((v.direction, v.modality, v.axis) != (first.direction, first.modality, first.axis) for v in views)
            or len({v.provenance()["task_id"] for v in views}) != len(views)):
        raise ValueError("same source direction/axis and unique task allocation required")
    return LincsNativeBatch(views, first.direction, first.modality, first.axis, tuple(v.native_profile() for v in views))


def _chemical_groups(smiles):
    # Derived nonstereo/scaffold guards never replace the source isomeric graph.
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("unparseable strict source reference graph")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    mol = Chem.RemoveHs(mol)
    iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if iso != smiles:
        raise ValueError("exact canonical source isomeric graph required; no lossy replacement")
    guards = {"chemical_exact_isomeric:" + _sha(smiles.encode())}
    for graph in (mol, *Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)):
        graph = Chem.Mol(graph)
        Chem.RemoveStereochemistry(graph)
        base = Chem.MolToSmiles(graph, canonical=True, isomericSmiles=False)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=graph, includeChirality=False) or base
        guards.update(("molecule:" + _sha(base.encode()), "scaffold:" + _sha(scaffold.encode())))
    return sorted(guards)


def _audit_payload(source_audit, modality):
    return dict(source_audit=source_audit,
        source_observed_or_model_generated="source_observation_with_computed_assay_features",
        profile_evidence="source_image_computed_CellPainting_well_profile" if modality == "CellPainting" else
            "source_preprocessed_L1000_probe_profile;measurement_vs_inference_perprobe_unresolved",
        observed_mask_role="forward target/loss only; inverse explicitly observed input mask; no target-derived predictor mask",
        native_int64_and_float64_preserved=True, all_feature_axes_preserved=True,
        source_nominal_units_not_assay_calibration=True, source_profile_not_single_cell_or_object_geometry=True,
        same_cell_crossassay_pair=False, matched_replicate_crossassay_pair=False,
        causal_or_independent_biological_validation=False, generated_data=False,
        desired_profile_inference_requires_separate_desired_evidence_role=True,
        source_target_loss="observed features only; missing is unknown, never zero/negative")


def _qualification(value):
    return dict(original_source_dose_condition=dict(value=value, unit="micromolar",
        evidence="author curated source field; no new conversion or concentration calibration"),
        primary_provenance=QUALIFICATION, source_curated_unit_label="micromolar",
        upstream_CP_unit_labels=["profiling README mmoles/L", "IDR author library uM"],
        source_unit_status="unresolved provenance; no automatic physical-unit or material equivalence",
        GE_primary_unit_confirmation="not asserted by this projection; source-specific coordinate retained",
        source_numeric_value_unchanged=True, physical_dose_condition_valid=False, cross_assay_unit_conversion_allowed=False)


def lincs_native_view(row, original_row, native_metadata, candidate, *, effective_groups,
                      corpus_binding, source_manifest, source_axis):
    """Validate qualified v2 against original full-native proof and exact source metadata.

    Direct projection assumes supplied memory objects; actual file byte integrity
    is checked only by read_lincs_native_views. Original v1 is proof, not a second
    observation or an allowed physical-dose inference interface.
    """
    try:
        return _project(row, original_row, native_metadata, candidate, effective_groups=effective_groups,
            corpus_binding=corpus_binding, source_manifest=source_manifest, source_axis=source_axis)
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError("malformed source-bound qualified LINCS task") from exc


def _project(row, original_row, native_metadata, candidate, *, effective_groups, corpus_binding, source_manifest, source_axis):
    direction = row["task_name"]
    modality = candidate["modality"]
    axis, types = _axis(source_axis, modality)
    pin = _PINS[modality]
    if _sha(_canonical(source_manifest)) != pin["source"]:
        raise ValueError("pinned full-native source QC manifest required")
    asset = source_manifest["asset_id"]
    rawref = source_manifest["original_payload"]
    index = native_metadata["source_row"]
    if type(index) is not int or not 0 <= index < rawref["rows"]:
        raise ValueError("exact native source row index required")

    index_receipts = [r for r in source_manifest["row_indexes"]
                      if r["source_row_start"] <= index < r["source_row_stop"]]
    if len(index_receipts) != 1 or corpus_binding["metadata_index_sha256"] != index_receipts[0]["sha256"]:
        raise ValueError("exact native metadata index receipt binding required")
    rid = asset + ":sha256:" + rawref["sha256"] + ":row:" + str(index)
    _same(native_metadata["source_occurrence_key"], rid, "source occurrence identity mismatch")
    metadata_keys = set(source_axis["metadata_columns"]) | {"source_occurrence_key", "source_row", "profile_null_count",
        "profile_nan_count", "profile_positive_infinity_count", "profile_negative_infinity_count", "finite_complete_profile"}
    if set(native_metadata) != metadata_keys:
        raise ValueError("complete native metadata/index fields required")
    if native_metadata["Metadata_pert_type"] != "trt":
        raise ValueError("control/sentinel profiles are not qualified coordinate tasks")
    _same(native_metadata["Metadata_cell_id"], "A549", "source cell mismatch")
    _same(native_metadata["Metadata_pert_timepoint"], pin["time"], "source time mismatch")
    value = native_metadata["Metadata_pert_dose_micromolar"]
    _dose(value)
    smiles = candidate["canonical_isomeric_source_reference_graph"]
    chemistry = _chemical_groups(smiles)
    _same(_json(candidate["source_native_metadata_json"]), native_metadata, "native metadata candidate/row binding mismatch")
    _same(_json(candidate["chemical_groups_json"]), chemistry, "chemical/isomeric/scaffold guard mismatch")
    ge_rows = _json(candidate["source_graph_binding_GE_rows_json"])
    if (type(ge_rows) is not list or not ge_rows or any(type(i) is not int or not 0 <= i < 27837 for i in ge_rows)
            or len(set(ge_rows)) != len(ge_rows)):
        raise ValueError("exact same-study source reference rows required")
    expected_candidate = dict(source_occurrence_key=rid, source_asset_id=asset, modality=modality,
        source_native_row=index, source_native_metadata_json=candidate["source_native_metadata_json"],
        source_native_perturbation_id=native_metadata["Metadata_pert_id"], chemical_binding_scope=pin["binding_scope"],
        unique_exact_source_reference_graph_sha=_sha(smiles.encode()), canonical_isomeric_source_reference_graph=smiles,
        source_graph_binding_GE_rows_json=candidate["source_graph_binding_GE_rows_json"],
        chemical_groups_json=candidate["chemical_groups_json"], source_nominal_dose_micromolar=value,
        source_nominal_dose_is_known=True, positive_treatment_dose_condition_mask=True, source_time_hours=pin["time"],
        source_cell_line="A549", source_feature_nulls=native_metadata["profile_null_count"], source_full_feature_axis=pin["n"],
        source_finite_complete_profile=native_metadata["finite_complete_profile"], existing_heldout_chemical_groups_json="[]",
        frozen_pilot_validation_groups_json="[]", chemical_dose_time_cell_task_candidate=True,
        candidate_status="candidate_for_new_task_specific_admission", task_admitted=False, serialized_task_views=0,
        original_source_feature_values_unmodified=True, raw_image_or_mask_available_for_task=False,
        crossassay_same_cell_or_matched_replicate_pair=False)
    _same(candidate, expected_candidate, "qualified source candidate disposition/type mismatch")
    source_ref = dict(path="outputs/" + rawref["path"].split("/outputs/", 1)[1],
        bytes=rawref["bytes"], sha256=rawref["sha256"], row_index=index)
    source_audit = dict(source_handoff=SOURCE_HANDOFF, source_native_record=source_ref,
        source_original_metadata=native_metadata, exact_reference_binding_scope=pin["binding_scope"],
        source_graph_binding_GE_rows=ge_rows, admission_audit=AUDIT,
        target_loss_mask_source="native source feature null bitmap; not predictor condition in forward direction",
        source_raw_images_masks_unavailable_for_task=True)
    original_raw = original_row["native_payload"]
    raw = row["native_payload"]
    if any(type(x) is not bytes for x in (raw, original_raw)):
        raise ValueError("exact serialized original/qualified task bytes required")
    if _sha(original_raw) != original_row["native_payload_sha256"] or _sha(raw) != row["native_payload_sha256"]:
        raise ValueError("native source task payload integrity mismatch")
    original = _json(original_raw)
    forward = direction == "molecule_to_profile"
    profile = NativeProfileArray.from_mapping(
        original["targets" if forward else "conditions"]["source_profile_native_array"], source_axis=source_axis)
    nulls = len(profile.values) - sum(profile.observed_mask)
    for k, expected in dict(profile_null_count=nulls, profile_nan_count=0, profile_positive_infinity_count=0,
                            profile_negative_infinity_count=0, finite_complete_profile=(nulls == 0)).items():
        _same(native_metadata[k], expected, "source null/nonfinite validity mismatch")
    common = dict(assay_modality=modality, cell_line="A549", source_exposure_time=_time(modality))
    nominal = dict(common, source_nominal_dose=_qualification(value)["original_source_dose_condition"])
    if forward:
        nominal.update(canonical_isomeric_smiles=smiles, known_profile_axis=list(axis), known_native_feature_types=list(types))
        targets = dict(source_profile_native_array=profile.as_mapping())
    else:
        nominal.update(source_profile_native_array=profile.as_mapping())
        targets = dict(canonical_isomeric_smiles=smiles)
    original_audit = _audit_payload(source_audit, modality)
    expected_original = dict(schema="mora3.CorpusTask/v1", task="source_observational_LINCS_" + direction,
        conditions=nominal, targets=targets, audit=original_audit, missing=MISSING[:-1])
    if original_raw != _canonical(expected_original):
        raise ValueError("original source task/metadata/axis/masks/graph proof mismatch")
    qualified = copy.deepcopy(expected_original)
    qualified["conditions"].pop("source_nominal_dose")
    qualified["conditions"]["source_export_dose_coordinate"] = _dose(value)
    qualified["audit"]["source_unit_semantic_qualification"] = _qualification(value)
    qualified["missing"] = MISSING
    if raw != _canonical(qualified):
        raise ValueError("exact qualified v2 task/target/audit/types/source coordinate mismatch")
    source_groups = sorted(set(chemistry) | {"source_study:LINCS-Pilot1",
        "source_parent:" + asset + ":" + rawref["sha256"],
        "source_perturbation:LINCS-Pilot1:" + _sha(_canonical(native_metadata["Metadata_pert_id"])),
        "source_plate:" + _sha(_canonical([asset, native_metadata["Metadata_Plate"]])),
        "source_well:" + _sha(_canonical([asset, native_metadata["Metadata_Plate"], native_metadata["Metadata_Well"]]))})
    effective = list(effective_groups)
    if any(type(g) is not str or not g for g in effective) or len(set(effective)) != len(effective) or not set(source_groups) <= set(effective):
        raise ValueError("unique effective groups must retain source chemical/study/plate/well guards")
    receipt = dict(source=SOURCE_HANDOFF, source_license="CC0-1.0", source_metadata_audit=AUDIT,
        source_native_null_mask_preserved=True, source_physical_types_preserved=True,
        source_parent_and_perturbation_groups_present=True, no_same_cell_crossassay_join=True,
        full_profile_axis=pin["n"], pilot_not_admitted=False)
    context = dict(cell_line="A549", source_nominal_dose_micromolar=value, source_exposure_time_hours=pin["time"],
        source_nominal_dose_calibrated=False, raw_image_available=False, source_mask_available=False,
        segmentation_available=False, source_generated=False)
    qualified_receipt = dict(receipt, source_dose_semantic_qualification=QUALIFICATION,
        source_original_native_payload_sha256=_sha(original_raw), physical_dose_task_admitted=False,
        source_export_coordinate_task_admitted=True, source_unit_status="unconfirmed/conflicting upstream labels; no conversion")
    qualified_context = {k:v for k,v in context.items() if k != "source_nominal_dose_micromolar"}
    qualified_context.update(source_export_dose_coordinate_value=value, source_export_claimed_unit="micromolar",
        physical_dose_condition_valid=False, cross_assay_unit_conversion_allowed=False)
    family = "Rosetta-LINCS-Pilot1-molecule-profile-" + modality
    tid = _sha(_canonical([family, rid, direction]))
    for envelope, payload, rec, ctx, missing in [(original_row, original_raw, receipt, context, MISSING[:-1]),
                                                (row, raw, qualified_receipt, qualified_context, MISSING)]:
        fields = {k:v for k,v in envelope.items() if k not in ("native_payload", "admission_receipt_json",
            "condition_context_json", "group_ids_json", "missing_context_json")}
        expected_fields = dict(task_id=tid, record_id=rid, source_family=family,
            source_revision="S3 Jan26_2025 entity " + source_manifest["source_release_entity"]["etag"] + ";author v1.0.0",
            source_sha256=SOURCE_HANDOFF["sha256"], task_name=direction, native_schema="mora3.CorpusTask/v1",
            native_payload_sha256=_sha(payload), split="train", split_policy=POLICY,
            evidence_class="source_image_computed_well_profile" if modality == "CellPainting" else
                "source_preprocessed_probe_profile;measured_vs_inferred_perprobe_unresolved",
            admitted_for_task=True, model_adapter_supported=False, full_axis_values=pin["n"])
        _same(fields, expected_fields, "source train/task/revision/axis admission envelope mismatch")
        _same(_json(envelope["admission_receipt_json"]), rec, "exact source admission receipt/type mismatch")
        _same(_json(envelope["condition_context_json"]), ctx, "source dose/time/missingness context mismatch")
        _same(_json(envelope["missing_context_json"]), missing, "source missingness mismatch")
        _same(_json(envelope["group_ids_json"]), source_groups, "source group conservation mismatch")
    keys = {"component_sha256", "shard_sha256", "groups_sha256", "original_shard_sha256",
        "metadata_index_sha256", "candidates_sha256", "source_axis_sha256", "source_manifest_sha256",
        "native_task_proof_sha256", "qualification_proof_sha256"}
    if (set(corpus_binding) != keys or not all(_is_sha(v) for v in corpus_binding.values())
            or corpus_binding["component_sha256"] != COMPONENT_SHA256
            or corpus_binding["source_manifest_sha256"] != pin["manifest"]
            or corpus_binding["source_axis_sha256"] != pin["axis_file"]
            or corpus_binding["candidates_sha256"] != CANDIDATES_SHA256
            or corpus_binding["native_task_proof_sha256"] != TASK_PROOF_SHA256
            or corpus_binding["qualification_proof_sha256"] != PROJECTION_PROOF_SHA256):
        raise ValueError("pinned qualified component/source/axis/proof bindings required")
    inputs = lincs_input_bytes(qualified, modality=modality, direction=direction, source_axis=source_axis)
    target_bytes = _canonical(targets)
    provenance = _canonical(dict(schema="mora3.LincsNativeAdmission/v1", task_id=tid, record_id=rid,
        source_family=family, source_revision=row["source_revision"], native_source_ref=source_ref,
        source_axis_sha256=pin["axis_file"], native_profile_sha256=_sha(_canonical(profile.as_mapping())),
        qualified_payload_sha256=_sha(raw), original_proof_payload_sha256=_sha(original_raw),
        source_metadata_sha256=_sha(_canonical(native_metadata)), effective_groups=effective,
        corpus_binding=corpus_binding, source_metadata_audit=AUDIT, dose_qualification=QUALIFICATION,
        observational_only=True, heldout_eligible=False, split="train", global_master_inclusion=False,
        master_manifest_sha256=None, component_verified=True, consumed_file_bytes_verified=False,
        raw_source_reread=False, full_native_values_bound_via_retained_independent_source_proof=True,
        source_reference_graph_scope="literal isomeric reference graph; not material/assay or same-cell equivalence",
        image_available=False, mask_available=False, source_time_assays_distinct=True,
        physical_dose_valid=False, physical_unit_confirmed=False, cross_assay_conversion_allowed=False,
        neural_objective_supported=False, model_token_counts=None, source_scaling_fit=None))
    return LincsNativeView(direction, modality, axis, inputs, target_bytes, provenance, _binding(inputs, target_bytes, provenance))


def _verified_bytes(root, receipt):
    path = (root / receipt["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("source path outside project")
    raw = path.read_bytes()
    if (type(receipt["bytes"]) is not int or len(raw) != receipt["bytes"] or _sha(raw) != receipt["sha256"]):
        raise ValueError("source/shard/group/retained-proof integrity mismatch")
    return raw


def read_lincs_native_views(manifest, *, root, expected_manifest_sha256, shard_paths, limit, task_ids=None):
    """Bounded component-only read: 1..8 explicit v2 shards, at most 64 views.

    Consumed v2/original-v1 proof shard, group, native metadata index, candidate,
    axis and retained proof bytes are verified. Raw exports are not scanned.
    """
    try:
        return _read(manifest, root=root, expected_manifest_sha256=expected_manifest_sha256,
            shard_paths=shard_paths, limit=limit, task_ids=task_ids)
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError("malformed qualified LINCS reader request/source") from exc


def _read(manifest, *, root, expected_manifest_sha256, shard_paths, limit, task_ids):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    if type(limit) is not int or not 1 <= limit <= 64 or expected_manifest_sha256 != COMPONENT_SHA256:
        raise ValueError("pinned qualified v2 component and explicit bounded limit required")
    if (type(shard_paths) not in (list, tuple) or not 1 <= len(shard_paths) <= 8
            or any(type(s) is not str for s in shard_paths) or len(set(shard_paths)) != len(shard_paths)):
        raise ValueError("unique bounded qualified shard selectors required")
    if task_ids is not None and (type(task_ids) not in (list, tuple) or not 1 <= len(task_ids) <= limit
            or any(not _is_sha(t) for t in task_ids) or len(set(task_ids)) != len(task_ids)):
        raise ValueError("unique exact bounded task selectors required")
    root = Path(root).resolve()
    handoff = _json(_verified_bytes(root, HANDOFF))
    expected_component = handoff["component"]
    path = (root / manifest).resolve()
    if not path.is_relative_to(root) or _sha(path.read_bytes()) != COMPONENT_SHA256:
        raise ValueError("qualified component manifest integrity mismatch; original v1 not accepted")
    m = _json(_verified_bytes(root, expected_component))
    for field, sha in [("source_full_native_independent_proof", TASK_PROOF_SHA256),
                       ("source_coordinate_projection_proof", PROJECTION_PROOF_SHA256)]:
        if m[field]["sha256"] != sha:
            raise ValueError("independent proof binding mismatch")
        _json(_verified_bytes(root, m[field]))
    _json(_verified_bytes(root, m["source_handoff"]))
    qualification = _json(_verified_bytes(root, m["source_dose_qualification"]))
    _json(_verified_bytes(root, qualification["source_primary"]))
    original = _json(_verified_bytes(root, m["source_original_component"]))
    audit = _json(_verified_bytes(root, m["source_metadata_admission_audit"]))
    candidate_ref = audit["candidate_rows"]
    candidates = pq.read_table(io.BytesIO(_verified_bytes(root, candidate_ref)), use_threads=False)
    if candidates.num_rows != candidate_ref["rows"]:
        raise ValueError("source candidate count mismatch")
    shards = {s["path"]:s for s in m["shards"]}
    if len(shards) != len(m["shards"]) or any(s not in shards for s in shard_paths):
        raise ValueError("selector outside qualified component or duplicate shard allocation")
    by_group = {original["effective_group_lookup"][s["path"]]["path"]:s for s in original["shards"]}
    wanted = set(task_ids) if task_ids is not None else None
    views = []; seen = set(); source_cache = {}; index_cache = {}
    for selected in shard_paths:
        shard = shards[selected]; side = m["effective_group_lookup"][selected]; old_shard = by_group[side["path"]]
        _same(side, original["effective_group_lookup"][old_shard["path"]], "replacement group proof binding mismatch")
        rows = pq.read_table(io.BytesIO(_verified_bytes(root, shard)), use_threads=False).to_pylist()
        olds = pq.read_table(io.BytesIO(_verified_bytes(root, old_shard)), use_threads=False).to_pylist()
        groups = pq.read_table(io.BytesIO(_verified_bytes(root, side)), use_threads=False).to_pylist()
        if (len(rows) != shard["rows"] or len(olds) != old_shard["rows"]
                or len(groups) != side["rows"] or len({len(rows),len(olds),len(groups)}) != 1):
            raise ValueError("qualified/original/group source row counts mismatch")
        for row, old_row, group in zip(rows, olds, groups):
            if (set(group) != {"task_id", "effective_groups_json"} or row["task_id"] != old_row["task_id"]
                    or row["task_id"] != group["task_id"] or row["task_id"] in seen):
                raise ValueError("duplicate allocation or source/group/task alignment mismatch")
            seen.add(row["task_id"])
        for row, old_row, group in zip(rows, olds, groups):
            if wanted is not None and row["task_id"] not in wanted:
                continue
            d = _json(row["native_payload"]); modality = d["conditions"]["assay_modality"]
            if modality not in source_cache:
                sr = next(r for r in m["source_components"] if r["sha256"] == _PINS[modality]["manifest"])
                source = _json(_verified_bytes(root, sr))
                axis_ref = m["source_full_axes"][modality]["axis_file"]
                axis = _json(_verified_bytes(root, axis_ref))
                source_cache[modality] = source, axis, sr, axis_ref
            source, axis, sr, axis_ref = source_cache[modality]
            idx = d["audit"]["source_audit"]["source_native_record"]["row_index"]
            ir = next(r for r in source["row_indexes"] if r["source_row_start"] <= idx < r["source_row_stop"])
            if ir["sha256"] not in index_cache:
                table = pq.read_table(io.BytesIO(_verified_bytes(root, ir)), use_threads=False)
                if table.num_rows != ir["rows"]:
                    raise ValueError("native source metadata index count mismatch")
                index_cache[ir["sha256"]] = table
            metadata = index_cache[ir["sha256"]].slice(idx-ir["source_row_start"], 1).to_pylist()[0]
            matches = candidates.filter(pc.equal(candidates["source_occurrence_key"], row["record_id"])).to_pylist()
            if len(matches) != 1:
                raise ValueError("one exact eligible source candidate required")
            binding = dict(component_sha256=COMPONENT_SHA256, shard_sha256=shard["sha256"], groups_sha256=side["sha256"],
                original_shard_sha256=old_shard["sha256"], metadata_index_sha256=ir["sha256"],
                candidates_sha256=candidate_ref["sha256"], source_axis_sha256=axis_ref["sha256"],
                source_manifest_sha256=sr["sha256"], native_task_proof_sha256=TASK_PROOF_SHA256,
                qualification_proof_sha256=PROJECTION_PROOF_SHA256)
            view = lincs_native_view(row, old_row, metadata, matches[0],
                effective_groups=_json(group["effective_groups_json"]), corpus_binding=binding,
                source_manifest=source, source_axis=axis)
            p = view.provenance()
            p.update(consumed_file_bytes_verified=True, source_metadata_index_sha256=ir["sha256"],
                source_full_native_proof_sha256=TASK_PROOF_SHA256,
                source_coordinate_projection_proof_sha256=PROJECTION_PROOF_SHA256,
                raw_source_proof_scope="retained full native replay; no raw export opened by this reader")
            pb = _canonical(p)
            views.append(replace(view, provenance_bytes=pb, binding_sha256=_binding(view.input_bytes, view.target_bytes, pb)))
            if wanted is None and len(views) == limit:
                return tuple(views)
    if wanted is not None and {v.provenance()["task_id"] for v in views} != wanted:
        raise ValueError("selected task absent from bounded qualified component scope")
    if not views:
        raise ValueError("no qualified matching source views")
    return tuple(views)
