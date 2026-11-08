"""NCI original assay reader tests; exact native rows and synthetic corruption cases."""
import copy
from decimal import Decimal
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest
from mora3.tokenization.shortest_path import ByteTokenizer

ROOT = Path(os.environ.get("MORA3_REPO_ROOT", Path.cwd()))
BASE = Path("outputs/data-intake/20261101-NCI-AIDS-original-v1")
MASTER = BASE / "independent-master-union-v19/master-corpus-manifest-v19-original-NCI-assays-v1.json"
SHA = "4fa73767f29703d851a487690e16327537af4029ac3bad654d41c382821c1d7e"
FAMILIES = {
    "conclusion": "NCI-AIDS-source-conclusion-prior",
    "endpoint": "NCI-AIDS-source-endpoint-prior",
    "paired": "NCI-AIDS-paired-assay-profile-prior",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_api():
    path = os.environ.get("MORA3_NCI_MODULE")
    if path:
        if not Path(path).is_file():
            pytest.fail("NCI original native reader is not implemented")
        existing = sys.modules.get("mora3.data.nci_aids")
        if existing is not None and Path(existing.__file__).resolve() == Path(path).resolve():
            return existing
        spec = importlib.util.spec_from_file_location("mora3.data.nci_aids", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module("mora3.data.nci_aids")
    except ModuleNotFoundError:
        pytest.fail("NCI original native reader is not implemented")


@pytest.fixture(scope="module")
def sources():
    if not (ROOT / MASTER).is_file():
        pytest.skip("original admitted NCI source is unavailable")
    import pyarrow.parquet as pq
    raw = (ROOT / MASTER).read_bytes()
    assert sha(raw) == SHA
    master = json.loads(raw)
    curated = json.loads((ROOT / BASE / "curated-native-v2/manifest.json").read_bytes())
    cache = {}
    def source_row(ref):
        if ref["path"] not in cache:
            raw = (ROOT / ref["path"]).read_bytes()
            assert sha(raw) == ref["sha256"]
            cache[ref["path"]] = pq.read_table(ROOT / ref["path"], use_threads=False)
        return cache[ref["path"]].slice(ref["row_index"], 1).to_pylist()[0]
    cases = {}
    for shard in master["shards"]:
        if not shard["path"].startswith(str(BASE / "admitted-source-tasks-v1")) or shard["split"] != "train":
            continue
        raw = (ROOT / shard["path"]).read_bytes()
        assert sha(raw) == shard["sha256"]
        rows = pq.read_table(ROOT / shard["path"], use_threads=False).to_pylist()
        side = master["effective_group_lookup"][shard["path"]]
        assert sha((ROOT / side["path"]).read_bytes()) == side["sha256"]
        groups = {r["task_id"]: json.loads(r["effective_groups_json"])
            for r in pq.read_table(ROOT / side["path"], use_threads=False).to_pylist()}
        for row in rows:
            kind = next(k for k, v in FAMILIES.items() if row["source_family"] == v)
            direction = "prediction" if "_prediction_" in row["task_name"] else "generation"
            key = (kind, direction)
            if key in cases:
                continue
            d = json.loads(row["native_payload"])
            ref = d["audit"]["source_ref"]
            outcomes = [source_row(x) for x in ref.get("outcomes", [ref.get("outcome")])]
            cases[key] = [row, source_row(ref["structure"]), outcomes,
                source_row(ref["context_link"]) if "context_link" in ref else None,
                groups[row["task_id"]], dict(manifest_sha256=SHA, shard_sha256=shard["sha256"],
                    groups_sha256=side["sha256"], split="train")]
        if len(cases) == 6:
            break
    assert len(cases) == 6
    return cases, curated, source_row


def project(api, case):
    row, structure, outcomes, context, groups, binding = copy.deepcopy(case)
    return api.nci_aids_view(row, structure, outcomes, context_link=context,
        effective_groups=groups, corpus_binding=binding)


def reseal(case, d):
    row = case[0]
    raw = canonical(d)
    row["native_payload"] = raw
    row["native_payload_sha256"] = sha(raw)
    row["task_id"] = sha(canonical([row["source_family"], row["record_id"], raw.decode()]))


@pytest.mark.parametrize("kind", list(FAMILIES))
@pytest.mark.parametrize("direction", ["prediction", "generation"])
def test_real_native_families_preserve_source_roles_and_complete_axes(sources, kind, direction):
    api = load_api()
    case = sources[0][kind, direction]
    view = project(api, case)
    assert view.direction == direction
    assert view.source_family == FAMILIES[kind]
    assert view.split == "train"
    assert view.provenance()["source_stereochemistry_known"] is False
    assert view.provenance()["unknown_stereo_is_not_achirality"] is True
    assert view.provenance()["source_geometry_observed"] is False
    assert view.provenance()["same_cell_or_replicate_pair"] is False
    assert view.provenance()["blind_or_new_study_evaluation"] is False
    assert set(view.inputs()).isdisjoint({"NSC", "CAS_RN", "source_flag", "StdDev", "NumExp"})
    target = view.targets()
    if direction == "generation":
        assert set(target) == {"molecule_source_connectivity_smiles"}
        assert len(target["molecule_source_connectivity_smiles"]) == case[0]["full_axis_values"]
        assert "molecule_source_connectivity_smiles" not in view.inputs()
    elif kind == "conclusion":
        assert target["source_conclusion"] in ("CA", "CM", "CI")
        assert view.class_axis == ("CA", "CM", "CI")
    else:
        assert len(view.source_point_values) == (2 if kind == "paired" else 1)
        assert all(type(x) is Decimal for x in view.source_point_values)
        assert view.source_point_mask == (True,) * len(view.source_point_values)
    if kind == "paired":
        assert view.axis == ("infected_cell_protection_EC50", "uninfected_cell_growth_IC50")
        assert view.provenance()["selectivity_ratio_derived"] is False
    tokenizer = ByteTokenizer((b"NCI", b"source_", b"CCC"))
    for raw in (view.input_bytes, view.target_bytes):
        assert tokenizer.decode(tokenizer.encode(raw)) == raw


@pytest.mark.parametrize("kind", list(FAMILIES))
@pytest.mark.parametrize("direction", ["prediction", "generation"])
@pytest.mark.parametrize("injected", ["source_NSC", "hidden_target", "source_flag", "source_native_fields"])
def test_condition_target_or_audit_injection_rejected(sources, kind, direction, injected):
    api = load_api()
    case = copy.deepcopy(sources[0][kind, direction])
    d = json.loads(case[0]["native_payload"])
    d["conditions"][injected] = "forbidden"
    reseal(case, d)
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("field,value", [
    ("split", "test"), ("admitted_for_task", False), ("record_id", "wrong"),
    ("native_payload_sha256", "0" * 64), ("source_sha256", "0" * 64),
    ("source_revision", "wrong"), ("full_axis_values", 999),
    ("task_name", "invented_direction"), ("evidence_class", "clinical_truth"),
    ("split_policy", "random"),
])
def test_tampered_envelope_identity_split_axis_or_direction_rejected(sources, field, value):
    api = load_api()
    case = copy.deepcopy(sources[0]["endpoint", "prediction"])
    case[0][field] = value
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("kind", list(FAMILIES))
def test_rehashed_target_drift_rejected_against_native_outcome(sources, kind):
    api = load_api()
    case = copy.deepcopy(sources[0][kind, "prediction"])
    d = json.loads(case[0]["native_payload"])
    d["targets"][next(iter(d["targets"]))] = "CA" if kind == "conclusion" else [1, 2] if kind == "paired" else 99
    reseal(case, d)
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("damage", ["bytes", "graph", "stereo", "geometry", "groups", "source_ref"])
def test_native_structure_group_and_source_binding_drift_rejected(sources, damage):
    api = load_api()
    case = copy.deepcopy(sources[0]["conclusion", "prediction"])
    if damage == "bytes":
        case[1]["all_native_record_bytes"] += b" "
    elif damage == "graph":
        d = json.loads(case[1]["projection_json"])
        d["derived_nonstereo_connectivity_smiles"] = "CCO"
        case[1]["projection_json"] = json.dumps(d)
    elif damage == "stereo":
        case[1]["source_stereochemistry_known"] = True
    elif damage == "geometry":
        case[1]["source_geometry_observed"] = True
    elif damage == "groups":
        case[4] = case[4][:-1]
    else:
        d = json.loads(case[0]["native_payload"])
        d["audit"]["source_ref"]["native_source"]["path"] = "other.json"
        receipt = json.loads(case[0]["admission_receipt_json"])
        receipt["source_ref"] = d["audit"]["source_ref"]
        case[0]["admission_receipt_json"] = json.dumps(receipt)
        reseal(case, d)
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("flag,unit", [(">", "M"), ("<", "M"), ("=", "V")])
def test_censored_or_unknown_unit_source_rows_remain_unavailable_points(sources, flag, unit):
    api = load_api()
    # Native-field mutation is a synthetic gate test, not admitted source coverage.
    row = copy.deepcopy(sources[0]["endpoint", "prediction"][2][0])
    fields = json.loads(row["source_native_fields_json"])
    fields["Flag"], fields["ConcUnit"] = flag, unit
    row["source_native_fields_json"] = json.dumps(fields)
    projection = json.loads(row["projection_json"])
    projection.update(source_native_fields=fields, source_flag=flag, source_ConcUnit=unit,
        point_target_available=False, positive_uncertainty_target_available=False,
        physical_unit_mapping_available=unit in ("M", "u"),
        source_documentation_gap=(row["endpoint_kind"] == "IC50" and (unit == "V" or flag == "<")),
        unit_meaning="unresolved_volumetric_definition" if unit == "V" else "molar",
        censoring_role="all_source_experiments_reached_endpoint" if flag == "=" else
            "source_aggregate_flag_at_least_one_experiment;not_bound_of_aggregate_mean")
    projection["context_key"][2] = unit
    projection["compound_highestdose_context_key"][1] = unit
    row["projection_json"] = json.dumps(projection)
    state = api.nci_endpoint_state(row)
    assert state.point_target_available is False
    case = copy.deepcopy(sources[0]["endpoint", "prediction"])
    case[2][0] = row
    with pytest.raises(ValueError):
        project(api, case)


def test_source_decimal_precision_and_postassay_statistics_never_enter_conditions(sources):
    api = load_api()
    case = copy.deepcopy(sources[0]["endpoint", "prediction"])
    before = project(api, case)
    row = case[2][0]
    p = json.loads(row["projection_json"])
    fields = json.loads(row["source_native_fields_json"])
    fields["NumExp"] = "999"
    p["source_NumExp"] = 999
    p["source_native_fields"] = fields
    row.update(projection_json=json.dumps(p), source_native_fields_json=json.dumps(fields))
    after = project(api, case)
    assert after.input_bytes == before.input_bytes
    assert after.provenance_bytes != before.provenance_bytes
    key = "Log10" + row["endpoint_kind"]
    fields[key] = "-5.7000000000000000000000000001"
    p["source_native_fields"] = fields
    row.update(projection_json=json.dumps(p), source_native_fields_json=json.dumps(fields))
    with pytest.raises(ValueError):
        api.nci_endpoint_state(row)


@pytest.mark.parametrize("damage", ["unit", "high", "point_mask", "ids", "samecell"])
def test_paired_profile_requires_exact_separate_assay_context(sources, damage):
    api = load_api()
    case = copy.deepcopy(sources[0]["paired", "prediction"])
    if damage == "unit":
        case[3]["source_ConcUnit"] = "u"
    elif damage == "high":
        case[3]["source_log10_highest_concentration_literal"] = "-99"
    elif damage == "point_mask":
        case[3]["exact_one_per_endpoint_and_both_source_point_masks"] = False
    elif damage == "ids":
        case[3]["IC50_source_record_ids_json"] = '["wrong"]'
    else:
        case[3]["same_cell_or_replicate_link"] = True
    with pytest.raises(ValueError):
        project(api, case)


@pytest.fixture(scope="module")
def real_views():
    cache = {}
    def load():
        if cache:
            return cache
        api = load_api()
        if not (ROOT / MASTER).is_file():
            pytest.skip("original admitted NCI source is unavailable")
        for kind, source in FAMILIES.items():
            for direction in ("prediction", "generation"):
                for split in ("train", "validation", "test"):
                    cache[kind, direction, split] = api.read_nci_aids_views(MASTER, root=ROOT,
                        expected_manifest_sha256=SHA, source_family=source, direction=direction, split=split, limit=2)
        return cache
    return load


@pytest.mark.parametrize("kind", list(FAMILIES))
@pytest.mark.parametrize("direction", ["prediction", "generation"])
@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_real_reader_covers_all_admitted_family_direction_splits(real_views, kind, direction, split):
    views = real_views()[kind, direction, split]
    assert len(views) == 2
    assert all(v.split == split and v.direction == direction for v in views)
    assert all(v.provenance()["corpus_binding"]["manifest_sha256"] == SHA for v in views)
    assert all(v.provenance()["raw_source_readback_verified"] is True for v in views)
    assert all(v.provenance()["source_point_mask"] == ([True, True] if kind == "paired" else [True] if kind == "endpoint" else []) for v in views)


def test_collation_preserves_native_decimals_masks_and_rejects_mixed_roles(real_views):
    api = load_api()
    real_views = real_views()
    views = real_views["paired", "prediction", "train"]
    batch = api.collate_nci_aids_views(views)
    assert batch["source_point_values"] == tuple(v.source_point_values for v in views)
    assert batch["source_point_masks"] == ((True, True), (True, True))
    assert batch["point_role"] == "observed_target"
    assert all(type(v) is Decimal for row in batch["source_point_values"] for v in row)
    assert batch["input_bytes"] == tuple(v.input_bytes for v in views)
    for other in [real_views["paired", "generation", "train"][0],
                  real_views["paired", "prediction", "test"][0],
                  real_views["endpoint", "prediction", "train"][0]]:
        with pytest.raises(ValueError):
            api.collate_nci_aids_views((views[0], other))


@pytest.mark.parametrize("kwargs", [
    dict(limit=0), dict(limit=True), dict(limit=257), dict(split="blind"),
    dict(direction="inverse"), dict(source_family="MolNet-HIV"), dict(expected_manifest_sha256="0" * 64),
])
def test_reader_rejects_unbounded_or_unpinned_requests(kwargs):
    api = load_api()
    params = dict(root=ROOT, expected_manifest_sha256=SHA, source_family=FAMILIES["endpoint"],
        direction="prediction", split="train", limit=1)
    params.update(kwargs)
    with pytest.raises(ValueError):
        api.read_nci_aids_views(MASTER, **params)


@pytest.mark.parametrize("field", ["point_target_available", "physical_unit_mapping_available",
    "positive_uncertainty_target_available", "source_documentation_gap",
    "zero_SD_is_not_perfect_precision", "same_cell_or_replicate_pair_asserted"])
def test_native_endpoint_masks_require_booleans_not_numeric_aliases(sources, field):
    api = load_api()
    row = copy.deepcopy(sources[0]["endpoint", "prediction"][2][0])
    p = json.loads(row["projection_json"])
    p[field] = int(p[field])
    row["projection_json"] = json.dumps(p)
    with pytest.raises(ValueError):
        api.nci_endpoint_state(row)


def test_rehashed_numeric_false_condition_is_not_a_native_boolean(sources):
    api = load_api()
    case = copy.deepcopy(sources[0]["conclusion", "prediction"])
    d = json.loads(case[0]["native_payload"])
    d["conditions"]["source_stereochemistry_known"] = 0
    reseal(case, d)
    with pytest.raises(ValueError):
        project(api, case)


def test_collation_rejects_view_direction_drift_even_with_unchanged_bytes(real_views):
    from dataclasses import replace
    api = load_api()
    view = real_views()["paired", "prediction", "train"][0]
    with pytest.raises(ValueError):
        api.collate_nci_aids_views((replace(view, direction="generation"),))


@pytest.mark.parametrize("endpoint,rows", [
    ("EC50", (0, 218, 11156)), ("IC50", (5, 33, 11135)),
])
def test_real_native_censor_and_unknown_unit_rows_keep_false_point_masks(sources, endpoint, rows):
    api = load_api()
    _, curated, source_row = sources
    file = next(f for f in curated["endpoint_files"] if f["endpoint"] == endpoint)
    states = []
    for index in rows:
        row = source_row(dict(file, row_index=index))
        states.append(api.nci_endpoint_state(row))
    assert tuple(s.source_flag for s in states[:2]) == (">", "<")
    assert states[2].unit == "V"
    assert all(s.point_target_available is False for s in states)
    assert all(s.value.is_finite() for s in states)


@pytest.mark.parametrize("tid,readout", [
    ("2e440b0786565f578ff91f5439973ed8f7f8bac8a4fe2b28f888d4c69fb97f46", "protection_of_infected_cells"),
    ("0906e351d43d0942c9ee97be1ceb447ab0240e7a5f74657de4f631952a6ca645", "growth_inhibition_of_uninfected_cells"),
])
def test_real_microgram_per_mL_means_preserve_units_without_molar_conversion(tid, readout):
    api = load_api()
    if not (ROOT / MASTER).is_file():
        pytest.skip("original admitted NCI source is unavailable")
    view, = api.read_nci_aids_views(MASTER, root=ROOT, expected_manifest_sha256=SHA,
        source_family=FAMILIES["endpoint"], direction="prediction", split="train",
        limit=1, task_ids=(tid,))
    assert view.inputs()["concentration_unit"] == "u"
    assert view.inputs()["assay_readout"] == readout
    assert view.provenance()["concentration_unit"] == "u"
    assert view.source_point_mask == (True,)


def test_real_generation_does_not_encode_hidden_molecule_or_identifiers(real_views):
    a, b = real_views()["conclusion", "generation", "train"]
    assert a.inputs() == b.inputs()
    assert a.targets() != b.targets()
    assert a.provenance()["envelope"]["record_id"] != b.provenance()["envelope"]["record_id"]
    assert a.inputs()["desired_source_conclusion"] == "CI"


@pytest.mark.parametrize("reference", ["structure", "outcome"])
def test_rehashed_source_row_index_boolean_alias_is_rejected(sources, reference):
    api = load_api()
    case = copy.deepcopy(sources[0]["conclusion", "prediction"])
    d = json.loads(case[0]["native_payload"])
    assert d["audit"]["source_ref"][reference]["row_index"] == 0
    d["audit"]["source_ref"][reference]["row_index"] = False
    receipt = json.loads(case[0]["admission_receipt_json"])
    receipt["source_ref"] = d["audit"]["source_ref"]
    case[0]["admission_receipt_json"] = json.dumps(receipt)
    reseal(case, d)
    with pytest.raises(ValueError):
        project(api, case)


def test_malformed_original_component_group_raises_value_error(sources):
    api = load_api()
    case = copy.deepcopy(sources[0]["conclusion", "prediction"])
    groups = json.loads(case[0]["group_ids_json"])
    groups = [g for g in groups if not g.startswith("component:")] + [123]
    case[0]["group_ids_json"] = json.dumps(groups)
    case[4] = [str(g) for g in groups]
    with pytest.raises(ValueError):
        project(api, case)
