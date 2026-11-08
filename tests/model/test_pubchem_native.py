"""PubChem source-bound reader tests; real-source coverage is explicitly bounded."""
import copy
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
BASE = Path("outputs/data-intake/20261101-PubChem-corpus-pilot-v1")
MASTER = BASE / "independent-master-union-v20/master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json"
SHA = "fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f"
LONG = {
    "reference_prior": ("303fc17e7bfcf0dd6fa15098acf1e16c910c1efd89c7146ec6d543d26ec23092", 3536),
    "smiles_to_selfies": ("1faabb9f90bfdfb2c5e109c03a09c2a8e1801a8374ae60065c1cd71da35c9500", 9290),
    "selfies_to_smiles": ("530faf5263c2640c4c335c8a0eeb4573ba4f5728ed7744b100be80bc7d3f1bc5", 3536),
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_api():
    path = os.environ.get("MORA3_PUBCHEM_MODULE")
    if path:
        if not Path(path).is_file():
            pytest.fail("PubChem native reader is not implemented")
        spec = importlib.util.spec_from_file_location("mora3.data.pubchem_native", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    try:
        return importlib.import_module("mora3.data.pubchem_native")
    except ModuleNotFoundError:
        pytest.fail("PubChem native reader is not implemented")


@pytest.fixture(scope="module")
def source_cases():
    if not (ROOT / MASTER).is_file():
        pytest.skip("bounded real admitted PubChem source is unavailable")
    import pyarrow.parquet as pq
    master = json.loads((ROOT / MASTER).read_bytes())
    assert sha((ROOT / MASTER).read_bytes()) == SHA
    cases = {}
    for name, indices in [("00000", [0]), ("00027", [0, 1])]:
        path = str(BASE / "shards" / (name + ".parquet"))
        shard = next(s for s in master["shards"] if s["path"] == path)
        group_receipt = master["effective_group_lookup"][path]
        assert sha((ROOT / path).read_bytes()) == shard["sha256"]
        assert sha((ROOT / group_receipt["path"]).read_bytes()) == group_receipt["sha256"]
        rows = pq.read_table(ROOT / path, use_threads=False).to_pylist()
        groups = pq.read_table(ROOT / group_receipt["path"], use_threads=False).to_pylist()
        for i in indices:
            row = rows[i]
            ref = json.loads(row["native_payload"])["audit"]["source_ref"]
            src = (ROOT / ref["native_source"]["path"]).read_bytes()
            raw = (ROOT / ref["native_record"]["path"]).read_bytes()
            assert sha(src) == ref["native_source"]["sha256"]
            assert sha(raw) == ref["native_record"]["sha256"]
            native = pq.read_table(ROOT / ref["native_record"]["path"], use_threads=False).slice(
                ref["native_record"]["row_index"], 1).to_pylist()[0]
            direction = "reference_prior" if name == "00000" else ("smiles_to_selfies" if i == 0 else "selfies_to_smiles")
            cases[direction] = (row, native, json.loads(groups[i]["effective_groups_json"]),
                dict(manifest_sha256=SHA, shard_sha256=shard["sha256"], groups_sha256=group_receipt["sha256"]),
                json.loads(src))
    return cases


def project(api, case):
    row, native, groups, binding, source = copy.deepcopy(case)
    return api.pubchem_native_view(row, native, effective_groups=groups,
        corpus_binding=binding, source_manifest=source)


@pytest.mark.parametrize("direction,input_keys,target_key", [
    ("reference_prior", {"prior_domain"}, "native_smiles"),
    ("smiles_to_selfies", {"representation_task", "native_smiles"}, "standard_selfies"),
    ("selfies_to_smiles", {"representation_task", "standard_selfies"}, "source_canonical_isomeric_smiles"),
])
def test_real_views_expose_only_declared_full_native_fields(source_cases, direction, input_keys, target_key):
    api = load_api()
    view = project(api, source_cases[direction])
    assert set(view.inputs()) == input_keys
    assert set(view.targets()) == {target_key}
    assert view.direction == direction
    assert view.target_native_bytes == view.targets()[target_key].encode()
    tokenizer = ByteTokenizer((b"[C]", b"standard_", b"SMILES"))
    assert tokenizer.decode(tokenizer.encode(view.input_bytes)) == view.input_bytes
    assert tokenizer.decode(tokenizer.encode(view.target_native_bytes)) == view.target_native_bytes
    p = view.provenance()
    assert p["heldout_eligible"] is False
    assert p["paired_cellular_observation"] is False
    assert p["corpus_binding"]["manifest_sha256"] == SHA
    assert p["canonical_graph_role"] == "derived_exact_isomeric_graph_not_original_native_bytes"
    assert p["coarse_guard_role"] == "leakage_guard_not_exact_stereo_join"
    assert p["effective_groups"] == source_cases[direction][2]
    assert sha(view.input_bytes) == p["condition_sha256"]


def test_unsupported_selfies_native_source_is_prior_only(source_cases):
    api = load_api()
    case = source_cases["reference_prior"]
    assert case[1]["graph_roundtrip_pass"] is False
    assert project(api, case).provenance()["standard_selfies_available"] is False
    codec = copy.deepcopy(source_cases["smiles_to_selfies"])
    codec[1]["graph_roundtrip_pass"] = False
    codec[1]["standard_selfies_target_present"] = False
    codec[1]["standard_selfies"] = None
    with pytest.raises(ValueError):
        project(api, codec)


@pytest.mark.parametrize("direction", ["reference_prior", "smiles_to_selfies", "selfies_to_smiles"])
@pytest.mark.parametrize("injected", ["record_id", "full_axis_values", "target_hash", "source_native_split", "hidden_target"])
def test_injected_target_or_provenance_condition_is_rejected(source_cases, direction, injected):
    api = load_api()
    case = copy.deepcopy(source_cases[direction])
    d = json.loads(case[0]["native_payload"])
    d["conditions"][injected] = "target-or-audit"
    case[0]["native_payload"] = canonical(d)
    case[0]["native_payload_sha256"] = sha(case[0]["native_payload"])
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("field,value", [
    ("split", "validation"), ("admitted_for_task", False),
    ("record_id", "injected"), ("source_revision", "other"),
    ("source_sha256", "0" * 64), ("task_id", "0" * 64),
    ("task_name", "computed_GroupSELFIES_smiles_to_selfies"),
    ("full_axis_values", 1), ("evidence_class", "experimental_response"),
    ("split_policy", "random"),
])
def test_tampered_envelope_allocation_identity_or_axis_rejected(source_cases, field, value):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    case[0][field] = value
    with pytest.raises(ValueError):
        project(api, case)


def test_payload_hash_tamper_rejected(source_cases):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    case[0]["native_payload"] += b" "
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("direction", ["reference_prior", "smiles_to_selfies", "selfies_to_smiles"])
def test_rehashed_target_label_drift_rejected_against_original_source(source_cases, direction):
    api = load_api()
    case = copy.deepcopy(source_cases[direction])
    d = json.loads(case[0]["native_payload"])
    d["targets"][next(iter(d["targets"]))] = "CC"
    case[0]["native_payload"] = canonical(d)
    case[0]["native_payload_sha256"] = sha(case[0]["native_payload"])
    case[0]["full_axis_values"] = 2
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("mutation", ["drop_effective", "wrong_exact", "drop_original", "split", "stereo", "constraints", "source_label"])
def test_source_group_runtime_or_stereo_drift_rejected(source_cases, mutation):
    api = load_api()
    case = list(copy.deepcopy(source_cases["smiles_to_selfies"]))
    if mutation == "drop_effective":
        case[2] = case[2][:-1]
    elif mutation == "wrong_exact":
        case[2] = ["chemical_exact_isomeric:" + "0" * 64] + case[2][1:]
    elif mutation == "drop_original":
        case[0]["group_ids_json"] = "[]"
    elif mutation == "split":
        case[1]["source_native_split"] = "validation"
    elif mutation == "stereo":
        case[1]["source_canonical_isomeric_smiles"] = "C[C@H](O)C(=O)O"
    elif mutation == "source_label":
        case[1]["standard_selfies"] = "[C][C]"
    else:
        case[4]["semantic_constraints"]["C"] = 3
    with pytest.raises(ValueError):
        project(api, case)


def test_audit_only_source_metadata_never_enters_conditions(source_cases):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    before = project(api, case)
    case[1]["exception_class"] = "audit-only-marker"
    after = project(api, case)
    assert after.input_bytes == before.input_bytes
    assert b"audit-only-marker" not in after.input_bytes
    assert b"audit-only-marker" in after.provenance_bytes


@pytest.mark.parametrize("direction", list(LONG))
def test_real_reader_roundtrips_longest_admitted_axes(direction):
    api = load_api()
    if not (ROOT / MASTER).is_file():
        pytest.skip("bounded real admitted PubChem source is unavailable")
    tid, size = LONG[direction]
    views = api.read_pubchem_native_views(MASTER, root=ROOT, expected_manifest_sha256=SHA,
        direction=direction, limit=1, task_ids=(tid,))
    assert len(views) == 1
    view = views[0]
    assert len(view.target_native_bytes) == size
    tokenizer = ByteTokenizer()
    assert tokenizer.decode(tokenizer.encode(view.target_native_bytes)) == view.target_native_bytes
    if direction != "reference_prior":
        assert view.provenance()["axes"]["standard_selfies_tokens"] == 1773
    assert view.provenance()["envelope"]["task_id"] == tid


@pytest.mark.parametrize("direction", ["reference_prior", "smiles_to_selfies", "selfies_to_smiles"])
def test_reader_real_short_views_are_source_bound(direction):
    api = load_api()
    if not (ROOT / MASTER).is_file():
        pytest.skip("bounded real admitted PubChem source is unavailable")
    views = api.read_pubchem_native_views(MASTER, root=ROOT, expected_manifest_sha256=SHA,
        direction=direction, limit=2)
    assert len(views) == 2
    assert all(v.direction == direction for v in views)
    assert len({v.provenance()["envelope"]["task_id"] for v in views}) == 2


@pytest.mark.parametrize("kwargs", [
    dict(limit=0), dict(limit=257), dict(limit=True), dict(direction="GroupSELFIES"),
    dict(expected_manifest_sha256="0" * 64), dict(task_ids=("missing",)),
])
def test_reader_rejects_unpinned_unsupported_or_unbounded_requests(kwargs):
    api = load_api()
    values = dict(root=ROOT, expected_manifest_sha256=SHA, direction="reference_prior", limit=1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        api.read_pubchem_native_views(MASTER, **values)


def test_rehashed_noncanonical_native_payload_is_rejected(source_cases):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    case[0]["native_payload"] += b" "
    case[0]["native_payload_sha256"] = sha(case[0]["native_payload"])
    with pytest.raises(ValueError):
        project(api, case)


def test_invalid_codec_symbols_raise_native_reader_value_error(source_cases):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    case[1]["standard_selfies"] = "[unsupported]"
    with pytest.raises(ValueError):
        project(api, case)


@pytest.fixture
def synthetic_stereo_case(source_cases):
    """Synthetic algorithm test, never evidence of admitted source coverage."""
    import selfies as sf
    case = list(copy.deepcopy(source_cases["smiles_to_selfies"]))
    row, native, groups, binding, manifest = case
    smiles = "[13CH3][C@H](O)C(=O)[O-].[Na+]"
    value = "[13CH3][C@H1][Branch1][C][O][C][=Branch1][C][=O][O-1].[Na+1]"
    native.update(native_smiles=smiles, source_canonical_isomeric_smiles=smiles,
        source_current_canonical_graph_sha256=sha(smiles.encode()),
        standard_selfies=value, standard_selfies_token_count=len(tuple(sf.split_selfies(value))),
        decoded_canonical_isomeric_smiles=smiles)
    base = "CC(O)C(=O)[O-].[Na+]"
    case[2] = sorted(["chemical_exact_isomeric:" + sha(smiles.encode()),
        "molecule:" + sha(base.encode()), "scaffold:" + sha(base.encode())])
    row["group_ids_json"] = json.dumps(case[2])
    d = json.loads(row["native_payload"])
    d["conditions"]["native_smiles"] = smiles
    d["targets"]["standard_selfies"] = value
    row.update(native_payload=canonical(d), native_payload_sha256=sha(canonical(d)),
        full_axis_values=len(value))
    return case


def test_synthetic_stereo_isotope_charge_components_keep_exact_graph(synthetic_stereo_case):
    api = load_api()
    view = project(api, synthetic_stereo_case)
    assert view.inputs()["native_smiles"] == "[13CH3][C@H](O)C(=O)[O-].[Na+]"
    assert view.targets()["standard_selfies"] == "[13CH3][C@H1][Branch1][C][O][C][=Branch1][C][=O][O-1].[Na+1]"
    assert view.provenance()["canonical_graph_role"] == "derived_exact_isomeric_graph_not_original_native_bytes"


def test_synthetic_stereo_loss_is_rejected_despite_same_coarse_guard(synthetic_stereo_case):
    api = load_api()
    case = synthetic_stereo_case
    case[1]["standard_selfies"] = "[13CH3][CH1][Branch1][C][O][C][=Branch1][C][=O][O-1].[Na+1]"
    with pytest.raises(ValueError):
        project(api, case)


def test_real_prior_hidden_source_identity_and_target_do_not_change_conditions():
    api = load_api()
    if not (ROOT / MASTER).is_file():
        pytest.skip("bounded real admitted PubChem source is unavailable")
    a, b = api.read_pubchem_native_views(MASTER, root=ROOT, expected_manifest_sha256=SHA,
        direction="reference_prior", limit=2)
    assert a.input_bytes == b.input_bytes == b'{"prior_domain":"molecular_reference_native_SMILES"}'
    assert a.target_native_bytes != b.target_native_bytes
    assert a.provenance()["envelope"]["record_id"] != b.provenance()["envelope"]["record_id"]
    assert a.binding_sha256 != b.binding_sha256
    inputs = a.inputs()
    inputs["hidden"] = "mutation"
    assert a.inputs() == {"prior_domain": "molecular_reference_native_SMILES"}


def test_reader_rejects_manifest_bytes_changed_under_expected_admission_hash(tmp_path):
    api = load_api()
    (tmp_path / "manifest.json").write_bytes(b'{"shards":[]}')
    with pytest.raises(ValueError, match="integrity"):
        api.read_pubchem_native_views("manifest.json", root=tmp_path, expected_manifest_sha256=SHA,
            direction="reference_prior", limit=1)


@pytest.mark.parametrize("field,value", [("path", "other-source.json"), ("bytes", 0)])
def test_rehashed_source_manifest_reference_drift_rejected(source_cases, field, value):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    d = json.loads(case[0]["native_payload"])
    d["audit"]["source_ref"]["native_source"][field] = value
    receipt = json.loads(case[0]["admission_receipt_json"])
    receipt["source_ref"] = d["audit"]["source_ref"]
    case[0]["admission_receipt_json"] = json.dumps(receipt)
    case[0]["native_payload"] = canonical(d)
    case[0]["native_payload_sha256"] = sha(case[0]["native_payload"])
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("direction,field", [
    ("reference_prior", "standard_SELFIES_available"),
    ("reference_prior", "source_geometry_present"),
    ("reference_prior", "future_generated_outputs_require_model_produced_provenance"),
    ("smiles_to_selfies", "standard_SELFIES_not_original_GroupSELFIES"),
    ("smiles_to_selfies", "atomic_charge_isotope_and_isomeric_graph_roundtrip_verified"),
    ("smiles_to_selfies", "source_geometry_present"),
    ("selfies_to_smiles", "standard_SELFIES_not_original_GroupSELFIES"),
])
def test_rehashed_audit_numeric_alias_cannot_replace_native_boolean(source_cases, direction, field):
    api = load_api()
    case = copy.deepcopy(source_cases[direction])
    d = json.loads(case[0]["native_payload"])
    assert type(d["audit"][field]) is bool
    d["audit"][field] = int(d["audit"][field])
    case[0]["native_payload"] = canonical(d)
    case[0]["native_payload_sha256"] = sha(case[0]["native_payload"])
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("direction", ["reference_prior", "smiles_to_selfies", "selfies_to_smiles"])
@pytest.mark.parametrize("field", ["full_native_axes", "no_3D_or_experimental_response_pair"])
def test_admission_receipt_numeric_alias_cannot_replace_native_boolean(source_cases, direction, field):
    api = load_api()
    case = copy.deepcopy(source_cases[direction])
    receipt = json.loads(case[0]["admission_receipt_json"])
    receipt[field] = 1
    case[0]["admission_receipt_json"] = json.dumps(receipt)
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("field", ["image_available", "expression_available", "segmentation_available"])
def test_condition_context_requires_native_boolean_missingness(source_cases, field):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    context = json.loads(case[0]["condition_context_json"])
    context[field] = 0
    case[0]["condition_context_json"] = json.dumps(context)
    with pytest.raises(ValueError):
        project(api, case)


@pytest.mark.parametrize("reference", ["native_source", "native_record"])
def test_nested_source_reference_byte_count_retains_integer_type(source_cases, reference):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    d = json.loads(case[0]["native_payload"])
    ref = d["audit"]["source_ref"]
    ref[reference]["bytes"] = float(ref[reference]["bytes"])
    receipt = json.loads(case[0]["admission_receipt_json"])
    receipt["source_ref"] = ref
    case[0]["admission_receipt_json"] = json.dumps(receipt)
    case[0]["native_payload"] = canonical(d)
    case[0]["native_payload_sha256"] = sha(case[0]["native_payload"])
    with pytest.raises(ValueError):
        project(api, case)


def test_native_selfies_symbol_axis_count_must_be_integer(source_cases):
    api = load_api()
    case = copy.deepcopy(source_cases["smiles_to_selfies"])
    case[1]["standard_selfies_token_count"] = float(case[1]["standard_selfies_token_count"])
    with pytest.raises(ValueError):
        project(api, case)


def test_file_reader_rejects_a_different_real_manifest_before_source_projection():
    api = load_api()
    wrong = Path("outputs/data-intake/20261101-NCI-AIDS-original-v1/independent-master-union-v19/master-corpus-manifest-v19-original-NCI-assays-v1.json")
    if not (ROOT / wrong).is_file():
        pytest.skip("independent wrong-manifest fixture is unavailable")
    with pytest.raises(ValueError, match="integrity"):
        api.read_pubchem_native_views(wrong, root=ROOT, expected_manifest_sha256=SHA,
            direction="reference_prior", limit=1)

# Whole-source extension tests. Real rows below are bounded admitted fixtures.
WHOLE_BASE = Path("outputs/data-intake/20261101-PubChem-native-corpus-v1")
WHOLE_MASTER = WHOLE_BASE / "materialized/independent-final-v1/master-corpus-manifest-v21-PubChem-whole-native-train-v1.json"
WHOLE_SHA = "0e85a28928c027b7855e3bf12713a172128e12bed1e17752bee5f1d27b3f70b2"
V22_MASTER = Path("outputs/data-intake/20261101-LINCS-corpus-v1/independent-master-union-v22/master-corpus-manifest-v22-LINCS-qualified-source-coordinate-v1.json")
V22_SHA = "82ef478d53ccbd2bb8a87607b1ce5dd35a6a0f58f6afa7d912aa214129c1e6f2"
WHOLE_FIXED_BINDING = dict(
    component_manifest_sha256="0303f6cc34ff3573335e9e8bc83197f94f24b33e1463657702ce5a7c0ff0b4a1",
    construction_proof_sha256="fe4b91c64bd3a5b3795bf897d1944e8b23422fe45cf0410a587f135a263f0623",
    census_manifest_sha256="ae4b120e6b68e2a213bb5bf59883253a8fc7841a30fcd4a4c1caf9758d64bdaf",
    census_proof_sha256="287314202c492f4ba54b0079a40a342e3fa7ab66f0a142938d4b9249d9e4d025")
WHOLE_SHARDS = dict(reference_prior=str(WHOLE_BASE / "materialized/shards/000001.parquet"),
    smiles_to_selfies=str(WHOLE_BASE / "materialized/shards/000000.parquet"),
    selfies_to_smiles=str(WHOLE_BASE / "materialized/shards/000000.parquet"))


@pytest.fixture(scope="module")
def whole_cases():
    if not (ROOT / WHOLE_MASTER).exists():
        pytest.skip("whole admitted PubChem fixture unavailable")
    import pyarrow.parquet as pq
    assert sha((ROOT / WHOLE_MASTER).read_bytes()) == WHOLE_SHA
    m = json.loads((ROOT / WHOLE_MASTER).read_bytes())
    source = json.loads((ROOT / "outputs/data-sources/20261031-legacy-mint-v1/curated/PubChem10m-standardSELFIES-full-v1/manifest.json").read_bytes())
    cases = {}
    for direction, path in WHOLE_SHARDS.items():
        sh = next(s for s in m["shards"] if s["path"] == path)
        gr = m["effective_group_lookup"][path]
        assert sha((ROOT / path).read_bytes()) == sh["sha256"]
        assert sha((ROOT / gr["path"]).read_bytes()) == gr["sha256"]
        rows = pq.read_table(ROOT / path, use_threads=False).to_pylist()
        row = rows[1 if direction == "selfies_to_smiles" else 0]
        groups = {g["task_id"]: json.loads(g["effective_groups_json"]) for g in
            pq.read_table(ROOT / gr["path"], use_threads=False).to_pylist()}
        ref = json.loads(row["native_payload"])["audit"]["source_ref"]
        nr = ref["native_record"]
        assert sha((ROOT / nr["path"]).read_bytes()) == nr["sha256"]
        native = pq.read_table(ROOT / nr["path"], use_threads=False).slice(nr["row_index"], 1).to_pylist()[0]
        cr = json.loads(row["admission_receipt_json"])["census_ref"]
        assert sha((ROOT / cr["path"]).read_bytes()) == cr["sha256"]
        census = pq.read_table(ROOT / cr["path"], use_threads=False).slice(cr["row_index"], 1).to_pylist()[0]
        binding = dict(manifest_sha256=WHOLE_SHA, shard_sha256=sh["sha256"],
            groups_sha256=gr["sha256"], census_part_sha256=cr["sha256"],
            census_part_path=cr["path"], census_row_index=cr["row_index"], **WHOLE_FIXED_BINDING)
        cases[direction] = (row, native, groups[row["task_id"]], binding, source, census)
    return cases


def project_whole(api, case):
    row, native, groups, binding, source, census = copy.deepcopy(case)
    return api.pubchem_native_view(row, native, effective_groups=groups,
        corpus_binding=binding, source_manifest=source, census_record=census)


@pytest.mark.parametrize("direction", list(WHOLE_SHARDS))
def test_whole_direct_projection_exact_native_roles(whole_cases, direction):
    v = project_whole(load_api(), whole_cases[direction])
    row, native = whole_cases[direction][:2]
    target = {"reference_prior": "native_smiles", "smiles_to_selfies": "standard_selfies",
        "selfies_to_smiles": "source_canonical_isomeric_smiles"}[direction]
    assert v.target_native_bytes == native[target].encode()
    assert set(v.inputs()) == ({"prior_domain"} if direction == "reference_prior" else
        {"representation_task", "native_smiles" if direction == "smiles_to_selfies" else "standard_selfies"})
    p = v.provenance()
    assert p["envelope"]["task_id"] == row["task_id"]
    assert p["corpus_binding"]["construction_proof_sha256"] == WHOLE_FIXED_BINDING["construction_proof_sha256"]
    assert p["model_training_support"] is False and p["full_corpus_usage"] is False
    assert p["whole_component_available"] is True


@pytest.mark.parametrize("manifest,digest", [(WHOLE_MASTER, WHOLE_SHA), (V22_MASTER, V22_SHA)])
@pytest.mark.parametrize("direction", list(WHOLE_SHARDS))
def test_whole_file_reader_exact_selected_new_shards(whole_cases, manifest, digest, direction):
    api = load_api()
    tid = whole_cases[direction][0]["task_id"]
    v, = api.read_pubchem_native_views(manifest, root=ROOT, expected_manifest_sha256=digest,
        direction=direction, task_ids=(tid,), limit=1, shard_paths=(WHOLE_SHARDS[direction],))
    expected = list(copy.deepcopy(whole_cases[direction]))
    expected[3]["manifest_sha256"] = digest
    assert v == project_whole(api, expected)


@pytest.mark.parametrize("mutation", [
    "target", "target_axis", "hidden_condition", "audit_bool_alias", "receipt_bool_alias",
    "receipt_graph", "receipt_census_path", "receipt_row_alias", "source_row_alias",
    "context_mask_alias", "source_split", "whole_policy", "original_group", "effective_group",
    "census_gate", "census_gate_alias", "census_native_hash", "census_graph",
    "census_stereo", "census_codec_mask", "census_parse_alias", "census_extra",
    "binding_component", "binding_proof", "binding_master", "binding_part",
    "model_flag", "no_census"])
def test_whole_source_type_scope_axis_and_leakage_mutations_rejected(whole_cases, mutation):
    api = load_api()
    case = list(copy.deepcopy(whole_cases["smiles_to_selfies"]))
    row, native, groups, binding, source, census = case
    payload = json.loads(row["native_payload"])
    receipt = json.loads(row["admission_receipt_json"])
    if mutation == "target": payload["targets"]["standard_selfies"] = "[C]"
    elif mutation == "target_axis": row["full_axis_values"] -= 1
    elif mutation == "hidden_condition": payload["conditions"]["hidden_target_hash"] = "0" * 64
    elif mutation == "audit_bool_alias": payload["audit"]["source_geometry_present"] = 0
    elif mutation == "receipt_bool_alias": receipt["strict_source_stereo_projection_available"] = 1
    elif mutation == "receipt_graph": receipt["original_source_graph_sha256"] = "0" * 64
    elif mutation == "receipt_census_path": receipt["census_ref"]["path"] = "other.parquet"
    elif mutation == "receipt_row_alias": receipt["census_ref"]["row_index"] = False
    elif mutation == "source_row_alias":
        payload["audit"]["source_ref"]["source_native_row"] = False
        receipt["source_ref"] = payload["audit"]["source_ref"]
    elif mutation == "context_mask_alias":
        context = json.loads(row["condition_context_json"]); context["mask_available"] = 0
        row["condition_context_json"] = json.dumps(context)
    elif mutation == "source_split": native["source_native_split"] = "validation"
    elif mutation == "whole_policy": row["split_policy"] = load_api().POLICY
    elif mutation == "original_group": row["group_ids_json"] = json.dumps(groups[:-1])
    elif mutation == "effective_group": case[2] = groups[:-1]
    elif mutation == "census_gate": census["admission_gate_code"] = 16
    elif mutation == "census_gate_alias": census["admission_gate_code"] = False
    elif mutation == "census_native_hash": census["source_native_SMILES_sha256"] = b"0" * 32
    elif mutation == "census_graph": census["canonical_isomeric_graph_sha256"] = b"0" * 32
    elif mutation == "census_stereo": census["lossy_stereo_warning"] = True
    elif mutation == "census_codec_mask": census["derived_SELFIES_graph_pass"] = False
    elif mutation == "census_parse_alias": census["source_native_parse_pass"] = 1
    elif mutation == "census_extra": census["hidden"] = "unexpected"
    elif mutation == "model_flag": row["model_adapter_supported"] = True
    elif mutation == "no_census": case[5] = None
    else:
        key = dict(binding_component="component_manifest_sha256", binding_proof="construction_proof_sha256",
            binding_master="manifest_sha256", binding_part="census_part_sha256")[mutation]
        binding[key] = "0" * 64
    row["native_payload"] = canonical(payload)
    row["native_payload_sha256"] = sha(row["native_payload"])
    row["admission_receipt_json"] = json.dumps(receipt)
    with pytest.raises(ValueError):
        project_whole(api, case)


@pytest.mark.parametrize("manifest,digest", [(WHOLE_MASTER, WHOLE_SHA), (V22_MASTER, V22_SHA)])
def test_whole_preserves_real_pilot_prior_only_without_duplicate_admission(source_cases, manifest, digest):
    api = load_api()
    row = source_cases["reference_prior"][0]
    tid = row["task_id"]
    path = str(BASE / "shards/00000.parquet")
    v, = api.read_pubchem_native_views(manifest, root=ROOT, expected_manifest_sha256=digest,
        direction="reference_prior", task_ids=(tid,), limit=1, shard_paths=(path,))
    old = project(api, source_cases["reference_prior"])
    assert v.input_bytes == old.input_bytes and v.target_native_bytes == old.target_native_bytes
    assert v.provenance()["standard_selfies_available"] is False
    assert v.provenance()["envelope"] == old.provenance()["envelope"]
    assert v.provenance()["corpus_binding"]["manifest_sha256"] == digest
    with pytest.raises(ValueError):
        api.read_pubchem_native_views(manifest, root=ROOT, expected_manifest_sha256=digest,
            direction="smiles_to_selfies", task_ids=(tid,), limit=1, shard_paths=(path,))


@pytest.mark.parametrize("selectors", [(), ("unknown.parquet",), (WHOLE_SHARDS["reference_prior"],)*2])
def test_whole_shard_selectors_are_exact_unique_and_admitted(selectors):
    with pytest.raises(ValueError):
        load_api().read_pubchem_native_views(V22_MASTER, root=ROOT, expected_manifest_sha256=V22_SHA,
            direction="reference_prior", limit=1, shard_paths=selectors)


def test_whole_manifest_substitution_is_rejected_before_native_io(tmp_path):
    wrong = tmp_path / "fake-master.json"
    wrong.write_bytes((ROOT / WHOLE_MASTER).read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        load_api().read_pubchem_native_views(wrong, root=tmp_path, expected_manifest_sha256=WHOLE_SHA,
            direction="reference_prior", limit=1)


def test_whole_native_metadata_stays_audit_only(whole_cases):
    case = copy.deepcopy(whole_cases["reference_prior"])
    api = load_api()
    before = project_whole(api, case)
    case[1]["exception_class"] = "audit-only-marker"
    after = project_whole(api, case)
    assert before.input_bytes == after.input_bytes
    assert b"audit-only-marker" in after.provenance_bytes
    assert before.binding_sha256 != after.binding_sha256

@pytest.mark.parametrize("artifact", ["proof", "component", "group", "native", "census"])
def test_whole_file_reader_rejects_corrupted_consumed_bytes(whole_cases, monkeypatch, artifact):
    api = load_api()
    case = whole_cases["smiles_to_selfies"]
    receipt = json.loads(case[0]["admission_receipt_json"])
    targets = dict(proof=str(WHOLE_BASE / "materialized/independent-final-v1/receipt.json"),
        component=str(WHOLE_BASE / "materialized/manifest.json"),
        group=str(WHOLE_BASE / "materialized/groups/000000.parquet"),
        native=receipt["source_ref"]["native_record"]["path"],
        census=receipt["census_ref"]["path"])
    target = (ROOT / targets[artifact]).resolve()
    read = Path.read_bytes
    touched = []
    def changed(path):
        raw = read(path)
        if path.resolve() == target:
            touched.append(True)
            return raw + b" "
        return raw
    monkeypatch.setattr(Path, "read_bytes", changed)
    with pytest.raises(ValueError):
        api.read_pubchem_native_views(V22_MASTER, root=ROOT, expected_manifest_sha256=V22_SHA,
            direction="smiles_to_selfies", limit=1, task_ids=(case[0]["task_id"],),
            shard_paths=(WHOLE_SHARDS["smiles_to_selfies"],))
    assert touched


def test_whole_synthetic_multicomponent_guards_preserve_original_stereo(synthetic_stereo_case, whole_cases):
    # Synthetic algorithm fixture; first bounded whole shard had no multicomponent forward row.
    case = list(copy.deepcopy(whole_cases["smiles_to_selfies"]))
    native = case[1]
    for key in ("native_smiles", "source_canonical_isomeric_smiles", "source_current_canonical_graph_sha256",
                "standard_selfies", "standard_selfies_token_count", "decoded_canonical_isomeric_smiles"):
        native[key] = synthetic_stereo_case[1][key]
    coarse = "CC(O)C(=O)[O-].[Na+]"
    components = ("CC(O)C(=O)[O-]", "[Na+]")
    groups = {"chemical_exact_isomeric:" + sha(native["source_canonical_isomeric_smiles"].encode())}
    for value in (coarse,) + components:
        groups.update(("molecule:" + sha(value.encode()), "scaffold:" + sha(value.encode())))
    case[2] = sorted(groups)
    row = case[0]
    row["group_ids_json"] = json.dumps(case[2])
    d = json.loads(row["native_payload"])
    d["conditions"]["native_smiles"] = native["native_smiles"]
    d["targets"]["standard_selfies"] = native["standard_selfies"]
    row["native_payload"] = canonical(d)
    row["native_payload_sha256"] = sha(row["native_payload"])
    row["full_axis_values"] = len(native["standard_selfies"])
    receipt = json.loads(row["admission_receipt_json"])
    receipt["original_source_graph_sha256"] = native["source_current_canonical_graph_sha256"]
    row["admission_receipt_json"] = json.dumps(receipt)
    case[5].update(source_native_SMILES_sha256=hashlib.sha256(native["native_smiles"].encode()).digest(),
        canonical_isomeric_graph_sha256=bytes.fromhex(native["source_current_canonical_graph_sha256"]),
        molecule_guard_sha256=hashlib.sha256(coarse.encode()).digest(),
        scaffold_guard_sha256=hashlib.sha256(coarse.encode()).digest())
    v = project_whole(load_api(), case)
    assert set(v.provenance()["effective_groups"]) == groups and len(groups) == 7
    assert v.inputs()["native_smiles"] == native["native_smiles"]
    assert v.target_native_bytes == native["standard_selfies"].encode()

@pytest.mark.parametrize("direction", list(WHOLE_SHARDS))
def test_whole_chemically_equivalent_target_reserialization_is_not_native_bytes(whole_cases, direction):
    from rdkit import Chem
    import selfies as sf
    case = copy.deepcopy(whole_cases[direction])
    row, native = case[:2]
    payload = json.loads(row["native_payload"])
    key = next(iter(payload["targets"]))
    if direction == "smiles_to_selfies":
        alternate = native["standard_selfies"] + "[nop]"
        assert sf.decoder(alternate) == sf.decoder(native["standard_selfies"])
    else:
        original = payload["targets"][key]
        mol = Chem.MolFromSmiles(original)
        alternate = Chem.MolToSmiles(mol, canonical=False, rootedAtAtom=1, isomericSmiles=True)
        assert alternate != original
        assert Chem.MolToSmiles(Chem.MolFromSmiles(alternate), isomericSmiles=True) == native["source_canonical_isomeric_smiles"]
    payload["targets"][key] = alternate
    row["full_axis_values"] = len(alternate)
    row["native_payload"] = canonical(payload)
    row["native_payload_sha256"] = sha(row["native_payload"])
    with pytest.raises(ValueError):
        project_whole(load_api(), case)
