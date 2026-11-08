"""Strict LINCS coordinate/native-array reader; bounded real component fixtures."""
import copy
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys

import pytest

ROOT = Path(os.environ.get("MORA3_REPO_ROOT", Path.cwd()))
BASE = Path("outputs/data-intake/20261101-LINCS-corpus-v1")
COMPONENT = BASE / "source-dose-coordinate-qualified-v2/manifest.json"
SHA = "52f5055af4e71297af793ac9ad55c6ae975654e5982341dd1f2cd9600bcfb29d"


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def sha(x):
    return hashlib.sha256(x).hexdigest()


def api():
    path = os.environ.get("MORA3_LINCS_MODULE")
    if path:
        if not Path(path).is_file():
            pytest.fail("qualified LINCS native reader is not implemented")
        spec = importlib.util.spec_from_file_location("mora3.data.lincs_native", path)
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod; spec.loader.exec_module(mod)
        return mod
    try:
        return importlib.import_module("mora3.data.lincs_native")
    except ModuleNotFoundError:
        pytest.fail("qualified LINCS native reader is not implemented")


@pytest.fixture(scope="module")
def cases():
    if not (ROOT / COMPONENT).is_file():
        pytest.skip("qualified LINCS component is unavailable")
    import pyarrow.parquet as pq
    m = json.loads((ROOT / COMPONENT).read_bytes())
    assert sha((ROOT / COMPONENT).read_bytes()) == SHA
    old = json.loads((ROOT / m["source_original_component"]["path"]).read_bytes())
    old_by_group = {old["effective_group_lookup"][s["path"]]["path"]: s for s in old["shards"]}
    audit = json.loads((ROOT / m["source_metadata_admission_audit"]["path"]).read_bytes())
    result = {}
    for label, name, rowindex in [("CP", "00000", 0), ("L1000", "00863", 0), ("CP_partial", "00685", 6)]:
        sh = next(s for s in m["shards"] if s["path"].endswith("/" + name + ".parquet"))
        side = m["effective_group_lookup"][sh["path"]]
        prev = old_by_group[side["path"]]
        for receipt in [sh, side, prev]:
            assert sha((ROOT / receipt["path"]).read_bytes()) == receipt["sha256"]
        rows = pq.read_table(ROOT / sh["path"], use_threads=False).to_pylist()
        originals = pq.read_table(ROOT / prev["path"], use_threads=False).to_pylist()
        groups = pq.read_table(ROOT / side["path"], use_threads=False).to_pylist()
        for i in [rowindex, rowindex + 1]:
            row, original = rows[i], originals[i]
            task = json.loads(row["native_payload"])
            ref = task["audit"]["source_audit"]["source_native_record"]
            mod = task["conditions"]["assay_modality"]
            sr = next(r for r in m["source_components"] if ("-CP/" in r["path"]) == (mod == "CellPainting"))
            source = json.loads((ROOT / sr["path"]).read_bytes())
            axis = json.loads((ROOT / m["source_full_axes"][mod]["axis_file"]["path"]).read_bytes())
            indexref = next(r for r in source["row_indexes"] if r["source_row_start"] <= ref["row_index"] < r["source_row_stop"])
            assert sha(Path(indexref["path"]).read_bytes()) == indexref["sha256"]
            native = pq.read_table(indexref["path"], use_threads=False).slice(ref["row_index"]-indexref["source_row_start"], 1).to_pylist()[0]
            cr = audit["candidate_rows"]
            assert sha((ROOT / cr["path"]).read_bytes()) == cr["sha256"]
            candidate, = pq.read_table(ROOT / cr["path"], filters=[("source_occurrence_key", "=", row["record_id"])], use_threads=False).to_pylist()
            binding = dict(component_sha256=SHA, shard_sha256=sh["sha256"], groups_sha256=side["sha256"],
                original_shard_sha256=prev["sha256"], metadata_index_sha256=indexref["sha256"],
                candidates_sha256=cr["sha256"], source_axis_sha256=m["source_full_axes"][mod]["axis_file"]["sha256"],
                source_manifest_sha256=sr["sha256"], native_task_proof_sha256=m["source_full_native_independent_proof"]["sha256"],
                qualification_proof_sha256=m["source_coordinate_projection_proof"]["sha256"])
            result[(label, row["task_name"])] = [row, original, native, candidate,
                json.loads(groups[i]["effective_groups_json"]), binding, source, axis]
    return result


def project(mod, case):
    row, original, metadata, candidate, groups, binding, source, axis = copy.deepcopy(case)
    return mod.lincs_native_view(row, original, metadata, candidate, effective_groups=groups,
        corpus_binding=binding, source_manifest=source, source_axis=axis)


def task_array(task, direction):
    return task["targets" if direction == "molecule_to_profile" else "conditions"]["source_profile_native_array"]


def edit_task(case, function, which=0):
    d = json.loads(case[which]["native_payload"]); function(d)
    case[which]["native_payload"] = canonical(d); case[which]["native_payload_sha256"] = sha(case[which]["native_payload"])


@pytest.mark.parametrize("name", ["CP", "L1000", "CP_partial"])
@pytest.mark.parametrize("direction", ["molecule_to_profile", "profile_to_molecule"])
def test_real_complete_axes_types_masks_and_role_allowlists(cases, name, direction):
    mod = api(); view = project(mod, cases[(name, direction)])
    d = json.loads(cases[(name, direction)][0]["native_payload"]); source = task_array(d, direction)
    n = 978 if name == "L1000" else 1783
    array = view.native_profile()
    assert len(view.axis) == n and array.as_mapping() == source
    assert canonical(array.as_mapping()) == canonical(source)
    assert tuple(array.observed_mask) == tuple(v is not None for v in array.values)
    assert sum(t == "int64" for t in array.native_types) == (0 if name == "L1000" else 21)
    for v, t in zip(array.values, array.native_types):
        if v is not None:
            assert type(v) is (int if t == "int64" else float)
    if name == "CP_partial":
        assert sum(v is None for v in array.values) == 36
    dose = view.inputs()["source_export_dose_coordinate"]
    assert dose["unit"] is None and dose["value"] == 10.0
    assert dose["physical_unit_confirmed"] is dose["cross_assay_conversion_allowed"] is False
    assert view.inputs()["source_exposure_time"]["value"] == (24 if name == "L1000" else 48)
    common = {"assay_modality", "cell_line", "source_export_dose_coordinate", "source_exposure_time"}
    expected = common | ({"canonical_isomeric_smiles", "known_profile_axis", "known_native_feature_types"}
                         if direction == "molecule_to_profile" else {"source_profile_native_array"})
    assert set(view.inputs()) == expected
    assert set(view.targets()) == ({"source_profile_native_array"} if direction == "molecule_to_profile" else {"canonical_isomeric_smiles"})
    p = view.provenance()
    assert p["global_master_inclusion"] is False and p["model_token_counts"] is None
    assert p["raw_source_reread"] is False and p["heldout_eligible"] is False


@pytest.mark.parametrize("direction", ["molecule_to_profile", "profile_to_molecule"])
def test_hidden_target_audit_and_missingness_mutations_cannot_change_projected_inputs(cases, direction):
    mod = api(); case = copy.deepcopy(cases[("CP_partial", direction)])
    task = json.loads(case[0]["native_payload"])
    before = mod.lincs_input_bytes(task, modality="CellPainting", direction=direction, source_axis=case[7])
    task["targets"] = {"hidden_molecule_profile_nullmask_hash_and_length": "mutated"}
    task["audit"] = {"BRD": "invented", "plate": "secret", "well": "A99", "source": "other", "unit": "M", "name": "other"}
    task["missing"] = ["changed"]
    assert mod.lincs_input_bytes(task, modality="CellPainting", direction=direction, source_axis=case[7]) == before
    edit_task(case, lambda d: d.update(targets=task["targets"], audit=task["audit"], missing=task["missing"]))
    with pytest.raises(ValueError):
        project(mod, case)


@pytest.mark.parametrize("key,value", [
    ("target_length", 1783), ("target_hash", "0"*64), ("observed_mask", [True]),
    ("source_ID", "x"), ("plate", "A"), ("well", "A01"), ("BRD", "id"),
    ("compound_name", "label"), ("audit", {}), ("source_nominal_dose", {"value":10,"unit":"micromolar"}),
])
def test_predictor_condition_injection_rejected(cases, key, value):
    mod = api(); case = copy.deepcopy(cases[("CP", "molecule_to_profile")])
    edit_task(case, lambda d: d["conditions"].__setitem__(key, value))
    with pytest.raises(ValueError):
        project(mod, case)
    with pytest.raises(ValueError):
        mod.lincs_input_bytes(json.loads(case[0]["native_payload"]), modality="CellPainting", direction="molecule_to_profile", source_axis=case[7])


@pytest.mark.parametrize("key,value", [
    ("unit", "micromolar"), ("physical_unit_confirmed", True), ("physical_unit_confirmed", 0),
    ("cross_assay_conversion_allowed", True), ("cross_assay_conversion_allowed", 0),
    ("value", 0), ("value", True), ("value", 10), ("scope", "converted concentration")
])
def test_dose_coordinate_values_types_units_and_conversion_flags_are_exact(cases, key, value):
    mod = api(); case = copy.deepcopy(cases[("CP", "molecule_to_profile")])
    edit_task(case, lambda d: d["conditions"]["source_export_dose_coordinate"].__setitem__(key, value))
    with pytest.raises(ValueError):
        project(mod, case)


@pytest.mark.parametrize("change", ["float_int64", "bool_int64", "int_double", "nan", "infinity", "mask_number", "mask_flip", "null_fill", "axis_reorder", "type_reorder", "axis_crop", "values_crop", "extra_key"])
@pytest.mark.parametrize("direction", ["molecule_to_profile", "profile_to_molecule"])
def test_native_array_mutations_rejected_without_cast_or_imputation(cases, change, direction):
    mod = api(); case = copy.deepcopy(cases[("CP_partial", direction)])
    task = json.loads(case[0]["native_payload"]); arr = task_array(task, direction)
    integer = next(i for i,t in enumerate(arr["native_types"]) if t=="int64" and arr["values"][i] is not None)
    null = arr["values"].index(None)
    floating = next(i for i,t in enumerate(arr["native_types"]) if t=="double" and arr["values"][i] is not None)
    if change == "float_int64": arr["values"][integer] = float(arr["values"][integer])
    elif change == "bool_int64": arr["values"][integer] = True
    elif change == "int_double": arr["values"][floating] = 1
    elif change == "nan": arr["values"][floating] = float("nan")
    elif change == "infinity": arr["values"][floating] = float("inf")
    elif change == "mask_number": arr["observed_mask"][floating] = 1
    elif change == "mask_flip": arr["observed_mask"][floating] = False
    elif change == "null_fill": arr["values"][null] = 0.0
    elif change == "axis_reorder": arr["axis_ids"][:2] = arr["axis_ids"][1::-1]
    elif change == "type_reorder": arr["native_types"][integer],arr["native_types"][floating] = arr["native_types"][floating],arr["native_types"][integer]
    elif change == "axis_crop": arr["axis_ids"].pop()
    elif change == "values_crop": arr["values"].pop()
    else: arr["target_hash"] = "0"*64
    # Nonfinite JSON is retained as a deliberate invalid fixture.
    case[0]["native_payload"] = json.dumps(task, sort_keys=True, separators=(",", ":")).encode()
    case[0]["native_payload_sha256"] = sha(case[0]["native_payload"])
    with pytest.raises(ValueError):
        project(mod, case)


@pytest.mark.parametrize("where,key,value", [
    ("row", "split", "test"), ("row", "source_family", "Rosetta-CDRP-molecule-profile-CellPainting"),
    ("row", "admitted_for_task", 1), ("row", "model_adapter_supported", 0),
    ("row", "task_id", "0"*64), ("row", "record_id", "invented"), ("row", "full_axis_values", 1782),
    ("row", "source_revision", "other"), ("row", "source_sha256", "0"*64),
    ("metadata", "source_row", False), ("metadata", "profile_null_count", True),
    ("metadata", "Metadata_pert_dose_micromolar", 1.0), ("metadata", "Metadata_Well", "A99"),
    ("candidate", "chemical_dose_time_cell_task_candidate", 1), ("candidate", "task_admitted", 0),
    ("candidate", "source_feature_nulls", False), ("candidate", "source_cell_line", "other"),
    ("candidate", "canonical_isomeric_source_reference_graph", "CC"),
    ("candidate", "existing_heldout_chemical_groups_json", '["heldout"]'),
])
def test_envelope_native_metadata_and_candidate_binding_mutations(cases, where, key, value):
    mod = api(); case = copy.deepcopy(cases[("CP", "molecule_to_profile")])
    case[{"row":0,"metadata":2,"candidate":3}[where]][key] = value
    with pytest.raises(ValueError):
        project(mod, case)


@pytest.mark.parametrize("field,key,value", [
    ("admission_receipt_json","source_native_null_mask_preserved",1),
    ("admission_receipt_json","physical_dose_task_admitted",0),
    ("admission_receipt_json","source_export_coordinate_task_admitted",1),
    ("admission_receipt_json","source_original_native_payload_sha256","0"*64),
    ("condition_context_json","source_generated",0),
    ("condition_context_json","physical_dose_condition_valid",0),
    ("condition_context_json","source_export_claimed_unit","M"),
])
def test_exact_receipt_context_boolean_types(cases, field, key, value):
    mod = api(); case = copy.deepcopy(cases[("CP", "molecule_to_profile")])
    d=json.loads(case[0][field]);d[key]=value;case[0][field]=json.dumps(d)
    with pytest.raises(ValueError):
        project(mod, case)


@pytest.mark.parametrize("change", ["group_missing","group_duplicate","binding","source_axis","source_manifest","old_v1","original_target","audit_numeric"])
def test_group_proof_original_v1_and_audit_guards(cases, change):
    mod=api();c=copy.deepcopy(cases[("CP","molecule_to_profile")])
    if change=="group_missing":c[4].pop()
    elif change=="group_duplicate":c[4].append(c[4][0])
    elif change=="binding":c[5]["component_sha256"]="0"*64
    elif change=="source_axis":c[7]["ordered_feature_names"][:2]=c[7]["ordered_feature_names"][1::-1]
    elif change=="source_manifest":c[6]["original_values_and_float_bits_preserved"]=1
    elif change=="old_v1":c[0]=copy.deepcopy(c[1])
    elif change=="original_target":
        edit_task(c,lambda d: task_array(d,"molecule_to_profile")["values"].__setitem__(0,9.0),which=1)
    else:
        edit_task(c,lambda d:d["audit"].__setitem__("native_int64_and_float64_preserved",1))
    with pytest.raises(ValueError):
        project(mod,c)


def test_synthetic_large_int64_and_signed_zero_are_lossless(cases):
    mod=api();d=json.loads(cases[("CP","molecule_to_profile")][0]["native_payload"])
    a=task_array(d,"molecule_to_profile")
    i=a["native_types"].index("int64")
    a["values"][i]=2**60+1;a["values"][0]=-0.0
    native=mod.NativeProfileArray.from_mapping(a,source_axis=cases[("CP","molecule_to_profile")][7])
    b=json.loads(canonical(native.as_mapping()))
    assert type(b["values"][i]) is int and b["values"][i]==2**60+1
    assert struct.pack("<d",b["values"][0])==struct.pack("<d",-0.0)


def test_collation_preserves_native_types_masks_and_rejects_duplicate_or_mixed_views(cases):
    mod=api();a=project(mod,cases[("CP","molecule_to_profile")]);b=project(mod,cases[("CP_partial","molecule_to_profile")])
    batch=mod.collate_lincs_native_views([a,b])
    assert batch.axis==a.axis and len(batch.profiles)==2
    assert batch.profiles[1].values.count(None)==36 and batch.inputs()==(a.inputs(),b.inputs())
    assert all(type(v) is int for p in batch.profiles for v,t in zip(p.values,p.native_types) if t=="int64" and v is not None)
    with pytest.raises(ValueError):mod.collate_lincs_native_views([a,a])
    with pytest.raises(ValueError):mod.collate_lincs_native_views([a,project(mod,cases[("L1000","molecule_to_profile")])])
    with pytest.raises(ValueError):mod.collate_lincs_native_views([a,project(mod,cases[("CP","profile_to_molecule")])])


@pytest.mark.parametrize("name,shard", [("CP","00000"),("L1000","00863"),("CP_partial","00685")])
def test_file_reader_bounded_real_views_and_inherited_raw_proof(cases,monkeypatch,name,shard):
    mod=api();old=Path.open
    def guarded(path,*a,**kw):
        assert not ("/raw/LINCS-Pilot1-" in str(path)), "whole source read forbidden"
        return old(path,*a,**kw)
    monkeypatch.setattr(Path,"open",guarded)
    ids=[cases[(name,d)][0]["task_id"] for d in ["molecule_to_profile","profile_to_molecule"]]
    views=mod.read_lincs_native_views(COMPONENT,root=ROOT,expected_manifest_sha256=SHA,
        shard_paths=[str(BASE/"source-dose-coordinate-qualified-v2/shards"/(shard+".parquet"))],task_ids=ids,limit=2)
    assert len(views)==2
    assert all(v.provenance()["consumed_file_bytes_verified"] is True for v in views)
    assert all(v.provenance()["global_master_inclusion"] is False for v in views)


@pytest.mark.parametrize("change",["old_manifest","bad_hash","duplicate_shard","foreign_shard","duplicate_task","missing_task","bool_limit"])
def test_file_selection_scope_and_duplicate_allocation(cases,change):
    mod=api();kw=dict(root=ROOT,expected_manifest_sha256=SHA,shard_paths=[str(BASE/"source-dose-coordinate-qualified-v2/shards/00000.parquet")],limit=1)
    path=COMPONENT
    if change=="old_manifest":path=BASE/"curated-profile-tasks-v1/manifest.json"
    elif change=="bad_hash":kw["expected_manifest_sha256"]="0"*64
    elif change=="duplicate_shard":kw["shard_paths"]*=2
    elif change=="foreign_shard":kw["shard_paths"]=["outputs/foreign.parquet"]
    elif change=="duplicate_task":kw["task_ids"]=[cases[("CP","molecule_to_profile")][0]["task_id"]]*2
    elif change=="missing_task":kw["task_ids"]=["0"*64]
    else:kw["limit"]=True
    with pytest.raises(ValueError):mod.read_lincs_native_views(path,**kw)


@pytest.mark.parametrize("suffix", ["shards/00000.parquet","groups/00000.parquet","row-index-00000.parquet",
    "independent-native-task-proof-v1/receipt.json","source-dose-coordinate-qualified-v2/independent-projection-receipt.json",
    "source-metadata-chemical-context-admission-candidates.parquet","DSE-LINCS-Pilot1-CP/full-axis-and-schema.json"])
def test_file_reader_rejects_consumed_native_proof_axis_or_payload_byte_tampering(cases,monkeypatch,suffix):
    mod=api();old=Path.read_bytes
    def mutated(path):
        raw=old(path)
        return bytes([raw[0]^1])+raw[1:] if str(path).endswith(suffix) else raw
    monkeypatch.setattr(Path,"read_bytes",mutated)
    with pytest.raises(ValueError):
        mod.read_lincs_native_views(COMPONENT,root=ROOT,expected_manifest_sha256=SHA,
            shard_paths=[str(BASE/"source-dose-coordinate-qualified-v2/shards/00000.parquet")],limit=1)


def test_direct_projection_binds_exact_native_metadata_index_receipt(cases):
    mod=api();c=copy.deepcopy(cases[("CP","molecule_to_profile")])
    c[5]["metadata_index_sha256"]="0"*64
    with pytest.raises(ValueError,match="index"):
        project(mod,c)


def test_native_array_direct_constructor_cannot_bypass_exact_types_and_null_mask(cases):
    mod=api();d=json.loads(cases[("CP_partial","molecule_to_profile")][0]["native_payload"])
    a=task_array(d,"molecule_to_profile")
    values=list(a["values"]);i=a["native_types"].index("int64");values[i]=True
    with pytest.raises(ValueError):
        mod.NativeProfileArray(tuple(a["axis_ids"]),tuple(a["native_types"]),tuple(values),
            tuple(a["observed_mask"]),a["feature_units"])
    mask=list(a["observed_mask"]);mask[0]=1
    with pytest.raises(ValueError):
        mod.NativeProfileArray(tuple(a["axis_ids"]),tuple(a["native_types"]),tuple(a["values"]),
            tuple(mask),a["feature_units"])
