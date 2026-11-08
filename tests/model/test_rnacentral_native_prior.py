"""RNAcentral pilot reader; synthetic cases never establish source coverage."""
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
BASE = Path("outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1")
MASTER = Path("outputs/data-intake/20261101-PubChem-corpus-pilot-v1/independent-master-union-v20/master-corpus-manifest-v20-PubChem-native-codec-pilot-v1.json")
SHA = "fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f"
INACTIVE = "5ccd3a5aba8003339f1b88f006826d491343117940c96577c1690487529f318d"


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(x):
    return hashlib.sha256(x).hexdigest()


def api():
    p = os.environ.get("MORA3_RNACENTRAL_PRIOR_MODULE")
    if p:
        if not Path(p).is_file():
            pytest.fail("RNAcentral native prior reader is not implemented")
        spec = importlib.util.spec_from_file_location("mora3.data.rnacentral_prior", p)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    try:
        return importlib.import_module("mora3.data.rnacentral_prior")
    except ModuleNotFoundError:
        pytest.fail("RNAcentral native prior reader is not implemented")


@pytest.fixture(scope="module")
def real_cases():
    if not (ROOT / MASTER).is_file():
        pytest.skip("admitted RNAcentral pilot is unavailable")
    import pyarrow.parquet as pq
    master = json.loads((ROOT / MASTER).read_bytes())
    assert digest((ROOT / MASTER).read_bytes()) == SHA
    component = json.loads((ROOT / BASE / "manifest.json").read_bytes())
    plan = json.loads((ROOT / BASE / "build-plan.json").read_bytes())
    cases = {}
    for name, selector in [("active", ("00000", 0)), ("ambiguity", ("00000", 12)), ("inactive", ("00105", INACTIVE))]:
        shname, sel = selector
        path = str(BASE / "shards/train" / (shname + ".parquet"))
        sh = next(s for s in master["shards"] if s["path"] == path)
        side = master["effective_group_lookup"][path]
        assert digest((ROOT / path).read_bytes()) == sh["sha256"]
        assert digest((ROOT / side["path"]).read_bytes()) == side["sha256"]
        rows = pq.read_table(ROOT / path, use_threads=False).to_pylist()
        i = sel if type(sel) is int else next(i for i, r in enumerate(rows) if r["task_id"] == sel)
        row = rows[i]
        groups = json.loads(pq.read_table(ROOT / side["path"], use_threads=False).slice(i, 1).to_pylist()[0]["effective_groups_json"])
        ref = json.loads(row["native_payload"])["audit"]["source_ref"]
        ordinal = next(i for i, p in enumerate(plan["selected"]) if p["native_payload"]["sha256"] == ref["native_payload_sha256"])
        srcpath = Path(plan["selected"][ordinal]["source_manifest"]["path"])
        source = json.loads((ROOT / srcpath).read_bytes())
        proofpath = srcpath.parent.parent / ("rnacentral28-" + ref["source_status"] + "-independent-replay-v1/manifest.json")
        proof = json.loads((ROOT / proofpath).read_bytes())
        raw = (ROOT / ref["native_payload"]).read_bytes()
        assert digest(raw) == ref["native_payload_sha256"]
        native = pq.read_table(ROOT / ref["native_payload"], use_threads=False).slice(ref["native_row_index"], 1).to_pylist()[0]
        occref = component["occurrences"][ordinal]
        assert digest((ROOT / occref["path"]).read_bytes()) == occref["sha256"]
        occ = pq.read_table(ROOT / occref["path"], filters=[("task_id", "=", row["task_id"])], use_threads=False).to_pylist()
        assert len(occ) == 1
        binding = dict(manifest_sha256=SHA, shard_sha256=sh["sha256"], groups_sha256=side["sha256"], occurrence_sha256=occref["sha256"])
        cases[name] = (row, native, occ[0], groups, binding, source, proof)
    return cases


def project(mod, case):
    row, native, occ, groups, binding, source, proof = copy.deepcopy(case)
    return mod.rnacentral_prior_view(row, native, occ, effective_groups=groups, corpus_binding=binding,
                                    source_manifest=source, source_proof=proof)


def rehash_task(case, edit):
    row = case[0]
    task = json.loads(row["native_payload"])
    edit(task)
    row["native_payload"] = canonical(task)
    row["native_payload_sha256"] = digest(row["native_payload"])


@pytest.mark.parametrize("name", ["active", "inactive", "ambiguity"])
def test_real_full_sequence_and_masks_without_header_status_or_target_conditions(real_cases, name):
    mod = api()
    view = project(mod, real_cases[name])
    original = real_cases[name][1]["native_sequence"]
    assert view.sequence == original
    assert view.inputs() == {"prior_domain": "RNA_reference_native_symbols", "source_organism": None}
    assert set(view.inputs()) == {"prior_domain", "source_organism"}
    ids, observed, known = view.target_tensors()
    assert len(ids) == len(original) == int(observed.sum())
    assert view.decode(ids) == original
    assert known.tolist() == [c.upper() in "ACGTU" for c in original]
    tok = ByteTokenizer((b"ACGT", b"NN"))
    assert tok.decode(tok.encode(original.encode())) == original.encode()
    p = view.provenance()
    assert p["heldout_eligible"] is False
    assert p["neural_model_token_count"] is None
    assert p["source_record_status_is_not_activity_label"] is True
    assert p["raw_FASTA_reread"] is False
    assert "native_sequence" not in p and "native_payload" not in p
    assert p["source_ref"]["source_status"] == ("inactive" if name == "inactive" else "active")


@pytest.mark.parametrize("field,value", [
    ("source_family", "Rfam15.1-native-reference-prior"), ("source_revision", "28"), ("task_name", "classification"),
    ("split", "test"), ("split_policy", "random"), ("evidence_class", "experimental"),
    ("admitted_for_task", 1), ("admitted_for_task", False), ("model_adapter_supported", 0),
    ("model_adapter_supported", True), ("full_axis_values", True), ("full_axis_values", 199),
    ("record_id", "invented"), ("source_sha256", "0" * 64), ("native_schema", "mora3.Other/v1"),
    ("task_id", "0" * 64), ("native_payload_sha256", "0" * 64)
])
def test_envelope_mutations_rejected(real_cases, field, value):
    mod = api()
    c = copy.deepcopy(real_cases["active"])
    c[0][field] = value
    with pytest.raises(ValueError):
        project(mod, c)


@pytest.mark.parametrize("where,key,value", [
    ("conditions", "source_status", "active"), ("conditions", "source_identifier", "URS0000000001"),
    ("conditions", "source_native_sequence", "ACGT"), ("conditions", "target_length", 200),
    ("conditions", "source_organism", "invented"), ("conditions", "prior_domain", "RNA_family_reference_native_symbols"),
    ("targets", "family", "RF00001"), ("targets", "source_native_sequence", "N"),
    ("audit", "native_reader_supported", 0), ("audit", "source_record_status_is_not_activity_label", 1),
    ("audit", "source_symbols", "uppercased"), ("audit", "extra", False)
])
def test_rehashed_task_leakage_and_type_aliases_rejected(real_cases, where, key, value):
    mod = api()
    c = copy.deepcopy(real_cases["active"])
    rehash_task(c, lambda d: d[where].__setitem__(key, value))
    with pytest.raises(ValueError):
        project(mod, c)


@pytest.mark.parametrize("field,value", [
    ("native_sequence", "A"), ("native_length", 199), ("source_record_index", False),
    ("sequence_alphabet_pass", 1), ("task_admitted", 0), ("native_T_count", True),
    ("native_U_count", False), ("ambiguity_symbol_count", False), ("explicit_Rfam_ID_in_header", "RF00001"),
    ("native_header", "URS0000000001 invented"), ("source_identifier", "URS9999999999"),
    ("source_payload", "/invented.fasta.gz"), ("uppercase_sequence_sha256", "0" * 64),
    ("source_role", "experiment")
])
def test_native_record_tampering_and_numeric_boolean_aliases_rejected(real_cases, field, value):
    mod = api()
    c = copy.deepcopy(real_cases["active"])
    c[1][field] = value
    with pytest.raises(ValueError):
        project(mod, c)


@pytest.mark.parametrize("field,value", [
    ("task_already_in_Rfam_corpus", True), ("task_already_in_Rfam_corpus", 0),
    ("first_occurrence_in_pilot", False), ("first_occurrence_in_pilot", 1),
    ("derived_RC_is_molecular_identity", True), ("derived_RC_is_molecular_identity", 0),
    ("source_status_is_not_negative_label", 1), ("native_length", True),
    ("task_id", "0" * 64), ("native_sequence_sha256", "0" * 64),
    ("derived_RC_baseorder_sha256", "0" * 64)
])
def test_occurrence_reuse_identity_type_and_allocation_mutations_rejected(real_cases, field, value):
    mod = api()
    c = copy.deepcopy(real_cases["active"])
    c[2][field] = value
    with pytest.raises(ValueError):
        project(mod, c)


@pytest.mark.parametrize("section,field,value", [
    ("admission_receipt_json", "exactfullaxis", 1), ("admission_receipt_json", "source_case_T_U_preserved", 1),
    ("admission_receipt_json", "source_license", "AGPL"), ("admission_receipt_json", "extra", 0),
    ("condition_context_json", "source_status_notactivitytruth", 1),
    ("condition_context_json", "sourceIDsheadersauditonly", 1),
    ("condition_context_json", "all_unlisted_annotations_unknown", 1),
    ("condition_context_json", "target_length", 200)
])
def test_receipt_context_exact_structural_types_rejected(real_cases, section, field, value):
    mod = api()
    c = copy.deepcopy(real_cases["active"])
    d = json.loads(c[0][section]); d[field] = value; c[0][section] = json.dumps(d)
    with pytest.raises(ValueError):
        project(mod, c)


@pytest.mark.parametrize("mutation", ["source_row_bool", "source_status", "release", "header", "missing", "group_missing", "group_duplicate", "group_foreign", "binding", "proof", "source_manifest"])
def test_source_ref_missingness_group_and_proof_mutations_rejected(real_cases, mutation):
    mod = api()
    c = copy.deepcopy(real_cases["active"])
    if mutation in ("source_row_bool", "source_status", "release", "header"):
        field, value = {"source_row_bool": ("native_row_index", False), "source_status": ("source_status", "inactive"),
                        "release": ("source_release", "RNAcentral27"), "header": ("native_header", "URS1")} [mutation]
        rehash_task(c, lambda d: d["audit"]["source_ref"].__setitem__(field, value))
    elif mutation == "missing":
        rehash_task(c, lambda d: d["missing"].pop())
    elif mutation == "group_missing":
        c[3].pop()
    elif mutation == "group_duplicate":
        c[3].append(c[3][0])
    elif mutation == "group_foreign":
        c[0]["group_ids_json"] = json.dumps(["nucleotide_native:" + "0" * 64])
    elif mutation == "binding":
        c[4]["manifest_sha256"] = "0" * 64
    elif mutation == "proof":
        c[6]["all_typed_columns_decoded"] = 1
    else:
        c[5]["no_T_U_conversion"] = 1
    with pytest.raises(ValueError):
        project(mod, c)


def synthetic_sequence_case(case, sequence):
    """Consistent in-memory source fixture only; cannot alter pinned actual file bytes."""
    c = copy.deepcopy(case)
    native = c[1]
    native["native_sequence"] = sequence
    native["native_length"] = len(sequence)
    native["native_sequence_sha256"] = digest(sequence.encode())
    native["uppercase_sequence_sha256"] = digest(sequence.upper().encode())
    native["native_T_count"] = sequence.upper().count("T")
    native["native_U_count"] = sequence.upper().count("U")
    native["ambiguity_symbol_count"] = sum(x.upper() not in "ACGTU" for x in sequence)
    task = json.loads(c[0]["native_payload"]); task["targets"]["source_native_sequence"] = sequence
    c[0]["native_payload"] = canonical(task); c[0]["native_payload_sha256"] = digest(c[0]["native_payload"])
    c[0]["full_axis_values"] = len(sequence)
    c[0]["task_id"] = digest(canonical(["RNA_reference_native_symbols", None, native["native_sequence_sha256"]]))
    receipt = json.loads(c[0]["admission_receipt_json"]); receipt["native_sequence_sha256"] = native["native_sequence_sha256"]
    c[0]["admission_receipt_json"] = json.dumps(receipt)
    base = sequence.upper().replace("U", "T")
    rc = base.translate(str.maketrans("ACGTRYSWKMBDHVNX", "TGCAYRSWMKVHDBNX"))[::-1]
    guard = digest(min(base, rc).encode())
    groups = ["nucleotide_native:" + native["native_sequence_sha256"], "nucleotide_upper:" + native["uppercase_sequence_sha256"],
              "nucleotide_derived_RC_baseorder:" + guard, "rnacentral_native_identifier:" + native["source_identifier"]]
    c[0]["group_ids_json"] = json.dumps(groups); c[3][:] = sorted(groups)
    c[2].update(task_id=c[0]["task_id"], native_sequence_sha256=native["native_sequence_sha256"],
                uppercase_sequence_sha256=native["uppercase_sequence_sha256"], derived_RC_baseorder_sha256=guard, native_length=len(sequence))
    return c


def test_synthetic_case_and_T_U_retained_with_separate_native_and_derived_groups(real_cases):
    mod = api()
    a = project(mod, synthetic_sequence_case(real_cases["active"], "acgTuRynN"))
    b = project(mod, synthetic_sequence_case(real_cases["active"], "ACGTTR YNN".replace(" ", "")))
    assert a.sequence == "acgTuRynN"
    assert a.provenance()["sequence_sha256"] != b.provenance()["sequence_sha256"]
    assert a.inputs() == b.inputs()
    ids, observed, known = a.target_tensors()
    assert a.decode(ids) == "acgTuRynN" and observed.all() and int(known.sum()) == 5
    ga = a.provenance()["effective_groups"]; gb = b.provenance()["effective_groups"]
    assert {g for g in ga if g.startswith("nucleotide_derived_RC_")} == {g for g in gb if g.startswith("nucleotide_derived_RC_")}


@pytest.mark.parametrize("seq", ["", "ACGX", "ACG-", "A C", "A1C", "AΨC"])
def test_synthetic_nonadmitted_alphabet_is_not_repaired(real_cases, seq):
    mod = api()
    with pytest.raises(ValueError):
        project(mod, synthetic_sequence_case(real_cases["active"], seq))


def test_file_reader_real_bound_selection_and_raw_FASTA_is_never_opened(real_cases, monkeypatch):
    mod = api()
    old = Path.open
    def guarded(path, *a, **kw):
        assert not str(path).endswith(".fasta.gz"), "raw source must use inherited proof"
        return old(path, *a, **kw)
    monkeypatch.setattr(Path, "open", guarded)
    rows = mod.read_rnacentral_priors(MASTER, root=ROOT, expected_manifest_sha256=SHA,
        shard_paths=[str(BASE / "shards/train/00000.parquet"), str(BASE / "shards/train/00105.parquet")],
        task_ids=[real_cases["active"][0]["task_id"], INACTIVE], limit=2)
    assert len(rows) == 2 and {v.provenance()["source_ref"]["source_status"] for v in rows} == {"active", "inactive"}
    assert all(v.provenance()["native_file_bytes_verified"] is True for v in rows)
    assert all(v.provenance()["source_release"] == "RNAcentral28.0" for v in rows)


@pytest.mark.parametrize("mutation", ["manifest_sha", "selector_duplicate", "selector_foreign", "task_duplicate", "missing_task", "bool_limit", "wrong_status"])
def test_file_reader_rejects_bad_selectors_and_duplicate_allocation(real_cases, mutation):
    mod = api()
    kw = dict(root=ROOT, expected_manifest_sha256=SHA, shard_paths=[str(BASE / "shards/train/00000.parquet")], limit=1)
    if mutation == "manifest_sha": kw["expected_manifest_sha256"] = "0" * 64
    elif mutation == "selector_duplicate": kw["shard_paths"] *= 2
    elif mutation == "selector_foreign": kw["shard_paths"] = ["outputs/no.parquet"]
    elif mutation == "task_duplicate": kw["task_ids"] = [real_cases["active"][0]["task_id"]] * 2
    elif mutation == "missing_task": kw["task_ids"] = ["0" * 64]
    elif mutation == "bool_limit": kw["limit"] = True
    else: kw["source_status"] = "positive"
    with pytest.raises(ValueError):
        mod.read_rnacentral_priors(MASTER, **kw)


@pytest.mark.parametrize("target", ["native", "proof", "occurrence", "group", "source_receipt"])
def test_file_reader_rejects_tampered_consumed_bytes(real_cases, monkeypatch, target):
    mod = api()
    c = real_cases["active"]
    paths = {"native": c[0]["source_sha256"], "proof": "active-independent-replay-v1/manifest.json",
             "occurrence": "occurrences/000.parquet", "group": "effective-groups/00000.parquet",
             "source_receipt": "asset-receipts-rnacentral28-recovery-v2/DSE3-RNACENTRAL28-active.json"}
    ref = json.loads(c[0]["native_payload"])["audit"]["source_ref"]
    original = Path.read_bytes
    def mutated(path):
        raw = original(path)
        hit = (str(path).endswith(ref["native_payload"]) if target == "native" else str(path).endswith(paths[target]))
        return bytes([raw[0] ^ 1]) + raw[1:] if hit else raw
    monkeypatch.setattr(Path, "read_bytes", mutated)
    with pytest.raises(ValueError):
        mod.read_rnacentral_priors(MASTER, root=ROOT, expected_manifest_sha256=SHA,
            shard_paths=[str(BASE / "shards/train/00000.parquet")], limit=1)


def test_direct_projection_rejects_unadmitted_full_source_QC_payload(real_cases):
    """Whole-source availability is not pilot admission, even with consistent metadata."""
    mod = api()
    c = copy.deepcopy(real_cases["active"])
    source = c[5]
    chosen = source["native_payloads"][1]
    assert chosen["sha256"] != c[0]["source_sha256"]
    task = json.loads(c[0]["native_payload"])
    ref = task["audit"]["source_ref"]
    old = ref["native_payload"]
    ref["native_payload"] = str(Path(old).parent / Path(chosen["path"]).name)
    ref["native_payload_sha256"] = chosen["sha256"]
    ref["native_row_index"] = 0
    ref["original_record_index"] = source["native_payloads"][0]["rows"]
    record = "official_RNAcentral28_active:DSE3-RNACENTRAL28-active:" + str(ref["original_record_index"])
    c[0]["native_payload"] = canonical(task); c[0]["native_payload_sha256"] = digest(c[0]["native_payload"])
    c[0]["record_id"] = record; c[0]["source_sha256"] = chosen["sha256"]
    c[1]["source_record_id"] = record; c[1]["source_record_index"] = ref["original_record_index"]
    receipt = json.loads(c[0]["admission_receipt_json"]); receipt["source_ref"] = ref
    c[0]["admission_receipt_json"] = json.dumps(receipt)
    c[2]["source_ref_json"] = json.dumps(ref)
    with pytest.raises(ValueError, match="pilot|occurrence"):
        project(mod, c)


def test_direct_projection_requires_exact_source_occurrence_file_binding(real_cases):
    mod = api()
    c = copy.deepcopy(real_cases["active"])
    c[4]["occurrence_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="occurrence"):
        project(mod, c)


@pytest.mark.parametrize("shard,task,length,status", [
    ("00073", "c82c7718f57411d417e7b4503359186179280a5cdddddc2355057b530fbe964f", 991425, "active"),
    ("00145", "9191df91fa6e3659928c865e8fab1e091f21d4299c51fac38fd194e1830eda34", 900074, "inactive"),
])
def test_real_complete_long_native_axes_and_byte_roundtrip(shard, task, length, status):
    if not (ROOT / MASTER).is_file():
        pytest.skip("admitted long-axis source unavailable")
    mod = api()
    view, = mod.read_rnacentral_priors(MASTER, root=ROOT, expected_manifest_sha256=SHA,
        shard_paths=[str(BASE / "shards/train" / (shard + ".parquet"))], task_ids=[task], source_status=status, limit=1)
    assert len(view.sequence) == length and digest(view.sequence.encode()) == view.provenance()["sequence_sha256"]
    ids, observed, known = view.target_tensors()
    assert view.decode(ids) == view.sequence and len(observed) == length and observed.all()
    tok = ByteTokenizer()
    assert tok.decode(tok.encode(view.sequence.encode())) == view.sequence.encode()
    assert view.provenance()["neural_model_token_count"] is None


def test_real_reused_Rfam_and_quarantines_are_outside_new_admission(real_cases):
    mod = api()
    import pyarrow.parquet as pq
    component = json.loads((ROOT / BASE / "manifest.json").read_bytes())
    counts = component["counts"]
    assert counts["source_rows"] == counts["distinct_new_task_views"] + counts["distinct_reused_existing_Rfam_tasks"] + counts["quarantined_occurrences"]
    assert counts["distinct_new_task_views"] == 445486 and counts["distinct_reused_existing_Rfam_tasks"] == 661
    q = component["quarantines"][0]
    assert digest((ROOT / q["path"]).read_bytes()) == q["sha256"]
    assert pq.read_table(ROOT / q["path"], use_threads=False).num_rows == q["rows"] == 2
    occ = component["occurrences"][0]
    reused = pq.read_table(ROOT / occ["path"], filters=[("task_already_in_Rfam_corpus", "=", True)], use_threads=False)
    assert reused.num_rows > 0
    old = reused.slice(0, 1).to_pylist()[0]
    c = copy.deepcopy(real_cases["active"]); c[2].update(old)
    with pytest.raises(ValueError):
        project(mod, c)
    with pytest.raises(ValueError, match="absent"):
        mod.read_rnacentral_priors(MASTER, root=ROOT, expected_manifest_sha256=SHA,
            shard_paths=[str(BASE / "shards/train/00000.parquet")], task_ids=[old["task_id"]], limit=1)


def test_neural_RNAcentral_domain_capability_preserves_legacy_source_registry(real_cases):
    """Target-free context capability; source admission is tested by the neural bridge."""
    mod = api()
    import torch
    from mora3.data.nucleotide_prior import SOURCES
    from mora3.model.annotation_multihead import SharedAnnotationModel
    from mora3.model.annotation_vocabulary import AnnotationVocabulary
    path = os.environ.get("MORA3_RNACENTRAL_DECODER_MODULE")
    if path:
        spec = importlib.util.spec_from_file_location("mora3.model._rnacentral_capability_decoder", path)
        decoder = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = decoder
        spec.loader.exec_module(decoder)
        NucleotideDecoder = decoder.NucleotideDecoder
    else:
        from mora3.model.nucleotide_decoder import NucleotideDecoder
    view = project(mod, real_cases["active"])
    assert mod.SOURCE not in SOURCES
    assert view.inputs()["prior_domain"] not in {v[0] for v in SOURCES.values()}
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(12501)
        vocabulary = AnnotationVocabulary(json.dumps(dict(learned_hex=[], controls=["pad", "node", "edge"])))
        shared = SharedAnnotationModel(vocabulary, ("GO:0000001",), ("1.1.1.1",), width=4, heads=2, layers=2)
        model = NucleotideDecoder(shared).eval()
        condition = model.condition([view.inputs()])
        assert condition.shape == (1, 4) and torch.isfinite(condition).all()
