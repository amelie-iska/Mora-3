"""Lossless admitted PubChem SMILES priors and standardSELFIES codec views.

Consumes the pinned v20 pilot and verified v21/v22 whole PubChem admission. Source references, canonical graph
identity and effective chemical/scaffold guards are audit metadata, never model
conditions. This reader grants no neural, assay, held-out or full-corpus support.
"""
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path

from .uniprot_annotations import _canonical, _verified_bytes

PILOT_MASTER_SHA256 = "fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f"
SOURCE_SHA256 = "00d6cb2cd0239c8c03e5a43671dccc432d6b692043d35290abf494e69703c33e"
SOURCE_REF = dict(
    path="outputs/data-sources/20261031-legacy-mint-v1/curated/PubChem10m-standardSELFIES-full-v1/manifest.json",
    bytes=8361, sha256=SOURCE_SHA256)
REVISION = "f83219601635a0a80fc99c13a9ca37f99ef34f0a"
CONSTRAINTS_SHA256 = "58d7c8c645ab64e6a6aa1fdc8df842491f7f2d66414f4c77015305f8978a760e"
POLICY = "native_source_train_only;existingmasterheldout+globalnativevalidationexact+selectedvalidationcoarseguards_excluded;no_newheldouts"
MISSING = ["experimental_property_activity_or_response_labels",
    "physical_assay_dose_time_lot_and_cell_context",
    "3D_geometry_images_expression_or_segmentation", "independent_blind_chemical_evaluation"]
CONTEXT = dict(assay=None, physical_dose=None, time=None, image_available=False,
    expression_available=False, segmentation_available=False)
DIRECTIONS = {
    "reference_prior": ("PubChem-native-SMILES-reference-prior", "source_native_molecular_SMILES_reference_prior"),
    "smiles_to_selfies": ("PubChem-standardSELFIES-codec", "computed_standard_SELFIES_smiles_to_selfies"),
    "selfies_to_smiles": ("PubChem-standardSELFIES-codec", "computed_standard_SELFIES_selfies_to_smiles"),
}

WHOLE_BASE = "outputs/data-intake/20261101-PubChem-native-corpus-v1"
WHOLE_MASTER_SHA256 = "0e85a28928c027b7855e3bf12713a172128e12bed1e17752bee5f1d27b3f70b2"
CURRENT_MASTER_SHA256 = "82ef478d53ccbd2bb8a87607b1ce5dd35a6a0f58f6afa7d912aa214129c1e6f2"
SUPPORTED_MASTER_SHA256 = (PILOT_MASTER_SHA256, WHOLE_MASTER_SHA256, CURRENT_MASTER_SHA256)
WHOLE_COMPONENT_REF = dict(path=WHOLE_BASE + "/materialized/manifest.json", bytes=10635486,
    sha256="0303f6cc34ff3573335e9e8bc83197f94f24b33e1463657702ce5a7c0ff0b4a1")
WHOLE_CENSUS_REF = dict(path=WHOLE_BASE + "/census/manifest.json", bytes=2097230,
    sha256="ae4b120e6b68e2a213bb5bf59883253a8fc7841a30fcd4a4c1caf9758d64bdaf")
WHOLE_PROOF_REF = dict(path=WHOLE_BASE + "/materialized/independent-final-v1/receipt.json", bytes=5710,
    sha256="fe4b91c64bd3a5b3795bf897d1944e8b23422fe45cf0410a587f135a263f0623")
WHOLE_CENSUS_PROOF_REF = dict(path=WHOLE_BASE + "/materialized/census-independent-proof-v1/receipt.json", bytes=5103,
    sha256="287314202c492f4ba54b0079a40a342e3fa7ab66f0a142938d4b9249d9e4d025")
WHOLE_BINDING = dict(component_manifest_sha256=WHOLE_COMPONENT_REF["sha256"],
    construction_proof_sha256=WHOLE_PROOF_REF["sha256"],
    census_manifest_sha256=WHOLE_CENSUS_REF["sha256"],
    census_proof_sha256=WHOLE_CENSUS_PROOF_REF["sha256"])
WHOLE_POLICY = "native_source_trainonly;globalnativevalidationexact+v20heldoutwhole/componentchemistry+frozenpilotvalidationguards_excluded;pilotpreservedonce"
WHOLE_CONTEXT = dict(assay=None, dose=None, time=None, image_available=False,
    mask_available=False, expression_available=False, segmentation_available=False)


def _whole_metadata(root, master):
    """Inherit exact full construction proofs, without repeating full ETL/QC."""
    component = json.loads(_verified_bytes(root, WHOLE_COMPONENT_REF))
    census = json.loads(_verified_bytes(root, WHOLE_CENSUS_REF))
    proof = json.loads(_verified_bytes(root, WHOLE_PROOF_REF))
    raw_proof = json.loads(_verified_bytes(root, WHOLE_CENSUS_PROOF_REF))
    prior = json.loads(_verified_bytes(root, proof["prior_master"]))
    plan = json.loads(_verified_bytes(root, proof["plan"]))
    _verified_bytes(root, plan["source_full_proof_handoff"])
    if (proof["status"] != "verified" or raw_proof["status"] != "verified"
            or _canonical(proof["new_component"]) != _canonical(WHOLE_COMPONENT_REF)
            or _canonical(proof["census"]) != _canonical(WHOLE_CENSUS_REF)
            or _canonical(proof["independent_original_raw_and_census_proof"]) != _canonical(WHOLE_CENSUS_PROOF_REF)
            or proof["prior_master"]["sha256"] != PILOT_MASTER_SHA256
            or any(_canonical(obj["source"]) != _canonical(SOURCE_REF) for obj in (plan, raw_proof, component))
            or _canonical(component["census"]) != _canonical(WHOLE_CENSUS_REF)
            or _canonical(component["independent_census_proof"]) != _canonical(WHOLE_CENSUS_PROOF_REF)
            or proof["old_pilot81155task_IDs_and_bytes_carried_once"] is not True
            or type(proof["old_new_task_ID_intersections"]) is not int
            or proof["old_new_task_ID_intersections"] != 0):
        raise ValueError("whole native construction/source/proof binding mismatch")
    families = {x[0] for x in DIRECTIONS.values()}
    old = [s for s in prior["shards"] if s.get("source_family") in families]
    wanted = old + component["shards"]
    actual = [s for s in master["shards"] if s.get("source_family") in families]
    if (_canonical(actual) != _canonical(wanted)
            or len({s["path"] for s in wanted}) != len(wanted)):
        raise ValueError("whole/pilot shard allocation mismatch; pilot must occur once")
    expected_groups = {s["path"]: prior["effective_group_lookup"][s["path"]] for s in old}
    expected_groups.update(component["effective_group_lookup"])
    if _canonical({s["path"]: master["effective_group_lookup"][s["path"]] for s in actual}) != _canonical(expected_groups):
        raise ValueError("whole authoritative group sidecar version mismatch")
    return component, census


def _whole_component_guards(native, guards):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(native["native_smiles"])
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) > 1:
        for fragment in fragments:
            for atom in fragment.GetAtoms():
                atom.SetAtomMapNum(0)
            fragment = Chem.RemoveHs(fragment)
            Chem.RemoveStereochemistry(fragment)
            base = Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=False)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=fragment, includeChirality=False) or base
            guards.update(("molecule:" + _sha(base.encode()), "scaffold:" + _sha(scaffold.encode())))
    return guards


def _verify_census_record(census, native, present, whole_guards):
    """Exact primitive types matter: integer zero is not the boolean False."""
    expected = dict(source_native_row=native["source_native_file_row_index"],
        source_native_SMILES_sha256=hashlib.sha256(native["native_smiles"].encode()).digest(),
        canonical_isomeric_graph_sha256=bytes.fromhex(native["source_current_canonical_graph_sha256"]),
        source_native_parse_pass=True, source_SELFIES_target_present=present,
        derived_native_graph_pass=True, derived_SELFIES_graph_pass=present,
        lossy_stereo_warning=False, admission_gate_code=0)
    if type(census) is not dict or set(census) != set(expected) | {"molecule_guard_sha256", "scaffold_guard_sha256"}:
        raise ValueError("complete exact source census row required")
    for k, v in expected.items():
        if type(census[k]) is not type(v) or census[k] != v:
            raise ValueError("source census native bytes/graph/mask/stereo/allocation mismatch")
    for field, prefix in (("molecule_guard_sha256", "molecule:"), ("scaffold_guard_sha256", "scaffold:")):
        value = census[field]
        if type(value) is not bytes or len(value) != 32 or prefix + value.hex() not in whole_guards:
            raise ValueError("native census whole-graph chemical/scaffold mismatch")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex_sha(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


@dataclass(frozen=True)
class PubChemNativeView:
    direction: str
    input_bytes: bytes
    target_bytes: bytes
    target_native_bytes: bytes
    provenance_bytes: bytes
    binding_sha256: str

    def inputs(self):
        return json.loads(self.input_bytes)

    def targets(self):
        return json.loads(self.target_bytes)

    def provenance(self):
        return json.loads(self.provenance_bytes)


def _source_graph(native, source_manifest):
    """Replay the source parser and exact isomeric decoder, never a stereo projection."""
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
    import selfies as sf
    versions = {"rdkit": "2026.3.1", "selfies": "2.2.0", "pyarrow": "24.0.0"}
    if (source_manifest.get("source_revision") != REVISION
            or source_manifest.get("source_data_license") != "Apache-2.0"
            or source_manifest.get("versions") != versions
            or any(importlib.metadata.version(k) != v for k, v in versions.items())
            or source_manifest.get("semantic_constraints_SHA256") != CONSTRAINTS_SHA256
            or _sha(_canonical(source_manifest.get("semantic_constraints"))) != CONSTRAINTS_SHA256
            or sf.get_semantic_constraints() != source_manifest["semantic_constraints"]):
        raise ValueError("source parser/codec version or semantic constraints mismatch")
    smiles = native.get("native_smiles")
    if (native.get("native_current_RDKit_parse_pass") is not True
            or native.get("coordinate_geometry_present") is not False
            or not isinstance(smiles, str) or not smiles):
        raise ValueError("valid complete native SMILES with no source geometry required")
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("native source graph parse failed")
    iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    graph_sha = _sha(iso.encode())
    if (iso != native["source_canonical_isomeric_smiles"]
            or graph_sha != native["source_current_canonical_graph_sha256"]):
        raise ValueError("original native/exact isomeric graph mismatch")
    present = native["standard_selfies_target_present"]
    if type(present) is not bool or native["graph_roundtrip_pass"] is not present:
        raise ValueError("explicit SELFIES availability/roundtrip masks required")
    selfies_tokens = None
    if present:
        value = native["standard_selfies"]
        if not isinstance(value, str) or not value:
            raise ValueError("available complete SELFIES required")
        try:
            decoded = sf.decoder(value)
            selfies_tokens = len(tuple(sf.split_selfies(value)))
        except (sf.DecoderError, ValueError) as exc:
            raise ValueError("invalid standardSELFIES source symbols") from exc
        with rdBase.BlockLogs():
            back = Chem.MolFromSmiles(decoded)
        if (back is None or Chem.MolToSmiles(back, canonical=True, isomericSmiles=True) != iso
                or native["decoded_canonical_isomeric_smiles"] != iso
                or type(native["standard_selfies_token_count"]) is not int
                or native["standard_selfies_token_count"] != selfies_tokens
                or native["status"] != "standard_SELFIES_graph_roundtrip_pass"):
            raise ValueError("SELFIES full exact isomeric graph/token-axis mismatch")
    elif native["standard_selfies"] is not None:
        raise ValueError("unsupported SELFIES cannot supply an admitted codec target")
    # Mirror the admission's explicitly coarser leakage policy; do not use it to
    # substitute an achiral graph for the source's exact isomeric representation.
    coarse = Chem.Mol(mol)
    for atom in coarse.GetAtoms():
        atom.SetAtomMapNum(0)
    coarse = Chem.RemoveHs(coarse)
    Chem.RemoveStereochemistry(coarse)
    base = Chem.MolToSmiles(coarse, canonical=True, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=coarse, includeChirality=False) or base
    guards = {"chemical_exact_isomeric:" + graph_sha,
        "molecule:" + _sha(base.encode()), "scaffold:" + _sha(scaffold.encode())}
    axes = dict(native_smiles_characters=len(smiles), canonical_smiles_characters=len(iso),
        standard_selfies_characters=len(native["standard_selfies"]) if present else None,
        standard_selfies_tokens=selfies_tokens)
    return present, guards, axes


def pubchem_native_view(row, native_record, *, effective_groups, corpus_binding, source_manifest, census_record=None):
    """Verify a task against its original source row; no independent re-admission.

    Direct callers must supply independently hash-verified source records and
    manifest/group bindings. The file reader below performs those IO checks.
    Full axes remain native bytes, directly consumable by ByteTokenizer.encode.
    """
    try:
        return _project(row, native_record, effective_groups, corpus_binding, source_manifest, census_record)
    except (KeyError, TypeError, IndexError, UnicodeError, OverflowError) as exc:
        raise ValueError("malformed source-native PubChem task") from exc


def _project(row, native, effective_groups, corpus_binding, source_manifest, census_record):
    whole_new = row["split_policy"] == WHOLE_POLICY
    whole_scope = corpus_binding.get("manifest_sha256") in (WHOLE_MASTER_SHA256, CURRENT_MASTER_SHA256)
    raw = row["native_payload"]
    if type(raw) is not bytes or _sha(raw) != row["native_payload_sha256"]:
        raise ValueError("native payload integrity mismatch")
    d = json.loads(raw)
    if _canonical(d) != raw:
        raise ValueError("native task payload must retain canonical admitted bytes")
    direction = next((k for k, v in DIRECTIONS.items() if v[1] == row["task_name"]), None)
    if direction is None:
        raise ValueError("unsupported PubChem codec/direction")
    family, task = DIRECTIONS[direction]
    prior = direction == "reference_prior"
    if (set(d) != {"schema", "task", "conditions", "targets", "audit", "missing"}
            or d["schema"] != "mora3.CorpusTask/v1" or row["native_schema"] != d["schema"]
            or d["task"] != task or row["source_family"] != family
            or row["source_revision"] != REVISION or row["source_sha256"] != SOURCE_SHA256
            or row["admitted_for_task"] is not True or row["split"] != "train"
            or row["split_policy"] != (WHOLE_POLICY if whole_new else POLICY)
            or (whole_new and (not whole_scope or row["model_adapter_supported"] is not False))
            or row["evidence_class"] != ("source_reference_structure" if prior else
                "computed_lossless_representation_not_experimental_response")):
        raise ValueError("source/train-only task admission mismatch")
    present, guards, axes = _source_graph(native, source_manifest)
    if whole_new:
        _verify_census_record(census_record, native, present, guards)
        guards = _whole_component_guards(native, guards)
    elif census_record is not None:
        raise ValueError("pilot rows do not carry whole-component census receipts")
    groups = tuple(effective_groups)
    original = json.loads(row["group_ids_json"])
    if (not groups or any(not isinstance(g, str) or not g for g in groups)
            or len(set(groups)) != len(groups) or not isinstance(original, list)
            or len(set(original)) != len(original) or set(original) != guards
            or not guards <= set(groups)):
        raise ValueError("complete source chemical/scaffold and effective groups required")
    base_keys = {"manifest_sha256", "shard_sha256", "groups_sha256"}
    extra_keys = {"census_part_sha256", "census_part_path", "census_row_index"} if whole_new else set()
    expected_keys = base_keys | (set(WHOLE_BINDING) | extra_keys if whole_scope else set())
    if (set(corpus_binding) != expected_keys
            or corpus_binding["manifest_sha256"] not in SUPPORTED_MASTER_SHA256
            or any(not _hex_sha(corpus_binding[k]) for k in base_keys | (set(WHOLE_BINDING) if whole_scope else set()))
            or (whole_scope and any(corpus_binding[k] != v for k, v in WHOLE_BINDING.items()))):
        raise ValueError("exact pinned admitted pilot/whole corpus bindings required")
    ref = d["audit"]["source_ref"]
    native_receipt = ref["native_record"]
    if (set(native_receipt) != {"path", "bytes", "sha256", "row_index"}
            or not isinstance(native_receipt["path"], str) or not native_receipt["path"]
            or type(native_receipt["bytes"]) is not int or native_receipt["bytes"] <= 0
            or not _hex_sha(native_receipt["sha256"])):
        raise ValueError("typed exact native source receipt required")
    rid = native["source_record_id"]
    index = native["source_native_file_row_index"]
    if (set(ref) != {"native_source", "native_record", "source_record_id", "source_native_split", "source_native_row"}
            or type(index) is not int or index < 0 or type(ref["source_native_row"]) is not int
            or ref["source_native_row"] != index
            or ref["source_record_id"] != rid or row["record_id"] != rid
            or rid != "HF:sagawa/pubchem-10m-canonicalized@" + REVISION + ":train:" + str(index)
            or native["source_native_split"] != "train" or ref["source_native_split"] != "train"
            or _canonical(ref["native_source"]) != _canonical(SOURCE_REF)
            or type(ref["native_record"]["row_index"]) is not int
            or ref["native_record"]["row_index"] < 0
            or row["task_id"] != _sha(_canonical([family, rid, direction]))):
        raise ValueError("original source record/view identity mismatch")
    receipt = dict(source_ref=ref, source_license="Apache-2.0",
        target_role="source_reference_SMILES" if prior else "computed_standard_SELFIES_codec",
        full_native_axes=True, no_3D_or_experimental_response_pair=True)
    if whole_new:
        cr = json.loads(row["admission_receipt_json"])["census_ref"]
        expected_cr = dict(path=corpus_binding["census_part_path"],
            sha256=corpus_binding["census_part_sha256"], row_index=native_receipt["row_index"])
        if (type(corpus_binding["census_row_index"]) is not int
                or corpus_binding["census_row_index"] != native_receipt["row_index"]
                or not _hex_sha(corpus_binding["census_part_sha256"])
                or type(corpus_binding["census_part_path"]) is not str
                or _canonical(cr) != _canonical(expected_cr)):
            raise ValueError("typed exact census part/source row binding required")
        receipt = dict(source_ref=ref, census_ref=expected_cr, source_license="Apache-2.0explicitDATA",
            strict_source_stereo_projection_available=True,
            original_source_graph_sha256=native["source_current_canonical_graph_sha256"],
            no_3D_or_experimental_response_pair=True)
    audit = dict(source_ref=ref, source_geometry_present=False, source_native_split_audit_only="train")
    if prior:
        conditions = dict(prior_domain="molecular_reference_native_SMILES")
        targets = dict(native_smiles=native["native_smiles"])
        audit.update(evidence_class="source_reference_connectivity_not_experimental_response",
            standard_SELFIES_available=present,
            source_dataset="sagawa/pubchem-10m-canonicalized@" + REVISION,
            future_generated_outputs_require_model_produced_provenance=True)
    else:
        if not present:
            raise ValueError("unsupported standardSELFIES remains prior-only")
        conditions = dict(representation_task="standard_SELFIES_codec")
        if direction == "smiles_to_selfies":
            conditions["native_smiles"] = native["native_smiles"]
            targets = dict(standard_selfies=native["standard_selfies"])
        else:
            conditions["standard_selfies"] = native["standard_selfies"]
            targets = dict(source_canonical_isomeric_smiles=native["source_canonical_isomeric_smiles"])
        audit.update(evidence_class="computed_representation_of_source_reference_graph_not_experiment",
            standard_SELFIES_not_original_GroupSELFIES=True,
            atomic_charge_isotope_and_isomeric_graph_roundtrip_verified=True)
    expected_task = dict(schema="mora3.CorpusTask/v1", task=task, conditions=conditions,
        targets=targets, audit=audit, missing=MISSING)
    if (raw != _canonical(expected_task)
            or _canonical(json.loads(row["admission_receipt_json"])) != _canonical(receipt)
            or _canonical(json.loads(row["missing_context_json"])) != _canonical(MISSING)
            or _canonical(json.loads(row["condition_context_json"])) != _canonical(WHOLE_CONTEXT if whole_new else CONTEXT)
            or type(row["full_axis_values"]) is not int
            or row["full_axis_values"] != len(next(iter(targets.values())))):
        raise ValueError("native source labels/full axis/allowlist mismatch; possible target leakage")
    ib, tb = _canonical(conditions), _canonical(targets)
    target_native = next(iter(targets.values())).encode()
    p = dict(schema="mora3.PubChemNativeAdmission/v1", direction=direction,
        envelope={k: v for k, v in row.items() if k != "native_payload"},
        source_ref=ref, native_payload=d, original_native_record=native,
        corpus_binding=corpus_binding, effective_groups=groups,
        condition_sha256=_sha(ib), target_sha256=_sha(tb), target_native_sha256=_sha(target_native),
        standard_selfies_available=present, axes=axes, split="train", heldout_eligible=False,
        paired_cellular_observation=False, independent_biological_validation=False,
        native_smiles_role="original_source_native_bytes",
        canonical_graph_role="derived_exact_isomeric_graph_not_original_native_bytes",
        coarse_guard_role="leakage_guard_not_exact_stereo_join",
        model_training_support=False, full_corpus_usage=False)
    if whole_scope:
        p.update(whole_component_available=True,
            whole_component_scope="verified v21 component inherited unchanged by v22;pilot once;train only")
    if whole_new:
        p["source_census_record"] = {k: v.hex() if type(v) is bytes else v for k, v in census_record.items()}
        p["census_binary_hash_serialization"] = "exact32byte_SHA256_as_hex"
    pb = _canonical(p)
    binding = _sha(_canonical(dict(schema="mora3.PubChemNativeView/v1",
        inputs=_sha(ib), targets=_sha(tb), native_target=_sha(target_native), provenance=_sha(pb))))
    return PubChemNativeView(direction, ib, tb, target_native, pb, binding)


def read_pubchem_native_views(manifest, *, root, expected_manifest_sha256, direction, limit, task_ids=None, shard_paths=None):
    """Read 1..256 complete train-only views from explicit pinned admissions, without cropping.

    Optional exact task IDs and admitted shard paths select bounded diagnostics.
    Whole component construction/QC is inherited via exact immutable proof refs;
    each consumed native source and census part is verified, never the full raw corpus. Every consumed
    full shard, group sidecar, source manifest, source chunk and native Parquet is
    verified before projection. No source scan, download or persistent cache.
    """
    import pyarrow.parquet as pq
    if (direction not in DIRECTIONS or type(limit) is not int or not 1 <= limit <= 256
            or expected_manifest_sha256 not in SUPPORTED_MASTER_SHA256):
        raise ValueError("known direction, bounded limit and pinned admitted v20/v21/v22 master required")
    requested = None
    if task_ids is not None:
        requested = tuple(task_ids)
        if (not requested or len(requested) > limit or len(set(requested)) != len(requested)
                or any(not _hex_sha(t) for t in requested)):
            raise ValueError("unique bounded exact task IDs required")
    root = Path(root).resolve()
    path = (root / manifest).resolve()
    if not path.is_relative_to(root) or _sha(path.read_bytes()) != expected_manifest_sha256:
        raise ValueError("manifest path/integrity mismatch")
    m = json.loads(path.read_bytes())
    family, task = DIRECTIONS[direction]
    whole_scope = expected_manifest_sha256 != PILOT_MASTER_SHA256
    component, census = _whole_metadata(root, m) if whole_scope else (None, None)
    new_paths = {s["path"] for s in component["shards"]} if component else set()
    shards = m["shards"]
    if shard_paths is not None:
        selected = tuple(shard_paths)
        index = {s["path"]: s for s in shards}
        if (not selected or len(selected) > 256 or any(type(p) is not str for p in selected)
                or len(set(selected)) != len(selected)
                or any(p not in index or index[p]["source_family"] != family for p in selected)):
            raise ValueError("unique bounded exact admitted direction-family shard paths required")
        shards = [index[p] for p in selected]
    out, seen, chunks, source_cache = [], set(), {}, {}
    # One native table cache prevents repeated IO for adjacent paired directions
    # while bounding memory independently of the number of source files visited.
    native_key, native_table = None, None
    census_key, census_table = None, None
    for shard in shards:
        if shard.get("source_family") != family:
            continue
        if shard.get("split") != "train":
            raise ValueError("PubChem pilot shard must be train-only")
        rows = pq.read_table(io.BytesIO(_verified_bytes(root, shard)), use_threads=False).to_pylist()
        side = m["effective_group_lookup"].get(shard["path"])
        if side is None:
            raise ValueError("missing authoritative group sidecar")
        group_rows = pq.read_table(io.BytesIO(_verified_bytes(root, side)), use_threads=False).to_pylist()
        groups = {g["task_id"]: g["effective_groups_json"] for g in group_rows}
        ids = [r["task_id"] for r in rows]
        if (len(rows) != shard["rows"] or len(group_rows) != side["rows"]
                or len(groups) != len(group_rows) or len(set(ids)) != len(ids) or set(ids) != set(groups)):
            raise ValueError("shard/group task allocation mismatch")
        for row in rows:
            if row["source_family"] != family or row["split"] != "train":
                raise ValueError("shard/source/train allocation mismatch")
            if row["task_name"] != task or (requested is not None and row["task_id"] not in requested):
                continue
            if row["task_id"] in seen:
                raise ValueError("duplicate requested task")
            seen.add(row["task_id"])
            if _sha(row["native_payload"]) != row["native_payload_sha256"]:
                raise ValueError("native payload integrity mismatch")
            ref = json.loads(row["native_payload"])["audit"]["source_ref"]
            src_receipt = ref["native_source"]
            if _canonical(src_receipt) != _canonical(SOURCE_REF):
                raise ValueError("unpinned original PubChem source")
            key = (src_receipt["path"], src_receipt["sha256"])
            if key not in source_cache:
                source_cache[key] = json.loads(_verified_bytes(root, src_receipt))
            source = source_cache[key]
            nr = ref["native_record"]
            native_path = (root / nr["path"]).resolve()
            candidates = [c for c in source["chunks"]
                if (root / c["path"]).resolve().parent == native_path.parent]
            if len(candidates) != 1:
                raise ValueError("native payload absent from original source chunk")
            ch = candidates[0]
            if ch["path"] not in chunks:
                cp = (root / ch["path"]).resolve()
                if not cp.is_relative_to(root) or _sha(cp.read_bytes()) != ch["sha256"]:
                    raise ValueError("original source chunk integrity mismatch")
                chunks[ch["path"]] = json.loads(cp.read_bytes())
            payloads = [p for p in chunks[ch["path"]]["payloads"]
                if (root / p["path"]).resolve() == native_path]
            if (len(payloads) != 1 or
                    _canonical({k: payloads[0][k] for k in ("bytes", "sha256")}) !=
                    _canonical({k: nr[k] for k in ("bytes", "sha256")})
                    or type(nr["row_index"]) is not int or not 0 <= nr["row_index"] < payloads[0]["rows"]):
                raise ValueError("original source payload/row binding mismatch")
            key = (str(native_path), nr["sha256"])
            if key != native_key:
                native_table = pq.read_table(io.BytesIO(_verified_bytes(root, nr)), use_threads=False)
                native_key = key
                if len(native_table) != payloads[0]["rows"]:
                    raise ValueError("original source row count mismatch")
            native = native_table.slice(nr["row_index"], 1).to_pylist()[0]
            binding = dict(manifest_sha256=expected_manifest_sha256,
                shard_sha256=shard["sha256"], groups_sha256=side["sha256"])
            qc = None
            if whole_scope:
                binding.update(WHOLE_BINDING)
            is_new = shard["path"] in new_paths
            if (row["split_policy"] == WHOLE_POLICY) is not is_new:
                raise ValueError("pilot versus whole component policy/shard substitution")
            if is_new:
                cr = json.loads(row["admission_receipt_json"])["census_ref"]
                parts = [p for p in census["parts"] if p["path"] == cr["path"]]
                if len(parts) != 1:
                    raise ValueError("census part absent from verified whole manifest")
                part = parts[0]
                source_part = part["source_payload"]
                if (part["mode"] != "train" or type(cr["row_index"]) is not int
                        or cr["row_index"] != nr["row_index"]
                        or not 0 <= cr["row_index"] < part["rows"]
                        or cr["sha256"] != part["sha256"]
                        or (root / source_part["path"]).resolve() != native_path
                        or _canonical({k: source_part[k] for k in ("bytes", "sha256")}) !=
                            _canonical({k: nr[k] for k in ("bytes", "sha256")})
                        or part["rows"] != len(native_table)
                        or native["source_native_file_row_index"] != part["source_native_row_start"] + nr["row_index"]):
                    raise ValueError("census/native file, split, row and byte binding mismatch")
                if census_key != part["path"]:
                    census_table = pq.read_table(io.BytesIO(_verified_bytes(root, part)), use_threads=False)
                    census_key = part["path"]
                    if len(census_table) != part["rows"]:
                        raise ValueError("census native row conservation mismatch")
                qc = census_table.slice(cr["row_index"], 1).to_pylist()[0]
                binding.update(census_part_sha256=part["sha256"], census_part_path=part["path"],
                    census_row_index=cr["row_index"])
            out.append(pubchem_native_view(row, native, source_manifest=source,
                effective_groups=json.loads(groups[row["task_id"]]), corpus_binding=binding, census_record=qc))
            if len(out) == limit:
                return tuple(out)
    if requested is not None and set(requested) != seen:
        raise ValueError("requested task IDs absent from admitted direction")
    if not out:
        raise ValueError("no admitted matching PubChem tasks")
    return tuple(out)
