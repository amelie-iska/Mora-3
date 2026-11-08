"""Lossless RNAcentral28 pilot priors bound to admitted corpus and native rows.

This reader establishes a native data interface. RNAcentral is absent from the
current neural nucleotide decoder source/domain allowlist. Full-native source QC
is inherited from retained independent proofs; this reader never opens FASTA.
"""
import hashlib
import io
import json
from pathlib import Path
import re

from .nucleotide_prior import NucleotidePriorView

SOURCE = "RNAcentral-reference-priors"
TASK = "source_native_nucleotide_sequence_prior"
DOMAIN = "RNA_reference_native_symbols"
REVISION = "officialRNAcentral28.0active+inactive"
POLICY = "referenceprior_trainonly;homology_andgenuineheldoutunresolved"
EVIDENCE = "releasedRNAreference_sequenceprior;notnewindependentexperiment"
ALPHABET = frozenset("ACGTURYSWKMBDHVNacgturyswkmbdhvn")
MASTER_SHA256 = "fc47edd361ce3a5dc7088995670323f0cd87803d6b97184fbb9703dcde4b2d2f"
BASE = "outputs/data-intake/20261101-RNAcentral-corpus-pilot-v1"
COMPONENT = dict(path=BASE + "/manifest.json", bytes=149225,
    sha256="e06af43a1a491782c70870e2f352035af0bb8afd768c778bec7deff088ad63fb")
PLAN = dict(path=BASE + "/build-plan.json", bytes=8242,
    sha256="3c61c0a688ecd761149b9934d67f554f8cf333bae35366fb94601e5ea2871b3b")
UNION = dict(path="outputs/data-intake/20261101-RNAcentral-pilot-union-v1/receipt.json", bytes=2628,
    sha256="60fce098a553cd0a0f82d12a8b0e4a0fbf5d22339805f27dcc6ca49613ae5d13")
_SOURCE_CANONICAL = {
    "active": ("832cb64d706c44623ceeba0cd38c513a27adbfb8a4899ad9fb187d38e93628aa",
               "69896c13306180c52afd5530d2f09c74d96225a1143e87608db962b755f7cd47"),
    "inactive": ("c08e50dff39bfcebc036e14f80b06493254b9f50413d27c4c1b9ed8605650fe3",
                 "1aa47ec9c7861a9a29fbb8204c570311208fd10e1b3483e4fcd1f0a986e457db"),
}
# Native source availability is broader than these ten admitted pilot allocations.
_PILOT_OCCURRENCE = {'a5d9a9db0084c79417118b43f0a1adb0ebaaa887af6ad5f8994fd4947457aba9': '0e99b7e6d8e29558f0c7044ccbe76b03e2b2d468987bf01a6576f6794e6b481d', '297103879da6b8daafeb3d8588d82efb180c6c79b0963ef87f5c02b6c70cc882': '57b6c1065392047869a71ab9381a93833d9b6659635043943fb43b63c53fea3f', '58068fb6b4403d795855accdb5bf9e7a9786d1a87cdf11224bd8322862bbdc23': 'daf827dccd7e5a0c3e3b47956263a06255a2f9178c31961e9b615cfa2c3308ac', 'ebdbfc481ac487b1d88354be0b1a64330b8e344e2c5ac65863806021e21b6364': '8bbeb84bbbb6c34ec69e8800ba87d93e0007a4acce53d5624bd2ec16d767355a', '46a76ea846ec4366e9fa595b96f7b43f1f49f79803a95683bb7abb2a2a7ecb24': 'f2a85c47d6b5f190abd963c417d1828b7cd71cb179186a0dce225af5127a838d', '2207fee9dfadd6afdbfc53e49d1fe8b6e1478da39587ec5373da6477ac7996dc': '8faa402e60a7224117032bc8ce70f151ab5a114a33bdc7e73a6c316f64126689', '9f87891c8bd302d20fca4fcac0624c1836b28c161c2be397a5f35f7e29c261e2': 'da6f5c4e2b13258949f4c46c2dffa82aaeeb02ef9d8766fbaf9c45de77983408', '2a099c43c0b471e35f8e1e26dcae1e5448bdaa83d7c0a8bd7cb83c344c298ea1': '69aef05b64cae877b8915f37fbb292bfae6c44c280c9e2d9e788adcba4c35ec3', '70f54cb3560c21058c0b9412d8cb5e2c963d46a8c698b6dc602e96e9c0c73063': '28024577221b1cd21fadf401e48b854a752cefc4d46201f30d8c2a992d475810', 'da67f646b38a10a592183ae90491cb0cdf265666bc4b98699d12e0d702097f72': '4c8824a4b54597c0828e5b51bc8af676f299ab930d0abccc6ad11606ce4c8760'}
MISSING = ["native_neural_RNAprior_reader", "nativeheader_taxonomy_function_scope_verification",
           "homology_and_genuine_heldout_protocol", "cellular_response_or_imagepair"]
CONDITIONS = dict(prior_domain=DOMAIN, source_organism=None)
CONTEXT = dict(source_status_notactivitytruth=True, sourceIDsheadersauditonly=True,
               all_unlisted_annotations_unknown=True)
_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVNX", "TGCAYRSWMKVHDBNX")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise ValueError("duplicate JSON key")
        out[k] = v
    return out


def _json(raw):
    if type(raw) not in (str, bytes):
        raise ValueError("serialized JSON required")
    def invalid_constant(value):
        raise ValueError("non-finite JSON value")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid_constant)


def _same(actual, expected, message):
    # Python equality aliases False/0 and True/1, including nested objects.
    if _canonical(actual) != _canonical(expected):
        raise ValueError(message)


def _sha(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def _groups(sequence, identifier):
    upper = sequence.upper()
    base = upper.replace("U", "T")
    rc = base.translate(_COMPLEMENT)[::-1]
    return ["nucleotide_native:" + _digest(sequence.encode()),
            "nucleotide_upper:" + _digest(upper.encode()),
            "nucleotide_derived_RC_baseorder:" + _digest(min(base, rc).encode()),
            "rnacentral_native_identifier:" + identifier]


def _source_part(source_manifest, source_proof, ref):
    status = ref["source_status"]
    if ref["native_payload_sha256"] not in _PILOT_OCCURRENCE:
        raise ValueError("native source is outside the admitted pilot occurrence scope")
    if status not in _SOURCE_CANONICAL:
        raise ValueError("unsupported source status")
    sh, ph = _SOURCE_CANONICAL[status]
    if _digest(_canonical(source_manifest)) != sh or _digest(_canonical(source_proof)) != ph:
        raise ValueError("pinned full-native source manifest/proof mismatch")
    # Exact manifest membership, including file offset, binds original FASTA order.
    offset = 0
    for payload in source_manifest["native_payloads"]:
        if payload["sha256"] == ref["native_payload_sha256"]:
            relative = ref["native_payload"]
            if (type(relative) is not str or Path(relative).is_absolute() or ".." in Path(relative).parts
                    or not payload["path"].endswith("/" + relative)):
                raise ValueError("native payload source path mismatch")
            if (type(ref["native_row_index"]) is not int
                    or not 0 <= ref["native_row_index"] < payload["rows"]):
                raise ValueError("native row outside source")
            return status, payload, offset
        offset += payload["rows"]
    raise ValueError("source payload absent from retained full-native proof")


def rnacentral_prior_view(row, native_record, occurrence, *, effective_groups,
                          corpus_binding, source_manifest, source_proof):
    """Validate an in-memory task against one native row and occurrence.

    The caller supplies source objects. File byte verification belongs to
    read_rnacentral_priors; projection alone makes no at-rest integrity claim.
    """
    try:
        return _project(row, native_record, occurrence, effective_groups=effective_groups,
            corpus_binding=corpus_binding, source_manifest=source_manifest, source_proof=source_proof)
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError("malformed source-bound RNAcentral view") from exc


def _project(row, native_record, occurrence, *, effective_groups, corpus_binding, source_manifest, source_proof):
    raw = row["native_payload"]
    if type(raw) is not bytes or _digest(raw) != row["native_payload_sha256"]:
        raise ValueError("native task payload integrity mismatch")
    task = _json(raw)
    ref = task["audit"]["source_ref"]
    status, source_payload, offset = _source_part(source_manifest, source_proof, ref)
    sequence = native_record["native_sequence"]
    if type(sequence) is not str or not sequence or set(sequence) - ALPHABET:
        raise ValueError("native RNAcentral alphabet required; no repair or truncation")
    asset = source_manifest["source_asset"]
    header = native_record["native_header"]
    if type(header) is not str or not header or not header.split():
        raise ValueError("full original source header required")
    identifier = header.split()[0]
    original_index = offset + ref["native_row_index"]
    record_id = "official_RNAcentral28_" + status + ":" + asset["id"] + ":" + str(original_index)
    sequence_sha = _digest(sequence.encode())
    expected_ref = dict(native_payload=ref["native_payload"], native_payload_sha256=source_payload["sha256"],
        native_row_index=ref["native_row_index"], original_record_index=original_index,
        original_FASTA=asset["path"], source_identifier=identifier, native_header=header,
        source_status=status, source_release="RNAcentral28.0", source_rights="CC0-1.0")
    _same(ref, expected_ref, "native source/header/occurrence/release binding mismatch")
    rfam = re.search(r"(?<![A-Za-z0-9])RF[0-9]{5}(?![0-9])", header)
    expected_native = dict(source_record_id=record_id, source_record_index=original_index,
        source_payload=asset["path"], source_identifier=identifier, native_header=header,
        native_sequence=sequence, native_length=len(sequence), native_sequence_sha256=sequence_sha,
        uppercase_sequence_sha256=_digest(sequence.upper().encode()), sequence_alphabet_pass=True,
        native_T_count=sequence.upper().count("T"), native_U_count=sequence.upper().count("U"),
        ambiguity_symbol_count=sum(c.upper() not in "ACGTU" for c in sequence),
        explicit_Rfam_ID_in_header=rfam[0] if rfam else None,
        source_role="released RNAfamily/reference-sequence evidence; exact family/region join requires source metadata",
        task_admitted=False)
    _same(native_record, expected_native, "native row values/types/axis mismatch")
    audit = dict(source_ref=expected_ref,
        source_role="releasedRNAreference_sequence_prior;notindependentexperimentalmeasurement",
        source_record_status_is_not_activity_label=True,
        source_symbols="nativecase_T_U_IUPACretainedwithoutcroporrepair",
        source_symbol_loss_mask="allnativevalidcharactersobserved;biologicalbase-resolutionlossonlyupperACGTU;ambiguitynotimputed",
        excluded_predictors=["sourceidentifiers_headers_status_classifications", "targetsequence_length_hash", "sourceauditmetadata"],
        native_reader_supported=False)
    expected_task = dict(schema="mora3.CorpusTask/v1", task=TASK, conditions=CONDITIONS,
        targets=dict(source_native_sequence=sequence), audit=audit, missing=MISSING)
    if raw != _canonical(expected_task):
        raise ValueError("exact native task/conditions/targets/audit/missingness mismatch")
    task_id = _digest(_canonical([DOMAIN, None, sequence_sha]))
    groups = _groups(sequence, identifier)
    receipt = dict(source_ref=expected_ref, native_sequence_sha256=sequence_sha, exactfullaxis=True,
        source_case_T_U_preserved=True, source_license="CC0-1.0",
        task_admission="eligible_referenceprior;master_unionindependentauditpending")
    expected_envelope = dict(task_id=task_id, record_id=record_id, source_family=SOURCE,
        source_revision=REVISION, source_sha256=source_payload["sha256"], task_name=TASK,
        native_schema="mora3.CorpusTask/v1", native_payload_sha256=_digest(raw), split="train",
        split_policy=POLICY, evidence_class=EVIDENCE, admitted_for_task=True, model_adapter_supported=False,
        full_axis_values=len(sequence))
    envelope = {k: v for k, v in row.items() if k not in
        ("native_payload", "admission_receipt_json", "condition_context_json", "missing_context_json", "group_ids_json")}
    _same(envelope, expected_envelope, "source/train/axis/task admission envelope mismatch")
    _same(_json(row["admission_receipt_json"]), receipt, "native source admission receipt mismatch")
    _same(_json(row["condition_context_json"]), CONTEXT, "condition context allowlist mismatch")
    _same(_json(row["missing_context_json"]), MISSING, "missingness mismatch")
    _same(_json(row["group_ids_json"]), groups, "native/upper/derived/URS group conservation mismatch")
    effective = list(effective_groups)
    if (not effective or any(type(g) is not str or not g for g in effective)
            or len(set(effective)) != len(effective) or not set(groups) <= set(effective)):
        raise ValueError("unique authoritative effective groups must preserve native guards")
    expected_occurrence = dict(source_ref_json=occurrence["source_ref_json"], task_id=task_id,
        task_already_in_Rfam_corpus=False, first_occurrence_in_pilot=True,
        native_sequence_sha256=sequence_sha, uppercase_sequence_sha256=expected_native["uppercase_sequence_sha256"],
        derived_RC_baseorder_sha256=groups[2].split(":", 1)[1], derived_RC_is_molecular_identity=False,
        native_length=len(sequence), source_status_is_not_negative_label=True)
    _same(_json(occurrence["source_ref_json"]), expected_ref, "source occurrence reference mismatch")
    _same(occurrence, expected_occurrence, "new pilot occurrence/allocation/type mismatch; old Rfam is not a new task")
    if (set(corpus_binding) != {"manifest_sha256", "shard_sha256", "groups_sha256", "occurrence_sha256"}
            or corpus_binding["manifest_sha256"] != MASTER_SHA256
            or corpus_binding["occurrence_sha256"] != _PILOT_OCCURRENCE[source_payload["sha256"]]
            or not all(_sha(v) for v in corpus_binding.values())):
        raise ValueError("authoritative v20 corpus/shard/group/occurrence bindings required")
    conditions = _canonical(CONDITIONS)
    provenance = dict(schema="mora3.RNAcentralPriorAdmission/v1", sequence_sha256=sequence_sha,
        condition_sha256=_digest(conditions), source_ref=expected_ref, task_id=task_id, record_id=record_id,
        source_family=SOURCE, source_release="RNAcentral28.0", task_native_payload_sha256=_digest(raw),
        native_length=len(sequence), effective_groups=effective, corpus_binding=dict(corpus_binding),
        source_manifest_sha256=source_proof["source_native_manifest_sha256"],
        source_proof_canonical_sha256=_digest(_canonical(source_proof)),
        source_receipt_sha256=source_manifest["source_receipt_sha256"],
        original_FASTA_sha256=asset["sha256"], inherited_full_native_QC=True, raw_FASTA_reread=False,
        native_file_bytes_verified=False, heldout_eligible=False, paired_cellular_observation=False,
        source_record_status_is_not_activity_label=True, derived_RC_is_molecular_identity=False,
        reused_Rfam_task=False, source_evidence=EVIDENCE, neural_model_token_count=None,
        neural_objective_supported=False)
    return NucleotidePriorView(sequence, conditions, _canonical(provenance))


def _verified_bytes(root, receipt):
    path = (root / receipt["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("source path outside project")
    raw = path.read_bytes()
    if (type(receipt.get("bytes")) is not int or len(raw) != receipt["bytes"]
            or not _sha(receipt["sha256"]) or _digest(raw) != receipt["sha256"]):
        raise ValueError("consumed source/shard/proof integrity mismatch")
    return raw


def _hash_document(root, path, sha):
    # Small retained JSON receipts; raw FASTA is never passed to this function.
    p = (root / path).resolve()
    if not p.is_relative_to(root):
        raise ValueError("receipt path outside project")
    raw = p.read_bytes()
    if _digest(raw) != sha:
        raise ValueError("retained proof integrity mismatch")
    return _json(raw)


def _parquet_row(pf, index):
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if index < n:
            return pf.read_row_group(rg, use_threads=False).slice(index, 1).to_pylist()[0]
        index -= n
    raise ValueError("source row outside native file")


def read_rnacentral_priors(manifest, *, root, expected_manifest_sha256, shard_paths,
                           limit, source_status=None, task_ids=None):
    """Read 1..64 native views from 1..8 explicit admitted pilot shards.

    Selected IDs must all occur in the selected shards/status scope. No corpus
    length scan, raw FASTA replay, neural encoding, or optimizer work occurs.
    """
    try:
        return _read(manifest, root=root, expected_manifest_sha256=expected_manifest_sha256,
            shard_paths=shard_paths, limit=limit, source_status=source_status, task_ids=task_ids)
    except (KeyError, TypeError, UnicodeError, OverflowError) as exc:
        raise ValueError("malformed RNAcentral reader request/source") from exc


def _read(manifest, *, root, expected_manifest_sha256, shard_paths, limit, source_status, task_ids):
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    if (expected_manifest_sha256 != MASTER_SHA256 or type(limit) is not int or not 1 <= limit <= 64
            or source_status not in (None, "active", "inactive")):
        raise ValueError("pinned master, bounded limit, and source status required")
    if (type(shard_paths) not in (list, tuple) or not 1 <= len(shard_paths) <= 8
            or any(type(x) is not str for x in shard_paths) or len(set(shard_paths)) != len(shard_paths)):
        raise ValueError("unique explicit bounded shard selectors required")
    if task_ids is not None and (type(task_ids) not in (list, tuple) or not 1 <= len(task_ids) <= limit
            or any(not _sha(x) for x in task_ids) or len(set(task_ids)) != len(task_ids)):
        raise ValueError("unique exact bounded task selectors required")
    root = Path(root).resolve()
    master = _hash_document(root, manifest, MASTER_SHA256)
    if master["additional_RNAcentral_native_prior_manifest"] != COMPONENT["path"]:
        raise ValueError("master RNAcentral component mismatch")
    component = _json(_verified_bytes(root, COMPONENT))
    plan = _json(_verified_bytes(root, PLAN))
    union = _json(_verified_bytes(root, UNION))
    if (union["status"] != "verified" or union["component"] != COMPONENT
            or union["source_tasks"][SOURCE] != 445486
            or component["counts"]["source_rows"] != 446149
            or component["counts"]["quarantined_occurrences"] != 2
            or component["counts"]["distinct_reused_existing_Rfam_tasks"] != 661):
        raise ValueError("independent pilot admission/reuse/quarantine scope mismatch")
    handoff = _json(_verified_bytes(root, plan["source_handoff"]))
    admitted = {s["path"]: s for s in component["shards"]}
    master_shards = {}
    for sh in master["shards"]:
        if sh["path"] in master_shards:
            raise ValueError("duplicate master shard allocation")
        master_shards[sh["path"]] = sh
    for path in shard_paths:
        if path not in admitted or path not in master_shards:
            raise ValueError("selector is not an admitted RNAcentral pilot shard")
        _same(master_shards[path], admitted[path], "master/component shard mismatch")
    selected = {}
    for i, p in enumerate(plan["selected"]):
        key = p["native_payload"]["sha256"]
        if key in selected:
            raise ValueError("duplicate native source allocation")
        selected[key] = (i, p)
    sources = {}
    native_files = {}
    occurrences = {}
    out = []
    seen = set()
    selected_occurrences = set()
    wanted = set(task_ids) if task_ids is not None else None
    for path in shard_paths:
        shard = master_shards[path]
        side = master["effective_group_lookup"][path]
        rows = pq.read_table(io.BytesIO(_verified_bytes(root, shard)), use_threads=False).to_pylist()
        group_rows = pq.read_table(io.BytesIO(_verified_bytes(root, side)), use_threads=False).to_pylist()
        if (len(rows) != shard["rows"] or len(group_rows) != side["rows"] or len(rows) != len(group_rows)):
            raise ValueError("shard/effective-group row count mismatch")
        # Check the complete consumed allocation before applying caller selection.
        for row, group in zip(rows, group_rows):
            if (set(group) != {"task_id", "effective_groups_json"} or row["task_id"] != group["task_id"]
                    or row["task_id"] in seen):
                raise ValueError("duplicate task allocation or group alignment mismatch")
            seen.add(row["task_id"])
        for row, group in zip(rows, group_rows):
            if wanted is not None and row["task_id"] not in wanted:
                continue
            task = _json(row["native_payload"])
            ref = task["audit"]["source_ref"]
            if source_status is not None and ref["source_status"] != source_status:
                continue
            ordinal, selection = selected[row["source_sha256"]]
            status = selection["component"]
            if status not in sources:
                component_ref = handoff["components"][status]
                source = _json(_verified_bytes(root, selection["source_manifest"]))
                proof = _hash_document(root, component_ref["independent_full_readback"],
                                       component_ref["independent_full_readback_sha256"])
                _hash_document(root, source["source_receipt"], source["source_receipt_sha256"])
                _same(source["source_family"], "official_RNAcentral28_" + status, "source component mismatch")
                sources[status] = (source, proof)
            source, proof = sources[status]
            _, native_receipt, _ = _source_part(source, proof, ref)
            key = native_receipt["sha256"]
            if key not in native_files:
                pf = pq.ParquetFile(io.BytesIO(_verified_bytes(root, native_receipt)))
                if pf.metadata.num_rows != native_receipt["rows"]:
                    raise ValueError("native source row count mismatch")
                native_files[key] = pf
            native = _parquet_row(native_files[key], ref["native_row_index"])
            occ_receipt = component["occurrences"][ordinal]
            if ordinal not in occurrences:
                pf = pq.ParquetFile(io.BytesIO(_verified_bytes(root, occ_receipt)))
                if pf.metadata.num_rows != occ_receipt["rows"]:
                    raise ValueError("occurrence row count mismatch")
                occurrences[ordinal] = pf.read(use_threads=False)
            matches = occurrences[ordinal].filter(
                pc.equal(occurrences[ordinal]["task_id"], row["task_id"])).to_pylist()
            if len(matches) != 1:
                raise ValueError("one exact new source occurrence required")
            occurrence_key = (key, ref["native_row_index"])
            if occurrence_key in selected_occurrences:
                raise ValueError("duplicate native occurrence allocation")
            selected_occurrences.add(occurrence_key)
            view = rnacentral_prior_view(row, native, matches[0],
                effective_groups=_json(group["effective_groups_json"]),
                corpus_binding=dict(manifest_sha256=MASTER_SHA256, shard_sha256=shard["sha256"],
                    groups_sha256=side["sha256"], occurrence_sha256=occ_receipt["sha256"]),
                source_manifest=source, source_proof=proof)
            provenance = view.provenance()
            provenance.update(native_file_bytes_verified=True, occurrence_file_bytes_verified=True,
                source_handoff_sha256=plan["source_handoff"]["sha256"], independent_union_sha256=UNION["sha256"],
                reader_scope="consumed_shards_groups_occurrences_native_bytes_and_rows;raw_FASTA_proof_inherited")
            out.append(NucleotidePriorView(view.sequence, view.condition_bytes, _canonical(provenance)))
            if len(out) == limit and wanted is None:
                return tuple(out)
    if wanted is not None and {v.provenance()["task_id"] for v in out} != wanted:
        raise ValueError("selected tasks absent from bounded admitted shard/status scope")
    return tuple(out)
