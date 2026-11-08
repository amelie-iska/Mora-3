from dataclasses import replace
from hashlib import sha256
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mora3.data.biogrid_native_assay_lists import (
    BioGRIDNativeAssayLists,
    SEQUENCE_TASK,
    SOURCE_TASK,
)


PRIMARY = "MCSA_guarded"
STRICTER = "MCSA_and_prospective_AFDB_guarded"


@pytest.fixture()
def native_fixture(tmp_path):
    def reference(path):
        raw = path.read_bytes()
        return {"path": str(path.relative_to(tmp_path)), "bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    def write_json(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return reference(path)

    source_rows = []
    coords = []
    for accession, sequence in (("P00001", "AAW"), ("P00002", "MKV")):
        full = json.dumps({
            "primaryAccession": accession, "organism": {"taxonId": 9606},
            "entryAudit": {"entryVersion": 3, "sequenceVersion": 2},
            "sequence": {"value": sequence},
        })
        digest = sha256(full.encode("utf-8")).hexdigest()
        source_rows.append(dict(primary_accession=accession, full_source_json=full, source_structured_sha256=digest))
        coords.append(dict(primary_accession=accession, canonical_sequence_sha256=sha256(sequence.encode("ascii")).hexdigest(),
                           canonical_sequence_length=len(sequence), entry_version=3, sequence_version=2,
                           source_release="2026_03", organism_taxon_id=9606, source_full_parquet_row_ordinal=len(coords),
                           source_structured_entry_sha256=digest))
    source = tmp_path / "actual-reference-part.parquet"
    pq.write_table(pa.Table.from_pylist(source_rows), source)
    source_ref = reference(source)
    for row in coords:
        row["source_full_parquet_path"] = source_ref["path"]
        row["source_full_parquet_sha256"] = source_ref["sha256"]
    coord_file = tmp_path / "coordinates.parquet"
    pq.write_table(pa.Table.from_pylist(coords), coord_file)

    def pair(a, b, ordinals, assay, primary, stricter, refs):
        return dict(
            native_BioGRID_interactor_A=a, native_BioGRID_interactor_B=b,
            known_component_min_original_ordinal=ordinals[0],
            prospective_MCSA_guarded_pair_role=min(primary),
            prospective_MCSA_and_AFDB_guarded_pair_role=min(stricter),
            original_source_row_ordinals=ordinals, original_assay_axis_ids=assay,
            original_publication_source_fields=[b"PUBMED:12345"] * len(ordinals),
            original_endpoint_order_reversed_from_pair=[False] + [True] * (len(ordinals) - 1),
            original_exposure_bits=[1] * len(ordinals),
            original_MCSA_event_role=primary, original_MCSA_and_AFDB_event_role=stricter,
            pair_A_current_reference_status_bits=0 if refs[0] is not None else 1,
            pair_B_current_reference_status_bits=0 if refs[1] is not None else 2,
            pair_A_unique_current_reference_index=refs[0], pair_B_unique_current_reference_index=refs[1],
        )

    pair_file = tmp_path / "pair-lists.parquet"
    pq.write_table(pa.Table.from_pylist([
        pair(10, 20, [0, 2], [0, 0], [1, 1], [1, 1], (0, 1)),
        pair(10, 30, [1], [1], [2], [2], (0, None)),
        pair(30, 40, [3, 5], [0, 1], [0, 3], [0, 3], (None, None)),
        pair(40, 50, [4], [1], [3], [0], (0, 1)),
    ]), pair_file, row_group_size=2)
    axis = write_json("axis.json", dict(
        schema="Mora3.BioGRIDLiteralSourceExperimentalSystemAxis/v1",
        symbols=[{"id": 0, "native_Experimental_System": "Affinity Capture-MS", "original_rows": 3},
                 {"id": 1, "native_Experimental_System": "Co-crystal Structure", "original_rows": 3}],
        EOS_id=2, BOS_input_only_id=3, output_classes=3, input_classes=4,
    ))
    manifest = write_json("manifest.json", dict(
        schema="Mora3.BioGRIDCompleteNativeAssayListCandidates/v1",
        source_release="5.0.262", optimizer_admission=False,
        prospective_pair_roles_review_pending=True,
        exact_native_unordered_pairs=4, exact_original_source_rows=6, original_human_physical_associations=6,
        original_native_BioGRID=source_ref, exact_native_assay_axis=axis, maximum_original_complete_assay_list=2,
        all_pairs=reference(pair_file), immutable_current_reference_coordinates=reference(coord_file),
        total_output_classes=3, total_additional_input_symbols=4,
    ))
    proof = write_json("proof.json", dict(
        schema="Mora3.IndependentOriginalBioGRIDAssayListProjectionVerification/v1", producer=manifest,
        all_source_pair_lists_verified=4, selected_original_native_events_verified=6,
        all_original_source_row_ordinals_accounted=6, original_native_BioGRID=source_ref,
        no_partially_masked_pair_is_eligible=True,
        source_reference_coordinates_and_optional_endpoint_conservative_checks_verified=True,
        optimizer_admission=False,
    ))

    def open_source():
        return BioGRIDNativeAssayLists(
            tmp_path, manifest["path"], manifest["sha256"],
            independent_proof_path=proof["path"], independent_proof_sha256=proof["sha256"],
        )

    return open_source, source, pair_file


def test_complete_ordered_duplicate_assays_and_audit_exclusion(native_fixture):
    open_source, _, _ = native_fixture
    source = open_source()
    pair = source.pair_at(0, arm=PRIMARY)
    assert pair.condition_inputs() == {"task": SOURCE_TASK, "native_pair": ("10", "20")}
    assert pair.target_ids() == (0, 0, source.EOS_id)
    assert source.literal_assay_axis == ("Affinity Capture-MS", "Co-crystal Structure")
    assert source.BOS_input_only_id == source.EOS_id + 1
    assert pair.events[1].original_order_reversed is True
    assert pair.events[1].original_publication_source == b"PUBMED:12345"
    assert pair.events[1].original_row_ordinal == 2
    assert pair.original_role is None and pair.optimizer_authorized is False
    source.require_pair(pair, arm=PRIMARY, prospective_role="train")
    with pytest.raises(ValueError, match="authentic complete"):
        source.require_pair(replace(pair, events=pair.events[:1]), arm=PRIMARY, prospective_role="train")


def test_partial_list_never_becomes_eligible_and_stricter_arm_does_not_reassign(native_fixture):
    source = native_fixture[0]()
    primary = list(source.iter_pairs(arm=PRIMARY, prospective_role="test"))
    assert [pair.source_pair_ordinal for pair in primary] == [3]
    assert list(source.iter_pairs(arm=STRICTER, prospective_role="test")) == []
    audits = list(source.iter_pairs(arm=PRIMARY, prospective_role="audit_only"))
    assert [row.source_pair_ordinal for row in audits] == [2]
    assert [event.original_prospective_event_role for event in audits[0].events] == ["audit_only", "test"]
    with pytest.raises(ValueError, match="no source objective"):
        audits[0].target_ids()
    with pytest.raises(ValueError, match="authentic complete"):
        source.require_pair(primary[0], arm=STRICTER, prospective_role="test")


def test_optional_complete_current_sequence_and_unknown_refusal(native_fixture):
    source = native_fixture[0]()
    train = source.pair_at(0, arm=PRIMARY)
    assert source.current_reference_condition_inputs(train, arm=PRIMARY, prospective_role="train") == {
        "task": SEQUENCE_TASK, "current_reference_pair_sequences": ("AAW", "MKV"),
    }
    val = source.pair_at(1, arm=PRIMARY)
    assert val.target_ids() == (1, source.EOS_id)
    with pytest.raises(ValueError, match="both original endpoints"):
        source.current_reference_condition_inputs(val, arm=PRIMARY, prospective_role="validation")


def test_native_part_hash_checked_when_optional_sequence_is_loaded(native_fixture):
    open_source, path, _ = native_fixture
    source = open_source()
    pair = source.pair_at(0, arm=PRIMARY)
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="pinned original"):
        source.current_reference_condition_inputs(pair, arm=PRIMARY, prospective_role="train")


def test_mutated_pair_artifact_is_refused_before_issuing_rows(native_fixture):
    open_source, _, path = native_fixture
    path.write_bytes(b"replacement")
    with pytest.raises(ValueError, match="pinned original"):
        open_source()
