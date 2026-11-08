from copy import deepcopy
from hashlib import sha256
import json

import pytest

from mora3.data.finite_union_inverse_literal_filter import TASK, _require_literal_witness


def make_witness(condition="substrate ACDEFGHIKL", target="VVACDEFGHIKLVV"):
    payload = {
        "schema": "mora3.CorpusTask/v1",
        "task": TASK,
        "conditions": {"source_kinetic_conditions": [{"substrate_label": condition}]},
        "targets": {"canonical_sequence": target},
    }
    original = {"results": [{
        "primaryAccession": "P00001",
        "entryAudit": {"entryVersion": 113, "sequenceVersion": 1},
        "sequence": {"value": target},
        "comments": [{"kineticParameters": {"michaelisConstants": [{"substrate": condition}]}}],
    }]}
    hashes = {
        name: sha256(json.dumps(payload[name], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        for name in ("conditions", "targets")
    }
    evidence = {
        "condition_field_JSON_pointer": "/conditions/source_kinetic_conditions/0/substrate_label",
        "target_field_JSON_pointer": "/targets/canonical_sequence",
        "exact_original_kinetic_substrate_pointer": "/results/0/comments/0/kineticParameters/michaelisConstants/0/substrate",
        "original_source_entry_row_ordinal": 0,
        "source_record_id": "P00001",
        "source_revision": "UniProt2026_03;entryVersion113;sequenceVersion1",
        "canonical_conditions_sha256": hashes["conditions"],
        "canonical_targets_sha256": hashes["targets"],
        "condition_UTF8_value_sha256": sha256(condition.encode()).hexdigest(),
        "target_UTF8_value_sha256": sha256(target.encode()).hexdigest(),
        "token_condition_character_offset0": len("substrate "),
        "target_canonical_residue_offset0": 2,
        "exact_literal_uppercase_token_length": 10,
        "exact_literal_uppercase_token_SHA256": sha256(b"ACDEFGHIKL").hexdigest(),
    }
    return payload, original, evidence


def test_exact_original_kinetic_and_inverse_literal_witness():
    _require_literal_witness(*make_witness())


@pytest.mark.parametrize("condition", ["substrate ACDEFGHIKL_", "substrate ACDEFGHIKLX", "substrate ACDEFGHIKLé"])
def test_nonstandalone_target_substring_is_not_the_literal_witness(condition):
    with pytest.raises(ValueError, match="standalone literal"):
        _require_literal_witness(*make_witness(condition=condition))


def test_nonmatching_target_residue_offset_refused():
    payload, original, evidence = make_witness()
    evidence["target_canonical_residue_offset0"] += 1
    with pytest.raises(ValueError, match="standalone literal"):
        _require_literal_witness(payload, original, evidence)


def test_pointer_and_source_version_remain_exact():
    payload, original, evidence = make_witness()
    bad = dict(evidence, exact_original_kinetic_substrate_pointer="/results/0/comments/0/summary")
    with pytest.raises(ValueError, match="exact declared kinetic condition"):
        _require_literal_witness(payload, original, bad)
    original["results"][0]["entryAudit"]["entryVersion"] += 1
    with pytest.raises(ValueError, match="entry or sequence version"):
        _require_literal_witness(payload, original, evidence)


def test_original_source_replacement_and_payload_gold_changes_refused():
    payload, original, evidence = make_witness()
    changed = deepcopy(original)
    changed["results"][0]["comments"][0]["kineticParameters"]["michaelisConstants"][0]["substrate"] += "!"
    with pytest.raises(ValueError, match="predictor/target bytes"):
        _require_literal_witness(payload, changed, evidence)
    payload["targets"]["canonical_sequence"] += "A"
    with pytest.raises(ValueError, match="predictor/target bytes"):
        _require_literal_witness(payload, original, evidence)
