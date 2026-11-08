"""Regression coverage for native scalar meaning and Float64 boundaries."""
from dataclasses import replace
import hashlib
import json

import pytest

from mora3.data.admet import NativeCell
from mora3.data.source_scalar_native_rows import SCOPES, ScalarTarget, _project


def project(family, conditions, targets, audit):
    task = SCOPES[family]
    schema = "mora3.CorpusTask/v1"
    raw = json.dumps(dict(schema=schema, task=task, conditions=conditions, targets=targets,
                          audit=audit, missing={})).encode()
    row = dict(native_payload=raw, native_payload_sha256=hashlib.sha256(raw).hexdigest(),
               task_name=task, native_schema=schema, source_family=family, split="train", task_id="fixture")
    return _project(row, {"path": "synthetic"}, 1)


def test_freesolv_uses_expt_only_and_errorbar_stays_target_metadata():
    audit = {
        "source_row": {"expt": -0.0, "d_expt": 0.6, "calc": 999},
        "label_contract": {
            "default_target_fields": ["expt"],
            "label_evidence": {"expt": "experimental", "d_expt": "experimental_uncertainty_may_be_default"},
        },
    }
    result = project("FreeSolv", {"smiles": "CCO"}, {"expt": -0.0, "unit": "kcal/mol"}, audit)
    assert result.target.reported_value_float64_bits == "8000000000000000"
    assert result.target.native_numeric_lexeme == "-0.0"
    assert result.target.uncertainty_value == 0.6
    assert result.target.uncertainty_kind == "experimental_uncertainty_may_be_default"
    assert result.condition_inputs()["known_context"] == {}
    assert result.condition_inputs()["native_entity"]["representation"] == "CCO"
    assert result.audit()["original_native_task"]["audit"]["source_row"]["calc"] == 999
    assert not result.training_authorized


@pytest.mark.parametrize(("quantity", "source_key", "numeric_key"), [
    ("Km", "substrate", "constant"), ("Vmax", "enzyme", "velocity"),
])
def test_kinetics_preserves_alias_meaning_and_literal_units(quantity, source_key, numeric_key):
    conditions = {"amino_acid_sequence": "AC", "measurement_type": quantity,
                  "substrate_alias": "literal alias", "unit": "source unit"}
    audit = {"original_measurement": {source_key: "literal alias", numeric_key: 0.125, "unit": "source unit"}}
    row = project("UniProt-kinetics", conditions, {"value": 0.125}, audit)
    assert row.condition_inputs()["known_context"] == {
        source_key + "_alias": "literal alias", "measurement_type": quantity, "literal_unit": "source unit",
    }
    assert row.partition_key[2:6] == (quantity, "source unit", None, None)
    assert row.target.kind == "reported_point"


def logd(value, *, relation=None):
    return project(
        "ChEMBL37-AZ-experimental-logD7_4",
        {"canonical_isomeric_smiles": "C", "pH": 7.4, "solvent_pair": ["octan-1-ol", "water"],
         "method": "shake_flask"},
        {"source_reported_logD7_4": value},
        {"source_relation": None, "source_units": None, "source_activity": {
            "assay_chembl_id": "CHEMBL3301363", "assay_type": "P", "type": "LogD7.4",
            "standard_type": "LogD7.4", "standard_flag": 1, "data_validity_comment": None,
            "relation": relation, "standard_relation": None, "units": None, "standard_units": None,
            "value": f"{value:.4f}", "standard_value": str(value),
        }},
    )


def test_logd_null_relation_is_nominal_and_original_decimals_stay_literal():
    row = logd(1.25)
    assert row.target.kind == "reported_nominal_coordinate" and row.target.raw_relation is None
    assert json.loads(row.target.provider_lexemes_json) == {"value": "1.2500", "standard_value": "1.25"}
    assert row.endpoint_definition.unit is None and row.endpoint_definition.unit_basis == "unresolved"
    with pytest.raises(ValueError, match="source relation"):
        replace(row.target, raw_relation="=")


@pytest.mark.parametrize(("value", "relation"), [(-1.5, None), (4.5, None), (1.25, "=")])
def test_logd_boundary_or_relation_changes_are_refused(value, relation):
    with pytest.raises(ValueError):
        logd(value, relation=relation)


def test_explicit_censor_interval_and_missing_keep_distinct_fields():
    upper = ScalarTarget.from_native_cell(NativeCell.parse("<= 3.00"))
    assert upper.kind == "censored_left" and upper.upper == 3.0 and upper.upper_inclusive
    assert upper.reported_value is None and upper.native_numeric_lexeme == "3.00"
    lower = ScalarTarget.from_native_cell(NativeCell.parse("> 1.2"))
    assert lower.kind == "censored_right" and lower.lower == 1.2 and not lower.lower_inclusive
    interval = ScalarTarget.from_native_cell(NativeCell.interval(
        "(1.2,3.00]", lower="1.2", upper="3.00", lower_inclusive=False, upper_inclusive=True,
    ))
    assert (interval.lower, interval.upper, interval.lower_inclusive, interval.upper_inclusive) == (1.2, 3.0, False, True)
    missing = ScalarTarget.from_native_cell(NativeCell.parse(""))
    assert missing.kind == "missing" and not missing.numeric_coordinate_available
    with pytest.raises(ValueError):
        replace(upper, reported_value=0.0)
    with pytest.raises(ValueError):
        replace(lower, lower=True)
