"""Focused tests for exact, inactive JATS summary evidence."""
from decimal import Decimal
from hashlib import sha256
import json

import pytest

from mora3.data.jats_group_summary import GRAIN, SCHEMA, ReviewedJATSGroupSummarySource

def _write_fixture(root, mean_sd, unit, *, unit_source="leaf", group_literal="Ctrl",
                   mismatched_group_row=False, table_id="TabX"):
    raw = ("<article><table-wrap id='TabX'><table><thead><tr><th>Group</th><th>V</th><th>Other</th></tr>"
           "<tr><th>Label</th><th>mM</th><th>V</th></tr></thead><tbody>"
           "<tr><td>Ctrl</td><td>-1.2 ± 0.3</td><td>4.5 ± 1.2</td></tr>"
           "<tr><td>Sham</td><td>V</td><td>2 ± 1</td></tr></tbody></table>"
           "<table-wrap-foot><p>Mean and standard deviation for groups.</p></table-wrap-foot></table-wrap></article>").encode()
    (root / "source.xml").write_bytes(raw)
    original = {"path": "source.xml", "bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    def span(path, text):
        digest = sha256(text.encode()).hexdigest()
        return dict(original_path="source.xml", original_bytes=len(raw), original_sha256=original["sha256"],
                    source_child_ordinal_path=path, slot="text", start_codepoint=0, end_codepoint=len(text),
                    whole_slot_sha256=digest, span_sha256=digest)

    doc = {
        "schema": SCHEMA, "grain": GRAIN, "active_role": None, "training_admission": False,
        "work_group": "synthetic-test-only", "inactive_prospective_role": "test", "original": original,
        "table": table_id,
        "literal_grain_statement": "Mean and standard deviation for groups.",
        "grain_span": span([0, 1, 0], "Mean and standard deviation for groups."),
        "rows": [{
            "source_group_literal": group_literal, "measurement": "concentration", "native_assay": "synthetic",
            "group_cell": {
                "source_child_ordinal_path": [0, 0, 1, 1, 0] if mismatched_group_row else [0, 0, 1, 0, 0],
                "visible_text_sha256": sha256(("Sham" if mismatched_group_row else "Ctrl").encode()).hexdigest(),
            },
            "literal_unit": unit, "numeric_lexemes": mean_sd, "conflict": None,
            "mean_SD_span": span([0, 0, 1, 0, 1], "-1.2 ± 0.3"),
            "unit_span": {
                "leaf": span([0, 0, 0, 1, 1], "mM"),
                "wrong_column": span([0, 0, 0, 1, 2], "V"),
                "wrong_header_row": span([0, 0, 0, 0, 1], "V"),
                "body_row": span([0, 0, 1, 1, 1], "V"),
            }[unit_source],
        }],
    }
    proof = json.dumps(doc, ensure_ascii=False).encode()
    (root / "proof.json").write_bytes(proof)
    return sha256(proof).hexdigest()


def test_signed_summary_lexemes_must_match_original_numeric_span(tmp_path):
    digest = _write_fixture(tmp_path, ["-1.2", "0.4"], "mM")
    with pytest.raises(ValueError, match="standard deviation differs from exact JATS"):
        ReviewedJATSGroupSummarySource(tmp_path, "proof.json", digest)


def test_signed_unit_must_match_original_header(tmp_path):
    digest = _write_fixture(tmp_path, ["-1.2", "0.3"], "V")
    with pytest.raises(ValueError, match="literal assay unit differs"):
        ReviewedJATSGroupSummarySource(tmp_path, "proof.json", digest)


def test_signed_unit_cannot_use_another_column(tmp_path):
    digest = _write_fixture(tmp_path, ["-1.2", "0.3"], "V", unit_source="wrong_column")
    with pytest.raises(ValueError, match="own leaf header column"):
        ReviewedJATSGroupSummarySource(tmp_path, "proof.json", digest)


def test_generic_negative_decimal_mean_is_exact(tmp_path):
    digest = _write_fixture(tmp_path, ["-1.2", "0.3"], "mM")
    src = ReviewedJATSGroupSummarySource(tmp_path, "proof.json", digest)
    (row,) = src.rows
    assert src.require_uncontested(row).mean == Decimal("-1.2")
    assert row.standard_deviation == Decimal("0.3")


def test_signed_group_label_must_match_native_group_cell(tmp_path):
    digest = _write_fixture(tmp_path, ["-1.2", "0.3"], "mM", group_literal="Sham")
    with pytest.raises(ValueError, match="reported group differs from exact JATS row label"):
        ReviewedJATSGroupSummarySource(tmp_path, "proof.json", digest)


def test_signed_group_cell_cannot_be_borrowed_from_another_row(tmp_path):
    digest = _write_fixture(tmp_path, ["-1.2", "0.3"], "mM", group_literal="Sham", mismatched_group_row=True)
    with pytest.raises(ValueError, match="own table, group row, header and table foot"):
        ReviewedJATSGroupSummarySource(tmp_path, "proof.json", digest)


@pytest.mark.parametrize("unit_source", ["wrong_header_row", "body_row"])
def test_signed_same_column_unit_requires_actual_leaf_header_row(tmp_path, unit_source):
    digest = _write_fixture(tmp_path, ["-1.2", "0.3"], "V", unit_source=unit_source)
    with pytest.raises(ValueError, match="exact JATS table body, leaf header and summary foot"):
        ReviewedJATSGroupSummarySource(tmp_path, "proof.json", digest)


def test_signed_table_id_must_match_original_wrapper(tmp_path):
    digest = _write_fixture(tmp_path, ["-1.2", "0.3"], "mM", table_id="DifferentTable")
    with pytest.raises(ValueError, match="exact JATS table body, leaf header and summary foot"):
        ReviewedJATSGroupSummarySource(tmp_path, "proof.json", digest)
