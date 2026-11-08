"""Reviewed data-side JATS group summaries. This module does not issue model targets."""
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from defusedxml.ElementTree import DefusedXMLParser

from mora3.data.bioelectric_evidence import BioelectricJATSEvidenceSpan, NativeBioelectricEndpoint

SCHEMA = "Mora3.ReviewedJATSMeanAndSD/v2"
GRAIN = "published_group_mean_and_standard_deviation"
NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
PAIR = re.compile(rf"({NUMBER})\s*±\s*({NUMBER})")


def _stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def _span(raw, original):
    span = BioelectricJATSEvidenceSpan(
        **{**raw, "source_child_ordinal_path": tuple(raw["source_child_ordinal_path"])}
    )
    if (span.original_path, span.original_bytes, span.original_sha256) != (
        original["path"], original["bytes"], original["sha256"]
    ):
        raise ValueError("summary evidence must share the exact original")
    return span


def _numeric_pair(span, root, expected):
    text = span.read(root)
    match = PAIR.fullmatch(text)
    if match is None or match.groups() != tuple(expected):
        raise ValueError("mean or standard deviation differs from exact JATS")
    mean, sd = map(Decimal, match.groups())
    if not mean.is_finite() or not sd.is_finite() or sd < 0:
        raise ValueError("finite mean and nonnegative standard deviation required")
    return mean, sd


def _node(document, path):
    for ordinal in path:
        if type(ordinal) is not int or not 0 <= ordinal < len(document):
            raise ValueError("structural JATS evidence path lies outside the original")
        document = document[ordinal]
    return document


def _require_native_group_and_header(document, table_id, grain, pair, unit, group):
    number_path = pair.source_child_ordinal_path
    unit_path = unit.source_child_ordinal_path
    group_path = tuple(group["source_child_ordinal_path"])
    if (
        len(number_path) < 5 or len(unit_path) != len(number_path)
        or number_path[:-3] != unit_path[:-3]
        or number_path[:-1] != group_path[:-1]
        or grain.source_child_ordinal_path[:-2] != number_path[:-4]
        or pair.slot != "text" or unit.slot != "text"
    ):
        raise ValueError("summary must bind to its own table, group row, header and table foot")
    wrapper = _node(document, number_path[:-4])
    table = _node(document, number_path[:-3])
    body = _node(document, number_path[:-2])
    header = _node(document, unit_path[:-2])
    row = _node(document, number_path[:-1])
    header_row = _node(document, unit_path[:-1])
    value_cell = _node(document, number_path)
    group_cell = _node(document, group_path)
    unit_cell = _node(document, unit_path)
    foot = _node(document, grain.source_child_ordinal_path[:-1])
    foot_item = _node(document, grain.source_child_ordinal_path)
    if (
        wrapper.tag != "table-wrap" or wrapper.get("id") != table_id or table.tag != "table"
        or body.tag != "tbody" or row.tag != "tr" or value_cell.tag != "td"
        or group_cell.tag != "td" or header.tag != "thead" or header_row.tag != "tr"
        or unit_cell.tag != "th" or header.findall("tr")[-1] is not header_row
        or foot.tag != "table-wrap-foot" or foot_item.tag != "p"
    ):
        raise ValueError("exact JATS table body, leaf header and summary foot required")
    body_cells = row.findall("td")
    header_cells = header_row.findall("th")
    if (
        len(body_cells) != len(header_cells) or body_cells[0] is not group_cell
        or value_cell is group_cell or body_cells.index(value_cell) != header_cells.index(unit_cell)
        or any(c.get("colspan", "1") != "1" or c.get("rowspan", "1") != "1"
               for c in body_cells + header_cells)
    ):
        raise ValueError("summary must bind to the first group cell and its own leaf header column")
    group_literal = "".join(group_cell.itertext())
    if sha256(group_literal.encode()).hexdigest() != group["visible_text_sha256"]:
        raise ValueError("group label fingerprint differs from exact JATS")
    return group_literal


@dataclass(frozen=True)
class NativeJATSGroupSummary:
    numeric_slot_path: tuple[int, ...]
    source_group_literal: str
    endpoint: NativeBioelectricEndpoint
    mean: Decimal
    standard_deviation: Decimal
    numeric_lexemes: tuple[str, str]
    grain: str
    conflict_status: str
    conflicting_numeric_lexemes: tuple[str, str] | None
    work_group: str
    inactive_prospective_role: str


class ReviewedJATSGroupSummarySource:
    """Issue an exact, inactive group summary view from a pinned review manifest.

    The manifest is the authority for the conflict census. The original JATS
    is the authority for the values and native unit on every issued row.
    """

    def __init__(self, root, manifest_path, expected_sha256):
        root = Path(root).resolve()
        manifest_path = (root / manifest_path).resolve()
        manifest_path.relative_to(root)
        manifest_stamp = _stamp(manifest_path)
        with manifest_path.open("rb") as f:
            raw = f.read(2**20 + 1)
        if len(raw) > 2**20 or sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("exact reviewed summary manifest required")
        doc = json.loads(raw)
        if (
            doc["schema"] != SCHEMA or doc["grain"] != GRAIN
            or doc["active_role"] is not None or doc["training_admission"] is not False
            or doc["inactive_prospective_role"] not in ("train", "validation", "test", "evaluation_only")
            or type(doc["work_group"]) is not str or not doc["work_group"].strip()
            or type(doc["table"]) is not str or not doc["table"].strip()
        ):
            raise ValueError("an inactive exact reviewed group-summary scope is required")
        original = doc["original"]
        original_path = (root / original["path"]).resolve()
        original_path.relative_to(root)
        original_stamp = _stamp(original_path)
        grain_span = _span(doc["grain_span"], original)
        grain = grain_span.read(root)
        if grain != doc["literal_grain_statement"] or not grain.strip():
            raise ValueError("source summary-statistic declaration differs")

        with original_path.open("rb") as f:
            xml = f.read(grain_span.original_bytes + 1)
        if len(xml) != original["bytes"] or sha256(xml).hexdigest() != original["sha256"]:
            raise ValueError("exact original JATS structure bytes required")
        parser = DefusedXMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True),
                                  forbid_dtd=False, forbid_entities=True, forbid_external=True)
        parser.feed(xml)
        document = parser.close()

        rows = {}
        for record in doc["rows"]:
            pair = _span(record["mean_SD_span"], original)
            mean, sd = _numeric_pair(pair, root, record["numeric_lexemes"])
            endpoint = NativeBioelectricEndpoint(
                record["measurement"], record["native_assay"], record["literal_unit"], "native_scalar", None
            )
            endpoint.binding()
            unit = _span(record["unit_span"], original)
            group_literal = _require_native_group_and_header(
                document, doc["table"], grain_span, pair, unit, record["group_cell"]
            )
            if group_literal != record["source_group_literal"]:
                raise ValueError("reported group differs from exact JATS row label")
            if unit.read(root) != endpoint.literal_unit:
                raise ValueError("literal assay unit differs from exact JATS header")
            conflict = record["conflict"]
            alternative_lexemes = None
            if conflict is not None:
                alternative_lexemes = tuple(conflict["numeric_lexemes"])
                alt_mean, alt_sd = _numeric_pair(_span(conflict["mean_SD_span"], original), root, alternative_lexemes)
                if (alt_mean, alt_sd) == (mean, sd):
                    raise ValueError("reviewed numeric conflict must contain a differing source value")
            coordinate = pair.source_child_ordinal_path
            if coordinate in rows:
                raise ValueError("duplicate native group summary coordinate")
            rows[coordinate] = NativeJATSGroupSummary(
                coordinate, group_literal, endpoint, mean, sd,
                tuple(record["numeric_lexemes"]), GRAIN, "unresolved" if conflict is not None else "none_recorded",
                alternative_lexemes, doc["work_group"], doc["inactive_prospective_role"],
            )
        if not rows:
            raise ValueError("reviewed native group summaries required")
        self._rows = rows
        self._files = ((manifest_path, manifest_stamp), (original_path, original_stamp))
        self._verify_files()

    def _verify_files(self):
        if any(_stamp(p) != stamp for p, stamp in self._files):
            raise ValueError("reviewed summary evidence changed after authentication")

    @property
    def rows(self):
        self._verify_files()
        return tuple(self._rows.values())

    def require_uncontested(self, candidate):
        """Authenticate one review-only row and withhold unresolved conflicts."""
        self._verify_files()
        if type(candidate) is not NativeJATSGroupSummary or self._rows.get(candidate.numeric_slot_path) != candidate:
            raise ValueError("exact issued group summary required")
        if candidate.conflict_status != "none_recorded":
            raise ValueError("reviewed source conflict remains unresolved")
        return candidate
