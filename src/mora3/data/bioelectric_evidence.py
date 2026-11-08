"""Native bioelectric study roles, target-free inputs, and original evidence spans.

This is a source/task contract. It never turns text mentions, simulations, or
research hypotheses into observed outcomes or authorizes a source or a split.
"""
from dataclasses import dataclass
from decimal import Decimal
from io import BytesIO
from pathlib import Path
from xml.etree import ElementTree
import hashlib
import json
import math
import re
import struct
import zipfile

from .source_scalar_native_rows import ScalarTarget


def _literal(value):
    if type(value) is not str or not value or not value.strip():
        raise ValueError('explicit nonempty native literal required')
    value.encode('utf8')
    return value


def _maybe(value):
    return None if value is None else _literal(value)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('exact source SHA256 required')


def _wire(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')


@dataclass(frozen=True)
class NativeBioelectricTime:
    value: float
    literal_unit: str
    event_reference: str

    def binding(self):
        if type(self.value) is not float or not math.isfinite(self.value):
            raise ValueError('known native time needs a finite Float64 coordinate')
        return dict(value=self.value,literal_unit=_literal(self.literal_unit),event_reference=_literal(self.event_reference))


@dataclass(frozen=True)
class NativeBioelectricBaseline:
    organism: str
    anatomical_system: str
    cell_or_tissue: str | None
    developmental_or_age_state: str | None
    injury_or_disease_state: str | None
    native_context: str | None = None

    def binding(self):
        return dict(organism=_literal(self.organism),anatomical_system=_literal(self.anatomical_system),
            cell_or_tissue=_maybe(self.cell_or_tissue),developmental_or_age_state=_maybe(self.developmental_or_age_state),
            injury_or_disease_state=_maybe(self.injury_or_disease_state),native_context=_maybe(self.native_context))


ACTION_TYPES=frozenset(('block','inhibit','open','modify_opening_rate','modify_closing_rate',
    'modify_conductance','modify_permeability','modify_expression','modify_trafficking',
    'alter_ionic_environment','alter_membrane_voltage','alter_junction_coupling','propose_linkage',
    'vehicle_control','sham_control','untreated_control','other_reported_perturbation'))
TARGET_TYPES=frozenset(('ion_channel','gap_junction','hemichannel','pump','transporter',
    'tissue_electrical_state','ionic_environment','unknown','not_applicable'))


@dataclass(frozen=True)
class NativeBioelectricEvent:
    action: str
    native_modality: str
    target_kind: str
    native_targets: tuple
    tool_or_construct: str | None
    site: str | None
    onset: NativeBioelectricTime | None
    duration: NativeBioelectricTime | None
    intended_coupling_partners: tuple | None = None

    def binding(self):
        if (self.action not in ACTION_TYPES or self.target_kind not in TARGET_TYPES
                or type(self.native_targets) is not tuple
                or self.target_kind in ('unknown','not_applicable') and self.native_targets):
            raise ValueError('explicit known action/target role or unavailable target required')
        if self.intended_coupling_partners is not None and (
                type(self.intended_coupling_partners) is not tuple or len(self.intended_coupling_partners)!=2
                or self.target_kind not in ('gap_junction','hemichannel')):
            raise ValueError('exact ordered pair intended for junction linkage required')
        for t in (self.onset,self.duration):
            if t is not None and type(t) is not NativeBioelectricTime:raise ValueError('exact native time or unknown required')
        return dict(action=self.action,native_modality=_literal(self.native_modality),target_kind=self.target_kind,
            native_targets=tuple(_literal(x)for x in self.native_targets),tool_or_construct=_maybe(self.tool_or_construct),
            site=_maybe(self.site),onset=None if self.onset is None else self.onset.binding(),
            duration=None if self.duration is None else self.duration.binding(),
            intended_coupling_partners=None if self.intended_coupling_partners is None else tuple(
                _literal(x) for x in self.intended_coupling_partners))


@dataclass(frozen=True)
class NativeBioelectricEndpoint:
    measurement: str
    native_assay: str
    literal_unit: str | None
    coordinate: str
    time: NativeBioelectricTime | None
    category_axis: tuple = ()

    def binding(self):
        if self.coordinate not in ('native_scalar','native_finite_category','native_functional_coupling_class'):
            raise ValueError('separate native measurement coordinate required')
        if type(self.category_axis) is not tuple or (bool(self.category_axis)==(self.coordinate=='native_scalar')):
            raise ValueError('literal categorical axis or native scalar coordinate required')
        axis=tuple(_literal(x)for x in self.category_axis)
        if len(set(axis))!=len(axis):raise ValueError('unique literal source outcome categories required')
        if self.time is not None and type(self.time) is not NativeBioelectricTime:raise ValueError('exact source measurement time required')
        return dict(measurement=_literal(self.measurement),native_assay=_literal(self.native_assay),
            literal_unit=_maybe(self.literal_unit),coordinate=self.coordinate,category_axis=axis,
            time=None if self.time is None else self.time.binding())


@dataclass(frozen=True)
class BioelectricJATSEvidenceSpan:
    original_path: str
    original_bytes: int
    original_sha256: str
    source_child_ordinal_path: tuple
    slot: str
    start_codepoint: int
    end_codepoint: int
    whole_slot_sha256: str
    span_sha256: str

    def validate(self):
        _literal(self.original_path)
        for digest in (self.original_sha256,self.whole_slot_sha256,self.span_sha256):_digest(digest)
        if (type(self.original_bytes) is not int or not 0<self.original_bytes<=16*2**20
                or type(self.source_child_ordinal_path) is not tuple
                or any(type(i) is not int or i<0 for i in self.source_child_ordinal_path)
                or self.slot not in ('text','tail')
                or type(self.start_codepoint) is not int or type(self.end_codepoint) is not int
                or not 0<=self.start_codepoint<self.end_codepoint):
            raise ValueError('exact bounded original JATS slot and nonempty Unicode span required')

    def read(self,root):
        self.validate()
        from defusedxml.ElementTree import DefusedXMLParser
        root=Path(root).resolve();p=(root/self.original_path).resolve();p.relative_to(root)
        with p.open('rb')as f:raw=f.read(self.original_bytes+1)
        if len(raw)!=self.original_bytes or _sha(raw)!=self.original_sha256:
            raise ValueError('complete immutable original evidence bytes differ')
        parser=DefusedXMLParser(target=ElementTree.TreeBuilder(insert_comments=True,insert_pis=True),
            forbid_dtd=False,forbid_entities=True,forbid_external=True)
        parser.feed(raw);node=parser.close()
        for i in self.source_child_ordinal_path:
            if i>=len(node):raise ValueError('evidence XML child ordinal outside original')
            node=node[i]
        value=getattr(node,self.slot)
        if (type(value) is not str or self.slot=='text' and not isinstance(node.tag,str)
                or self.end_codepoint>len(value) or _sha(value.encode('utf8'))!=self.whole_slot_sha256):
            raise ValueError('native character-data slot or bounds differ')
        span=value[self.start_codepoint:self.end_codepoint]
        if _sha(span.encode('utf8'))!=self.span_sha256:raise ValueError('evidence span fingerprint differs')
        return span


@dataclass(frozen=True)
class BioelectricWorkbookNumericCell:
    """An exact original Open XML numeric coordinate; never a formula cache."""
    original_path: str
    original_bytes: int
    original_sha256: str
    native_sheet_member: str
    native_sheet_sha256: str
    cell_coordinate: str
    native_cell_type: str | None
    literal_decimal: str

    def validate(self):
        _literal(self.original_path);_digest(self.original_sha256);_digest(self.native_sheet_sha256)
        if (type(self.original_bytes) is not int or not 0<self.original_bytes<=32*2**20
                or type(self.native_sheet_member) is not str or not re.fullmatch(r'xl/worksheets/sheet[1-9][0-9]*\.xml',self.native_sheet_member)
                or type(self.cell_coordinate) is not str or not re.fullmatch(r'[A-Z]+[1-9][0-9]*',self.cell_coordinate)
                or self.native_cell_type not in (None,'n')
                or type(self.literal_decimal) is not str
                or not re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',self.literal_decimal)):
            raise ValueError('exact original numeric workbook cell evidence required')

    def read(self,root):
        from defusedxml.ElementTree import fromstring
        self.validate();root=Path(root).resolve();p=(root/self.original_path).resolve();p.relative_to(root)
        with p.open('rb') as f:raw=f.read(self.original_bytes+1)
        if len(raw)!=self.original_bytes or _sha(raw)!=self.original_sha256:
            raise ValueError('complete immutable original workbook bytes differ')
        with zipfile.ZipFile(BytesIO(raw)) as package:
            entries=[m for m in package.infolist() if m.filename==self.native_sheet_member]
            if len(entries)!=1 or entries[0].file_size>32*2**20:raise ValueError('one bounded native worksheet member required')
            sheet=package.read(entries[0])
        if _sha(sheet)!=self.native_sheet_sha256:raise ValueError('exact original worksheet member differs')
        ns='{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        root=fromstring(sheet,forbid_dtd=True,forbid_entities=True,forbid_external=True)
        cells=[c for c in root.findall(f'{ns}sheetData/{ns}row/{ns}c') if c.get('r')==self.cell_coordinate]
        if (len(cells)!=1 or cells[0].get('t')!=self.native_cell_type or len(cells[0])!=1
                or cells[0][0].tag!=ns+'v' or len(cells[0][0])!=0
                or cells[0][0].text!=self.literal_decimal):
            raise ValueError('raw native numeric cell differs or contains a formula/unknown payload')
        original=Decimal(self.literal_decimal);value=float(original)
        if not math.isfinite(value) or original!=0 and value==0:
            raise ValueError('native literal outside finite Float64 model coordinate')
        return value


@dataclass(frozen=True)
class NativeBioelectricObservation:
    baseline: NativeBioelectricBaseline
    interventions: tuple
    endpoint: NativeBioelectricEndpoint
    evidence_class: str
    control_evidence: str
    exact_work_group: str
    exact_experiment_parent: str | None
    original_version: str
    evidence_spans: tuple
    numeric_target: ScalarTarget | None = None
    category_target: str | None = None

    def condition_inputs(self):
        """Only known assay, subject, intended interventions, and query coordinates."""
        if (type(self.baseline) is not NativeBioelectricBaseline or type(self.endpoint) is not NativeBioelectricEndpoint
                or type(self.interventions) is not tuple or not self.interventions
                or any(type(e) is not NativeBioelectricEvent for e in self.interventions)):
            raise ValueError('exact native baseline, intervention sequence, and measurement query required')
        return dict(task='native_bioelectric_perturbation_response',baseline=self.baseline.binding(),
            interventions=tuple(e.binding()for e in self.interventions),endpoint=self.endpoint.binding())

    def target(self):
        """Observed-source target only. No data split or scientific admission."""
        self.condition_inputs();_literal(self.exact_work_group);_maybe(self.exact_experiment_parent);_literal(self.original_version)
        if (self.evidence_class not in ('primary_observation','primary_reported_contrast','simulation','review','hypothesis')
                or self.control_evidence not in ('randomized','paired','reported_comparator','none','unresolved')
                or type(self.evidence_spans) is not tuple or not self.evidence_spans
                or any(type(e) not in (BioelectricJATSEvidenceSpan,BioelectricWorkbookNumericCell) for e in self.evidence_spans)):
            raise ValueError('literal evidence category, control status, and exact source spans required')
        for span in self.evidence_spans:span.validate()
        if self.evidence_class not in ('primary_observation','primary_reported_contrast'):
            raise ValueError('simulations, reviews and hypotheses are not observed outcomes')
        if self.evidence_class=='primary_reported_contrast' and self.control_evidence in ('none','unresolved'):
            raise ValueError('a reported contrast needs an explicit source comparator')
        if any(event.action=='propose_linkage' for event in self.interventions):
            raise ValueError('a proposed linkage is not a measured junction intervention')
        if self.endpoint.coordinate=='native_scalar':
            if self.category_target is not None or type(self.numeric_target) is not ScalarTarget:
                raise ValueError('one typed native scalar observation or missing coordinate required')
            self.numeric_target.__post_init__()
            return self.numeric_target if self.numeric_target.numeric_coordinate_available else None
        if self.numeric_target is not None or self.category_target is not None and self.category_target not in self.endpoint.category_axis:
            raise ValueError('only literal observed outcome categories; unknown is not negative')
        return self.category_target

    def verify_evidence(self,root):
        """Check original evidence and the exact typed numeric point where applicable."""
        target=self.target();numeric=[]
        for span in self.evidence_spans:
            value=span.read(root)
            if type(span) is BioelectricWorkbookNumericCell:numeric.append((span,value))
        if numeric:
            if len(numeric)!=1 or type(target) is not ScalarTarget or target.kind!='reported_point':
                raise ValueError('one original numeric workbook cell must bind one reported point')
            span,value=numeric[0]
            if (span.literal_decimal!=target.native_numeric_lexeme
                    or struct.pack('>d',value)!=struct.pack('>d',target.reported_value)):
                raise ValueError('source workbook point, literal or Float64 bits changed')
        return target

    def condition_sha256(self):
        return _sha(_wire(self.condition_inputs()))
