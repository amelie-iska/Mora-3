"""Safe, source-bound ProCyon ORIGINAL relation/sequence candidates.

This is not a corpus task admission or neural/text objective. Opaque metadata
are never opened or deserialized. Native IDs/index transformations and official
splits remain audit-only; only complete known native sequence enters projection.
"""
import csv
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import re

REVISION = "2b99a397917282b3184d616bf01825e340488c88"
HANDOFF_PATH = "outputs/data-sources/20261101-ProCyon-source-v1/handoff-ProCyon-open-native-components-v1.json"
HANDOFF_SHA256 = "0cc11e0d7be06d59c0ffbeea4f8d5a3eb38631258af4892508c513d488cc43ef"
QC_SHA256 = "e6304ee6b187692e0ea5913d9ff4ddac63169615ccb8cdda99e974f43673f2f0"
TRANSFORMATION_SHA256 = "0f900c0afa122446f05735d8f578d7366d0b2c8bd1e7874d50caffe6562a810f"
LICENSE_SHA256 = "dc2f92509076dd07c29fc8b7596ffbfb21d1f8dcf36ac7627735d575bc8438c4"
PROOF_BINDING = dict(handoff_sha256=HANDOFF_SHA256, QC_sha256=QC_SHA256,
    transformation_sha256=TRANSFORMATION_SHA256, license_sha256=LICENSE_SHA256)
FAMILY_PATHS = {
    "protein_GO": "integrated_data/v1/protein_go/sample_aware_ontology_go_centric/protein_go_relations.unified.csv",
    "protein_Reactome": "integrated_data/v1/protein_reactome/random_reactome_centric/protein_reactome_relations.unified.csv",
    "domain_GO": "integrated_data/v1/domain_go/sample_aware_ontology_go_centric/domain_go_relations.unified.csv",
    "domain_Pfam": "integrated_data/v1/domain_pfam/clan_aware_pfam_centric/domain_pfam_relations.unified.csv",
    "protein_GtoP": "integrated_data/v1/protein_gtop/random_gtop_centric/protein_gtop_relations.unified.csv",
}
RIGHTS = dict(protein_GO=("UniProt CC-BY-4.0", "GO CC-BY-4.0"),
    protein_Reactome=("UniProt CC-BY-4.0", "Reactome CC0"),
    domain_GO=("UniProt CC-BY-4.0", "Pfam CC0", "GO CC-BY-4.0"),
    domain_Pfam=("UniProt CC-BY-4.0", "Pfam CC0",),
    protein_GtoP=("UniProt CC-BY-4.0", "GtoP ODbL database", "GtoP content CC-BY-SA-4.0"))
OFFICIAL_SPLITS = ("CL_train", "CL_val_five_shot", "CL_val_two_shot", "CL_val_pt_ft",
    "CL_val_zero_shot", "eval_five_shot", "eval_two_shot", "eval_pt_ft", "eval_zero_shot", "eval_zero_shot_hard")
EVIDENCE_ROLE = "original_source_database_relation_not_independent_experimental_measurement"


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hex(value):
    return type(value) is str and len(value)==64 and not set(value)-set("0123456789abcdef")


def _typed(value):
    if type(value) is bytes:return {"bytes_hex":value.hex()}
    if type(value) is dict:return {k:_typed(v) for k,v in value.items()}
    if type(value) is tuple:return {"tuple":[_typed(v) for v in value]}
    if type(value) is list:return [_typed(v) for v in value]
    return value


def _json_bytes(raw):
    if type(raw) is not bytes:raise ValueError("immutable canonical native field bytes required")
    value=json.loads(raw)
    if _canonical(value)!=raw:raise ValueError("exact canonical native field bytes required")
    return value


def _indices(values):
    if (type(values) is not tuple or not 1<=len(values)<=32
            or any(type(i) is not int or i<0 for i in values) or len(set(values))!=len(values)):
        raise ValueError("1..32 unique exact nonnegative integer native ordinals required")


@dataclass(frozen=True)
class NativeFileRef:
    path: str
    bytes: int
    sha256: str

    def __post_init__(self):
        if (type(self.path) is not str or not self.path or type(self.bytes) is not int
                or self.bytes<=0 or not _hex(self.sha256)):
            raise ValueError("typed exact native file path/bytes/SHA256 required")

    def as_mapping(self):
        return asdict(self)


@dataclass(frozen=True)
class RelationSelector:
    family: str
    row_indices: tuple

    def __post_init__(self):
        if type(self.family) is not str or not self.family:raise ValueError("explicit original relation family required")
        _indices(self.row_indices)


@dataclass(frozen=True)
class SequenceSelector:
    entity_kind: str
    native_ordinals: tuple

    def __post_init__(self):
        if self.entity_kind not in ("protein","domain"):raise ValueError("native protein/domain FASTA only")
        _indices(self.native_ordinals)


def _record_sequence(raw, header):
    if type(raw) is not bytes or type(header) is not bytes:
        raise ValueError("complete literal native FASTA record/header bytes required")
    lines=raw.splitlines(keepends=True)
    def end(line):
        return line[:-2] if line.endswith(b"\r\n") else line[:-1] if line.endswith(b"\n") else line
    if not lines or end(lines[0])!=header or not header.startswith(b">"):
        raise ValueError("native FASTA header/record mismatch")
    seq=b"".join(end(line) for line in lines[1:])
    if not seq or any(not (65<=x<=90 or 97<=x<=122 or x in b"*-?") for x in seq):
        raise ValueError("native FASTA axis contains unsupported non-sequence bytes; no replacement")
    return seq


@dataclass(frozen=True)
class NativeSequenceCandidate:
    entity_kind: str
    native_ordinal: int
    native_id: str
    native_header_bytes: bytes
    sequence_bytes: bytes
    raw_record_bytes: bytes
    source_ref: NativeFileRef
    payload_ref: NativeFileRef
    proof_binding_bytes: bytes

    def __post_init__(self):
        if (self.entity_kind not in ("protein","domain") or type(self.native_ordinal) is not int
                or self.native_ordinal<0 or type(self.native_id) is not str or not self.native_id
                or type(self.sequence_bytes) is not bytes
                or type(self.source_ref) is not NativeFileRef or type(self.payload_ref) is not NativeFileRef
                or _json_bytes(self.proof_binding_bytes)!=PROOF_BINDING
                or _record_sequence(self.raw_record_bytes,self.native_header_bytes)!=self.sequence_bytes
                or self.native_header_bytes[1:].split(maxsplit=1)[0].decode()!=self.native_id):
            raise ValueError("exact typed native FASTA sequence/header/axis/source binding required")

    @property
    def parent_protein_id(self):
        return None

    @property
    def parent_coordinates(self):
        return None

    @property
    def binding_sha256(self):
        return _sha(_canonical(_typed(asdict(self))))

    def native_observed_mask(self):
        return (True,)*len(self.sequence_bytes)

    def audit(self):
        return dict(entity_kind=self.entity_kind,native_ordinal=self.native_ordinal,native_id=self.native_id,
            source_ref=self.source_ref.as_mapping(),payload_ref=self.payload_ref.as_mapping(),
            header_sha256=_sha(self.native_header_bytes),sequence_sha256=_sha(self.sequence_bytes),
            raw_record_sha256=_sha(self.raw_record_bytes),native_axis_length=len(self.sequence_bytes),
            internal_index_asserted=False,parent_protein_id=None,parent_coordinates=None,
            dataset_revision=REVISION,proof_binding=_json_bytes(self.proof_binding_bytes),
            current_UniProt_qualification_verified=False,task_admitted=False)


def _raw_csv_fields(header, row):
    if type(header) is not bytes or type(row) is not bytes:raise ValueError("literal CSV bytes required")
    reader=csv.DictReader(io.StringIO((header+row).decode("utf8"),newline=""),strict=True)
    values=list(reader)
    if len(values)!=1 or any(type(k) is not str or type(v) is not str for k,v in values[0].items()):
        raise ValueError("one complete literal CSV source record required")
    return values[0]


@dataclass(frozen=True)
class NativeRelationCandidate:
    family: str
    source_row_index: int
    source_ref: NativeFileRef
    indexed_source_ref: NativeFileRef
    mapping_ref: NativeFileRef
    raw_csv_header: bytes
    raw_csv_row: bytes
    native_fields_bytes: bytes
    indexed_fields_bytes: bytes
    native_internal_index: str
    sequence: NativeSequenceCandidate
    proof_binding_bytes: bytes
    evidence_role: str = EVIDENCE_ROLE

    def __post_init__(self):
        fields=_json_bytes(self.native_fields_bytes)
        indexed=_json_bytes(self.indexed_fields_bytes)
        if (self.family not in FAMILY_PATHS or type(self.source_row_index) is not int or self.source_row_index<0
                or any(type(v) is not NativeFileRef for v in (self.source_ref,self.indexed_source_ref,self.mapping_ref))
                or type(self.sequence) is not NativeSequenceCandidate
                or self.evidence_role!=EVIDENCE_ROLE
                or _json_bytes(self.proof_binding_bytes)!=PROOF_BINDING
                or type(fields) is not dict or type(indexed) is not dict
                or any(type(v) is not str for v in (*fields.values(),*indexed.values()))
                or _raw_csv_fields(self.raw_csv_header,self.raw_csv_row)!=fields
                or fields.get("seq_id")!=self.sequence.native_id
                or fields.get("seq_type")!=self.sequence.entity_kind
                or fields.get("split") not in OFFICIAL_SPLITS
                or fields.get("split")!=indexed.get("split")
                or fields.get("seq_type")!=indexed.get("seq_type")
                or type(self.native_internal_index) is not str
                or not re.fullmatch(r"0|[1-9][0-9]*",self.native_internal_index)
                or self.native_internal_index!=indexed.get("seq_id")
                or int(self.native_internal_index)!=self.sequence.native_ordinal):
            raise ValueError("exact original relation/FASTA ID-index/split/evidence binding required")
        shared=set(fields)&set(indexed)-{"seq_id","text_id","relation","text_type"}
        if any(fields[k]!=indexed[k] for k in shared):raise ValueError("native original/indexed shared context mismatch")
        if not self.relation_assertions:raise ValueError("literal original source relation required")

    def native_fields(self):
        return _json_bytes(self.native_fields_bytes)

    def indexed_fields(self):
        return _json_bytes(self.indexed_fields_bytes)

    @property
    def status(self):
        return "native_source_candidate_unadmitted"

    @property
    def native_split(self):
        return self.native_fields()["split"]

    @property
    def relation_assertions(self):
        literal=self.native_fields()["relation"]
        if self.family=="protein_GO":
            # Inert native list-of-identifiers grammar; preserve order/multiplicity.
            if not re.fullmatch(r"\['[A-Za-z_]+'(?:, '[A-Za-z_]+')*\]",literal):
                raise ValueError("unsupported native GO relation literal")
            return tuple(re.findall(r"'([A-Za-z_]+)'",literal))
        return (literal,) if literal else ()

    @property
    def gates(self):
        base=("no_verified_native_task_admission","current_UniProt_sequence_release_qualification_pending",
            "opaque_text_and_instruction_parent_metadata","cross_corpus_homology_and_phenotype_split_audit_pending")
        return base+(("domain_parent_coordinates_unresolved",) if self.sequence.entity_kind=="domain" else ())

    @property
    def binding_sha256(self):
        return _sha(_canonical(_typed(asdict(self))))

    def prediction_inputs(self):
        return procyon_prediction_inputs(self.sequence)

    def audit(self):
        fields=self.native_fields()
        return dict(schema="mora3.ProCyonNativeCandidate/v1",status=self.status,dataset_revision=REVISION,
            family=self.family,source_row_index=self.source_row_index,native_split=self.native_split,
            source_ref=self.source_ref.as_mapping(),indexed_source_ref=self.indexed_source_ref.as_mapping(),
            mapping_ref=self.mapping_ref.as_mapping(),native_fields=fields,indexed_fields=self.indexed_fields(),
            native_internal_index=self.native_internal_index,sequence=self.sequence.audit(),
            source_native_cell_present_mask={k:v!="" for k,v in fields.items()},
            unlisted_relations="unknown_not_negative",negative_labels_available=False,
            native_source_row_multiplicity=1,independent_experimental_measurement_count=None,
            original_relation_assertions=self.relation_assertions,evidence_role=self.evidence_role,
            indexed_relation_role="source_transformed_index_not_global_relation2id",
            instruction_parent=None,LLM_rephrase_available=False,phenotype_description_available=False,
            current_UniProt_qualification_verified=False,source_rights=RIGHTS[self.family],
            author_processing_rights="Apache-2.0 separate from original component rights",
            gates=self.gates,proof_binding=_json_bytes(self.proof_binding_bytes),binding_sha256=self.binding_sha256)

    def require_model_use(self,purpose,*,admission=None):
        if purpose=="training" and self.native_split!="CL_train":
            raise ValueError("official heldout source split cannot enter training; no task admission")
        raise ValueError("native candidate has no verified constructor/model task admission; model targets unavailable")


def procyon_prediction_inputs(sequence,*,request=None):
    """Candidate input projection only; does not construct an admitted task."""
    if type(sequence) is not NativeSequenceCandidate:raise ValueError("typed full native sequence candidate required")
    if request is not None and request!={"query_role":"known_sequence_for_candidate_relation_query"}:
        raise ValueError("exact target-free query allowlist; no target/gold/audit inputs")
    return dict(entity_kind=sequence.entity_kind,native_sequence=sequence.sequence_bytes.decode("ascii"),
        query_role="known_sequence_for_candidate_relation_query")


@dataclass(frozen=True)
class DesiredRelation:
    namespace: str
    native_identifier: str
    role: str = "desired"

    def __post_init__(self):
        if (self.namespace not in ("GO","Reactome","Pfam","GtoP") or self.role!="desired"
                or type(self.native_identifier) is not str or not self.native_identifier):
            raise ValueError("explicit desired native relation identifier required; never an observed label")


def desired_procyon_inputs(conditions):
    if (type(conditions) is not tuple or not 1<=len(conditions)<=32
            or any(type(c) is not DesiredRelation for c in conditions) or len(set(conditions))!=len(conditions)):
        raise ValueError("unique explicit typed desired conditions required")
    return dict(desired_source_relations=[asdict(c) for c in conditions])


class ProCyonNativeSource:
    """Read the pinned open components as native candidates, never task admission."""
    def __init__(self,*,root,handoff=HANDOFF_PATH,expected_handoff_sha256=HANDOFF_SHA256):
        self.root=Path(root).resolve()
        self.handoff=Path(handoff)
        if expected_handoff_sha256!=HANDOFF_SHA256:raise ValueError("exact pinned source handoff required")
        self._snapshot()

    def _bytes(self,reference,suffixes):
        ref=reference if type(reference) is NativeFileRef else NativeFileRef(**{k:reference[k] for k in ("path","bytes","sha256")})
        p=(self.root/ref.path).resolve()
        if (not p.is_relative_to(self.root) or p.suffix.lower() not in suffixes
                or p.suffix.lower() not in {".json",".csv",".fa",".fasta",".parquet"}):
            raise ValueError("unsafe/opaque or unsupported source format/path")
        raw=p.read_bytes()
        if len(raw)!=ref.bytes or _sha(raw)!=ref.sha256:raise ValueError("source byte integrity mismatch")
        return raw

    def _snapshot(self):
        p=(self.root/self.handoff).resolve()
        if not p.is_relative_to(self.root) or p.suffix.lower()!=".json":raise ValueError("safe pinned JSON handoff only")
        raw=p.read_bytes()
        if _sha(raw)!=HANDOFF_SHA256:raise ValueError("source handoff integrity mismatch")
        hand=json.loads(raw)
        refs=hand["references"]
        qc=json.loads(self._bytes(refs["native_QC"],{".json"}))
        transform=json.loads(self._bytes(refs["relation_transformations"],{".json"}))
        self._bytes(refs["license_receipts"],{".json"})
        if (refs["native_QC"]["sha256"]!=QC_SHA256
                or refs["relation_transformations"]["sha256"]!=TRANSFORMATION_SHA256
                or refs["license_receipts"]["sha256"]!=LICENSE_SHA256
                or hand["dataset_revision"]!=REVISION or qc["dataset_revision"]!=REVISION
                or type(hand["task_admitted_records"]) is not int or hand["task_admitted_records"]!=0
                or transform["no_model_admission"] is not True
                or transform["source_QC_manifest"]["sha256"]!=QC_SHA256
                or qc["native_mapping_conflicts"] or qc["pickle_semantics_not_extracted"] is not True):
            raise ValueError("exact native source/QC/transformation/no-admission contract mismatch")
        return hand,qc,transform

    def _catalog(self,family,qc,transform):
        if family not in FAMILY_PATHS:
            raise ValueError("unsupported/opaque source; only approved original-ID relations are candidates")
        path=FAMILY_PATHS[family]
        entry=next(c for c in qc["CSV_files"] if c["source_path"]==path)
        audit=next(a for a in transform["paired_component_audits"] if a["original"]==entry["source"])
        indexed=next(c for c in qc["CSV_files"] if c["source"]==audit["indexed"])
        if (audit["shared_field_mismatches"] or audit["stable_ID_to_index_mapping_conflicts"]
                or audit["unchanged_shared_fields_excluding_explicitly_transformed_IDs_relation_texttype"] is not True):
            raise ValueError("quarantined unresolved original/indexed relation mapping")
        return entry,indexed

    def _csv(self,entry,indices):
        if max(indices)>=entry["rows"]:raise ValueError("source row selector outside exact native axis")
        raw=self._bytes(entry["source"],{".csv"})
        reader=csv.DictReader(io.StringIO(raw.decode("utf8"),newline=""),strict=True)
        if reader.fieldnames!=entry["native_field_order"]:raise ValueError("native CSV ordered field schema mismatch")
        # Preserve exact quoted/multiline physical row bytes as well as cell values.
        boundaries=(m.end() for m in re.finditer(b"\n",raw))
        position=0;offset=0;exhausted=False
        def boundary(index):
            nonlocal position,offset,exhausted
            while position<index:
                if exhausted:raise IndexError("list index out of range")
                try:offset=next(boundaries)
                except StopIteration:offset=len(raw);exhausted=True
                position+=1
            return offset
        result={};wanted=set(indices);last_line=1
        header=raw[:boundary(1)]
        for ordinal,fields in enumerate(reader):
            stop=reader.line_num
            if ordinal in wanted:
                if any(type(k) is not str or type(v) is not str for k,v in fields.items()):
                    raise ValueError("unsupported native CSV missing/extra cells")
                start=boundary(last_line);end=boundary(stop)
                result[ordinal]=(fields,header,raw[start:end])
                if len(result)==len(wanted):break
            last_line=stop
        if set(result)!=wanted:raise ValueError("native CSV selected row missing")
        return result

    def _sequence(self,kind,ordinal,qc,table_cache=None):
        import pyarrow.parquet as pq
        entry=next(f for f in qc["FASTA_files"] if f["source_path"]==f"integrated_data/v1/{kind}/{kind}_sequences.fa")
        # Preserve fresh full-byte verification in the original order even
        # when this call already decoded the same immutable payload.
        source=self._bytes(entry["source"],{".fa",".fasta"})
        payload=self._bytes(entry["payload"],{".parquet"})
        key=(entry["source"]["sha256"],entry["payload"]["sha256"])
        if table_cache is not None and key in table_cache:
            table=table_cache[key]
        else:
            table=pq.read_table(io.BytesIO(payload),use_threads=False)
            if table_cache is not None:table_cache[key]=table
        if len(table)!=entry["native_records"] or not 0<=ordinal<len(table):raise ValueError("native FASTA ordinal outside source axis")
        row=table.slice(ordinal,1).to_pylist()[0]
        if (type(row["native_ordinal"]) is not int or row["native_ordinal"]!=ordinal
                or row["source_sha256"]!=entry["source"]["sha256"] or row["source_asset"]!=entry["source_path"]
                or row["task_admitted"] is not False or row["adapter_support"] is not False
                or type(row["sequence_length"]) is not int
                or row["sequence_length"]!=len(row["native_sequence"])
                or _sha(row["native_sequence"].encode())!=row["sequence_SHA256"]):
            raise ValueError("native FASTA payload/source/length/mask binding mismatch")
        marker=b">"+row["native_header"].encode()
        start=source.find(marker)
        if (start<0 or (start and source[start-1:start]!=b"\n") or source.find(marker,start+1)>=0
                or source[start+len(marker):start+len(marker)+1] not in (b"\n",b"\r")):
            raise ValueError("exact unique native FASTA source header unavailable")
        end=source.find(b"\n>",start)
        record=source[start:end+1 if end>=0 else len(source)]
        return NativeSequenceCandidate(kind,ordinal,row["native_ID"],marker,row["native_sequence"].encode(),
            record,NativeFileRef(**entry["source"]),NativeFileRef(**entry["payload"]),_canonical(PROOF_BINDING))

    def read_sequences(self,selector):
        if type(selector) is not SequenceSelector:raise ValueError("typed bounded native FASTA selector required")
        _,qc,_=self._snapshot()
        table_cache={}
        return tuple(self._sequence(selector.entity_kind,i,qc,table_cache=table_cache) for i in selector.native_ordinals)

    def read_relations(self,selector):
        import pyarrow.parquet as pq
        if type(selector) is not RelationSelector:raise ValueError("typed bounded original relation selector required")
        _,qc,trans=self._snapshot()
        original,indexed=self._catalog(selector.family,qc,trans)
        originals=self._csv(original,selector.row_indices);indices=self._csv(indexed,selector.row_indices)
        mapping=pq.read_table(io.BytesIO(self._bytes(qc["native_ID_index_map"],{".parquet"})),use_threads=False)
        lookup={}
        names=("source_namespace","native_original_ID","native_internal_index")
        if all(mapping.column_names.count(name)==1 for name in names):
            # Convert columns, not a complete retained list of row dictionaries.
            # Every key is still checked, including unselected namespaces/IDs.
            columns=[mapping[name].to_pylist() for name in names]
            for namespace,native_id,internal_index in zip(*columns):
                key=(namespace,native_id)
                if key in lookup:raise ValueError("quarantined duplicate native ID/index allocation")
                lookup[key]=internal_index
        else:
            # Retain the original first-error behavior for malformed schemas.
            for row in mapping.to_pylist():
                key=(row["source_namespace"],row["native_original_ID"])
                if key in lookup:raise ValueError("quarantined duplicate native ID/index allocation")
                lookup[key]=row["native_internal_index"]
        out=[];table_cache={}
        for i in selector.row_indices:
            fields,header,raw=originals[i];mapped=indices[i][0]
            seqindex=lookup.get((fields["seq_type"],fields["seq_id"]))
            textindex=lookup.get((fields["text_type"],fields["text_id"]))
            if (seqindex is None or type(seqindex) is not str or not re.fullmatch(r"0|[1-9][0-9]*",seqindex)
                    or seqindex!=mapped["seq_id"] or textindex!=mapped["text_id"]):
                raise ValueError("quarantined unresolved native original-ID/internal-index mapping")
            seq=self._sequence(fields["seq_type"],int(seqindex),qc,table_cache=table_cache)
            out.append(NativeRelationCandidate(selector.family,i,NativeFileRef(**original["source"]),
                NativeFileRef(**indexed["source"]),NativeFileRef(**qc["native_ID_index_map"]),
                header,raw,_canonical(fields),_canonical(mapped),seqindex,seq,_canonical(PROOF_BINDING)))
        return tuple(out)

    def verify(self,candidate):
        if type(candidate) is not NativeRelationCandidate:raise ValueError("typed native candidate required")
        checked,=self.read_relations(RelationSelector(candidate.family,(candidate.source_row_index,)))
        if candidate!=checked:raise ValueError("source candidate replay/binding mismatch")
        return checked

    def verify_sequence(self,candidate):
        if type(candidate) is not NativeSequenceCandidate:raise ValueError("typed native FASTA candidate required")
        checked,=self.read_sequences(SequenceSelector(candidate.entity_kind,(candidate.native_ordinal,)))
        if candidate!=checked:raise ValueError("source FASTA candidate replay/binding mismatch")
        return checked

    def quarantines(self):
        return dict(opaque_pickle_files=28,indexed_only_UniProt_views=True,
            text_description_and_instruction_ancestry="opaque_unavailable",
            domain_parent_coordinates="unresolved",current_UniProt_qualification="pending",
            cross_corpus_homology_or_phenotype_guards="not_established",task_admitted_records=0,
            unacquired_or_unqualified_components=("DrugBank","DisGeNET","OMIM","generated instructions",
                "peptide/PDB binding","protein-protein","complete released text descriptions"))

    def require_text_capability(self,purpose):
        raise ValueError("phenotype descriptions/instruction ancestry remain opaque; text tasks unavailable")

    def source_references(self,family):
        hand,qc,trans=self._snapshot()
        entry,_=self._catalog(family,qc,trans);kind="domain" if family.startswith("domain") else "protein"
        fasta=next(f for f in qc["FASTA_files"] if f["source_path"]==f"integrated_data/v1/{kind}/{kind}_sequences.fa")
        p=(self.root/self.handoff).resolve()
        return dict(handoff=dict(path=str(p),bytes=p.stat().st_size,sha256=HANDOFF_SHA256),
            QC=hand["references"]["native_QC"],transformation=hand["references"]["relation_transformations"],
            license=hand["references"]["license_receipts"],source_csv=entry["source"],
            mapping=qc["native_ID_index_map"],FASTA=fasta["source"],sequence_payload=fasta["payload"])
