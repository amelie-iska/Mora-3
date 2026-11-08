"""X-Pert native expression/context CANDIDATES; no model task admission.

Only safe pinned JSON/gzip/numeric-NPY projections are consumed. No H5AD, ZIP,
pickle, foreign inspection environment, model weights, response loss or training.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import numpy as np

REVISION="623c7f8034fccf1655cac0c99d8ad6fbd9a73c9f"
BASE="outputs/data-sources/20261101-XPert-source-v1"
FIXTURE_REF=dict(path=BASE+"/curated/bounded-native-reader-fixture-v1/handoff-bounded-native-reader-fixture-v2-role-masks.json",
    bytes=27049,sha256="fbec896835fb84516447bebc4a928c4b06686796c1664e35835d4acf26f6388b")
FULL_REF=dict(path=BASE+"/handoff-XPert-full-native-source-v1.json",bytes=9656,
    sha256="9a41dc42a2fa5e1fa57945c7499e7d69a51b63fb883ce8b455689c57864f9b2f")
QC_SHA256="c5d30d2854765d5c59d15340226a042fa109c9e3edcb48b5251e9c73c79046df"
COORD_SHA256="97a72b164af4627f516e1088751f78c8706c7b1e4858a0a514432ff70aa25260"
GRAPH_SHA256="6a5f9a46ddbb4d5ca5f3f42d937179471a40189370cfd43304cbbb3f4b618449"
PREPARATION_SHA256="9e71d25e9a217fe25853237e085af5c193eb3184b37db40814b18932007c9487"
SOURCES={32221275:dict(n=33694,rows=111445,encoding="csc_matrix",stored=361582621,
    source_sha256="efde6f5301fe256725dce1d980f37bd96a13481a9a16135515897368e631affc"),
    32221302:dict(n=978,rows=678401,encoding="dense",stored=663476178,
    source_sha256="676671c8a79d2461938f45229abe63a09c5d559e7432381ff3bd06492362df38")}


def _canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError("duplicate native JSON key")
            result[key]=value
        return result
    def invalid(value):raise ValueError("nonfinite JSON numeric literal")
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid)


def _canonical_json(raw):
    if type(raw) is not bytes:raise ValueError("immutable canonical source bytes required")
    value=_json(raw)
    if _canonical(value)!=raw:raise ValueError("exact canonical native JSON required")
    return value


def _same(a,b,message):
    if _canonical(a)!=_canonical(b):raise ValueError(message)


def _article(value):
    if type(value) is not int or value not in SOURCES:
        raise ValueError("exact supported original native source article required")


@dataclass(frozen=True)
class Float32Profile:
    values_bytes: bytes
    observed_mask: bytes
    finite_mask: bytes
    explicit_storage_mask: bytes
    source_position_bytes: bytes
    matrix_encoding: str

    def __post_init__(self):
        if (type(self.values_bytes) is not bytes or not self.values_bytes or len(self.values_bytes)%4
                or self.matrix_encoding not in ("dense","csc_matrix")):
            raise ValueError("complete native little-endian float32 bytes required")
        n=len(self.values_bytes)//4
        if (any(type(x) is not bytes or len(x)!=n or set(x)-{0,1}
                for x in (self.observed_mask,self.finite_mask,self.explicit_storage_mask))
                or self.observed_mask!=b"\1"*n
                or type(self.source_position_bytes) is not bytes or len(self.source_position_bytes)!=8*n):
            raise ValueError("complete observed axis and exact numeric/storage masks/positions required")
        values=self.native_values()
        if np.isfinite(values).tobytes()!=self.finite_mask:
            raise ValueError("numeric validity must match exact original float32 bits")
        positions=self.source_positions();explicit=[p for p,s in zip(positions,self.explicit_storage_mask) if s]
        if (len(set(explicit))!=len(explicit)
                or any((p<0 if s else p!=-1) for p,s in zip(positions,self.explicit_storage_mask))
                or (self.matrix_encoding=="dense" and self.explicit_storage_mask!=b"\1"*n)):
            raise ValueError("native unique storage positions/masks required; no coalescing/reordering")
        for i,stored in enumerate(self.explicit_storage_mask):
            if not stored and self.values_bytes[4*i:4*i+4]!=b"\0"*4:
                raise ValueError("implicit sparse value must be original mathematical positivezero, not missing or negativezero")

    @property
    def axis_length(self):return len(self.values_bytes)//4

    def native_values(self):
        """Read-only native float32 view; never cast or reconstruct via JSON floats."""
        return np.frombuffer(self.values_bytes,dtype="<f4")

    def source_positions(self):
        return tuple(x[0] for x in struct.iter_unpack("<q",self.source_position_bytes))

    @property
    def binding_sha256(self):
        return _sha(_canonical(dict(dtype="<f4",encoding=self.matrix_encoding,
            values=_sha(self.values_bytes),observed=_sha(self.observed_mask),finite=_sha(self.finite_mask),
            storage=_sha(self.explicit_storage_mask),positions=_sha(self.source_position_bytes))))

    def audit(self):
        return dict(native_dtype="<f4",native_axis=self.axis_length,matrix_encoding=self.matrix_encoding,
            values_sha256=_sha(self.values_bytes),observed_mask_sha256=_sha(self.observed_mask),
            numeric_finite_mask_sha256=_sha(self.finite_mask),storage_mask_sha256=_sha(self.explicit_storage_mask),
            storage_positions_sha256=_sha(self.source_position_bytes),explicit_storage=sum(self.explicit_storage_mask),
            implicit_observed_zeros=self.axis_length-sum(self.explicit_storage_mask),
            finite_values=sum(self.finite_mask),finite_validity_is_not_endpoint_applicability=True,
            implicit_zero_is_not_instrument_detection_assertion=True)


@dataclass(frozen=True)
class XPertSelector:
    source_article_id: int
    native_rows: tuple

    def __post_init__(self):
        _article(self.source_article_id)
        if (type(self.native_rows) is not tuple or not 1<=len(self.native_rows)<=16
                or any(type(i) is not int or not 0<=i<SOURCES[self.source_article_id]["rows"] for i in self.native_rows)
                or len(set(self.native_rows))!=len(self.native_rows)):
            raise ValueError("1..16unique exact native row indices required; no split allocation")


def _category(value):
    if value is None:return None
    if (type(value) is not dict or set(value)!={"c","v"} or type(value["c"]) is not int
            or value["c"]<-1 or (value["c"]==-1 and value["v"] is not None)
            or (value["c"]>=0 and type(value["v"]) is not str)):
        raise ValueError("exact native categorical code/value with explicit missingness required")
    return value["v"]


GENETIC_DEFAULT=("perturbation","nperts","cell_line","organism","perturbation_type")
GENETIC_ALLOWED=GENETIC_DEFAULT+("guide_id","celltype","tissue_type")
CHEMICAL_DEFAULT=("canonical_smiles","cell_id","pert_dose","pert_dose_unit","pert_time","pert_time_unit","pert_type")


@dataclass(frozen=True)
class XPertContext:
    source_article_id: int
    native_fields_bytes: bytes

    def __post_init__(self):
        _article(self.source_article_id)
        if type(_canonical_json(self.native_fields_bytes)) is not dict:raise ValueError("literal native obs mapping required")
        self.prediction_inputs()  # Validate default known native categorical/types at construction.

    def native_fields(self):return _canonical_json(self.native_fields_bytes)

    def prediction_inputs(self,*,known_fields=None):
        """Pure target-free projection; source verification and admission are separate."""
        genetic=self.source_article_id==32221275
        default=GENETIC_DEFAULT if genetic else CHEMICAL_DEFAULT
        allowed=GENETIC_ALLOWED if genetic else CHEMICAL_DEFAULT
        keys=default if known_fields is None else known_fields
        if (type(keys) is not tuple or not keys or len(set(keys))!=len(keys)
                or any(type(k) is not str or k not in allowed for k in keys)):
            raise ValueError("strict known biological input allowlist; targets/QC/IDs/selection outcomes excluded")
        raw=self.native_fields()
        fields={k:raw.get(k) for k in keys}
        if genetic:
            out=dict(guide_efficacy=None,guide_efficacy_known=False)
            names=dict(perturbation="native_genetic_perturbation",guide_id="native_design_guide_identity",
                cell_line="cell_line",organism="organism",celltype="native_cell_type",
                tissue_type="native_tissue_type",perturbation_type="native_perturbation_type")
            for key,name in names.items():
                if key in fields:out[name]=_category(fields[key])
            if "nperts" in fields:
                n=fields["nperts"]
                if n is not None and (type(n) is not int or n<0):raise ValueError("native perturbation count is an exact known design integer")
                out["native_perturbation_count"]=n
            return out
        out={}
        for key,name in (("canonical_smiles","native_source_smiles"),("cell_id","cell_line"),("pert_type","native_perturbation_type")):
            if key in fields:out[name]=_category(fields[key])
        if "pert_dose" in fields or "pert_dose_unit" in fields:
            value=_category(fields.get("pert_dose"));unit=_category(fields.get("pert_dose_unit"))
            if unit not in (None,"","um","%","-666"):raise ValueError("unsupported native dose unit; no converter")
            known=False
            if value not in (None,"") and unit in ("um","%"):
                try:
                    numeric=Decimal(value)
                    known=numeric.is_finite() and numeric!=Decimal("-666")
                except InvalidOperation:known=False
            out["native_dose_coordinate"]=dict(value=value,unit=unit,coordinate_known=bool(known),
                physical_unit_confirmed=False,conversion_allowed=False)
        if "pert_time" in fields or "pert_time_unit" in fields:
            value=fields.get("pert_time");unit=_category(fields.get("pert_time_unit"))
            if value is not None and type(value) is not int:raise ValueError("exact native integer exposure-time coordinate required")
            out["native_exposure_time"]=dict(value=value,unit=unit,
                coordinate_known=value is not None and value>=0 and bool(unit) and unit!="-666")
        return out

    def require_guide_efficacy_target(self):
        raise ValueError("sgRNA_new control is a protocol placeholder; measured guide efficacy unavailable")


@dataclass(frozen=True)
class ControlSemantics:
    role: str

    def __post_init__(self):
        if self.role not in ("native_population_control","recipe_random_with_replacement_pool","recipe_synthetic_untreated_mean"):
            raise ValueError("explicit source control role required; no same-cell baseline inference")

    @property
    def same_cell_before_after(self):return False

    @property
    def independent_experimental_measurement_count(self):return None

    @property
    def is_synthetic_aggregate(self):return self.role=="recipe_synthetic_untreated_mean"


def require_DE_supervision(purpose):
    raise ValueError("literal0.1statistics and all-gene-DElists are placeholders; no measured DE or ranking targets")


def require_XPert_model_use(purpose,*,admission=None):
    if purpose=="split_use":
        raise ValueError("official saved split semantics remain opaque; no qualified native task admission")
    raise ValueError("source candidates have no qualified constructor/model native task admission; response supervision unavailable")


def load_numeric_npy(raw,*,dtype,shape):
    """Preflight numeric NPY, then load with pickle OFF; never cast/reshape.

    This pure format function grants no source identity or task admission.
    """
    if (type(raw) is not bytes or dtype not in ("<f4","|b1","<i8")
            or type(shape) is not tuple or not 1<=len(shape)<=2
            or any(type(n) is not int or n<1 for n in shape)):
        raise ValueError("exact bounded native numeric dtype/shape required")
    stream=io.BytesIO(raw)
    try:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):
            actual,order,native_dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=4096)
        elif version==(2,0):
            actual,order,native_dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=4096)
        else:raise ValueError("unsupported NPY version")
        if (any(type(n) is not int for n in actual) or actual!=shape or order is not False
                or native_dtype.hasobject or native_dtype.str!=dtype):
            raise ValueError("numeric NPY native dtype/shape/Corder mismatch; no object/pickle")
        count=1
        for n in shape:count*=n
        if len(raw)!=stream.tell()+count*native_dtype.itemsize:
            raise ValueError("exact native NPY payload extent required; no trailing/opaque bytes")
        array=np.load(io.BytesIO(raw),allow_pickle=False,max_header_size=4096)
    except (EOFError,TypeError,OverflowError) as exc:
        raise ValueError("unsafe or malformed numeric NPY") from exc
    if type(array) is not np.ndarray or array.dtype.str!=dtype or array.shape!=shape or not array.flags.c_contiguous:
        raise ValueError("complete exact native NPY array required")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class XPertGeneAxis:
    source_article_id: int
    native_rows: tuple
    source_ref_bytes: bytes

    def __post_init__(self):
        _article(self.source_article_id)
        if (type(self.native_rows) is not tuple or len(self.native_rows)!=SOURCES[self.source_article_id]["n"]
                or type(_canonical_json(self.source_ref_bytes)) is not dict):
            raise ValueError("complete original ordered gene/var axis required")
        labels=[]
        for i,raw in enumerate(self.native_rows):
            row=_canonical_json(raw)
            if (row.get("native_axis")!="var" or type(row.get("native_row_index")) is not int
                    or row["native_row_index"]!=i or row.get("task_admitted") is not False
                    or row.get("adapter_support") is not False or type(row.get("native_fields")) is not dict
                    or type(row["native_fields"].get("_index")) is not str or not row["native_fields"]["_index"]):
                raise ValueError("exact typed ordered native var/source row required")
            labels.append(row["native_fields"]["_index"])
        if len(set(labels))!=len(labels):raise ValueError("duplicate original gene-axis allocation")

    @property
    def axis_length(self):return len(self.native_rows)

    @property
    def ordered_labels(self):return tuple(_json(row)["native_fields"]["_index"] for row in self.native_rows)

    @property
    def binding_sha256(self):
        return _sha(_canonical(dict(article=self.source_article_id,
            source_ref=_canonical_json(self.source_ref_bytes),ordered_native_row_sha256=[_sha(r) for r in self.native_rows])))

    def audit(self):
        return dict(source_article_id=self.source_article_id,native_gene_axis=self.axis_length,
            complete_native_var_ref=_canonical_json(self.source_ref_bytes),binding_sha256=self.binding_sha256,
            biological_gene_values_or_QC_not_predictor_fields=True)


@dataclass(frozen=True)
class XPertCandidate:
    source_article_id: int
    native_row_index: int
    native_role: str
    gene_axis: XPertGeneAxis
    profile: Float32Profile
    context: XPertContext
    provenance_bytes: bytes

    def __post_init__(self):
        _article(self.source_article_id)
        n=SOURCES[self.source_article_id]["n"]
        if (type(self.native_row_index) is not int or not 0<=self.native_row_index<SOURCES[self.source_article_id]["rows"]
                or type(self.gene_axis) is not XPertGeneAxis or type(self.profile) is not Float32Profile
                or type(self.context) is not XPertContext or self.gene_axis.source_article_id!=self.source_article_id
                or self.context.source_article_id!=self.source_article_id or self.profile.axis_length!=n
                or self.profile.matrix_encoding!=SOURCES[self.source_article_id]["encoding"]):
            raise ValueError("exact source/row/complete gene-axis/matrix-role binding required")
        p=_canonical_json(self.provenance_bytes)
        if (p.get("schema")!="mora3.XPertNativeCandidate/v1" or p.get("fixture_sha256")!=FIXTURE_REF["sha256"]
                or type(p.get("source_article_id")) is not int or p["source_article_id"]!=self.source_article_id
                or p.get("source_H5AD_SHA256")!=SOURCES[self.source_article_id]["source_sha256"]
                or p.get("source_value_role")!="native_expression_target_only"
                or type(p.get("native_row_index")) is not int or p["native_row_index"]!=self.native_row_index
                or p.get("native_role")!=self.native_role or p.get("profile_binding_sha256")!=self.profile.binding_sha256
                or p.get("gene_axis_binding_sha256")!=self.gene_axis.binding_sha256
                or p.get("context_sha256")!=_sha(self.context.native_fields_bytes)
                or p.get("official_split") is not None or p.get("task_admitted") is not False
                or p.get("biological_samecell_link") is not None):
            raise ValueError("exact native candidate/provenance/type/axis/source binding required")
        if self.source_article_id==32221275:
            count=self.context.prediction_inputs()["native_perturbation_count"]
            roles={0:("native_genetic_population_control_nperts0",),
                1:("native_single_genetic_perturbation_nperts1",),
                2:("native_two_genetic_combination_nperts2",)}.get(count,())
        else:
            native_type=_category(self.context.native_fields().get("pert_type"))
            roles=(("native_bulk_vehicle_population_control",) if native_type=="ctl_vehicle" else
                ("native_bulk_chemical_perturbation","other_native_bulk_chemical_perturbation") if native_type=="trt_cp" else ())
        if type(self.native_role) is not str or self.native_role not in roles:
            raise ValueError("native control/intervention/evidence role cannot be relabeled as efficacy or derived truth")

    @property
    def status(self):return "native_expression_candidate_no_task_admission"

    @property
    def official_split(self):return None

    @property
    def control_semantics(self):
        return ControlSemantics("native_population_control") if "population_control" in self.native_role else None

    def prediction_inputs(self,*,known_fields=None):
        return self.context.prediction_inputs(known_fields=known_fields)

    @property
    def binding_sha256(self):
        return _sha(_canonical(dict(profile=self.profile.binding_sha256,axis=self.gene_axis.binding_sha256,
            context=_sha(self.context.native_fields_bytes),provenance=_sha(self.provenance_bytes))))

    def audit(self):
        p=_canonical_json(self.provenance_bytes)
        p.update(status=self.status,profile=self.profile.audit(),gene_axis=self.gene_axis.audit(),
            native_obs=self.context.native_fields(),independent_experimental_measurement_count=None,
            control_input_automatically_supplied=False,all_gene_DE_lists_are_not_measured_DE=True,
            guide_efficacy=None,task_admitted=False,binding_sha256=self.binding_sha256)
        return p

    def require_model_use(self,purpose,*,admission=None):
        if purpose=="DE_statistics":return require_DE_supervision(purpose)
        if purpose=="guide_efficacy":return self.context.require_guide_efficacy_target()
        return require_XPert_model_use(purpose,admission=admission)


@dataclass(frozen=True)
class XPertDesiredExpression:
    gene_axis: XPertGeneAxis
    profile: Float32Profile
    role: str = "desired"

    def __post_init__(self):
        if (self.role!="desired" or type(self.gene_axis) is not XPertGeneAxis or type(self.profile) is not Float32Profile
                or self.gene_axis.axis_length!=self.profile.axis_length):
            raise ValueError("complete explicit caller-desired expression/axis required; never an observed label")

    def as_mapping(self):
        return dict(evidence_role="caller_desired_expression",native_dtype="<f4",
            ordered_gene_axis=list(self.gene_axis.ordered_labels),native_values_hex=self.profile.values_bytes.hex(),
            finite_validity_hex=self.profile.finite_mask.hex(),observed_experimental_label=False)

    def require_model_use(self,purpose):
        return require_XPert_model_use(purpose)


class XPertNativeSource:
    """Pinned six-row native interoperability fixture; not a whole-matrix reader."""
    def __init__(self,*,root,manifest=FIXTURE_REF["path"],expected_manifest_sha256=FIXTURE_REF["sha256"]):
        self.root=Path(root).resolve();self.manifest=Path(manifest)
        if expected_manifest_sha256!=FIXTURE_REF["sha256"]:raise ValueError("exact pinned safe fixture required")
        self._snapshot()

    def _verified(self,ref):
        if (type(ref) is not dict or type(ref.get("path")) is not str
                or type(ref.get("bytes")) is not int or ref["bytes"]<=0
                or type(ref.get("sha256")) is not str or len(ref["sha256"])!=64):
            raise ValueError("typed exact source projection reference required")
        path=(self.root/ref["path"]).resolve()
        if not path.is_relative_to(self.root) or path.suffix.lower() not in (".json",".gz",".npy"):
            raise ValueError("unsafe or unsupported source format; no H5AD/ZIP/pickle/external links")
        raw=path.read_bytes()
        if len(raw)!=ref["bytes"] or _sha(raw)!=ref["sha256"]:raise ValueError("source/projection byte integrity mismatch")
        return raw

    def _snapshot(self):
        path=(self.root/self.manifest).resolve()
        if not path.is_relative_to(self.root) or path.suffix.lower()!=".json":
            raise ValueError("safe pinned fixture JSON required")
        raw=path.read_bytes()
        if _sha(raw)!=FIXTURE_REF["sha256"]:raise ValueError("fixture integrity mismatch")
        m=_json(raw);full=_json(self._verified(FULL_REF))
        qc=_json(self._verified(m["source_QC"]))
        coordinates=_json(self._verified(m["whole_CSC_and_identity_proof"]))
        graph=_json(self._verified(m["chemical_reference_graph_QC"]))
        gates=_json(self._verified(m["source_preparation_gates"]))
        original=_json(self._verified(m["original_fixture_handoff"]))
        self._verified(full["whole_original_archives_and_all_member_receipts"])
        self._verified(full["source_release_rights"])
        _same(m["bundles"],original["bundles"],"original bounded numeric payload/source roles must remain unchanged")
        expected_fixture=dict(FIXTURE_REF,path=str((self.root/FIXTURE_REF["path"]).resolve()))
        _same(full["bounded_fullaxis_reader_fixture"],expected_fixture,"full source handoff/fixture version mismatch")
        expected={"source_QC":QC_SHA256,"whole_CSC_and_identity_proof":COORD_SHA256,
            "chemical_reference_graph_QC":GRAPH_SHA256,"source_preparation_gates":PREPARATION_SHA256}
        if (any(m[k]["sha256"]!=v for k,v in expected.items())
                or m["schema"]!="mora3.XPert_bounded_native_reader_fixture.v2"
                or m["task_admitted_records"]!=0 or type(m["task_admitted_records"]) is not int
                or m["model_training_use_authorized"] is not False
                or full["task_admitted_records"]!=0 or type(full["task_admitted_records"]) is not int
                or full["substantive_training_launched"] is not False
                or qc["opaque_pert_data_pickles_never_unpickled"] is not True
                or coordinates["Norman_CSC_coordinate_audit"]["native_CSC_coordinate_unique"] is not True
                or gates["raw_H5AD_metadata_remains_distinct_from_recipe_outputs"] is not True):
            raise ValueError("qualified safe projection/source/scientific-role/no-admission proof required")
        return m,full,qc,coordinates,graph,gates

    @staticmethod
    def _metadata_ref(entry):
        return dict(path=entry["payload_path"],bytes=entry["payload_bytes"],sha256=entry["payload_SHA256"])

    def _metadata(self,entry,rows=None):
        raw=self._verified(self._metadata_ref(entry))
        wanted=None if rows is None else set(rows)
        out=[]
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
            for i,line in enumerate(stream):
                if wanted is None or i in wanted:out.append(_json(line))
                if wanted is not None and i>=max(wanted):break
        return out

    def _array(self,ref,*,dtype,shape):
        if (ref.get("native_dtype")!=dtype or _canonical(ref.get("native_shape"))!=_canonical(list(shape))
                or ref.get("NPY_allow_pickle_false_full_readback") is not True):
            raise ValueError("native dtype/fullshape/pickle-free projection binding required")
        array=load_numeric_npy(self._verified(ref),dtype=dtype,shape=shape)
        if _sha(array.tobytes(order="C"))!=ref["native_logical_value_bits_SHA256"]:
            raise ValueError("native logical numeric bit hash mismatch")
        return array

    def read_profiles(self,selector):
        if type(selector) is not XPertSelector:raise ValueError("typed bounded native row selector required")
        m,full,qc,coordinates,graph,gates=self._snapshot()
        article=selector.source_article_id
        bundle=next(b for b in m["bundles"] if b["source_article_ID"]==article)
        native=next(s for s in qc["sources"] if s["source_article_ID"]==article)
        expected=SOURCES[article]
        if (bundle["source_native_H5AD_SHA256"]!=expected["source_sha256"]
                or native["source_H5AD_SHA256"]!=expected["source_sha256"]
                or bundle["complete_native_gene_axis"]!=expected["n"]
                or native["native_gene_axis"]!=expected["n"]
                or native["native_matrix"]["encoding"]!=expected["encoding"]
                or bundle["no_original_gene_axis_crop_or_dtype_downcast"] is not True):
            raise ValueError("exact original matrix/geneaxis/source/role binding required")
        rows=bundle["rows"];by_native={}
        for i,row in enumerate(rows):
            r=row["native_source_row_index"]
            if (type(r) is not int or type(row["fixture_row_axis"]) is not int or row["fixture_row_axis"]!=i
                    or r in by_native or row["task_admitted"] is not False
                    or any(row[k] is not None for k in ("official_XPert_train_val_test_split","guide_efficacy",
                        "DE_statistics","biological_match_or_samecell_link"))):
                raise ValueError("exact native fixture row/unknown split/efficacy/DE role required")
            by_native[r]=row
        if not set(selector.native_rows)<=set(by_native):
            raise ValueError("native row not available in bounded fixture; no raw-matrix fallback")
        gene_meta=bundle["complete_gene_metadata"]
        _same(gene_meta,self._gene_meta(native["var"]),"exact original var metadata reference required")
        genes=self._metadata(gene_meta)
        if len(genes)!=expected["n"]:raise ValueError("complete original gene-axis readback required")
        axis=XPertGeneAxis(article,tuple(_canonical(r) for r in genes),_canonical(self._metadata_ref(gene_meta)))
        observations=self._metadata(native["obs"],selector.native_rows)
        observations={r["native_row_index"]:r for r in observations}
        if set(observations)!=set(selector.native_rows):raise ValueError("native obs row allocation mismatch")
        payloads=bundle["numeric_payloads"];shape=(len(rows),expected["n"])
        arrays={k:self._array(payloads[k],dtype=dtype,shape=shape) for k,dtype in
            (("values","<f4"),("numeric_validity","|b1"),("source_explicit_storage","|b1"),("source_storage_positions","<i8"))}
        graph_lookup={g["source_SMILES"]:g for g in graph["chemical_graph_records"]}
        out=[]
        for native_row in selector.native_rows:
            row=by_native[native_row];i=row["fixture_row_axis"];observation=observations[native_row]
            _same(row["source_native_observation"],observation,"exact source obs/native row/typed values required")
            if (observation["native_axis"]!="obs" or type(observation["native_row_index"]) is not int
                    or observation["native_row_index"]!=native_row or observation["task_admitted"] is not False
                    or observation["adapter_support"] is not False
                    or set(observation["native_fields"])!=set(native["obs"]["native_column_order"])|{native["obs"]["index_column"]}):
                raise ValueError("exact original typed source obs schema/role required")
            context=XPertContext(article,_canonical(observation["native_fields"]))
            profile=Float32Profile(arrays["values"][i].tobytes(),b"\1"*expected["n"],
                arrays["numeric_validity"][i].tobytes(),arrays["source_explicit_storage"][i].tobytes(),
                arrays["source_storage_positions"][i].tobytes(),expected["encoding"])
            positions=np.frombuffer(profile.source_position_bytes,dtype="<i8")
            if np.any(positions>=expected["stored"]):raise ValueError("source storage position outside complete original matrix")
            if article==32221302:
                if not np.array_equal(positions,np.arange(native_row*expected["n"],(native_row+1)*expected["n"],dtype="<i8")):
                    raise ValueError("native dense rowmajor source position mismatch")
                smiles=context.prediction_inputs()["native_source_smiles"]
                if smiles not in graph_lookup:raise ValueError("source reference graph candidate unavailable")
                _same(row["source_reference_graph"],graph_lookup[smiles],"original SMILES/reference graph/role mismatch")
            elif row["source_reference_graph"] is not None:
                raise ValueError("genetic source provides no chemical graph target")
            if not all(profile.finite_mask):raise ValueError("current whole native QC establishes finite expression; masks cannot drift")
            prov=dict(schema="mora3.XPertNativeCandidate/v1",fixture_sha256=FIXTURE_REF["sha256"],
                source_article_id=article,native_row_index=native_row,native_role=row["native_role"],
                profile_binding_sha256=profile.binding_sha256,gene_axis_binding_sha256=axis.binding_sha256,
                context_sha256=_sha(context.native_fields_bytes),source_H5AD_SHA256=expected["source_sha256"],
                source_observation_unit=bundle["source_observation_unit"],source_observation_ref=self._metadata_ref(native["obs"]),
                numeric_payload_refs=payloads,proof_refs={k:m[k] for k in
                    ("source_QC","whole_CSC_and_identity_proof","chemical_reference_graph_QC","source_preparation_gates")},
                source_reference_graph=row["source_reference_graph"],official_split=None,task_admitted=False,
                biological_samecell_link=None,source_value_role="native_expression_target_only",
                source_matrix_units="author_prepared_native_X_as_released;not_reinterpreted",
                known_missing_gates=full["remaining_gates"],source_repository_revision=REVISION,
                protocol_control_roles=["recipe_random_with_replacement_pool","recipe_synthetic_untreated_mean"],
                source_predictor_role_policy=m["explicit_roles_and_predictor_masks"])
            out.append(XPertCandidate(article,native_row,row["native_role"],axis,profile,context,_canonical(prov)))
        return tuple(out)

    @staticmethod
    def _gene_meta(entry):
        return {k:entry[k] for k in ("native_rows","index_column","native_column_order","payload_path","payload_bytes","payload_SHA256")}

    def verify(self,candidate):
        if type(candidate) is not XPertCandidate:raise ValueError("typed source candidate required")
        checked,=self.read_profiles(XPertSelector(candidate.source_article_id,(candidate.native_row_index,)))
        if candidate!=checked:raise ValueError("exact native source candidate replay/binding mismatch")
        return checked

    def source_references(self,article):
        _article(article);m,full,qc,*_=self._snapshot()
        b=next(b for b in m["bundles"] if b["source_article_ID"]==article)
        source=next(s for s in qc["sources"] if s["source_article_ID"]==article)
        path=(self.root/self.manifest).resolve()
        return dict(fixture=dict(path=str(path),bytes=path.stat().st_size,sha256=FIXTURE_REF["sha256"]),
            fullsource=FULL_REF,QC=m["source_QC"],coordinates=m["whole_CSC_and_identity_proof"],
            graph=m["chemical_reference_graph_QC"],preparation=m["source_preparation_gates"],
            gene_axis=self._metadata_ref(b["complete_gene_metadata"]),obs=self._metadata_ref(source["obs"]),
            values=b["numeric_payloads"]["values"],finite=b["numeric_payloads"]["numeric_validity"],
            storage=b["numeric_payloads"]["source_explicit_storage"],positions=b["numeric_payloads"]["source_storage_positions"])
