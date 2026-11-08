"""Immutable native ADMET contracts; generic schemas never grant task admission.

Matches the corpus owner's ADMETNativeObservation/v1 wire and qualified v2
axis-cell count semantics. No unit conversion, thresholding, chemistry rewrite,
assay pooling, split assignment or experimental-label propagation occurs.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import csv
import hashlib
import io
import json
from pathlib import Path
import re

MOLECULE_CLASSES = ("small_molecule", "peptide", "protein", "RNA", "DNA")
CANDIDATE_V2_SHA256 = "cd2ff3bde8dc01f92fdb6a88ff46d09a10d4d058f07989584f2912fe8a2f79f0"
CONTEXT_KEYS = frozenset(("species","strain","cell_line","tissue","matrix","temperature","pH","exposure_time",
    "dose","route","formulation","carrier","assay_system","assay_direction","source_measurement_coordinate"))
IDENTITY_KEYS = frozenset(("molecule_class","representation_type","representation","native_modifications","construct","source_identity_missingness"))
ENDPOINT_KEYS = frozenset(("endpoint_key","source_endpoint","description","readout","unit","unit_basis","scale","measurement_type","assay_protocol_context"))
EVIDENCE_ROLES = frozenset(("experimental_measurement","source_curated_measurement","computed","predicted","model_generated","desired"))
GATES = ("source_rights","exact_identity","endpoint_definition","task_units_or_declared_native_coordinate","split_group_review")
_CHEMISTRY_KEYS = frozenset(("component","components","chain","chains","chain_role","position","positions","site","sites",
    "native_chemistry","native_residue","stereochemistry","bond","bonds","topology","attachment","attachments",
    "terminal_group","terminal_groups","representation","representation_type","status","source_literal",
    "isotype","Fc","glycosylation","conjugates","backbone","cap","polyA","duplex","linkers","structure","sugar","base"))
_CONTEXT_NESTED_KEYS = frozenset(("value","unit","unit_basis","status","native_literal","minimum","maximum","lower","upper",
    "relation","carrier","composition","native_chemistry","representation","representation_type","temperature","pH","duration"))
_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_CELL = re.compile(r"^\s*(<=|>=|<|>|=|≤|≥)?\s*("+_NUM+r")\s*$")


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d:raise ValueError("duplicate native JSON key")
        d[k]=v
    return d


def _json(raw):
    def invalid(x):raise ValueError("nonfinite JSON cannot represent a native scalar")
    return json.loads(raw,object_pairs_hook=_pairs,parse_constant=invalid)


def _same(a,b,message):
    if canonical(a)!=canonical(b):raise ValueError(message)


def _text(x,name):
    if type(x) is not str or not x:raise ValueError(name+" requires exact nonempty text")
    x.encode("utf8")


def _nested(value,allowed):
    if isinstance(value,dict):
        if set(value)-allowed:raise ValueError("unsupported predictor metadata; keep source IDs/targets/audit outside inputs")
        for item in value.values():_nested(item,allowed)
    elif isinstance(value,list):
        for item in value:_nested(item,allowed)


def _decimal(literal):
    if type(literal) is not str or not re.fullmatch(_NUM,literal):raise ValueError("exact finite native decimal literal required")
    try:value=Decimal(literal)
    except InvalidOperation as exc:raise ValueError("invalid native decimal") from exc
    if not value.is_finite():raise ValueError("finite native decimal required")
    return value


def _parse_cell(literal):
    if literal is not None and type(literal) is not str:raise ValueError("native string or explicit null required")
    if literal is None or literal=="":
        return dict(native_literal=literal,status="missing",decimal_literal=None,relation=None,
            observation_present=False,numeric_loss_eligible=False)
    match=_CELL.fullmatch(literal)
    if match:
        relation={"≤":"<=","≥":">="}.get(match[1],match[1]) or "="
        _decimal(match[2])
        return dict(native_literal=literal,status="exact" if relation=="=" else "censored_bound",
            decimal_literal=match[2],relation=relation,observation_present=True,numeric_loss_eligible=True)
    return dict(native_literal=literal,status="unparsed",decimal_literal=None,relation=None,
        observation_present=True,numeric_loss_eligible=False)


@dataclass(frozen=True)
class NativeCell:
    native_bytes: bytes

    def __post_init__(self):
        if type(self.native_bytes) is not bytes:raise ValueError("immutable serialized native cell required")
        d=_json(self.native_bytes)
        if d.get("status")=="interval_bound":
            keys={"native_literal","status","decimal_literal","relation","observation_present","numeric_loss_eligible",
                  "lower_decimal_literal","upper_decimal_literal","lower_inclusive","upper_inclusive"}
            if set(d)!=keys:raise ValueError("explicit complete interval fields required")
            _text(d["native_literal"],"native interval")
            if _decimal(d["lower_decimal_literal"])>=_decimal(d["upper_decimal_literal"]):raise ValueError("positive-width ordered source interval required")
            expected=dict(d,decimal_literal=None,relation="interval",observation_present=True,numeric_loss_eligible=True)
            if any(type(d[k]) is not bool for k in ("lower_inclusive","upper_inclusive")):raise ValueError("typed native interval inclusion flags required")
        else:
            expected=_parse_cell(d["native_literal"])
            if d.get("status")=="missing" and d.get("missing_lexicon_source") is not None:
                source=d["missing_lexicon_source"]
                if type(source) is str:_text(source,"missing lexicon source")
                elif type(source) is dict:
                    if not source or not isinstance(source.get("path"),str) or not source.get("path"):
                        raise ValueError("explicit auditable missing lexicon source required")
                else:raise ValueError("missing lexicon is source metadata, never a boolean mask grant")
                expected=dict(native_literal=d["native_literal"],status="missing",decimal_literal=None,relation=None,
                    observation_present=False,numeric_loss_eligible=False,missing_lexicon_source=d["missing_lexicon_source"])
        _same(d,expected,"source literal/Decimal/censor/missing/type mismatch")
        if canonical(d)!=self.native_bytes:raise ValueError("canonical native cell required")

    @classmethod
    def parse(cls,literal):return cls(canonical(_parse_cell(literal)))

    @classmethod
    def from_mapping(cls,value):return cls(canonical(value))

    @classmethod
    def interval(cls,native_literal,*,lower,upper,lower_inclusive=True,upper_inclusive=True):
        return cls(canonical(dict(native_literal=native_literal,status="interval_bound",decimal_literal=None,relation="interval",
            observation_present=True,numeric_loss_eligible=True,lower_decimal_literal=lower,upper_decimal_literal=upper,
            lower_inclusive=lower_inclusive,upper_inclusive=upper_inclusive)))

    def as_mapping(self):return _json(self.native_bytes)

    def __getattr__(self,name):
        if name in ("native_literal","status","decimal_literal","relation","observation_present","numeric_loss_eligible",
                    "lower_decimal_literal","upper_decimal_literal","lower_inclusive","upper_inclusive"):
            return self.as_mapping().get(name)
        raise AttributeError(name)

    @property
    def censor_kind(self):
        r=self.relation
        return "equal" if r=="=" else "left" if r in ("<","<=") else "right" if r in (">",">=") else "interval" if r=="interval" else None

    def decimal_value(self):
        return None if self.decimal_literal is None else _decimal(self.decimal_literal)

    def constraint_fields(self):
        return {k:v for k,v in self.as_mapping().items() if k not in ("observation_present","numeric_loss_eligible","missing_lexicon_source")}


@dataclass(frozen=True)
class NativeEntity:
    native_bytes: bytes

    def __post_init__(self):
        d=_json(self.native_bytes)
        if type(self.native_bytes) is not bytes or set(d)-IDENTITY_KEYS or d.get("molecule_class") not in MOLECULE_CLASSES:
            raise ValueError("explicit supported class and exact native identity fields required")
        _text(d.get("representation_type"),"native representation type")
        _text(d.get("representation"),"full native chemical/polymer identity")
        for key in ("native_modifications","construct"):_nested(d.get(key),_CHEMISTRY_KEYS)
        missing=d.get("source_identity_missingness")
        if missing is not None and (type(missing) is not list or any(type(x) is not str for x in missing)):
            raise ValueError("explicit native identity missingness required")
        if canonical(d)!=self.native_bytes:raise ValueError("canonical native identity required")

    @classmethod
    def from_mapping(cls,value):return cls(canonical(value))
    def as_mapping(self):return _json(self.native_bytes)
    def canonical_bytes(self):return self.native_bytes
    @property
    def molecule_class(self):return self.as_mapping()["molecule_class"]


@dataclass(frozen=True)
class AssayContext:
    native_bytes: bytes

    def __post_init__(self):
        d=_json(self.native_bytes)
        if type(self.native_bytes) is not bytes or type(d) is not dict or set(d)-CONTEXT_KEYS:
            raise ValueError("only known pre-assay biological/physical context allowed")
        for v in d.values():_nested(v,_CONTEXT_NESTED_KEYS)
        if canonical(d)!=self.native_bytes:raise ValueError("canonical native context required")

    @classmethod
    def from_mapping(cls,value):return cls(canonical(value))
    def as_mapping(self):return _json(self.native_bytes)


@dataclass(frozen=True)
class EndpointDefinition:
    native_bytes: bytes
    objective_kind: str
    applicability: tuple
    binary_readout: str | None = None
    semantic_name: str | None = None
    unit_applicability: str | None = None

    def __post_init__(self):
        d=_json(self.native_bytes)
        if (type(self.native_bytes) is not bytes or set(d)-ENDPOINT_KEYS
                or self.objective_kind not in ("unresolved","binary","regression")):
            raise ValueError("explicit native endpoint/objective definition required")
        _text(d.get("endpoint_key"),"native endpoint key");_text(d.get("source_endpoint"),"native endpoint")
        if d.get("unit_basis") not in ("source_literal","official_source_schema","unresolved"):
            raise ValueError("native source versus unknown unit basis required")
        if d.get("unit") is not None:_text(d["unit"],"native unit")
        if d.get("unit_basis")=="unresolved" and d.get("unit") is not None:
            raise ValueError("unresolved unit cannot be supplied or converted")
        if self.objective_kind=="binary":_text(self.binary_readout,"explicit positive binary readout definition")
        if self.unit_applicability is None:
            object.__setattr__(self,"unit_applicability","declared_native_unit" if d.get("unit") is not None else "unknown")
        if self.unit_applicability not in ("declared_native_unit","unknown","not_applicable"):
            raise ValueError("explicit known/unknown/inapplicable unit state required")
        if self.unit_applicability=="declared_native_unit" and d.get("unit") is None:
            raise ValueError("declared native unit must be present")
        if self.unit_applicability=="not_applicable" and (self.objective_kind!="binary" or d.get("unit") is not None):
            raise ValueError("only explicitly reviewed binary class unit can be inapplicable")
        if self.semantic_name is not None:_text(self.semantic_name,"semantic query")
        if (type(self.applicability) is not tuple or tuple(k for k,v in self.applicability)!=MOLECULE_CLASSES
                or any(v not in ("applicable","inapplicable","unknown") for k,v in self.applicability)):
            raise ValueError("explicit applicability/unknown state for all five classes required")
        if canonical(d)!=self.native_bytes:raise ValueError("canonical native endpoint required")

    @classmethod
    def from_native(cls,value,*,objective_kind="unresolved",applicability=None,binary_readout=None,semantic_name=None,unit_applicability=None):
        mapping={c:"unknown" for c in MOLECULE_CLASSES} if applicability is None else applicability
        if type(mapping) is not dict or set(mapping)!=set(MOLECULE_CLASSES):raise ValueError("all five class applicability states required")
        return cls(canonical(value),objective_kind,tuple((c,mapping[c]) for c in MOLECULE_CLASSES),binary_readout,semantic_name,unit_applicability)

    def native(self):return _json(self.native_bytes)
    @property
    def key(self):return self.native()["endpoint_key"]
    @property
    def unit(self):return self.native().get("unit")
    @property
    def unit_basis(self):return self.native()["unit_basis"]
    def applicability_for(self,molecule_class):
        if molecule_class not in MOLECULE_CLASSES:raise ValueError("unsupported molecule class")
        return dict(self.applicability)[molecule_class]
    def query(self):
        d=self.native()
        return dict(semantic_name=self.semantic_name or d["source_endpoint"],description=d.get("description"),
            readout=d.get("readout"),unit=d.get("unit"),unit_basis=d["unit_basis"],scale=d.get("scale"),
            measurement_type=d.get("measurement_type"),binary_readout=self.binary_readout,unit_applicability=self.unit_applicability)
    def binding(self):
        return digest(canonical(dict(native=self.native(),objective_kind=self.objective_kind,
            applicability=self.applicability,binary_readout=self.binary_readout,semantic_name=self.semantic_name,unit_applicability=self.unit_applicability)))


@dataclass(frozen=True)
class EndpointRegistry:
    definitions: tuple

    def __post_init__(self):
        if type(self.definitions) is not tuple or any(not isinstance(d,EndpointDefinition) for d in self.definitions):
            raise ValueError("immutable endpoint definitions required")
        if len({d.key for d in self.definitions})!=len(self.definitions):raise ValueError("duplicate endpoint axes")
    def get(self,key):
        for d in self.definitions:
            if d.key==key:return d
        raise ValueError("endpoint definition is not registered")
    def binding(self):return digest(canonical([d.binding() for d in self.definitions]))
    @property
    def supported_molecule_classes(self):return MOLECULE_CLASSES


@dataclass(frozen=True)
class AdmetObservation:
    native_bytes: bytes
    applicability: str = "unknown"

    def __post_init__(self):
        if type(self.native_bytes) is not bytes:raise ValueError("immutable native observation bytes required")
        d=self.native()
        if d.get("schema")!="mora3.ADMETNativeObservation/v1" or self.applicability not in ("applicable","inapplicable","unknown"):
            raise ValueError("native ADMET observation and applicability state required")
        NativeEntity.from_mapping(d["entity"]);AssayContext.from_mapping(d["known_context"])
        EndpointDefinition.from_native(d["endpoint"]);cell=NativeCell.from_mapping(d["cell"])
        if d["evidence_role"] not in EVIDENCE_ROLES:raise ValueError("explicit source evidence role required")
        if (d.get("absence_is_not_negative") is not True or d.get("native_fields_not_projected_remain_in_source_record") is not True
                or type(d.get("source_native_axis_cell_multiplicity")) is not int or d["source_native_axis_cell_multiplicity"]!=1
                or d.get("independent_experimental_measurement_count") is not None or d.get("inferred_unit_conversions")!=[]):
            raise ValueError("native axis cells are not independent experiments or converted observations")
        if any(type(d["admission"].get(k)) is not bool for k in GATES):raise ValueError("typed source gate metadata required; flags are not verification")
        if d["endpoint"]["source_endpoint"] not in d["native_source_record"]:
            raise ValueError("complete original record must retain the endpoint literal")
        _same(d["native_source_record"][d["endpoint"]["source_endpoint"]],cell.native_literal,"native original cell/literal mismatch")
        if canonical(d)!=self.native_bytes:raise ValueError("exact canonical native observation required")

    @classmethod
    def from_native(cls,raw,*,applicability="unknown"):
        if type(raw) is not bytes:raise ValueError("native observation bytes required")
        return cls(raw,applicability)
    def native(self):return _json(self.native_bytes)
    @property
    def cell(self):return NativeCell.from_mapping(self.native()["cell"])
    @property
    def entity(self):return NativeEntity.from_mapping(self.native()["entity"])
    @property
    def context(self):return AssayContext.from_mapping(self.native()["known_context"])
    @property
    def evidence_role(self):return self.native()["evidence_role"]
    @property
    def experimental_measurement_candidate(self):
        return (self.evidence_role in ("experimental_measurement","source_curated_measurement")
            and self.applicability=="applicable" and self.cell.observation_present and self.cell.numeric_loss_eligible)
    @property
    def training_eligibility(self):return "requires_verified_native_task_admission"
    def require_source_admission(self,*,proof=None):
        # No actual source task reader/admission adapter is installed. A schema,
        # hash, dataclass, rights flag or all-True gate dict cannot confer it.
        raise ValueError("verified native ADMET task reader/admission integration required; native candidates are not supervision")


@dataclass(frozen=True)
class AdmetPredictionView:
    input_bytes: bytes
    target_bytes: bytes
    provenance_bytes: bytes
    def __post_init__(self):
        for raw in (self.input_bytes,self.target_bytes,self.provenance_bytes):
            if type(raw) is not bytes or canonical(_json(raw))!=raw:
                raise ValueError("immutable canonical prediction view bytes required")
        p=self.provenance();targets=self.targets();inputs=self.inputs()
        if (p.get("task_admitted") is not False or p.get("model_supervision_supported") is not False
                or p.get("training_eligibility")!="requires_verified_native_task_admission"):
            raise ValueError("prediction view cannot promote source admission or supervision")
        raw=canonical(p["native_observation"])
        obs=AdmetObservation.from_native(raw,applicability=targets.get("applicability"))
        if p.get("native_observation_sha256")!=digest(raw):
            raise ValueError("prediction view source payload mismatch")
        _same(targets,dict(native_measurement=obs.cell.as_mapping(),applicability=obs.applicability,
            evidence_role=obs.evidence_role),"prediction view target/source mismatch")
        admet_condition_graph(inputs)
        if inputs.get("role")!="source_native_ADMET_prediction_query":
            raise ValueError("observed prediction view requires a prediction query")
        _same(inputs["native_entity"],obs.entity.as_mapping(),"prediction identity/source mismatch")
        _same(inputs["assay_context"],obs.context.as_mapping(),"prediction context/source mismatch")
        if any(type(p.get(k)) is not str or not re.fullmatch("[0-9a-f]{64}",p[k])
               for k in ("registry_binding","endpoint_binding")):
            raise ValueError("canonical registry/endpoint digest types required")
    def inputs(self):return _json(self.input_bytes)
    def targets(self):return _json(self.target_bytes)
    def provenance(self):return _json(self.provenance_bytes)


def admet_prediction_view(observation,registry):
    if not isinstance(observation,AdmetObservation) or not isinstance(registry,EndpointRegistry):
        raise ValueError("typed native observation and registry required")
    d=observation.native();endpoint=registry.get(d["endpoint"]["endpoint_key"])
    _same(endpoint.native(),d["endpoint"],"registry native endpoint/readout/unit binding mismatch")
    _validate_endpoint_query(endpoint.query())
    conditions=dict(role="source_native_ADMET_prediction_query",native_entity=observation.entity.as_mapping(),
        endpoint=endpoint.query(),assay_context=observation.context.as_mapping())
    targets=dict(native_measurement=observation.cell.as_mapping(),applicability=observation.applicability,evidence_role=observation.evidence_role)
    p=dict(native_observation=d,registry_binding=registry.binding(),endpoint_binding=endpoint.binding(),
        native_observation_sha256=digest(observation.native_bytes),training_eligibility=observation.training_eligibility,
        task_admitted=False,model_supervision_supported=False)
    return AdmetPredictionView(canonical(conditions),canonical(targets),canonical(p))


@dataclass(frozen=True)
class DesiredProperty:
    endpoint_key: str
    native_constraint: NativeCell
    assay_context: AssayContext
    request_provenance_bytes: bytes

    def __post_init__(self):
        _text(self.endpoint_key,"desired endpoint")
        if (not isinstance(self.native_constraint,NativeCell) or not self.native_constraint.numeric_loss_eligible
                or not isinstance(self.assay_context,AssayContext) or type(self.request_provenance_bytes) is not bytes):
            raise ValueError("explicit typed desired constraint/context and separate provenance required")
        _json(self.request_provenance_bytes)
    @property
    def observed_label_count(self):return 0
    @property
    def evidence_role(self):return "desired"


def desired_admet_inputs(molecule_class,constraints,registry):
    constraints=tuple(constraints)
    if molecule_class not in MOLECULE_CLASSES or not constraints or not isinstance(registry,EndpointRegistry):
        raise ValueError("explicit class, registry and desired constraints required")
    result=[];seen=set()
    for c in constraints:
        if not isinstance(c,DesiredProperty) or c.endpoint_key in seen:raise ValueError("unique desired properties, never observed labels required")
        seen.add(c.endpoint_key);definition=registry.get(c.endpoint_key)
        if definition.applicability_for(molecule_class)=="inapplicable":raise ValueError("endpoint explicitly inapplicable to requested chemistry class")
        _validate_endpoint_query(definition.query())
        result.append(dict(endpoint=definition.query(),native_constraint=c.native_constraint.constraint_fields(),assay_context=c.assay_context.as_mapping(),
            applicability=definition.applicability_for(molecule_class)))
    return dict(role="desired_ADMET_properties",molecule_class=molecule_class,desired_properties=result)



def _validate_endpoint_query(query):
    keys={"semantic_name","description","readout","unit","unit_basis","scale","measurement_type","binary_readout","unit_applicability"}
    if type(query) is not dict or set(query)!=keys:
        raise ValueError("only fixed endpoint query semantics allowed; no source IDs, outcomes or hashes")
    _text(query["semantic_name"],"semantic endpoint query")
    for k in keys-{"semantic_name"}:
        if query[k] is not None and type(query[k]) is not str:
            raise ValueError("native query definitions must remain text or explicit unknown")
    if query["unit_basis"] not in ("source_literal","official_source_schema","unresolved"):
        raise ValueError("native query unit basis required")
    if query["unit_applicability"] not in ("declared_native_unit","unknown","not_applicable"):
        raise ValueError("unknown physical units and inapplicable class units are separate")
    if query["unit_basis"]=="unresolved" and query["unit"] is not None:
        raise ValueError("unresolved physical unit cannot be supplied")
    if query["unit_applicability"]=="not_applicable" and (query["unit"] is not None or not query["binary_readout"]):
        raise ValueError("unit-inapplicable classification requires explicit binary readout")


def _validate_desired_properties(properties):
    if type(properties) is not list or not properties:
        raise ValueError("explicit desired property requests required")
    for prop in properties:
        if type(prop) is not dict or set(prop)!={"endpoint","native_constraint","assay_context","applicability"}:
            raise ValueError("desired query contains hidden source/target metadata")
        _validate_endpoint_query(prop["endpoint"])
        AssayContext.from_mapping(prop["assay_context"])
        if prop["applicability"] not in ("applicable","unknown"):
            raise ValueError("inapplicable property cannot become a desired objective")
        constraint=prop["native_constraint"]
        if type(constraint) is not dict or set(constraint)-{
                "native_literal","status","decimal_literal","relation","lower_decimal_literal",
                "upper_decimal_literal","lower_inclusive","upper_inclusive"}:
            raise ValueError("only explicit native desired bounds allowed")
        NativeCell.from_mapping(dict(constraint,observation_present=True,numeric_loss_eligible=True))


def admet_condition_graph(inputs):
    from .schema import GraphRecord,Node,Measurement
    from .provenance import Provenance
    from .tasks import ConditionalTask
    role=inputs.get("role")
    if role=="source_native_ADMET_prediction_query":
        if set(inputs)!={"role","native_entity","endpoint","assay_context"}:raise ValueError("target-free prediction fields required")
        NativeEntity.from_mapping(inputs["native_entity"]);AssayContext.from_mapping(inputs["assay_context"])
        _validate_endpoint_query(inputs["endpoint"])
    elif role=="desired_ADMET_properties":
        if set(inputs)!={"role","molecule_class","desired_properties"}:raise ValueError("separate desired fields required")
        if inputs["molecule_class"] not in MOLECULE_CLASSES:raise ValueError("known chemistry class required")
        _validate_desired_properties(inputs["desired_properties"])
    else:raise ValueError("explicit prediction or desired role required")
    record=GraphRecord("input",(Node("molecular-endpoint-query",(
        Measurement("condition",canonical(inputs).decode(),None,"observed",True),
        Measurement("requested_output",None,None,"missing",False))),),(),
        (Provenance("supplied-context","ADMET-v1","input","0"*64,"not-observational-admission"),))
    task=ConditionalTask(((("node",0,"condition"),"condition"),(("node",0,"requested_output"),"target")),topology_known=True)
    return record,task


def _verified_bytes(root,receipt):
    path=(root/receipt["path"]).resolve()
    if not path.is_relative_to(root):raise ValueError("source outside project")
    raw=path.read_bytes()
    if type(receipt["bytes"]) is not int or len(raw)!=receipt["bytes"] or digest(raw)!=receipt["sha256"]:
        raise ValueError("native candidate/source receipt integrity mismatch")
    return raw


def read_admet_native_candidates(manifest,*,root,expected_manifest_sha256,shard_paths,limit,observation_ids=None):
    """Bounded qualified v2 native contract readback. Never a task/admission reader."""
    import pyarrow.parquet as pq
    if (expected_manifest_sha256!=CANDIDATE_V2_SHA256 or type(limit) is not int or not 1<=limit<=32
            or type(shard_paths) not in (tuple,list) or not 1<=len(shard_paths)<=4
            or any(type(x) is not str for x in shard_paths) or len(set(shard_paths))!=len(shard_paths)):
        raise ValueError("qualified v2 manifest and explicit bounded unique selectors required")
    if observation_ids is not None and (type(observation_ids) not in (list,tuple) or not 1<=len(observation_ids)<=limit
            or len(set(observation_ids))!=len(observation_ids)):
        raise ValueError("unique bounded observation IDs required")
    root=Path(root).resolve();path=(root/manifest).resolve()
    if not path.is_relative_to(root) or digest(path.read_bytes())!=CANDIDATE_V2_SHA256:
        raise ValueError("qualified native candidate manifest mismatch")
    m=_json(path.read_bytes())
    if type(m["task_admitted_observations"]) is not int or m["task_admitted_observations"]!=0:
        raise ValueError("this reader is native candidates only, not an admission extension")
    _json(_verified_bytes(root,m["source_handoff"]));_json(_verified_bytes(root,m["source_manifest"]))
    shards={s["path"]:s for s in m["shards"]}
    if len(shards)!=len(m["shards"]) or any(p not in shards for p in shard_paths):raise ValueError("unqualified or duplicate shard allocation")
    wanted=None if observation_ids is None else set(observation_ids)
    out=[];seen=set();source_readers={}

    def source_row(source,index):
        # Per-call only: verify/decode each source once, retain requested rows
        # (at most32), and never parse beyond the current requested ordinal.
        key=source["sha256"]
        if key not in source_readers:
            source_readers[key]=dict(text=_verified_bytes(root,source).decode("utf8"),
                parser=None,position=-1,rows={})
        state=source_readers[key]
        if index in state["rows"]:return state["rows"][index]
        if state["parser"] is None or index<=state["position"]:
            # Explicit shard order may go backwards; preserve the original
            # prefix/error semantics without caching every skipped CSV row.
            state["parser"]=enumerate(csv.DictReader(io.StringIO(state["text"],newline="")))
            state["position"]=-1
        for i,original in state["parser"]:
            state["position"]=i
            if i==index:
                state["rows"][index]=original
                return original
        raise ValueError("native source row absent")

    for selected in shard_paths:
        sh=shards[selected];table=pq.read_table(io.BytesIO(_verified_bytes(root,sh)),use_threads=False)
        ids=table["observation_id"].to_pylist()
        if table.num_rows!=sh["rows"] or len(set(ids))!=len(ids) or set(ids)&seen:raise ValueError("native observation count/allocation mismatch")
        seen.update(ids)
        for row in table.to_pylist():
            if wanted is not None and row["observation_id"] not in wanted:continue
            raw=row["native_observation"]
            if type(raw) is not bytes or digest(raw)!=row["native_observation_sha256"]:raise ValueError("native observation payload mismatch")
            obs=AdmetObservation.from_native(raw);d=obs.native();cell=obs.cell;ref=d["source_ref"];source=ref["source"]
            index=ref["native_row_index"]
            if type(index) is not int or index<0:raise ValueError("native source row index required")
            expected=dict(observation_id=digest(canonical([source["sha256"],index,d["endpoint"]["source_endpoint"]])),
                source_family=ref["native_entity_namespace"],source_native_row_index=index,source_endpoint=d["endpoint"]["source_endpoint"],
                source_native_entity_id=ref["native_entity_id"],source_parent_protocol=d["source_split"]["parent_protocol"],
                native_observation_sha256=digest(raw),observation_present=cell.observation_present,
                native_numeric_parse_pass=cell.numeric_loss_eligible,native_censored=cell.status=="censored_bound",
                physical_unit_source_defined=d["endpoint"].get("unit") is not None,task_admitted=False,model_adapter_supported=False)
            _same({k:v for k,v in row.items() if k!="native_observation"},expected,"native source/cell/mask/split/type envelope mismatch")
            original=source_row(source,index)
            _same(original,d["native_source_record"],"complete original CSV/unknown native columns/literals mismatch")
            out.append(obs)
            if wanted is None and len(out)==limit:return tuple(out)
    if wanted is not None:
        found={digest(canonical([o.native()["source_ref"]["source"]["sha256"],o.native()["source_ref"]["native_row_index"],o.native()["endpoint"]["source_endpoint"]])) for o in out}
        if found!=wanted:raise ValueError("requested native candidates absent from bounded source")
    return tuple(out)
