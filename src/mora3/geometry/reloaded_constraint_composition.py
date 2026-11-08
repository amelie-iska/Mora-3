"""Request-bound synthetic reload -> numeric/freeze diagnostic composition.

No source IO, writer invocation, pre-export coordinate callback, model/optimizer
or authority grant. Frozen leaf flags/hardholds are never rewritten.
"""
from dataclasses import dataclass,field,asdict
from enum import Enum
import hashlib
import json
import time
import numpy as np
from ..data import spatial_design_constraints as request_types
from . import rigid_constraints as rigid
from . import spatial_constraint_evaluation as numeric_leaf
from . import fixed_region_preservation as fixed_leaf
from . import structure_export as export_leaf

REVISION="Mora3.TSE.ReloadedConstraintComposition/v1"

class Status(str,Enum):
    DIAGNOSTIC_PASS="PASS_SYNTHETIC_DIAGNOSTIC_ONLY"
    VIOLATED_CANDIDATE="SUPPLIED_CANDIDATE_VIOLATES_HARD_CONSTRAINT"
    NOT_EVALUABLE="NOT_EVALUABLE"
    UNSUPPORTED="UNSUPPORTED"
    BUDGET_EXHAUSTED="BUDGET_EXHAUSTED"

class Use(str,Enum):
    OUTPUT="numeric_and_or_frozen_candidate_output"
    NUMERIC_REFERENCE="numeric_original_reference"
    FROZEN_SOURCE="original_supplied_frozen_input"
    FROZEN_OUTPUT="frozen_only_candidate_output"

def _text(x,label,empty=False):
    if type(x) is not str or len(x)>128 or (not x and not empty) or any(ord(c)<32 for c in x):
        raise ValueError("bounded exact "+label)

def _integer(x,label,minimum=0,maximum=2**63-1):
    if type(x) is not int or not minimum<=x<=maximum:raise ValueError("exact bounded "+label)

def _typed(items,cls,label,limit=128):
    if type(items) is not tuple or len(items)>limit or any(type(x) is not cls for x in items):
        raise ValueError("exact bounded "+label+" tuple")
    for x in items:x.__post_init__()

def _hash(raw):return hashlib.sha256(raw).hexdigest()

def _digest(value):
    return _hash(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8"))

def _hash_string(x):
    if type(x) is not str or len(x)!=64 or any(c not in "0123456789abcdef" for c in x):
        raise ValueError("exact SHA256 declaration")

@dataclass(frozen=True)
class Slot:
    chain:str
    index:int
    insertion:str=""
    def __post_init__(self):
        _text(self.chain,"chain");_integer(self.index,"native slot");_text(self.insertion,"insertion",True)

@dataclass(frozen=True)
class RecordBinding:
    key:str
    artifact:str
    namespace:str
    atoms:tuple[request_types.AtomKey,...]
    body:numeric_leaf.Body
    purpose:numeric_leaf.Purpose
    role:request_types.Role
    use:Use
    def __post_init__(self):
        for x in (self.key,self.artifact,self.namespace):_text(x,"record/artifact/namespace")
        _typed(self.atoms,request_types.AtomKey,"complete native axis")
        if not self.atoms or any(a.entity!=self.namespace for a in self.atoms):
            raise ValueError("explicit complete native namespace")
        if len({(a.entity,a.ordinal) for a in self.atoms})!=len(self.atoms):
            raise ValueError("duplicate native atom binding")
        if type(self.body) is not numeric_leaf.Body or type(self.purpose) is not numeric_leaf.Purpose or type(self.role) is not request_types.Role or type(self.use) is not Use:
            raise ValueError("exact source/body/purpose/use roles")
        if self.use in (Use.OUTPUT,Use.FROZEN_OUTPUT):
            if self.artifact!="candidate" or self.purpose is not numeric_leaf.Purpose.CANDIDATE_OUTPUT:
                raise ValueError("candidate output must originate only from candidate reload")
        elif self.artifact=="candidate" or self.purpose is numeric_leaf.Purpose.CANDIDATE_OUTPUT:
            raise ValueError("original reference cannot alias candidate output")
        if self.use is Use.FROZEN_SOURCE and (self.body is not numeric_leaf.Body.PROTEIN or self.purpose is not numeric_leaf.Purpose.GUIDANCE_REFERENCE):
            raise ValueError("frozen source must be explicit supplied protein input")

@dataclass(frozen=True)
class IdentityBinding:
    native:request_types.ResidueKey
    chain_identifier:str
    output_slot:Slot
    def __post_init__(self):
        if type(self.native) is not request_types.ResidueKey:raise ValueError("exact native identity subject")
        self.native.__post_init__();_text(self.chain_identifier,"original output chain identifier")
        if type(self.output_slot) is not Slot:raise ValueError("explicit identity slot, not gold identity")
        self.output_slot.__post_init__()

@dataclass(frozen=True)
class GeometryBinding:
    native:request_types.ResidueKey
    input_key:str
    source_key:str
    output_key:str
    declared_atoms:tuple[request_types.AtomKey,...]
    pairs:tuple[request_types.AtomPair,...]
    coverage:fixed_leaf.Coverage
    frame_policy:fixed_leaf.FramePolicy
    def __post_init__(self):
        if type(self.native) is not request_types.ResidueKey:raise ValueError("exact frozen native subject")
        self.native.__post_init__()
        for x in (self.input_key,self.source_key,self.output_key):_text(x,"geometry binding key")
        _typed(self.declared_atoms,request_types.AtomKey,"explicit full subject declaration")
        _typed(self.pairs,request_types.AtomPair,"frozen correspondence")
        if type(self.coverage) is not fixed_leaf.Coverage or type(self.frame_policy) is not fixed_leaf.FramePolicy:
            raise ValueError("exact coverage/frame declarations")

@dataclass(frozen=True)
class EvaluationScope:
    records:tuple[RecordBinding,...]
    numeric_scopes:tuple[numeric_leaf.Scope,...]
    numerics:rigid.Numerics
    max_input_bytes:int
    identities:tuple[IdentityBinding,...]=()
    geometries:tuple[GeometryBinding,...]=()
    def __post_init__(self):
        _typed(self.records,RecordBinding,"record bindings",16)
        _typed(self.numeric_scopes,numeric_leaf.Scope,"numeric scopes",16)
        _typed(self.identities,IdentityBinding,"identity bindings")
        _typed(self.geometries,GeometryBinding,"geometry bindings")
        if type(self.numerics) is not rigid.Numerics:raise ValueError("exact explicit numerical guard config")
        self.numerics.__post_init__()
        _integer(self.max_input_bytes,"per-invocation decode budget",1,16777216)
        if len({r.key for r in self.records})!=len(self.records) or len({(r.artifact,r.namespace) for r in self.records})!=len(self.records):
            raise ValueError("duplicate records/views; no double accounting")
        if len({(r.native.chain,r.native.index,r.native.insertion) for r in self.identities})!=len(self.identities):
            raise ValueError("duplicate identity subject")
        if len({(r.native.chain,r.native.index,r.native.insertion) for r in self.geometries})!=len(self.geometries):
            raise ValueError("duplicate geometry subject")

@dataclass(frozen=True)
class ReferenceFiles:
    key:str
    files:export_leaf.ExportBundle
    def __post_init__(self):
        _text(self.key,"original reference artifact key")
        if self.key=="candidate" or type(self.files) is not export_leaf.ExportBundle:
            raise ValueError("distinct exact original reference export")

@dataclass(frozen=True)
class ReferenceExpectation:
    key:str
    expectation:export_leaf.ExportExpectation
    def __post_init__(self):
        _text(self.key,"external original reference key")
        if self.key=="candidate" or type(self.expectation) is not export_leaf.ExportExpectation:
            raise ValueError("distinct exact external reference expectation")
        self.expectation.__post_init__()

@dataclass(frozen=True)
class JointExpectation:
    request_sha256:str
    compiled_sha256:str
    scope_sha256:str
    candidate:export_leaf.ExportExpectation
    references:tuple[ReferenceExpectation,...]
    binding_sha256:str
    revision:str=REVISION
    def __post_init__(self):
        for x in (self.request_sha256,self.compiled_sha256,self.scope_sha256,self.binding_sha256):_hash_string(x)
        if type(self.revision) is not str or self.revision!=REVISION or type(self.candidate) is not export_leaf.ExportExpectation:
            raise ValueError("exact current external declaration")
        self.candidate.__post_init__();_typed(self.references,ReferenceExpectation,"original expectations",3)
        if len({x.key for x in self.references})!=len(self.references):
            raise ValueError("duplicate original reference keys")
        identities=[self.candidate.identity_sha256]+[r.expectation.identity_sha256 for r in self.references]
        if len(set(identities))!=len(identities):
            raise ValueError("candidate/reference original identity alias")
        if _digest(_expectation_fields(self.request_sha256,self.compiled_sha256,self.scope_sha256,self.candidate,self.references))!=self.binding_sha256:
            raise ValueError("changed/stale external expectation declaration")

def _expectation_fields(request_sha,compiled_sha,scope_sha,candidate,references):
    return dict(revision=REVISION,request_sha256=request_sha,compiled_sha256=compiled_sha,scope_sha256=scope_sha,
        candidate=asdict(candidate),references=[asdict(x) for x in references])

def _fresh(request,compiled):
    if type(request) is not request_types.DesignRequest or type(compiled) is not request_types.CompiledDesignPlan:
        raise ValueError("exact raw request/current compiled type")
    fresh=request_types.compile_design_request(request)
    if compiled.canonical_bytes()!=fresh.canonical_bytes():
        raise ValueError("forged/stale compiled decisions/predictors/statuses")
    return fresh

def declare_expectation(request,compiled,scope,candidate_expectation,reference_expectations):
    """Metadata-only caller preparation, NOT approval/source/runtime authority.

    This deliberately takes original expectations, never export bytes or their
    embedded receipts. Execution receives this independently supplied binding.
    """
    fresh=_fresh(request,compiled)
    if type(scope) is not EvaluationScope:raise ValueError("exact declared evaluation scope")
    scope.__post_init__()
    if type(candidate_expectation) is not export_leaf.ExportExpectation:raise ValueError("external original candidate expectation")
    candidate_expectation.__post_init__();_typed(reference_expectations,ReferenceExpectation,"original expectations",3)
    a,b,c=fresh.request_sha256,_hash(fresh.canonical_bytes()),_digest(asdict(scope))
    seal=_digest(_expectation_fields(a,b,c,candidate_expectation,reference_expectations))
    return JointExpectation(a,b,c,candidate_expectation,reference_expectations,seal)

@dataclass(frozen=True)
class ReloadedArtifact:
    key:str
    result:export_leaf.ReloadedPacket
    fasta_status:export_leaf.FastaStatus

@dataclass(frozen=True)
class Consumption:
    key:str
    artifact:str
    namespace:str
    original_identity_sha256:str
    native_axis_sha256:str
    points:int
    origin:str="strict_reloaded_native_packet"

@dataclass(frozen=True)
class CompositionReport:
    request_sha256:str
    binding_sha256:str
    status:Status
    reasons:tuple[str,...]
    artifacts:tuple[ReloadedArtifact,...]
    consumed:tuple[Consumption,...]
    numeric:numeric_leaf.RequestEvaluation|None
    fixed:fixed_leaf.FreezeReport|None
    unsupported_profiles:tuple[str,...]
    authorization_status:str=field(default="HOLD",init=False)
    actual_export_acceptance:bool=field(default=False,init=False)
    numeric_adapter_hold_cleared:bool=field(default=False,init=False)
    source_native_model_authority:None=field(default=None,init=False)
    def require_native_or_model_use(self):
        raise RuntimeError("synthetic diagnostic only; no source/native/model/training/export authorization")
    def require_export_constraint_acceptance(self):
        raise RuntimeError("genuine independent source/context/coverage authority remains unavailable")

def _file_size(files):
    total=0
    for raw in (files.xyz,files.mmcif,files.fasta,files.sidecar):
        if raw is not None:
            if type(raw) is not bytes:raise ValueError("exact export bytes, not file/callback")
            total+=len(raw)
    return total

def _report(fresh,expected,status,reasons,artifacts=(),consumed=(),numeric=None,fixed=None,unsupported=()):
    return CompositionReport(fresh.request_sha256,expected.binding_sha256,status,tuple(reasons),tuple(artifacts),tuple(consumed),numeric,fixed,tuple(unsupported))

def _expired(start,request):
    return time.monotonic()-start>request.budget.wall_seconds

def evaluate_reloaded_constraints(request,compiled,candidate_files,reference_files,scope,expectation,*,software):
    """One bounded composition. Every coordinate comes from strict reload."""
    started=time.monotonic()
    if type(software) is not numeric_leaf.SoftwareBinding:raise ValueError("exact pinned software declaration")
    software.__post_init__()
    fresh=_fresh(request,compiled)
    if type(scope) is not EvaluationScope or type(expectation) is not JointExpectation:
        raise ValueError("explicit external scope+expectation required")
    scope.__post_init__();expectation.__post_init__()
    if (fresh.request_sha256,_hash(fresh.canonical_bytes()),_digest(asdict(scope)))!=(expectation.request_sha256,expectation.compiled_sha256,expectation.scope_sha256):
        raise ValueError("actual request/map/context/limit/scope changed from external binding")
    if type(candidate_files) is not export_leaf.ExportBundle:raise ValueError("exact candidate bytes")
    _typed(reference_files,ReferenceFiles,"original reference files",3)
    if tuple(x.key for x in reference_files)!=tuple(x.key for x in expectation.references):
        raise ValueError("complete ordered original reference expectations required")
    artifact_keys={"candidate"}|{r.key for r in reference_files}
    if any(r.artifact not in artifact_keys for r in scope.records):
        raise ValueError("record bound to missing original artifact")
    if any(r.key not in {b.artifact for b in scope.records} for r in reference_files):
        raise ValueError("unbound/unused original reference artifact")
    size=_file_size(candidate_files)+sum(_file_size(r.files) for r in reference_files)
    if size>scope.max_input_bytes or size>request.budget.memory_bytes or _expired(started,request):
        return _report(fresh,expectation,Status.BUDGET_EXHAUSTED,("declared_decode_or_wall_budget_exhausted_before_reload",))
    # Never consult ExportBundle.expectation. Only EXTERNAL expectations select identity.
    artifacts=[ReloadedArtifact("candidate",export_leaf.reload_bundle(candidate_files,expectation.candidate),candidate_files.fasta_status)]
    for original,bound in zip(reference_files,expectation.references):
        artifacts.append(ReloadedArtifact(original.key,export_leaf.reload_bundle(original.files,bound.expectation),original.files.fasta_status))
        if _expired(started,request):
            return _report(fresh,expectation,Status.BUDGET_EXHAUSTED,("wall_budget_exhausted_during_reload",),artifacts)
    by_artifact={x.key:x for x in artifacts}
    records={};components={};consumed=[]
    expected_ids={"candidate":expectation.candidate.identity_sha256}
    expected_ids.update({r.key:r.expectation.identity_sha256 for r in expectation.references})
    for binding in scope.records:
        packet=by_artifact[binding.artifact].result.packet
        selected=tuple(a for a in packet.atoms if a.native.entity==binding.namespace)
        if tuple(a.native for a in selected)!=binding.atoms:
            raise ValueError("bound native namespace/order/metadata incomplete; no intersection/crop")
        roles=tuple(a.component for a in selected)
        if binding.body is numeric_leaf.Body.TS and any(x is not export_leaf.Component.TS for x in roles):
            raise ValueError("TS record contains nonTS role")
        if binding.body is numeric_leaf.Body.PROTEIN and any(x is not export_leaf.Component.ENZYME for x in roles):
            raise ValueError("protein record contains nonenzyme/cofactor role")
        if binding.body is numeric_leaf.Body.COMPLEX and (export_leaf.Component.TS not in roles or export_leaf.Component.ENZYME not in roles):
            raise ValueError("explicit complex requires TS and enzyme roles")
        coordinates=np.array([a.xyz for a in selected],dtype=np.float64)
        records[binding.key]=numeric_leaf.Coordinates(binding.namespace,binding.body,binding.purpose,binding.role,binding.atoms,coordinates,packet.frame)
        components[binding.key]={a.native:a.component for a in selected}
        consumed.append(Consumption(binding.key,binding.artifact,binding.namespace,expected_ids[binding.artifact],_digest([asdict(a) for a in binding.atoms]),len(selected)))
    if sum(r.xyz.nbytes for r in records.values())>request.budget.memory_bytes or _expired(started,request):
        return _report(fresh,expectation,Status.BUDGET_EXHAUSTED,("coordinate_view_or_wall_budget_exhausted",),artifacts,consumed)
    # Explicit component provenance prevents a residue-labelled cofactor from
    # masquerading as a protein pose anchor inside an otherwise valid complex.
    numeric_bindings=tuple(b for b in scope.records if b.use in (Use.OUTPUT,Use.NUMERIC_REFERENCE))
    by_namespace={b.namespace:b for b in numeric_bindings}
    if len(by_namespace)!=len(numeric_bindings):raise ValueError("ambiguous numeric namespace role")
    for c in request.constraints:
        if c.metric is request_types.Metric.TS_POSE:
            for p in c.atoms:
                for a in (p.source,p.reference):
                    bound=by_namespace.get(a.entity)
                    if bound is not None and components[bound.key].get(a) is not export_leaf.Component.TS:
                        raise ValueError("pose-scored atom not exact reloaded TS role")
            for p in c.protein_anchors:
                for a in (p.source,p.reference):
                    bound=by_namespace.get(a.entity)
                    if bound is not None and components[bound.key].get(a) is not export_leaf.Component.ENZYME:
                        raise ValueError("cofactor/nonprotein cannot be pose anchor")
    unsupported_numeric=any(c.alignment is request_types.Alignment.FIXED_FRAME for c in request.constraints)
    numeric=None
    if not unsupported_numeric:
        numeric=numeric_leaf.evaluate_request(request,fresh,numeric_leaf.Registry(tuple(records[b.key] for b in numeric_bindings)),
            scope.numeric_scopes,software=software,numerics=scope.numerics)
    if _expired(started,request):
        return _report(fresh,expectation,Status.BUDGET_EXHAUSTED,("wall_budget_exhausted_after_numeric",),artifacts,consumed,numeric)
    identities=[]
    candidate_packet=by_artifact["candidate"].result.packet
    for binding in scope.identities:
        hits=tuple(r for chain in candidate_packet.chains if chain.identifier==binding.chain_identifier for r in chain.residues
            if (r.chain,r.index,r.insertion)==(binding.output_slot.chain,binding.output_slot.index,binding.output_slot.insertion))
        if len(hits)>1:raise ValueError("ambiguous reloaded output identity")
        identities.append(fixed_leaf.IdentityWitness(binding.native,hits[0] if hits else None))
    geometry=[]
    binding_map={b.key:b for b in scope.records}
    for binding in scope.geometries:
        src=binding_map.get(binding.source_key);out=binding_map.get(binding.output_key)
        if src is None or out is None or src.use is not Use.FROZEN_SOURCE or out.use not in (Use.OUTPUT,Use.FROZEN_OUTPUT):
            raise ValueError("frozen original/candidate role binding required")
        if src.namespace!=binding.input_key:raise ValueError("frozen source exact raw-input namespace mismatch")
        declaration=fixed_leaf.AtomDeclaration(binding.native,binding.input_key,binding.declared_atoms,binding.coverage)
        geometry.append(fixed_leaf.GeometryWitness(declaration,records[binding.source_key],records[binding.output_key],binding.pairs,binding.frame_policy))
    fixed=fixed_leaf.evaluate_fixed_regions(request,fresh,tuple(identities),tuple(geometry),software=software)
    if _expired(started,request):
        return _report(fresh,expectation,Status.BUDGET_EXHAUSTED,("wall_budget_exhausted_after_freeze",),artifacts,consumed,numeric,fixed)
    unsupported=tuple(x.key+":canonical_fasta" for x in artifacts if x.result.packet.chains and x.fasta_status is export_leaf.FastaStatus.UNSUPPORTED)
    has_freeze=any(d.freeze_sequence or d.freeze_all_atom_geometry for d in fresh.sequence_decisions+fresh.template_decisions)
    hard_numeric=[] if numeric is None else [x.evaluation.status for x in numeric.results if x.constraint.required]
    reasons=[]
    if rigid.Status.VIOLATED in hard_numeric or (has_freeze and fixed.status is fixed_leaf.Status.VIOLATED):
        status=Status.VIOLATED_CANDIDATE;reasons.append("provided_candidate_violates_required_constraint_not_global_infeasibility")
    elif unsupported_numeric or unsupported or (has_freeze and fixed.status is fixed_leaf.Status.UNSUPPORTED):
        status=Status.UNSUPPORTED;reasons.append("explicit_unsupported_profile_or_frame")
    elif rigid.Status.NOT_EVALUABLE in hard_numeric or (has_freeze and fixed.status is fixed_leaf.Status.NOT_EVALUABLE):
        status=Status.NOT_EVALUABLE;reasons.append("required_numeric_or_frozen_evidence_unavailable")
    elif numeric is not None and any(x!="all_atom_freeze_unverified" for x in numeric.policy_unknowns):
        status=Status.NOT_EVALUABLE;reasons.append("unhandled_identity_policy_unknown")
    elif not hard_numeric and not has_freeze:
        status=Status.NOT_EVALUABLE;reasons.append("no_required_numeric_or_frozen_constraint")
    else:
        status=Status.DIAGNOSTIC_PASS;reasons.append("synthetic_required_checks_pass_source_authorization_still_HOLD")
    # The numeric leaf's all_atom_freeze_unverified field is preserved verbatim.
    # Independent fixed checks inform ONLY this diagnostic status; no gate mutates.
    return _report(fresh,expectation,status,reasons,artifacts,consumed,numeric,fixed,unsupported)
