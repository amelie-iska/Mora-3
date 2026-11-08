"""Synthetic in-memory request-v2 -> bounded rigid geometry adapter.

No parser/file IO/model/source/export qualification. Install only with the
exact dependency versions reviewed in this package; no live-package import here.
"""
from dataclasses import dataclass, field
from enum import Enum
import json
import math
import numpy as np
from ..data import spatial_design_constraints as request_types
from . import rigid_constraints as rigid

REQUEST_SHA256 = "0c06abe200dd41144c9c9deea5b5812dc1dd395b9a1846abb2a8e12a2cf94817"
RIGID_SHA256 = "4b1a25c701ca0142962ed0a6b46846de5b8c6fc687eefe55e2239aa6b7cc61d8"
COMPILER_REVISION = "Mora3.TSE.RequestRegionCompiler/v2"
ADAPTER_REVISION = "Mora3.TSE.InMemoryConstraintEvaluation/v1"
_VISIBLE_ROLES = frozenset((request_types.Role.SOURCE_OBSERVED,request_types.Role.USER_SUPPLIED,
    request_types.Role.DESIRED,request_types.Role.GENERATED_INTERMEDIATE))
_BACKBONE = frozenset(("N","CA","C","O","OXT"))

class Body(str,Enum):
    TS = "explicit_transition_state_atoms"
    PROTEIN = "explicit_enzyme_protein_atoms"
    COMPLEX = "explicit_reference_TS_enzyme_complex"

class Purpose(str,Enum):
    CANDIDATE_OUTPUT = "candidate_geometry_output_not_condition"
    GUIDANCE_REFERENCE = "caller_visible_guidance_reference"
    EVALUATION_ONLY_REFERENCE = "post_generation_evaluation_reference"

@dataclass(frozen=True)
class SoftwareBinding:
    request_sha256:str
    rigid_sha256:str
    compiler_revision:str
    def __post_init__(self):
        if (type(self.request_sha256) is not str or self.request_sha256!=REQUEST_SHA256
            or type(self.rigid_sha256) is not str or self.rigid_sha256!=RIGID_SHA256
            or type(self.compiler_revision) is not str or self.compiler_revision!=COMPILER_REVISION):
            raise ValueError("exact reviewed software declaration required; no old/alternative hashes")
        if request_types.COMPILER_REVISION!=COMPILER_REVISION or rigid.MAX_POINTS!=128:
            raise ValueError("dependency API/profile revision mismatch")
        # This is a declaration, not code-file inspection/source/runtime authority.
        # The isolated bootstrap/owner must separately verify actual source bytes.

def native_math_id(atom):
    if type(atom) is not request_types.AtomKey:raise ValueError("exact native AtomKey required")
    atom.__post_init__()
    residue=atom.residue
    key=None if residue is None else json.dumps([residue.chain,residue.index,residue.insertion],
        ensure_ascii=False,separators=(",",":"),allow_nan=False)
    return rigid.AtomID(atom.entity,atom.ordinal,atom.element,atom.name,key,
        None if residue is None else residue.identity,atom.isotope)

@dataclass(frozen=True,eq=False)
class Coordinates:
    namespace:str
    body:Body
    purpose:Purpose
    role:request_types.Role
    atoms:tuple[request_types.AtomKey,...]
    xyz:np.ndarray
    frame:str
    units:str="angstrom"
    def __post_init__(self):
        if type(self.namespace) is not str or not self.namespace:
            raise ValueError("exact coordinate namespace")
        if type(self.body) is not Body or type(self.purpose) is not Purpose or type(self.role) is not request_types.Role:
            raise ValueError("exact body/purpose/role required")
        if type(self.atoms) is not tuple or not 1<=len(self.atoms)<=128:
            raise ValueError("unsupported full-structure profile:1..128 points; no cropping")
        if any(type(a) is not request_types.AtomKey or a.entity!=self.namespace for a in self.atoms):
            raise ValueError("exact native coordinate axis/namespace required")
        for a in self.atoms:a.__post_init__()
        if len({(a.entity,a.ordinal) for a in self.atoms})!=len(self.atoms):
            raise ValueError("duplicate/ambiguous native atom coordinate")
        if self.body is Body.TS and any(a.residue is not None for a in self.atoms):
            raise ValueError("TS-scored geometry cannot masquerade as protein")
        if self.body is Body.PROTEIN and any(a.residue is None for a in self.atoms):
            raise ValueError("enzyme coordinates require declared protein residue identities")
        if self.body is Body.COMPLEX:
            if self.purpose is Purpose.CANDIDATE_OUTPUT:
                raise ValueError("candidate TS/protein bodies must be separate explicit registries")
            if not any(a.residue is None for a in self.atoms) or not any(a.residue is not None for a in self.atoms):
                raise ValueError("complex reference must explicitly contain TS and protein atoms")
        if self.purpose is Purpose.CANDIDATE_OUTPUT and self.role not in (
            request_types.Role.USER_SUPPLIED,request_types.Role.GENERATED_INTERMEDIATE):
            raise ValueError("target/evaluation/source labels cannot be candidate output")
        if self.purpose is Purpose.GUIDANCE_REFERENCE and self.role not in _VISIBLE_ROLES:
            raise ValueError("hidden reference cannot guide")
        point=rigid.PointSet(tuple(native_math_id(a) for a in self.atoms),self.xyz,self.frame,128,self.units)
        object.__setattr__(self,"xyz",point.xyz)

@dataclass(frozen=True)
class Registry:
    records:tuple[Coordinates,...]
    origin:str="handcrafted_in_memory"
    def __post_init__(self):
        if type(self.origin) is not str or self.origin!="handcrafted_in_memory":
            raise ValueError("native/source registry entrypoints unqualified")
        if type(self.records) is not tuple or len(self.records)>16 or any(type(r) is not Coordinates for r in self.records):
            raise ValueError("bounded exact handcrafted coordinate registry required")
        if len({r.namespace for r in self.records})!=len(self.records):
            raise ValueError("duplicate registry namespace")
        for r in self.records:
            if not 1<=len(r.atoms)<=128:raise ValueError("unsupported128 point profile")

@dataclass(frozen=True)
class Scope:
    constraint_key:str
    coverage:rigid.Coverage
    weights:tuple[int|float,...]
    anchor_coverage:rigid.Coverage|None=None
    anchor_weights:tuple[int|float,...]=()
    def __post_init__(self):
        if type(self.constraint_key) is not str or not self.constraint_key or type(self.coverage) is not rigid.Coverage:
            raise ValueError("exact constraint key and explicit source coverage")
        if self.anchor_coverage is not None and type(self.anchor_coverage) is not rigid.Coverage:
            raise ValueError("explicit anchor source coverage")
        for weights in (self.weights,self.anchor_weights):
            if type(weights) is not tuple or len(weights)>128:
                raise ValueError("explicit immutable bounded weights")
            for w in weights:
                if type(w) not in (int,float):
                    raise ValueError("positive finite weights, not bool/type aliases")
                try:valid=math.isfinite(w) and w>0
                except (ValueError,OverflowError):valid=False
                if not valid:raise ValueError("positive finite weights required")

@dataclass(frozen=True)
class ConstraintResult:
    constraint:request_types.ConstraintSpec
    evaluation:rigid.Evaluation
    source_registry_total:int
    reference_registry_total:int
    selected_reference_count:int
    protein_anchor_count:int
    native_pairs:tuple[request_types.AtomPair,...]
    native_anchor_pairs:tuple[request_types.AtomPair,...]

@dataclass(frozen=True)
class RequestEvaluation:
    request_sha256:str
    compiler_revision:str
    adapter_revision:str
    results:tuple[ConstraintResult,...]
    raw_numeric_overall:rigid.OverallReport
    policy_unknowns:tuple[str,...]
    predictor_input_keys:tuple[str,...]
    predictor_reference_keys:tuple[str,...]
    sequence_decisions:tuple[request_types.ResolvedResidueDecision,...]
    template_decisions:tuple[request_types.ResolvedResidueDecision,...]
    export_reload_verified:bool=field(default=False,init=False)
    source_or_model_authority:bool=field(default=False,init=False)
    def require_native_or_model_use(self):
        raise RuntimeError("handcrafted numeric adapter; native/model/training authority unconditionally closed")
    def require_export_verified_pass(self):
        raise RuntimeError("export/reload proof unavailable; raw numeric PASS is insufficient")

def _point(record,selection=None):
    atoms=record.atoms if selection is None else selection
    lookup={a:i for i,a in enumerate(record.atoms)}
    if any(a not in lookup for a in atoms):raise ValueError("exact native atom metadata missing; no remapping/intersection")
    xyz=record.xyz[[lookup[a] for a in atoms]]
    return rigid.PointSet(tuple(native_math_id(a) for a in atoms),xyz,record.frame,128,record.units)

def _mapping(pairs,coverage):
    for p in pairs:
        p.__post_init__()
        if p.source.isotope!=p.reference.isotope or p.source.formal_charge!=p.reference.formal_charge:
            raise ValueError("unknown/known isotope or charge mismatch; no implicit defaults")
    return rigid.Mapping(tuple(rigid.Pair(native_math_id(p.source),native_math_id(p.reference)) for p in pairs),coverage)

def _subject_atom(atom,compiled):
    residue=atom.residue
    if residue is None:raise ValueError("protein anchor must be a declared residue atom")
    # Exact explicit native slots retain chain/index/insertion; no coordinate inference.
    slot=(residue.chain,residue.index,residue.insertion)
    unique=tuple(dict.fromkeys(compiled.sequence_decisions+compiled.template_decisions))
    matches=tuple(d for d in unique if any(r is not None and (r.chain,r.index,r.insertion)==slot
        for r in (d.sequence,d.template)))
    if len(matches)!=1:raise ValueError("unbound/ambiguous output protein residue")
    d=matches[0]
    if d.redesign and atom.name not in _BACKBONE:
        raise ValueError("unknown redesigned sidechain cannot be scored or anchor")
    if d.chosen_identity is not None:
        if residue.identity!=d.chosen_identity:raise ValueError("candidate violates chosen sequence/freeze identity")
    elif not d.redesign or residue not in (d.sequence,d.template):
        raise ValueError("output identity unresolved; no invented supplied sequence")
    return d

def _unknown(metric,threshold,reason):
    return rigid.Evaluation(metric,rigid.Status.NOT_EVALUABLE,float(threshold) if threshold is not None else None,
        None,None,None,(),reason)

def evaluate_request(request,compiled,registry,scopes,*,software,numerics):
    """Recompile authoritative raw request; preflight ALL jobs before any fitting."""
    if type(software) is not SoftwareBinding:raise ValueError("explicit exact software declaration")
    software.__post_init__()
    if type(numerics) is not rigid.Numerics:raise ValueError("explicit numerical rank/orthogonality guards")
    numerics.__post_init__()
    if type(request) is not request_types.DesignRequest or type(compiled) is not request_types.CompiledDesignPlan:
        raise ValueError("exact request-v2 compiled types")
    fresh=request_types.compile_design_request(request)
    if compiled.compiler_revision!=COMPILER_REVISION or compiled.canonical_bytes()!=fresh.canonical_bytes():
        raise ValueError("compiled metadata/masks/predictors/statuses not derived from raw request")
    compiled=fresh
    if type(registry) is not Registry:raise ValueError("handcrafted registry required")
    registry.__post_init__()
    # Fresh immutable validation snapshots; do not mutate caller record fields.
    registry=Registry(tuple(Coordinates(r.namespace,r.body,r.purpose,r.role,r.atoms,r.xyz,r.frame,r.units)
        for r in registry.records),registry.origin)
    if type(scopes) is not tuple or any(type(s) is not Scope for s in scopes):
        raise ValueError("exact ordered explicit coverage scopes")
    for s in scopes:s.__post_init__()
    if len(scopes)!=len(request.constraints):raise ValueError("complete explicit scopes")
    for c,s in zip(request.constraints,scopes):
        if len(s.weights)!=len(c.atoms) or len(s.anchor_weights)!=len(c.protein_anchors):
            raise ValueError("weights must cover exact scored/anchor pair axes")
    if tuple(s.constraint_key for s in scopes)!=tuple(c.key for c in request.constraints):
        raise ValueError("complete ordered scopes required; no implicit partial evaluation")
    refs={r.key:r for r in request.references}
    records={r.namespace:r for r in registry.records}
    total_bytes=sum(r.xyz.nbytes for r in registry.records)
    if total_bytes>request.budget.memory_bytes:raise ValueError("complete coordinate registry exceeds explicit memory budget")
    outputs=tuple(r for r in registry.records if r.purpose is Purpose.CANDIDATE_OUTPUT)
    if sum(len(r.atoms) for r in outputs)>request.budget.max_atoms:
        raise ValueError("complete candidate geometry exceeds caller atom budget")
    if any(r.namespace in {i.key for i in request.inputs} for r in outputs):
        raise ValueError("output/condition namespace collision")
    for rec in registry.records:
        if rec.purpose is Purpose.CANDIDATE_OUTPUT:
            if rec.body is Body.PROTEIN:
                for a in rec.atoms:_subject_atom(a,compiled)
        else:
            ref=refs.get(rec.namespace)
            if ref is None or not ref.available or rec.role is not ref.role or rec.atoms!=ref.atoms:
                raise ValueError("coordinate reference must bind complete exact declared role/axis")
            expected_body={request_types.ReferenceKind.TS:Body.TS,
                request_types.ReferenceKind.ENZYME_TEMPLATE:Body.PROTEIN,
                request_types.ReferenceKind.BOUND_COMPLEX:Body.COMPLEX}[ref.kind]
            if rec.body is not expected_body:raise ValueError("reference body role mismatch")
    jobs=[]
    for c,scope,declared in zip(request.constraints,scopes,compiled.constraint_statuses):
        metric={request_types.Metric.TS_SHAPE:rigid.Metric.TS_SHAPE,
            request_types.Metric.ENZYME:rigid.Metric.ENZYME_TEMPLATE,
            request_types.Metric.TS_POSE:rigid.Metric.TS_POSE}[c.metric]
        ref=records.get(c.reference_key)
        if declared.status=="NOT_EVALUABLE" or ref is None:
            jobs.append((c,metric,None,"reference_registry_unavailable",0,0,0,0));continue
        purpose=Purpose.GUIDANCE_REFERENCE if c.use is request_types.ConstraintUse.GUIDANCE else Purpose.EVALUATION_ONLY_REFERENCE
        if ref.purpose is not purpose:raise ValueError("guidance/evaluation reference purpose mismatch")
        if c.use is request_types.ConstraintUse.GUIDANCE and ref.role not in _VISIBLE_ROLES:
            raise ValueError("hidden reference must never condition")
        if not c.atoms or len({p.source.entity for p in c.atoms})!=1:
            raise ValueError("one exact scored candidate body required")
        src=records.get(c.atoms[0].source.entity)
        if src is None:
            jobs.append((c,metric,None,"candidate_registry_unavailable",0,len(ref.atoms),len(c.expected_reference_atoms),len(c.protein_anchors)));continue
        if any(p.source not in src.atoms or p.reference not in ref.atoms for p in c.atoms):
            raise ValueError("exact source/reference atom metadata including charge/isotope required")
        if src.purpose is not Purpose.CANDIDATE_OUTPUT:raise ValueError("target/reference cannot alias candidate output")
        expected_source=Body.PROTEIN if c.metric is request_types.Metric.ENZYME else Body.TS
        if src.body is not expected_source:raise ValueError("scored candidate TS/enzyme role mismatch")
        if c.metric is not request_types.Metric.TS_POSE and c.selection is request_types.Selection.ALL_ATOM:
            if c.expected_reference_atoms!=ref.atoms:raise ValueError("all-atom reference axis must be complete and in native order")
        target=_point(ref,c.expected_reference_atoms)
        source=_point(src)
        pairs=_mapping(c.atoms,scope.coverage)
        selection=rigid.Selection.BACKBONE if c.selection is request_types.Selection.BACKBONE else rigid.Selection.ALL
        if c.metric is request_types.Metric.TS_POSE:
            if ref.body is not Body.COMPLEX or scope.anchor_coverage is None or len({p.source.entity for p in c.protein_anchors})!=1:
                raise ValueError("pose requires explicit complex/protein-anchor coverage")
            enzyme=records.get(c.protein_anchors[0].source.entity)
            if enzyme is None:
                jobs.append((c,metric,None,"enzyme_anchor_registry_unavailable",len(src.atoms),len(ref.atoms),len(c.expected_reference_atoms),len(c.protein_anchors)));continue
            if enzyme.body is not Body.PROTEIN or enzyme.purpose is not Purpose.CANDIDATE_OUTPUT or enzyme.namespace==src.namespace:
                raise ValueError("protein anchors cannot be ligand/self/reference aliases")
            if src.frame!=enzyme.frame:raise ValueError("candidate TS/protein common frame required")
            if any(p.source.residue is None or p.reference.residue is None for p in c.protein_anchors):
                raise ValueError("pose anchors require explicit protein residues")
            if any(p.source.residue is not None or p.reference.residue is not None for p in c.atoms):
                raise ValueError("TS-scored atoms cannot be protein anchors")
            scored_src={(p.source.entity,p.source.ordinal) for p in c.atoms}
            scored_ref={(p.reference.entity,p.reference.ordinal) for p in c.atoms}
            if any((p.source.entity,p.source.ordinal) in scored_src or (p.reference.entity,p.reference.ordinal) in scored_ref for p in c.protein_anchors):
                raise ValueError("scored TS/protein anchor ID collision")
            if any(p.source not in enzyme.atoms or p.reference not in ref.atoms for p in c.protein_anchors):
                raise ValueError("exact protein anchor metadata required")
            for p in c.protein_anchors:_subject_atom(p.source,compiled)
            anchor_points=_point(enzyme)
            anchor_target=_point(ref,tuple(p.reference for p in c.protein_anchors))
            anchors=_mapping(c.protein_anchors,scope.anchor_coverage)
            payload=("pose",source,target,anchor_points,anchor_target,pairs,anchors)
        else:
            if c.alignment is not request_types.Alignment.PROPER_RIGID:
                raise ValueError("fixed-frame evaluation unsupported; never silently fit instead")
            if scope.anchor_coverage is not None:raise ValueError("anchor coverage only for pose")
            if src.body is Body.PROTEIN:
                for p in c.atoms:_subject_atom(p.source,compiled)
            payload=("template" if c.metric is request_types.Metric.ENZYME else "shape",source,target,pairs,selection)
        jobs.append((c,metric,payload,None,len(src.atoms),len(ref.atoms),len(c.expected_reference_atoms),len(c.protein_anchors)))
    results=[];requirements=[]
    for (c,metric,payload,reason,ns,nr,nselected,nanchors),scope in zip(jobs,scopes):
        if payload is None:result=_unknown(metric,c.threshold,reason)
        elif payload[0]=="pose":
            _,x,y,a,b,pairs,anchors=payload
            result=rigid.evaluate_pose(x,y,a,b,pairs,anchors,np.array(scope.weights,dtype=np.float64),
                np.array(scope.anchor_weights,dtype=np.float64),c.threshold,numerics=numerics)
        else:
            _,x,y,pairs,selection=payload
            function=rigid.evaluate_template if payload[0]=="template" else rigid.evaluate_shape
            result=function(x,y,pairs,np.array(scope.weights,dtype=np.float64),c.threshold,numerics=numerics,selection=selection)
        results.append(ConstraintResult(c,result,ns,nr,nselected,nanchors,c.atoms,c.protein_anchors))
        requirements.append(rigid.Requirement(result,c.required))
    # Freeze preservation is not automatically an RMSD threshold or verified
    # export. Keep it hard-unknown until a separately reviewed fixed-region check.
    unknowns=[]
    all_decisions=compiled.sequence_decisions+compiled.template_decisions
    if any(d.freeze_all_atom_geometry for d in all_decisions):
        unknowns.append("all_atom_freeze_unverified")
    if any(d.sequence is None or (d.chosen_identity is None and not d.redesign) for d in all_decisions):
        unknowns.append("supplied_identity_resolution_unavailable")
    for reason in unknowns:
        requirements.append(rigid.Requirement(_unknown(rigid.Metric.ENZYME_TEMPLATE,None,reason),True))
    return RequestEvaluation(compiled.request_sha256,COMPILER_REVISION,ADAPTER_REVISION,tuple(results),
        rigid.overall_status(tuple(requirements)),tuple(unknowns),compiled.predictor_input_keys,
        compiled.predictor_reference_keys,compiled.sequence_decisions,compiled.template_decisions)
