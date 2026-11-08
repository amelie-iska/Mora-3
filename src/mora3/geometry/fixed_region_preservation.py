"""Stateless literal fixed-region preservation, distinct from RMSD.

No registration/fitting, tolerance, parser/IO/source/model/export authority.
Complete means complete within the declared supplied handcrafted subject axis;
it is never biological/chemical/native completeness inferred from a schema.
"""
from dataclasses import dataclass,field
from enum import Enum
import numpy as np
from ..data import spatial_design_constraints as request_types
from . import spatial_constraint_evaluation as coordinates

REQUEST_SHA256="0c06abe200dd41144c9c9deea5b5812dc1dd395b9a1846abb2a8e12a2cf94817"
ADAPTER_SHA256="9b40448d9210f29a2f8f56a7bcf564e5c3ed421800c948faa17dd59d920ca2b7"
COMPILER_REVISION="Mora3.TSE.RequestRegionCompiler/v2"
REVISION="Mora3.TSE.FixedRegionPreservation/v1"
EQUALITY_RULE="exact_float64_bits_and_literal_atom_identity"

class Status(str,Enum):
    PASS="PASS"
    VIOLATED="VIOLATED"
    NOT_EVALUABLE="NOT_EVALUABLE"
    UNSUPPORTED="UNSUPPORTED"

class Coverage(str,Enum):
    COMPLETE_SUPPLIED_SUBJECT="complete_declared_supplied_subject"
    UNKNOWN="unknown_or_incomplete_subject_coverage"

class FramePolicy(str,Enum):
    EXACT_COMMON_FRAME="explicit_same_native_frame"
    UNAVAILABLE="common_frame_unavailable"
    TRANSFORM_REQUESTED="transform_needed_not_supported"

def _residue(r):
    if type(r) is not request_types.ResidueKey:raise ValueError("exact native residue required")
    r.__post_init__()

def _slot(r):
    return r.chain,r.index,r.insertion

def _atom_id(a):
    return a.entity,a.ordinal

def _typed_tuple(items,cls,label,limit=128):
    if type(items) is not tuple or len(items)>limit or any(type(x) is not cls for x in items):
        raise ValueError("exact bounded "+label+" tuple; profile128")
    for item in items:item.__post_init__()

@dataclass(frozen=True)
class IdentityWitness:
    native:request_types.ResidueKey
    output:request_types.ResidueKey|None
    def __post_init__(self):
        _residue(self.native)
        if self.output is not None:_residue(self.output)

@dataclass(frozen=True)
class AtomDeclaration:
    native:request_types.ResidueKey
    input_key:str
    atoms:tuple[request_types.AtomKey,...]
    coverage:Coverage
    def __post_init__(self):
        _residue(self.native)
        if type(self.input_key) is not str or not self.input_key:raise ValueError("exact supplied input key")
        if type(self.coverage) is not Coverage:raise ValueError("explicit coverage evidence required")
        _typed_tuple(self.atoms,request_types.AtomKey,"declared atom")
        if not self.atoms:raise ValueError("nonempty explicit selected atom axis")
        if any(a.entity!=self.input_key or a.residue!=self.native for a in self.atoms):
            raise ValueError("declared atoms must belong to exact supplied native subject")
        if len({_atom_id(a) for a in self.atoms})!=len(self.atoms):
            raise ValueError("duplicate declared atom identity")

@dataclass(frozen=True)
class GeometryWitness:
    declaration:AtomDeclaration
    source:coordinates.Coordinates|None
    output:coordinates.Coordinates|None
    pairs:tuple[request_types.AtomPair,...]
    frame_policy:FramePolicy
    def __post_init__(self):
        if type(self.declaration) is not AtomDeclaration:raise ValueError("exact full-atom declaration")
        self.declaration.__post_init__()
        for c in (self.source,self.output):
            if c is not None and type(c) is not coordinates.Coordinates:
                raise ValueError("exact bounded handcrafted coordinates")
        _typed_tuple(self.pairs,request_types.AtomPair,"atom correspondence")
        if len({_atom_id(p.source) for p in self.pairs})!=len(self.pairs) or len({_atom_id(p.reference) for p in self.pairs})!=len(self.pairs):
            raise ValueError("ambiguous/duplicate frozen atom correspondence")
        if type(self.frame_policy) is not FramePolicy:raise ValueError("explicit frame availability/policy")

@dataclass(frozen=True)
class Check:
    status:Status
    reason:str
    checked_atoms:int=0

@dataclass(frozen=True)
class SubjectResult:
    decision:request_types.ResolvedResidueDecision
    identity:Check|None
    geometry:Check|None
    declared_atoms:tuple[request_types.AtomKey,...]
    pairs:tuple[request_types.AtomPair,...]

@dataclass(frozen=True)
class FreezeReport:
    request_sha256:str
    compiler_revision:str
    revision:str
    equality_rule:str
    subjects:tuple[SubjectResult,...]
    status:Status
    numeric_adapter_hold_cleared:bool=field(default=False,init=False)
    export_reload_verified:bool=field(default=False,init=False)
    source_or_model_authority:bool=field(default=False,init=False)
    def require_native_or_model_use(self):
        raise RuntimeError("handcrafted freeze checks cannot grant native/source/model/training authority")
    def clear_numeric_adapter_hold(self):
        raise RuntimeError("separate numeric adapter hardhold/integration must remain unchanged")
    def require_export_verified_pass(self):
        raise RuntimeError("export/reload proof remains unavailable")

def _snapshot(c):
    if c is None:return None
    return coordinates.Coordinates(c.namespace,c.body,c.purpose,c.role,c.atoms,c.xyz,c.frame,c.units)

def _bind_geometry(w,d,inputs):
    """Validate literal source/role/subject/correspondence metadata before results."""
    expected=d.template if d.template is not None else d.sequence
    if w.declaration.native!=expected:raise ValueError("geometry witness not exact selected native subject")
    if d.template is None:
        raise ValueError("no mapped supplied geometry may be invented for sequence-only subject")
    declared=inputs.get(w.declaration.input_key)
    if declared is None or not declared.available or declared.kind is not request_types.InputKind.ENZYME_STRUCTURE or expected not in declared.residues:
        raise ValueError("freeze source must bind exact available supplied enzyme structure")
    source,output=_snapshot(w.source),_snapshot(w.output)
    if source is not None:
        if source.namespace!=declared.key or source.body is not coordinates.Body.PROTEIN or source.purpose is not coordinates.Purpose.GUIDANCE_REFERENCE or source.role is not declared.role:
            raise ValueError("target/evaluation/unrelated source cannot stand in for supplied fixed geometry")
        if any(a.residue not in declared.residues for a in source.atoms):
            raise ValueError("source protein axis contains undeclared native subject")
    if output is not None:
        if output.body is not coordinates.Body.PROTEIN or output.purpose is not coordinates.Purpose.CANDIDATE_OUTPUT:
            raise ValueError("freeze output must be explicit candidate protein geometry")
        if source is not None and output.namespace==source.namespace:
            raise ValueError("supplied input/output namespace alias")
    if any(p.reference.entity!=w.declaration.input_key or p.reference.residue!=expected for p in w.pairs):
        raise ValueError("correspondence outside exact supplied native subject")
    # An incomplete declared set/correspondence is a hard unknown below, never
    # intersected into a smaller passing set. All referenced subject IDs are
    # explicit; source-row membership is established before coordinate indexing.
    return GeometryWitness(w.declaration,source,output,w.pairs,w.frame_policy)

def _identity_check(d,w):
    if d.sequence is None:
        return Check(Status.NOT_EVALUABLE,"no_supplied_sequence_subject")
    if w is None or w.output is None:
        return Check(Status.NOT_EVALUABLE,"output_identity_unavailable")
    if w.output!=d.sequence:
        return Check(Status.VIOLATED,"frozen_literal_sequence_identity_or_native_key_changed")
    return Check(Status.PASS,"exact_supplied_sequence_identity_preserved")

def _geometry_check(d,w):
    if d.template is None:
        return Check(Status.NOT_EVALUABLE,"no_mapped_supplied_geometry")
    if w is None or w.source is None or w.output is None:
        return Check(Status.NOT_EVALUABLE,"geometry_witness_unavailable")
    if w.declaration.coverage is not Coverage.COMPLETE_SUPPLIED_SUBJECT:
        return Check(Status.NOT_EVALUABLE,"complete_selected_subject_coverage_unavailable")
    if w.frame_policy is FramePolicy.TRANSFORM_REQUESTED:
        return Check(Status.UNSUPPORTED,"input_transform_unsupported_no_fitting")
    if w.frame_policy is FramePolicy.UNAVAILABLE or w.source.frame!=w.output.frame:
        return Check(Status.NOT_EVALUABLE,"declared_common_frame_unavailable")
    native=d.template
    source_group=tuple(a for a in w.source.atoms if a.residue==native)
    if source_group!=w.declaration.atoms:
        return Check(Status.NOT_EVALUABLE,"declared_atomset_not_complete_native_order")
    if len(w.pairs)!=len(w.declaration.atoms) or set(p.reference for p in w.pairs)!=set(w.declaration.atoms):
        return Check(Status.NOT_EVALUABLE,"complete_reference_correspondence_unavailable")
    expected_output=d.sequence if d.sequence is not None else d.template
    if any(p.source.residue!=expected_output for p in w.pairs):
        return Check(Status.VIOLATED,"frozen_output_residue_identity_or_key_changed")
    # Namespace/ordinals differ only through explicit supplied pair identities.
    # All chemical identity annotations must be literally unchanged, including unknowns.
    for p in w.pairs:
        a,z=p.source,p.reference
        if (a.element,a.name,a.isotope,a.formal_charge)!=(z.element,z.name,z.isotope,z.formal_charge):
            return Check(Status.VIOLATED,"frozen_literal_atom_identity_changed")
    output_group=tuple(a for a in w.output.atoms if a.residue==expected_output)
    wanted=set(p.source for p in w.pairs)
    if any(a not in output_group for a in wanted):
        return Check(Status.NOT_EVALUABLE,"complete_output_atom_coverage_unavailable")
    if set(output_group)!=wanted:
        return Check(Status.VIOLATED,"additional_or_substituted_output_atom_in_frozen_subject")
    source_index={a:i for i,a in enumerate(w.source.atoms)}
    output_index={a:i for i,a in enumerate(w.output.atoms)}
    source_rows=w.source.xyz[[source_index[p.reference] for p in w.pairs]]
    output_rows=w.output.xyz[[output_index[p.source] for p in w.pairs]]
    # Exact freeze invariant, not RMSD0 or an inferred tolerance. Signed zero bits
    # are deliberately preserved; finite arrays were validated by Coordinates.
    if not np.array_equal(source_rows.view(np.uint64),output_rows.view(np.uint64)):
        return Check(Status.VIOLATED,"frozen_coordinate_bits_changed",len(w.pairs))
    return Check(Status.PASS,"exact_common_frame_atom_and_coordinate_bits_preserved",len(w.pairs))

def evaluate_fixed_regions(request,compiled,identities,geometries,*,software):
    """Authoritative raw-request recompile, strict all-witness preflight, no fitting."""
    if type(software) is not coordinates.SoftwareBinding:raise ValueError("exact immutable prerequisite declaration")
    software.__post_init__()
    if coordinates.REQUEST_SHA256!=REQUEST_SHA256 or coordinates.COMPILER_REVISION!=COMPILER_REVISION:
        raise ValueError("stale prerequisite API")
    if type(request) is not request_types.DesignRequest or type(compiled) is not request_types.CompiledDesignPlan:
        raise ValueError("exact raw request and compiled-v2 types")
    fresh=request_types.compile_design_request(request)
    if compiled.canonical_bytes()!=fresh.canonical_bytes() or compiled.compiler_revision!=COMPILER_REVISION:
        raise ValueError("compiled freeze/predictor/status fields not proven by raw request")
    decisions=tuple(dict.fromkeys(fresh.sequence_decisions+fresh.template_decisions))
    selected=tuple(d for d in decisions if d.freeze_sequence or d.freeze_all_atom_geometry)
    if len(selected)>128:raise ValueError("unsupported128 subject implementation profile; no crop")
    _typed_tuple(identities,IdentityWitness,"identity witness")
    _typed_tuple(geometries,GeometryWitness,"geometry witness")
    if len({_slot(w.native) for w in identities})!=len(identities):
        raise ValueError("ambiguous/duplicate native identity witness")
    if len({_slot(w.declaration.native) for w in geometries})!=len(geometries):
        raise ValueError("ambiguous/duplicate native geometry witness")
    sequence_needed={d.sequence for d in selected if d.freeze_sequence and d.sequence is not None}
    geometry_needed={d.template if d.template is not None else d.sequence for d in selected if d.freeze_all_atom_geometry}
    if any(w.native not in sequence_needed for w in identities):
        raise ValueError("identity witness outside selected supplied sequence; no template-only fabrication")
    if any(w.declaration.native not in geometry_needed for w in geometries):
        raise ValueError("geometry witness outside selected native subject")
    known_outputs=tuple(w.output for w in identities if w.output is not None)
    if len({_slot(r) for r in known_outputs})!=len(known_outputs):
        raise ValueError("duplicate/ambiguous output residue identity")
    by_identity={w.native:w for w in identities}
    inputs={i.key:i for i in request.inputs}
    by_geometry={}
    for w in geometries:
        d=next(d for d in selected if d.freeze_all_atom_geometry and (d.template if d.template is not None else d.sequence)==w.declaration.native)
        by_geometry[w.declaration.native]=_bind_geometry(w,d,inputs)
    supplied_bytes=sum((w.source.xyz.nbytes if w.source is not None else 0)+(w.output.xyz.nbytes if w.output is not None else 0) for w in by_geometry.values())
    if supplied_bytes>request.budget.memory_bytes:raise ValueError("complete supplied geometry view byte budget exceeded")
    if any(w.output is not None and len(w.output.atoms)>request.budget.max_atoms for w in by_geometry.values()):
        raise ValueError("full candidate structure exceeds declared atom budget; no cropping")
    # ALL declarations/roles/maps have been validated before making any result.
    results=[];statuses=[]
    for d in selected:
        iw=by_identity.get(d.sequence) if d.sequence is not None else None
        gw=by_geometry.get(d.template if d.template is not None else d.sequence)
        identity=_identity_check(d,iw) if d.freeze_sequence else None
        geometry=_geometry_check(d,gw) if d.freeze_all_atom_geometry else None
        for item in (identity,geometry):
            if item is not None:statuses.append(item.status)
        results.append(SubjectResult(d,identity,geometry,() if gw is None else gw.declaration.atoms,() if gw is None else gw.pairs))
    if Status.VIOLATED in statuses:status=Status.VIOLATED
    elif Status.UNSUPPORTED in statuses:status=Status.UNSUPPORTED
    elif not statuses or Status.NOT_EVALUABLE in statuses:status=Status.NOT_EVALUABLE
    else:status=Status.PASS
    return FreezeReport(fresh.request_sha256,COMPILER_REVISION,REVISION,EQUALITY_RULE,tuple(results),status)
