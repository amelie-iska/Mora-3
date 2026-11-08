"""Explicit reference-profile dispatch around unchanged TSE leaves.

Synthetic request-bound diagnostics only; candidate remains old TS profile.
No serializer, solver, source IO, authority lift or native/model admission.
"""
from dataclasses import dataclass,field,asdict
from enum import Enum
import time
from . import reloaded_constraint_composition as base
from . import protein_reference_wire as protein

REVISION="Mora3.TSE.ExplicitReferenceDispatch/v1"
COMPOSITION_SHA256="94c22ff9e490afc242527390acc21d8df65928288cc9d01e4f9f46c32515e660"
PROTEIN_SHA256="bd353d4582639e6def0f6a7686178d311b99e1cfe0cbccf7d251cbe0d7d225d2"
Status=base.Status
Use=base.Use
request_types=base.request_types
numeric_leaf=base.numeric_leaf
fixed_leaf=base.fixed_leaf
rigid=base.rigid
export_leaf=base.export_leaf
np=base.np
_text=base._text
_typed=base._typed
_hash=base._hash
_digest=base._digest
_hash_string=base._hash_string
_fresh=base._fresh
_expired=base._expired

class Profile(str,Enum):
    TS_STRUCTURE="explicit_original_TS_structure_profile_v1"
    PROTEIN_REFERENCE="explicit_original_protein_reference_profile_v1"

def _profile(profile,files,budget):
    if type(profile) is not Profile:raise ValueError("explicit exact profile tag required; never infer content")
    if profile is Profile.TS_STRUCTURE:
        if type(files) not in (export_leaf.ExportBundle,export_leaf.ExportExpectation) or budget is not None:
            raise ValueError("old TS structure profile must retain exact original types/no protein budget")
    else:
        if type(files) not in (protein.ReferenceBundle,protein.ReferenceExpectation) or type(budget) is not protein.ReferenceBudget:
            raise ValueError("distinct protein wire/expectation/caller budget required")
        budget.__post_init__()

@dataclass(frozen=True)
class ReferenceFiles:
    profile:Profile
    key:str
    files:export_leaf.ExportBundle|protein.ReferenceBundle
    budget:protein.ReferenceBudget|None
    def __post_init__(self):
        _text(self.key,"explicit original artifact key");_profile(self.profile,self.files,self.budget)
        if self.key=="candidate":raise ValueError("reference/candidate artifact alias")
        expected=export_leaf.ExportBundle if self.profile is Profile.TS_STRUCTURE else protein.ReferenceBundle
        if type(self.files) is not expected:raise ValueError("exact explicitly tagged files")

@dataclass(frozen=True)
class ReferenceExpectation:
    profile:Profile
    key:str
    expectation:export_leaf.ExportExpectation|protein.ReferenceExpectation
    budget:protein.ReferenceBudget|None
    def __post_init__(self):
        _text(self.key,"explicit external original key");_profile(self.profile,self.expectation,self.budget)
        if self.key=="candidate":raise ValueError("reference/candidate expectation alias")
        expected=export_leaf.ExportExpectation if self.profile is Profile.TS_STRUCTURE else protein.ReferenceExpectation
        if type(self.expectation) is not expected:raise ValueError("exact explicitly tagged external expectation")
        self.expectation.__post_init__()
        if self.profile is Profile.PROTEIN_REFERENCE and self.expectation.budget_sha256!=self.budget.digest():
            raise ValueError("external protein budget does not match original")

def _expectation_fields(a,b,c,candidate,references):
    return dict(revision=REVISION,request_sha256=a,compiled_sha256=b,scope_sha256=c,candidate=asdict(candidate),references=[asdict(x) for x in references])

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
            raise ValueError("exact current dispatch/candidate declaration")
        self.candidate.__post_init__();_typed(self.references,ReferenceExpectation,"external tagged expectations",3)
        if len({x.key for x in self.references})!=len(self.references):raise ValueError("duplicate original reference keys")
        hashes=[self.candidate.identity_sha256]+[x.expectation.identity_sha256 for x in self.references]
        if len(set(hashes))!=len(hashes):raise ValueError("candidate/original reference identity alias")
        if _digest(_expectation_fields(self.request_sha256,self.compiled_sha256,self.scope_sha256,self.candidate,self.references))!=self.binding_sha256:
            raise ValueError("changed/stale external dispatch declaration")

def declare_expectation(request,compiled,scope,candidate_expectation,reference_expectations):
    fresh=_fresh(request,compiled)
    if type(scope) is not base.EvaluationScope:raise ValueError("exact unchanged evaluation scope")
    scope.__post_init__()
    if type(candidate_expectation) is not export_leaf.ExportExpectation:raise ValueError("candidate remains old TS profile")
    candidate_expectation.__post_init__();_typed(reference_expectations,ReferenceExpectation,"external reference expectations",3)
    a,b,c=fresh.request_sha256,_hash(fresh.canonical_bytes()),_digest(asdict(scope))
    return JointExpectation(a,b,c,candidate_expectation,reference_expectations,_digest(_expectation_fields(a,b,c,candidate_expectation,reference_expectations)))

@dataclass(frozen=True)
class ReloadedArtifact:
    key:str
    result:export_leaf.ReloadedPacket|protein.ReloadedReference
    fasta_status:export_leaf.FastaStatus
    profile:str="candidate_TS_structure_v1"

@dataclass(frozen=True)
class DispatchReport:
    request_sha256:str
    binding_sha256:str
    status:Status
    reasons:tuple[str,...]
    artifacts:tuple[ReloadedArtifact,...]
    consumed:tuple[base.Consumption,...]
    numeric:numeric_leaf.RequestEvaluation|None
    fixed:fixed_leaf.FreezeReport|None
    unsupported_profiles:tuple[str,...]
    authorization_status:str=field(default="HOLD",init=False)
    actual_export_acceptance:bool=field(default=False,init=False)
    numeric_adapter_hold_cleared:bool=field(default=False,init=False)
    source_native_model_authority:None=field(default=None,init=False)
    require_native_or_model_use=base.CompositionReport.require_native_or_model_use
    require_export_constraint_acceptance=base.CompositionReport.require_export_constraint_acceptance

def _report(fresh,expected,status,reasons,artifacts=(),consumed=(),numeric=None,fixed=None,unsupported=()):
    return DispatchReport(fresh.request_sha256,expected.binding_sha256,status,tuple(reasons),tuple(artifacts),tuple(consumed),numeric,fixed,tuple(unsupported))

def _signature(x):
    return x.profile,x.key,None if x.budget is None else x.budget.digest()

def _file_size(files):
    if type(files) is export_leaf.ExportBundle:return base._file_size(files)
    if type(files) is not protein.ReferenceBundle:raise ValueError("explicit exact file class")
    total=0
    for raw in (files.mmcif,files.fasta,files.sidecar):
        if raw is not None:
            if type(raw) is not bytes:raise ValueError("exact reference bytes")
            total+=len(raw)
    return total

def _reload(original,bound):
    if original.profile is Profile.TS_STRUCTURE:
        result=export_leaf.reload_bundle(original.files,bound.expectation)
    else:
        result=protein.reload_reference(original.files,bound.expectation,budget=original.budget)
    return ReloadedArtifact(original.key,result,original.files.fasta_status,original.profile.value)

def evaluate_dispatched_constraints(request,compiled,candidate_files,reference_files,scope,expectation,*,software):
    """One bounded composition. Every coordinate comes from strict reload."""
    started=time.monotonic()
    if type(software) is not numeric_leaf.SoftwareBinding:raise ValueError("exact pinned software declaration")
    software.__post_init__()
    fresh=_fresh(request,compiled)
    if type(scope) is not base.EvaluationScope or type(expectation) is not JointExpectation:
        raise ValueError("explicit external scope+expectation required")
    scope.__post_init__();expectation.__post_init__()
    if (fresh.request_sha256,_hash(fresh.canonical_bytes()),_digest(asdict(scope)))!=(expectation.request_sha256,expectation.compiled_sha256,expectation.scope_sha256):
        raise ValueError("actual request/map/context/limit/scope changed from external binding")
    if type(candidate_files) is not export_leaf.ExportBundle:raise ValueError("exact candidate bytes")
    _typed(reference_files,ReferenceFiles,"original reference files",3)
    if tuple(_signature(x) for x in reference_files)!=tuple(_signature(x) for x in expectation.references):
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
        artifacts.append(_reload(original,bound))
        if _expired(started,request):
            return _report(fresh,expectation,Status.BUDGET_EXHAUSTED,("wall_budget_exhausted_during_reload",),artifacts)
    # A larger wire budget never changes downstream point/full-axis profiles.
    for artifact in artifacts:
        if len(artifact.result.packet.atoms)>128:
            return _report(fresh,expectation,Status.UNSUPPORTED,("reference_wire_axis_exceeds_unchanged128_point_profile_no_crop",),artifacts,
                unsupported=(artifact.key+":axis_over128",))
    profiles={x.key:x.profile for x in reference_files}
    for binding in scope.records:
        if profiles.get(binding.artifact) is not Profile.PROTEIN_REFERENCE:continue
        packet=next(x.result.packet for x in artifacts if x.key==binding.artifact)
        if packet.purpose is protein.ReferencePurpose.EVALUATION:
            if binding.use is not Use.NUMERIC_REFERENCE or binding.purpose is not numeric_leaf.Purpose.EVALUATION_ONLY_REFERENCE or binding.role is not request_types.Role.EVALUATION_REFERENCE:
                raise ValueError("evaluation protein bytes cannot become predictor/guidance/frozen input")
            if binding.namespace in {i.key for i in request.inputs}:raise ValueError("evaluation bytes cannot bind condition input namespace")
        else:
            if binding.use not in (Use.NUMERIC_REFERENCE,Use.FROZEN_SOURCE) or binding.purpose is not numeric_leaf.Purpose.GUIDANCE_REFERENCE or binding.role is not request_types.Role.USER_SUPPLIED:
                raise ValueError("guidance protein profile/purpose/role mismatch")
        if binding.body is numeric_leaf.Body.COMPLEX or any(c.metric is request_types.Metric.TS_POSE and c.reference_key==binding.namespace for c in request.constraints):
            return _report(fresh,expectation,Status.UNSUPPORTED,("protein_only_reference_cannot_supply_bound_TS_pose_complex",),artifacts,
                unsupported=(binding.artifact+":protein_only_not_bound_complex",))
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
        consumed.append(base.Consumption(binding.key,binding.artifact,binding.namespace,expected_ids[binding.artifact],_digest([asdict(a) for a in binding.atoms]),len(selected)))
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
