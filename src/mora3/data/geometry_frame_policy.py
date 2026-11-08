"""Parameterless R2 metadata eligibility; no geometry or accepted authority.

Native source/model/registration bridge is unimplemented. Synthetic declarations
only qualify metadata diagnostics, never source, features, NN, loss or export.
"""
from dataclasses import dataclass,field,asdict
from enum import Enum
import hashlib,json
from . import spatial_design_constraints as request_types

REQUEST_SHA256="0c06abe200dd41144c9c9deea5b5812dc1dd395b9a1846abb2a8e12a2cf94817"
COMPILER_REVISION="Mora3.TSE.RequestRegionCompiler/v2"
POLICY_SHA256="779e2cab3ee1da21a3acb040dda83861b71a159a861456d0ea602f692709e45a"
REVISION="Mora3.TSE.FramePolicyCompiler/v1-PREPARATION"
RECIPE_REVISION="Mora3.TSE.FrameRecipe/v1"
MAX_AXIS_ENTRIES=32768
MAX_REQUEST_BYTES=1048576

class FramePolicyError(ValueError):
    def __init__(self,code):
        self.code=code
        super().__init__(code)

def _fail(code):raise FramePolicyError(code)
def _text(x):
    if type(x) is not str or not x or len(x)>256 or any(ord(c)<32 for c in x):_fail("exact_bounded_text")
def _sha_string(x):
    if type(x) is not str or len(x)!=64 or any(c not in "0123456789abcdef" for c in x):_fail("exact_sha256_declaration")
def _enum(x,cls):
    if type(x) is not cls:_fail("exact_enum_no_alias")
def _integer(x,low,high):
    if type(x) is not int or not low<=x<=high:_fail("exact_bounded_integer_no_bool")
def _typed(xs,cls,limit):
    if type(xs) is not tuple or len(xs)>limit or any(type(x) is not cls for x in xs):_fail("exact_bounded_tuple")
    for x in xs:x.__post_init__()
def _digest(x):
    return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf8")).hexdigest()

class SlotName(str,Enum):
    REACTANT="reactant"
    PRODUCT="product"
    TS_BODY="TS_body"
    ENZYME_TEMPLATE="enzyme_template"
SLOT_ORDER=tuple(SlotName)

class FrameState(str,Enum):
    INDEPENDENT_LOCAL="independent_local"
    REGISTERED_COMMON="claimed_registered_common_not_accepted_proof"
    UNKNOWN="unknown_frame"
    UNAVAILABLE="unavailable"

class Feature(str,Enum):
    INTERNAL_DISTANCE="internal_distance"
    INTERNAL_DOT="internal_dot"
    MIXED_DOT="current_registered_reference_dot"
    REFERENCE_VECTOR="registered_reference_vector"
    CROSS_VECTOR="registered_reference_cross_vector"
INTERNAL=frozenset((Feature.INTERNAL_DISTANCE,Feature.INTERNAL_DOT))

class Origin(str,Enum):
    UNQUALIFIED_SOURCE="unqualified_external_declaration"
    SYNTHETIC_LITERAL="synthetic_literal_metadata"

class Purpose(str,Enum):
    CONDITION_INPUT="declared_condition_input_not_authority"
    EVALUATION_REFERENCE="evaluation_reference"
    WITHHELD_TARGET="withheld_target"

class MapOrigin(str,Enum):
    SOURCE_PROVIDED="source_provided_declaration_unaccepted"
    CALLER_EXPLICIT="caller_explicit_declaration_unaccepted"
    SYNTHETIC_EXPLICIT="synthetic_explicit"
    TARGET_DERIVED="target_derived"
    POSTHOC_RMSD="posthoc_RMSD_pass_alignment"
    GOLD_ALIGNMENT="gold_alignment"

class Status(str,Enum):
    HOLD_AUTHORITY="HOLD_source_model_authority_bridge_unimplemented"
    HOLD_REGISTRATION="HOLD_registration_authority_unavailable"
    INTERNAL_CANDIDATE="internal_group_metadata_candidate_only"
    UNSUPPORTED_FRAME="unsupported_frame"
    NOT_EVALUABLE="not_evaluable_required_or_partial_XYZ"
    SYNTHETIC_ELIGIBLE="synthetic_metadata_diagnostic_eligible_only"
    SYNTHETIC_PARTIAL="synthetic_partial_metadata_diagnostic_only"
    INAPPLICABLE="inapplicable_no_eligible_metadata_features"

@dataclass(frozen=True)
class PolicyBinding:
    request_sha256:str
    compiler_revision:str
    policy_sha256:str
    def __post_init__(self):
        if (type(self.request_sha256) is not str or self.request_sha256!=REQUEST_SHA256
            or type(self.compiler_revision) is not str or self.compiler_revision!=COMPILER_REVISION
            or type(self.policy_sha256) is not str or self.policy_sha256!=POLICY_SHA256
            or request_types.COMPILER_REVISION!=COMPILER_REVISION):_fail("exact_current_software_policy_binding")

@dataclass(frozen=True)
class MetadataBudget:
    max_slots:int
    max_axis_entries:int
    max_recipe_features:int
    def __post_init__(self):
        _integer(self.max_slots,1,4);_integer(self.max_axis_entries,1,MAX_AXIS_ENTRIES);_integer(self.max_recipe_features,1,256)

@dataclass(frozen=True)
class SourceDeclaration:
    source_id:str
    release:str
    reader_version:str
    reader_sha256:str
    original_identity_sha256:str
    group_sha256:str|None
    heldout_sha256:str|None
    origin:Origin
    purpose:Purpose
    def __post_init__(self):
        for x in (self.source_id,self.release,self.reader_version):_text(x)
        for x in (self.reader_sha256,self.original_identity_sha256):_sha_string(x)
        for x in (self.group_sha256,self.heldout_sha256):
            if x is not None:_sha_string(x)
        _enum(self.origin,Origin);_enum(self.purpose,Purpose)
        if self.origin is Origin.SYNTHETIC_LITERAL and not self.source_id.startswith("synthetic:"):_fail("synthetic_namespace_required")

def axis_digest(axis):
    _typed(axis,request_types.AtomKey,MAX_AXIS_ENTRIES)
    ids=[(x.entity,x.ordinal) for x in axis]
    if len(set(ids))!=len(ids):_fail("duplicate_or_conflicting_native_atom_key")
    return _digest([asdict(x) for x in axis])

@dataclass(frozen=True)
class SlotDeclaration:
    name:SlotName
    input_key:str
    role:request_types.Role
    state:FrameState
    units:str
    frame_id:str|None
    source:SourceDeclaration|None
    native_axis:tuple[request_types.AtomKey,...]|None
    xyz_available:tuple[bool,...]|None
    native_axis_sha256:str|None
    map_sha256:str|None
    map_origin:MapOrigin|None
    registration_policy_sha256:str|None
    def __post_init__(self):
        _enum(self.name,SlotName);_text(self.input_key);_enum(self.role,request_types.Role);_enum(self.state,FrameState)
        if type(self.units) is not str or self.units!="angstrom":_fail("exact_angstrom_units")
        if self.frame_id is not None:_text(self.frame_id)
        if self.source is not None:
            if type(self.source) is not SourceDeclaration:_fail("exact_source_declaration")
            self.source.__post_init__()
        if self.native_axis is None:
            if self.xyz_available is not None or self.native_axis_sha256 is not None:_fail("unknown_axis_cannot_invent_mask_or_hash")
        else:
            actual=axis_digest(self.native_axis)
            if self.native_axis_sha256!=actual:_fail("native_axis_order_or_hash_mismatch")
            if type(self.xyz_available) is not tuple or len(self.xyz_available)!=len(self.native_axis) or any(type(x) is not bool for x in self.xyz_available):
                _fail("whole_XYZ_exact_boolean_per_native_atom")
        if self.map_sha256 is not None:_sha_string(self.map_sha256)
        if self.map_origin is not None:_enum(self.map_origin,MapOrigin)
        if self.registration_policy_sha256 is not None:_sha_string(self.registration_policy_sha256)
        if self.state is FrameState.UNAVAILABLE and self.xyz_available is not None and any(self.xyz_available):_fail("unavailable_slot_cannot_claim_observed_XYZ")

@dataclass(frozen=True)
class CurrentFrame:
    input_key:str
    frame_id:str
    units:str
    source:SourceDeclaration
    def __post_init__(self):
        _text(self.input_key);_text(self.frame_id)
        if type(self.units) is not str or self.units!="angstrom":_fail("exact_current_frame_units")
        if type(self.source) is not SourceDeclaration:_fail("exact_current_source_declaration")
        self.source.__post_init__()

@dataclass(frozen=True)
class SlotRecipe:
    name:SlotName
    required:bool
    features:tuple[Feature,...]
    def __post_init__(self):
        _enum(self.name,SlotName)
        if type(self.required) is not bool:_fail("exact_required_boolean")
        if type(self.features) is not tuple or not self.features or len(self.features)>64 or any(type(x) is not Feature for x in self.features) or len(set(self.features))!=len(self.features):
            _fail("exact_ordered_unique_feature_recipe")

@dataclass(frozen=True)
class FrameRecipe:
    recipe_id:str
    slots:tuple[SlotRecipe,...]
    other_context_input_keys:tuple[str,...]
    revision:str=RECIPE_REVISION
    def __post_init__(self):
        _text(self.recipe_id);_typed(self.slots,SlotRecipe,4)
        if not self.slots or type(self.revision) is not str or self.revision!=RECIPE_REVISION:_fail("exact_recipe_revision")
        names=tuple(x.name for x in self.slots)
        if len(set(names))!=len(names) or tuple(sorted(names,key=SLOT_ORDER.index))!=names:_fail("named_slot_order_or_duplicate")
        if type(self.other_context_input_keys) is not tuple or len(self.other_context_input_keys)>64:_fail("bounded_other_context")
        for x in self.other_context_input_keys:_text(x)
        if len(set(self.other_context_input_keys))!=len(self.other_context_input_keys):_fail("duplicate_context_input")

@dataclass(frozen=True)
class SyntheticWitness:
    slot_name:SlotName
    declaration_sha256:str
    policy_sha256:str=POLICY_SHA256
    revision:str="Mora3.TSE.SyntheticFrameWitness/v1"
    def __post_init__(self):
        _enum(self.slot_name,SlotName);_sha_string(self.declaration_sha256)
        if self.policy_sha256!=POLICY_SHA256 or self.revision!="Mora3.TSE.SyntheticFrameWitness/v1":_fail("exact_synthetic_witness_not_native_proof")

@dataclass(frozen=True)
class SlotAssessment:
    name:SlotName
    state:FrameState
    required:bool
    requested_features:tuple[Feature,...]
    candidate_features:tuple[Feature,...]
    status:Status
    reasons:tuple[str,...]
    native_axis:tuple[request_types.AtomKey,...]|None
    xyz_available:tuple[bool,...]|None
    declaration:SlotDeclaration

@dataclass(frozen=True)
class FramePlan:
    revision:str
    request_sha256:str
    compiled_sha256:str
    declaration_sha256:str
    recipe_revision:str
    status:Status
    reasons:tuple[str,...]
    slots:tuple[SlotAssessment,...]
    declared_input_order:tuple[str,...]
    synthetic_diagnostic:bool
    current_frame:CurrentFrame|None
    recipe:FrameRecipe
    software:PolicyBinding
    metadata_budget:MetadataBudget
    metadata_only:bool=field(default=True,init=False)
    actual_source_authority:None=field(default=None,init=False)
    actual_registration_authority:None=field(default=None,init=False)
    actual_model_authority:None=field(default=None,init=False)
    actual_feature_arrays:None=field(default=None,init=False)
    actual_model_token_positions:None=field(default=None,init=False)
    native_model_authorized:bool=field(default=False,init=False)
    fullready:bool=field(default=False,init=False)
    def require_native_or_model_use(self):raise RuntimeError("P0/P3/native_model_bridge_unimplemented")
    def require_source_registration(self):raise RuntimeError("no_accepted_source_registration_authority")
    def require_loss_or_training(self):raise RuntimeError("no_loss_or_training_authority")
    def require_export_acceptance(self):raise RuntimeError("no_native_export_acceptance_authority")

def _prepare(request,compiled,recipe,current,slots,budget,software):
    if type(software) is not PolicyBinding:_fail("exact_policy_software_declaration")
    software.__post_init__()
    if type(request) is not request_types.DesignRequest or type(compiled) is not request_types.CompiledDesignPlan:_fail("exact_current_request_types")
    fresh=request_types.compile_design_request(request)
    raw=fresh.canonical_bytes()
    if compiled.canonical_bytes()!=raw:_fail("compiled_not_derived_from_raw_request")
    if len(raw)>min(MAX_REQUEST_BYTES,request.budget.memory_bytes):_fail("metadata_request_budget_exceeded")
    if type(recipe) is not FrameRecipe or type(budget) is not MetadataBudget:_fail("exact_recipe_budget")
    recipe.__post_init__();budget.__post_init__();_typed(slots,SlotDeclaration,4)
    if current is not None:
        if type(current) is not CurrentFrame:_fail("exact_optional_current_frame")
        current.__post_init__()
    for slot in slots:
        if slot.role in (request_types.Role.EVALUATION_REFERENCE,request_types.Role.WITHHELD_TARGET) or (slot.source is not None and slot.source.purpose is not Purpose.CONDITION_INPUT):
            _fail("target_or_evaluation_forbidden")
        if slot.map_origin in (MapOrigin.TARGET_DERIVED,MapOrigin.POSTHOC_RMSD,MapOrigin.GOLD_ALIGNMENT):_fail("target_derived_registration_forbidden")
    if current is not None and current.source.purpose is not Purpose.CONDITION_INPUT:_fail("target_or_evaluation_forbidden")
    if tuple(x.name for x in slots)!=tuple(x.name for x in recipe.slots):_fail("complete_named_slot_order_required")
    if len(slots)>budget.max_slots or sum(len(x.native_axis or ()) for x in slots)>budget.max_axis_entries or sum(len(x.features) for x in recipe.slots)>budget.max_recipe_features:
        _fail("metadata_profile_budget_exceeded_no_crop")
    inputs={x.key:x for x in request.inputs if x.available and x.role in request_types._INPUT_ROLES}
    consumed=set(recipe.other_context_input_keys)
    for key in recipe.other_context_input_keys:
        if key not in inputs or inputs[key].kind in (request_types.InputKind.TS_GEOMETRY,request_types.InputKind.ENZYME_STRUCTURE):
            _fail("spatial_input_must_not_be_silently_other_context")
    for slot in slots:
        item=inputs.get(slot.input_key)
        if item is None or slot.role is not item.role:_fail("slot_role_or_input_binding_mismatch")
        if slot.name is SlotName.TS_BODY and item.kind is not request_types.InputKind.TS_GEOMETRY:_fail("TS_slot_input_kind")
        if slot.name is SlotName.ENZYME_TEMPLATE and item.kind is not request_types.InputKind.ENZYME_STRUCTURE:_fail("template_slot_input_kind")
        if slot.name in (SlotName.REACTANT,SlotName.PRODUCT) and item.kind not in (request_types.InputKind.REACTION,request_types.InputKind.CONTEXT):_fail("reaction_slot_input_kind")
        consumed.add(slot.input_key)
    if current is not None:
        item=inputs.get(current.input_key)
        if item is None or item.kind not in (request_types.InputKind.TS_GEOMETRY,request_types.InputKind.ENZYME_STRUCTURE,request_types.InputKind.CONTEXT):_fail("current_frame_requires_explicit_allowed_input")
        consumed.add(current.input_key)
    if consumed!=set(fresh.predictor_input_keys):_fail("complete_input_context_accounting_required")
    payload=dict(revision=REVISION,policy=asdict(software),request=fresh.request_sha256,compiled=hashlib.sha256(raw).hexdigest(),
        recipe=asdict(recipe),current=None if current is None else asdict(current),slots=[asdict(x) for x in slots],budget=asdict(budget))
    return fresh,payload

def _synthetic_only(slots,current):
    sources=[x.source for x in slots]+([] if current is None else [current.source])
    if any(x is None or x.origin is not Origin.SYNTHETIC_LITERAL for x in sources):_fail("synthetic_diagnostic_requires_only_literal_declarations")
    if any(x.map_origin not in (None,MapOrigin.SYNTHETIC_EXPLICIT) for x in slots):_fail("synthetic_diagnostic_cannot_adopt_source_map_claim")

def declare_synthetic_witness(request,compiled,recipe,current,slots,budget,software,*,slot_name):
    """Unsigned metadata declaration; NEVER an accepted source/frame proof."""
    fresh,payload=_prepare(request,compiled,recipe,current,slots,budget,software)
    _synthetic_only(slots,current);_enum(slot_name,SlotName)
    slot=next((x for x in slots if x.name is slot_name),None)
    if slot is None or slot.state is not FrameState.REGISTERED_COMMON:_fail("synthetic_witness_only_for_claimed_common_slot")
    return SyntheticWitness(slot_name,_digest(dict(declaration=payload,slot=slot_name.value)))

def _compile(request,compiled,recipe,current,slots,budget,software,*,synthetic,witnesses):
    fresh,payload=_prepare(request,compiled,recipe,current,slots,budget,software)
    _typed(witnesses,SyntheticWitness,4)
    if len({x.slot_name for x in witnesses})!=len(witnesses):_fail("duplicate_synthetic_witness")
    if synthetic:_synthetic_only(slots,current)
    valid={}
    for witness in witnesses:
        slot=next((x for x in slots if x.name is witness.slot_name),None)
        if not synthetic or slot is None or slot.state is not FrameState.REGISTERED_COMMON:_fail("unexpected_synthetic_witness")
        if witness.declaration_sha256!=_digest(dict(declaration=payload,slot=witness.slot_name.value)):_fail("stale_synthetic_witness_binding")
        valid[witness.slot_name]=True
    assessments=[]
    for slot,want in zip(slots,recipe.slots):
        status=Status.HOLD_AUTHORITY;reasons=[];features=()
        if slot.state is FrameState.UNAVAILABLE:
            status=Status.NOT_EVALUABLE;reasons.append("explicit_slot_unavailable_retained")
        elif slot.state is FrameState.UNKNOWN or slot.frame_id is None:
            status=Status.UNSUPPORTED_FRAME;reasons.append("unknown_single_local_or_common_frame")
        elif slot.source is None or not slot.native_axis or slot.xyz_available is None or not all(slot.xyz_available):
            status=Status.NOT_EVALUABLE;reasons.append("whole_declared_axis_XYZ_or_source_declaration_unavailable")
        elif slot.state is FrameState.INDEPENDENT_LOCAL:
            if any(x not in INTERNAL for x in want.features):
                status=Status.UNSUPPORTED_FRAME;reasons.append("independent_local_internal_invariants_only")
            else:features=want.features;status=Status.INTERNAL_CANDIDATE;reasons.append("internal_group_candidate_not_source_admission")
        elif slot.state is FrameState.REGISTERED_COMMON:
            mixed=any(x not in INTERNAL for x in want.features)
            if mixed and (current is None or slot.map_sha256 is None or slot.map_origin is None):
                status=Status.UNSUPPORTED_FRAME;reasons.append("common_frame_input_or_map_declaration_missing")
            elif mixed and slot.registration_policy_sha256!=POLICY_SHA256:
                status=Status.HOLD_REGISTRATION;reasons.append("exact_registration_policy_declaration_unavailable")
            elif mixed and not valid.get(slot.name,False):
                status=Status.HOLD_REGISTRATION;reasons.append("matching_ids_hashes_claims_are_not_accepted_registration")
            else:
                features=want.features;status=Status.SYNTHETIC_ELIGIBLE if mixed else Status.INTERNAL_CANDIDATE;reasons.append("synthetic_common_candidate_only" if mixed else "internal_group_candidate_only")
        assessments.append(SlotAssessment(slot.name,slot.state,want.required,want.features,features,status,tuple(reasons),slot.native_axis,slot.xyz_available,slot))
    required=[x for x in assessments if x.required]
    if any(x.status is Status.UNSUPPORTED_FRAME for x in required):overall=Status.UNSUPPORTED_FRAME
    elif any(x.status is Status.NOT_EVALUABLE for x in required):overall=Status.NOT_EVALUABLE
    elif any(x.status is Status.HOLD_REGISTRATION for x in assessments):overall=Status.HOLD_REGISTRATION if synthetic else Status.HOLD_AUTHORITY
    elif not synthetic:overall=Status.HOLD_AUTHORITY
    elif not any(x.candidate_features for x in assessments):overall=Status.INAPPLICABLE
    elif any(x.candidate_features!=x.requested_features for x in assessments):overall=Status.SYNTHETIC_PARTIAL
    else:overall=Status.SYNTHETIC_ELIGIBLE
    return FramePlan(REVISION,fresh.request_sha256,payload["compiled"],_digest(payload),recipe.revision,overall,
        ("metadata_only_P0_P3_and_registration_bridge_unimplemented",),tuple(assessments),fresh.predictor_input_keys,synthetic,current,recipe,software,budget)

def compile_frame_policy(request,compiled,recipe,current,slots,budget,software,*,registration_authority=None,source_authority=None,model_authority=None):
    # Do not inspect/duck-type any caller receipt/flag/object.
    if registration_authority is not None or source_authority is not None or model_authority is not None:_fail("authority_bridge_unimplemented_no_caller_grant")
    return _compile(request,compiled,recipe,current,slots,budget,software,synthetic=False,witnesses=())

def compile_synthetic_frame_policy(request,compiled,recipe,current,slots,budget,software,*,witnesses=()):
    return _compile(request,compiled,recipe,current,slots,budget,software,synthetic=True,witnesses=witnesses)
