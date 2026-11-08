"""Pure TSE request/mapping/constraint metadata. No geometry/model/source IO.

Declared reference availability is not inspected native geometry or admission.
All compiled constraints remain pending/unevaluable; this leaf never returns PASS.
"""
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math

COMPILER_REVISION = 'Mora3.TSE.RequestRegionCompiler/v2'
_MAX_COUNT = 2**63-1
_BACKBONE = frozenset(('N','CA','C','O','OXT'))

def _text(value, label, *, empty=False):
    if type(value) is not str or (not value and not empty):
        raise ValueError(label+' requires an exact string')
    try:value.encode('utf8')
    except UnicodeError as error:raise ValueError(label+' must be UTF8') from error

def _count(value,label,minimum=0):
    if type(value) is not int or not minimum<=value<=_MAX_COUNT:
        raise ValueError(label+' requires a bounded exact integer')

def _finite(value,label,*,positive=False):
    if type(value) not in (int,float):
        raise ValueError(label+' requires a finite caller numeric value')
    try:valid=math.isfinite(value)
    except (OverflowError,ValueError):valid=False
    if not valid or value<0 or (positive and value==0):
        raise ValueError(label+' requires a finite nonnegative value')

def _bool(value,label):
    if type(value) is not bool:raise ValueError(label+' requires exact bool')

def _enum(value,cls):
    if type(value) is not cls:raise ValueError('exact '+cls.__name__+' required')

def _typed(value,cls,label):
    if type(value) is not tuple or any(type(x) is not cls for x in value):
        raise ValueError(label+' requires an immutable typed tuple')
    for x in value:x.__post_init__()

def _residue_id(r):
    return r.chain,r.index,r.insertion

def _atom_id(a):
    return a.entity,a.ordinal

def _unique(items,key,label):
    ids=tuple(key(x) for x in items)
    if len(set(ids))!=len(ids):raise ValueError('duplicate or ambiguous '+label)

def _json(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

class Direction(str,Enum):
    REACTION_TS='reaction_to_transition_state'
    DOCK_KNOWN='dock_ts_to_known_enzyme'
    DE_NOVO='design_de_novo_enzyme_for_ts'
    COMPLETION='complete_or_redesign_enzyme_for_ts'

class Role(str,Enum):
    SOURCE_OBSERVED='source_observed'
    USER_SUPPLIED='user_supplied'
    DESIRED='desired'
    GENERATED_INTERMEDIATE='generated_intermediate'
    WITHHELD_TARGET='withheld_target'
    EVALUATION_REFERENCE='evaluation_reference'

class InputKind(str,Enum):
    REACTION='reaction'
    TS_GEOMETRY='ts_geometry'
    ENZYME_SEQUENCE='enzyme_sequence'
    ENZYME_STRUCTURE='enzyme_structure'
    CATALYTIC_LABELS='catalytic_labels'
    CONTEXT='context'

class ReferenceKind(str,Enum):
    TS='TS'
    ENZYME_TEMPLATE='enzyme_template'
    BOUND_COMPLEX='bound_complex'

class Metric(str,Enum):
    TS_SHAPE='TS_internal_shape_RMSD'
    TS_POSE='TS_pose_in_protein_frame_RMSD'
    ENZYME='enzyme_template_RMSD'

class Selection(str,Enum):
    ALL_ATOM='explicit_all_selected_atoms'
    BACKBONE='mapped_backbone_atoms'
    COMPATIBLE_ALL_ATOM='explicit_compatible_all_atom_subset'

class Alignment(str,Enum):
    PROPER_RIGID='proper_rotation_translation'
    PROTEIN_ANCHORS='protein_anchor_transform_only'
    FIXED_FRAME='declared_fixed_common_frame'

class ConstraintUse(str,Enum):
    GUIDANCE='caller_visible_guidance'
    EVALUATION='blinded_post_generation_evaluation'

class InputPolicy(str,Enum):
    DEFAULT='sequence-identities/template-backbone'
    TEMPLATE='structure-priority'
    PER_REGION='per-region'
    STRICT='strict-consistency'

_INPUT_ROLES=frozenset((Role.SOURCE_OBSERVED,Role.USER_SUPPLIED,Role.DESIRED,Role.GENERATED_INTERMEDIATE))

@dataclass(frozen=True)
class ResidueKey:
    chain:str
    index:int
    identity:str
    insertion:str=''
    def __post_init__(self):
        _text(self.chain,'chain');_count(self.index,'sequence slot')
        _text(self.identity,'literal residue identity');_text(self.insertion,'insertion code',empty=True)

@dataclass(frozen=True)
class AtomKey:
    entity:str
    ordinal:int
    element:str
    name:str
    residue:ResidueKey|None=None
    isotope:int|None=None
    formal_charge:int|None=None
    def __post_init__(self):
        _text(self.entity,'atom namespace');_count(self.ordinal,'native atom ordinal')
        _text(self.element,'literal element');_text(self.name,'native atom name')
        if self.residue is not None:
            if type(self.residue) is not ResidueKey:raise ValueError('exact residue key')
            self.residue.__post_init__()
        if self.isotope is not None:_count(self.isotope,'isotope',1)
        if self.formal_charge is not None and (type(self.formal_charge) is not int or abs(self.formal_charge)>_MAX_COUNT):
            raise ValueError('exact bounded signed formal charge')

@dataclass(frozen=True)
class AtomPair:
    source:AtomKey
    reference:AtomKey
    def __post_init__(self):
        for x in (self.source,self.reference):
            if type(x) is not AtomKey:raise ValueError('exact atom correspondence')
            x.__post_init__()
        a,b=self.source,self.reference
        if a.element!=b.element:raise ValueError('atom element correspondence conflict')
        if a.isotope is not None and b.isotope is not None and a.isotope!=b.isotope:
            raise ValueError('known isotope correspondence conflict')
        if a.formal_charge is not None and b.formal_charge is not None and a.formal_charge!=b.formal_charge:
            raise ValueError('known charge correspondence conflict')
        if (a.residue is None)!=(b.residue is None):
            raise ValueError('unrelated protein/ligand atom sets')
        if a.residue is not None and a.residue.identity!=b.residue.identity:
            if a.name not in _BACKBONE or b.name not in _BACKBONE or a.name!=b.name:
                raise ValueError('incompatible sidechain comparison after identity mismatch')

@dataclass(frozen=True)
class ResiduePair:
    sequence:ResidueKey
    template:ResidueKey
    def __post_init__(self):
        for x in (self.sequence,self.template):
            if type(x) is not ResidueKey:raise ValueError('exact native residue pair')
            x.__post_init__()

@dataclass(frozen=True)
class InputSpec:
    key:str
    kind:InputKind
    role:Role
    available:bool=True
    residues:tuple[ResidueKey,...]=()
    complete:bool=False
    def __post_init__(self):
        _text(self.key,'input key');_enum(self.kind,InputKind);_enum(self.role,Role)
        if self.role not in _INPUT_ROLES:raise ValueError('target/evaluation cannot be a predictor input')
        _bool(self.available,'input availability');_bool(self.complete,'complete sequence declaration')
        _typed(self.residues,ResidueKey,'input residue axis')
        _unique(self.residues,_residue_id,'input residue axis')
        if not self.available and self.residues:raise ValueError('unavailable input cannot carry hidden residue values')
        if self.residues and self.kind not in (InputKind.ENZYME_SEQUENCE,InputKind.ENZYME_STRUCTURE):
            raise ValueError('residue axis needs an explicit sequence/structure role')
        if self.complete and self.kind is InputKind.ENZYME_SEQUENCE and not self.residues:
            raise ValueError('complete sequence needs its declared identities')

@dataclass(frozen=True)
class ReferenceSpec:
    key:str
    kind:ReferenceKind
    role:Role
    available:bool
    atoms:tuple[AtomKey,...]=()
    def __post_init__(self):
        _text(self.key,'reference key');_enum(self.kind,ReferenceKind);_enum(self.role,Role)
        _bool(self.available,'declared reference availability')
        _typed(self.atoms,AtomKey,'reference atom axis')
        _unique(self.atoms,_atom_id,'reference atom axis')
        if any(x.entity!=self.key for x in self.atoms):raise ValueError('reference atom namespace mismatch')

@dataclass(frozen=True)
class ConstraintSpec:
    key:str
    metric:Metric
    threshold:int|float
    reference_key:str|None
    required:bool=True
    units:str='angstrom'
    selection:Selection=Selection.ALL_ATOM
    alignment:Alignment=Alignment.PROPER_RIGID
    use:ConstraintUse=ConstraintUse.GUIDANCE
    atoms:tuple[AtomPair,...]=()
    protein_anchors:tuple[AtomPair,...]=()
    expected_reference_atoms:tuple[AtomKey,...]=()
    def __post_init__(self):
        _text(self.key,'constraint key');_enum(self.metric,Metric);_finite(self.threshold,'caller RMSD threshold')
        if self.reference_key is not None:_text(self.reference_key,'constraint reference')
        _bool(self.required,'hard/required constraint')
        if type(self.units) is not str or self.units!='angstrom':raise ValueError('exact angstrom units required; no conversion')
        _enum(self.selection,Selection);_enum(self.alignment,Alignment);_enum(self.use,ConstraintUse)
        _typed(self.atoms,AtomPair,'scored atom correspondence')
        _typed(self.protein_anchors,AtomPair,'protein alignment anchors')
        _typed(self.expected_reference_atoms,AtomKey,'explicit reference selection')
        for pairs,label in ((self.atoms,'scored'),(self.protein_anchors,'anchor')):
            _unique(pairs,lambda x:_atom_id(x.source),label+' source correspondence')
            _unique(pairs,lambda x:_atom_id(x.reference),label+' reference correspondence')
        _unique(self.expected_reference_atoms,_atom_id,'explicit reference selection')
        if self.metric is Metric.TS_POSE:
            if self.alignment is not Alignment.PROTEIN_ANCHORS:raise ValueError('pose uses protein anchors, never independent ligand fit')
        elif self.alignment is Alignment.PROTEIN_ANCHORS or self.protein_anchors:
            raise ValueError('protein-anchor pose policy cannot alias a shape/template metric')
        if self.selection is Selection.BACKBONE:
            if self.metric is not Metric.ENZYME or any(
                x.source.residue is None or x.source.name not in _BACKBONE
                or x.reference.name not in _BACKBONE or x.source.name!=x.reference.name for x in self.atoms):
                raise ValueError('explicit compatible backbone names required')

@dataclass(frozen=True)
class RegionRule:
    residues:tuple[ResidueKey,...]
    policy:InputPolicy
    freeze_sequence:bool=False
    redesign:bool=False
    freeze_all_atom_geometry:bool=False
    def __post_init__(self):
        _typed(self.residues,ResidueKey,'region residue selection')
        if not self.residues:raise ValueError('nonempty exact region')
        _unique(self.residues,_residue_id,'region selection');_enum(self.policy,InputPolicy)
        if self.policy is InputPolicy.PER_REGION:raise ValueError('no recursive priority policy')
        for x in (self.freeze_sequence,self.redesign,self.freeze_all_atom_geometry):_bool(x,'region mask')
        if self.redesign and (self.freeze_sequence or self.freeze_all_atom_geometry):
            raise ValueError('contradictory frozen/redesign region')
        if self.redesign and self.policy is InputPolicy.TEMPLATE:
            raise ValueError('redesign identity cannot simultaneously be fixed by template priority')

@dataclass(frozen=True)
class Budget:
    candidates:int
    max_atoms:int
    max_residues:int
    max_tokens:int
    max_forward_calls:int
    memory_bytes:int
    wall_seconds:int|float
    coordinate_steps:int=0
    pose_steps:int=0
    projection_iterations:int=0
    def __post_init__(self):
        for name in ('candidates','max_atoms','max_residues','max_tokens','max_forward_calls','memory_bytes'):
            _count(getattr(self,name),name,1)
        for name in ('coordinate_steps','pose_steps','projection_iterations'):_count(getattr(self,name),name)
        _finite(self.wall_seconds,'finite wall budget',positive=True)

@dataclass(frozen=True)
class DesignRequest:
    direction:Direction
    inputs:tuple[InputSpec,...]
    budget:Budget
    references:tuple[ReferenceSpec,...]=()
    constraints:tuple[ConstraintSpec,...]=()
    residue_map:tuple[ResiduePair,...]=()
    regions:tuple[RegionRule,...]=()
    policy:InputPolicy=InputPolicy.DEFAULT
    sequence_change_authorizations:tuple[ResidueKey,...]=()
    version:int=1
    def __post_init__(self):
        _enum(self.direction,Direction);_enum(self.policy,InputPolicy)
        if type(self.version) is not int or self.version!=1:raise ValueError('exact request version1')
        if type(self.budget) is not Budget:raise ValueError('explicit finite budget required')
        self.budget.__post_init__()
        for value,cls,label in ((self.inputs,InputSpec,'input declarations'),(self.references,ReferenceSpec,'references'),
            (self.constraints,ConstraintSpec,'constraints'),(self.residue_map,ResiduePair,'residue map'),
            (self.regions,RegionRule,'region priorities'),(self.sequence_change_authorizations,ResidueKey,'specific sequence-change authorization')):
            _typed(value,cls,label)
        _unique(self.inputs,lambda x:x.key,'input keys')
        _unique(self.references,lambda x:x.key,'reference keys')
        _unique(self.constraints,lambda x:x.key,'constraint keys')
        if set(x.key for x in self.inputs)&set(x.key for x in self.references):
            raise ValueError('input/reference namespace collision')
        _unique(self.residue_map,lambda x:_residue_id(x.sequence),'sequence-to-template map')
        _unique(self.residue_map,lambda x:_residue_id(x.template),'template-to-sequence map')
        _unique(self.sequence_change_authorizations,_residue_id,'sequence authorization')

@dataclass(frozen=True)
class MappingDecision:
    sequence:ResidueKey
    template:ResidueKey
    supplied_identity:str
    template_identity:str
    chosen_identity:str|None
    template_backbone:bool
    sidechain_scope:str
    sequence_change_authorized:bool
    effective_policy:InputPolicy

@dataclass(frozen=True)
class ResolvedResidueDecision:
    """One exact mapped subject, or an explicitly unmapped native endpoint.

    A mapped decision appears in both axis views, nonadditively. No missing
    supplied sequence/template or unknown redesigned identity is synthesized.
    """
    sequence:ResidueKey|None
    template:ResidueKey|None
    chosen_identity:str|None
    effective_policy:InputPolicy
    freeze_sequence:bool
    redesign:bool
    freeze_all_atom_geometry:bool
    sequence_change_authorized:bool|None
    region_endpoints:tuple[ResidueKey,...]
    geometry_scope:str
    sidechain_scope:str
    def __post_init__(self):
        if self.sequence is None and self.template is None:
            raise ValueError('at least one exact native subject required')
        for r in (self.sequence,self.template):
            if r is not None:
                if type(r) is not ResidueKey:raise ValueError('exact subject endpoint')
                r.__post_init__()
        if self.chosen_identity is not None:_text(self.chosen_identity,'chosen literal identity')
        _enum(self.effective_policy,InputPolicy)
        if self.effective_policy is InputPolicy.PER_REGION:raise ValueError('priority must be resolved')
        for x in (self.freeze_sequence,self.redesign,self.freeze_all_atom_geometry):_bool(x,'resolved mask')
        if self.redesign and (self.freeze_sequence or self.freeze_all_atom_geometry):
            raise ValueError('resolved freeze/redesign conflict')
        if self.sequence is None:
            if self.chosen_identity is not None or self.sequence_change_authorized is not None:
                raise ValueError('no fabricated supplied sequence or its authorization')
        else:_bool(self.sequence_change_authorized,'exact supplied-sequence authorization')
        if self.redesign and self.chosen_identity is not None:
            raise ValueError('redesigned identity remains unknown')
        _typed(self.region_endpoints,ResidueKey,'selected exact native endpoints')
        if any(r not in (self.sequence,self.template) for r in self.region_endpoints):
            raise ValueError('region endpoint not bound to this subject')
        _text(self.geometry_scope,'geometry scope');_text(self.sidechain_scope,'sidechain comparison scope')

@dataclass(frozen=True)
class ConstraintStatus:
    key:str
    metric:Metric
    threshold:int|float
    units:str
    required:bool
    use:ConstraintUse
    status:str
    measured_rmsd:None=None
    geometry_or_authority_verified:bool=False
    def __post_init__(self):
        if self.status not in ('PENDING_GEOMETRY','NOT_EVALUABLE') or self.measured_rmsd is not None or self.geometry_or_authority_verified is not False:
            raise ValueError('pure leaf cannot compute/verify/pass a geometry constraint')

@dataclass(frozen=True)
class CompiledDesignPlan:
    direction:Direction
    policy:InputPolicy
    predictor_input_keys:tuple[str,...]
    predictor_reference_keys:tuple[str,...]
    decisions:tuple[MappingDecision,...]
    constraint_statuses:tuple[ConstraintStatus,...]
    unmapped_sequence:tuple[ResidueKey,...]
    unmapped_template:tuple[ResidueKey,...]
    request_json:str
    request_sha256:str
    sequence_decisions:tuple[ResolvedResidueDecision,...]=()
    template_decisions:tuple[ResolvedResidueDecision,...]=()
    compiler_revision:str=COMPILER_REVISION
    overall_status:str='NOT_EVALUATED'
    source_or_model_authority:bool=False
    def __post_init__(self):
        if type(self.compiler_revision) is not str or self.compiler_revision!=COMPILER_REVISION:
            raise ValueError('exact compiler output revision required')
        _typed(self.sequence_decisions,ResolvedResidueDecision,'native sequence decisions')
        _typed(self.template_decisions,ResolvedResidueDecision,'native template decisions')
        if any(x.sequence is None for x in self.sequence_decisions) or any(x.template is None for x in self.template_decisions):
            raise ValueError('decision in wrong native-axis view')
        if self.overall_status!='NOT_EVALUATED' or self.source_or_model_authority is not False:
            raise ValueError('normalization is not geometry/source/model approval')
        if type(self.request_json) is not str or hashlib.sha256(self.request_json.encode('utf8')).hexdigest()!=self.request_sha256:
            raise ValueError('immutable normalized request binding')
    def canonical_bytes(self):
        self.__post_init__()
        result=asdict(self);result.pop('request_json');result['request']=json.loads(self.request_json)
        return _json(result).encode('utf8')
    def require_overall_pass(self,*args,**kwargs):
        raise RuntimeError('No RMSD computation/geometry validation in pure request leaf')
    def require_model_use(self,*args,**kwargs):
        raise RuntimeError('No source/model/runtime/training authority in pure request leaf')

def _region_for(request,residue,counterpart=None):
    endpoints=(residue,) if counterpart is None else (residue,counterpart)
    matches=tuple(r for r in request.regions if any(x in r.residues for x in endpoints))
    if not matches:return None
    if len({r.policy for r in matches})!=1:raise ValueError('contradictory overlapping priority regions')
    if any(r.freeze_sequence for r in matches) and any(r.redesign for r in matches):
        raise ValueError('overlapping frozen and redesign sequence')
    if any(r.freeze_all_atom_geometry for r in matches) and any(r.redesign for r in matches):
        raise ValueError('overlapping frozen all-atom geometry and redesign')
    first=matches[0]
    return RegionRule((residue,),first.policy,
        any(r.freeze_sequence for r in matches),any(r.redesign for r in matches),
        any(r.freeze_all_atom_geometry for r in matches))

def _resolve_subject(request,sequence,template):
    endpoint=sequence if sequence is not None else template
    rule=_region_for(request,endpoint,template if sequence is not None else None)
    policy=(rule.policy if rule else InputPolicy.DEFAULT) if request.policy is InputPolicy.PER_REGION else request.policy
    mismatch=sequence is not None and template is not None and sequence.identity!=template.identity
    if mismatch and request.policy is InputPolicy.PER_REGION and rule is None:
        raise ValueError('mismatching region lacks explicit priority')
    if mismatch and policy is InputPolicy.STRICT:
        raise ValueError('strict sequence/template identity conflict')
    freeze_sequence=bool(rule and rule.freeze_sequence)
    redesign=bool(rule and rule.redesign)
    freeze_geometry=bool(rule and rule.freeze_all_atom_geometry)
    if freeze_geometry and mismatch:
        raise ValueError('all-atom freeze incompatible with identity mismatch')
    authorized=sequence in request.sequence_change_authorizations if sequence is not None else None
    if sequence is None:
        chosen=None
    elif policy is InputPolicy.TEMPLATE:
        chosen=template.identity if template is not None else None
    else:
        chosen=sequence.identity
    if sequence is not None:
        change_known=chosen is not None and chosen!=sequence.identity
        if (change_known or redesign) and not authorized:
            raise ValueError('specific sequence-change authorization required')
        if freeze_sequence and (chosen is None or change_known or redesign):
            raise ValueError('frozen supplied identity conflicts with requested priority')
    if redesign:chosen=None
    if redesign:scope='redesigned_unknown_not_comparable'
    elif sequence is None:scope='template_only_no_supplied_sequence'
    elif template is None and policy is InputPolicy.TEMPLATE:scope='template_priority_identity_unavailable'
    elif template is None:scope='no_template_geometry'
    elif mismatch:scope='identity_mismatch_not_comparable'
    else:scope='explicit_compatible_identity'
    geometry_scope=('mapped_template_backbone' if sequence is not None else 'unmapped_template_native_only') if template is not None else 'not_supplied'
    endpoints=tuple(dict.fromkeys(r for r in (sequence,template) if r is not None))
    selected=tuple(r for r in endpoints if any(r in rr.residues for rr in request.regions))
    return ResolvedResidueDecision(sequence,template,chosen,policy,freeze_sequence,redesign,freeze_geometry,
        authorized,selected,geometry_scope,scope)

def _constraint_status(c,references):
    ref=references.get(c.reference_key)
    if ref is None or not ref.available:
        if c.required:raise ValueError('required reference unavailable or unknown: '+c.key)
        return ConstraintStatus(c.key,c.metric,c.threshold,c.units,c.required,c.use,'NOT_EVALUABLE')
    if c.use is ConstraintUse.GUIDANCE and ref.role not in _INPUT_ROLES:
        raise ValueError('hidden target/evaluation reference cannot guide conditioning')
    wanted={Metric.TS_SHAPE:ReferenceKind.TS,Metric.TS_POSE:ReferenceKind.BOUND_COMPLEX,
        Metric.ENZYME:ReferenceKind.ENZYME_TEMPLATE}[c.metric]
    if ref.kind is not wanted:raise ValueError('incompatible shape/pose/enzyme reference kind')
    if not c.atoms or not c.expected_reference_atoms:raise ValueError('complete explicit atom correspondence required')
    expected=tuple(c.expected_reference_atoms)
    if any(a not in ref.atoms for a in expected):raise ValueError('selected atom not in exact reference declaration')
    if set(x.reference for x in c.atoms)!=set(expected):
        raise ValueError('scored correspondence must cover exact selected reference set')
    if any(x.reference.entity!=ref.key for x in c.atoms):raise ValueError('constraint reference namespace mismatch')
    if c.metric is Metric.TS_POSE:
        if len(c.protein_anchors)<3 or any(x.source.residue is None or x.reference.residue is None for x in c.protein_anchors):
            raise ValueError('pose requires three distinct protein anchors; geometry still unverified')
        if any(x.reference not in ref.atoms for x in c.protein_anchors):
            raise ValueError('protein anchor outside declared complex reference')
        if set(_atom_id(x.source) for x in c.atoms)&set(_atom_id(x.source) for x in c.protein_anchors):
            raise ValueError('TS score atoms cannot be protein alignment anchors')
        if set(_atom_id(x.reference) for x in c.atoms)&set(_atom_id(x.reference) for x in c.protein_anchors):
            raise ValueError('TS score/reference anchors overlap')
    return ConstraintStatus(c.key,c.metric,c.threshold,c.units,c.required,c.use,'PENDING_GEOMETRY')

def compile_design_request(request):
    """Validate metadata only; no solver, file/parser, arrays, model or RNG hook."""
    if type(request) is not DesignRequest:raise ValueError('exact pure DesignRequest required')
    request.__post_init__()  # Revalidate forged frozen/nested fields before deriving anything.
    active=tuple(x for x in request.inputs if x.available)
    kinds={x.kind for x in active}
    needs={Direction.REACTION_TS:{InputKind.REACTION},Direction.DOCK_KNOWN:{InputKind.TS_GEOMETRY,InputKind.ENZYME_SEQUENCE},
        Direction.DE_NOVO:{InputKind.TS_GEOMETRY},Direction.COMPLETION:{InputKind.TS_GEOMETRY}}[request.direction]
    if not needs<=kinds:raise ValueError('missing required available direction input')
    seq=tuple(r for x in active if x.kind is InputKind.ENZYME_SEQUENCE for r in x.residues)
    template=tuple(r for x in active if x.kind is InputKind.ENZYME_STRUCTURE for r in x.residues)
    if request.direction is Direction.DOCK_KNOWN and not any(x.kind is InputKind.ENZYME_SEQUENCE and x.complete for x in active):
        raise ValueError('known enzyme needs complete supplied sequence; partial belongs to completion')
    if request.direction is Direction.DE_NOVO and kinds & {InputKind.ENZYME_SEQUENCE,InputKind.ENZYME_STRUCTURE}:
        raise ValueError('de novo cannot disguise supplied enzyme as an unknown target')
    if request.direction is Direction.REACTION_TS and kinds & {InputKind.TS_GEOMETRY,InputKind.ENZYME_SEQUENCE,InputKind.ENZYME_STRUCTURE}:
        raise ValueError('reaction TS direction cannot alias TS/enzyme target conditioning')
    if request.direction is Direction.COMPLETION and not kinds & {InputKind.ENZYME_SEQUENCE,InputKind.ENZYME_STRUCTURE}:
        raise ValueError('completion requires partial/full sequence or template')
    _unique(seq,_residue_id,'combined supplied sequence')
    _unique(template,_residue_id,'combined supplied template')
    if seq and template and not request.residue_map:raise ValueError('explicit partial sequence/template correspondence required')
    if any(x.sequence not in seq or x.template not in template for x in request.residue_map):
        raise ValueError('residue mapping does not bind exact supplied identities')
    known=set(seq)|set(template)
    if any(r not in known for region in request.regions for r in region.residues):raise ValueError('region outside supplied native subjects')
    if any(r not in seq for r in request.sequence_change_authorizations):raise ValueError('sequence-change authorization outside supplied sequence')
    if request.policy is InputPolicy.PER_REGION and not request.regions:raise ValueError('explicit region policy required')
    if request.policy is not InputPolicy.PER_REGION and any(r.policy is not request.policy for r in request.regions):
        raise ValueError('priority override must use explicit per-region policy')
    for r in known:_region_for(request,r)
    # RegionRule has no separate axis tag. A literal key present in both axes
    # can be jointly selected only if those endpoints are explicitly self-mapped.
    # Otherwise the caller must disambiguate native namespaces; no implicit join.
    for region in request.regions:
        for residue in region.residues:
            if residue in seq and residue in template and not any(
                p.sequence==residue and p.template==residue for p in request.residue_map):
                raise ValueError('ambiguous region key across native axes without exact self correspondence')
    by_sequence={p.sequence:p.template for p in request.residue_map}
    by_template={p.template:p.sequence for p in request.residue_map}
    sequence_decisions=tuple(_resolve_subject(request,r,by_sequence.get(r)) for r in seq)
    sequence_lookup={d.sequence:d for d in sequence_decisions}
    template_decisions=tuple(sequence_lookup[by_template[r]] if r in by_template
        else _resolve_subject(request,None,r) for r in template)
    decisions=[]
    # Preserve the original explicit map order as well as separate native views.
    for pair in request.residue_map:
        d=sequence_lookup[pair.sequence]
        decisions.append(MappingDecision(pair.sequence,pair.template,pair.sequence.identity,pair.template.identity,
            d.chosen_identity,True,d.sidechain_scope,d.sequence_change_authorized,d.effective_policy))
    references={r.key:r for r in request.references}
    statuses=tuple(_constraint_status(c,references) for c in request.constraints)
    redesigned_slots={_residue_id(r) for d in sequence_decisions+template_decisions if d.redesign
        for r in (d.sequence,d.template) if r is not None}
    if any(x.source.residue is not None and _residue_id(x.source.residue) in redesigned_slots
        and x.source.name not in _BACKBONE for c in request.constraints for x in c.atoms+c.protein_anchors):
        raise ValueError('redesigned unknown sidechain cannot be a prequalified scored or anchor atom')
    output_atoms={_atom_id(x.source) for c in request.constraints for x in c.atoms+c.protein_anchors}
    if len(output_atoms)>request.budget.max_atoms or len(seq)+len(template)>request.budget.max_residues:
        raise ValueError('declared mapping/residue records exceed explicit budget')
    requested_refs=tuple(r.key for r in request.references if r.available and r.role in _INPUT_ROLES
        and any(c.reference_key==r.key and c.use is ConstraintUse.GUIDANCE for c in request.constraints))
    raw=_json(asdict(request))
    return CompiledDesignPlan(request.direction,request.policy,tuple(x.key for x in active),requested_refs,
        tuple(decisions),statuses,tuple(r for r in seq if r not in {x.sequence for x in request.residue_map}),
        tuple(r for r in template if r not in {x.template for x in request.residue_map}),raw,
        hashlib.sha256(raw.encode('utf8')).hexdigest(),
        sequence_decisions=sequence_decisions,template_decisions=template_decisions)
