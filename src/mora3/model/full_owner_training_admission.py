"""Process-local owner authorization from explicitly reviewed complete evidence.

The repository-reviewed release pins are the authority boundary. A caller's
path and digest, a diagnostic receipt, and checkpoint state convey no authority.
No approved full-project release is present in this source revision.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import os
from pathlib import Path
import threading
from weakref import WeakKeyDictionary

from mora3.data.admet import canonical, digest

_RELEASE_SCHEMA = 'Mora3.RootReviewedPrimaryCampaignTrainingRelease/v1'
_LAUNCH_RELEASE_SCHEMA = 'Mora3.RootReviewedPrimaryCampaignTrainingRelease/v2'
_REVIEWED_RELEASES: frozenset[tuple[str, str]] = frozenset()
_PROCESS = os.getpid()
_ACTIVE = WeakKeyDictionary()
_LOCK = threading.Lock()
_HEADS = tuple(sorted((
    'core','admet','mcsa','pubchem','cartesian','source_annotations','cell','text','lincs_inverse',
    'native_priors','scalar','pdb','rosetta_flow','gray_reverse','gray_forward','bioelectric','nci',
    'expansionrx','cema_trajectory','reactzyme','tox21_source','hcn4_laterality','reactome_static',
    'biogrid_chemical','lincx_fetch','string_score','uniprot_native_annotation','uniprot_full_record',
)))
_NATIVE29_HEADS = tuple(sorted((set(_HEADS)-{'admet'})|{'betse','raw_cell'}))


@dataclass(frozen=True)
class _Admission:
    release_sha256: str
    campaign_sha256: str
    trial_id: str
    state_sha256: str
    layout_sha256: str
    window_capacity_sha256: str
    heads: tuple[str, ...]


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _same(actual, expected, label):
    _require(type(actual) is dict and canonical(actual)==canonical(expected),label)


def _metadata(root, reference):
    from mora3.research.primary_campaign_assembly import _document,_read

    path=Path(reference.path)
    _require(path.is_relative_to(Path('outputs')) and path.suffix=='.json'
             and path.name!='private-packet.json'
             and not {'data-intake','data-sources','shards','validation','heldout','held-out','test'}.intersection(path.parts),
             'Only bounded review metadata outside source payload roots is supported')
    return _document(_read(root,reference))


def _owner_identity(owner, heads=_HEADS):
    from .full_owner import SourceBoundOwner,unique_parameter_inventory

    _require(tuple(sorted(owner._modules))==heads and all(owner._modules.values()),
             'Full native owner branch inventory required for admission')
    state=SourceBoundOwner.get_extra_state(owner)
    _require(state['training_admission'] is False
             and state['primary_arm']=='ordinary-attention-mup-primary-v1','Exact checkpoint-neutral primary owner required')
    inventory=unique_parameter_inventory(owner)
    # Device placement changes no aliases, extents, trainability or parameter dtype.
    rows=[{k:v for k,v in row.items() if k!='device'} for row in inventory['parameters']]
    return state,inventory['total_parameters'],_layout(rows)


def _layout(rows):
    rows=[dict(row,aliases=tuple(sorted(row['aliases']))) for row in rows]
    return digest(canonical(sorted(rows,key=lambda row:row['aliases'])))


def _base_family_layout(base):
    return _layout([dict(aliases=r.aliases,shape=r.shape,numel=r.numel,trainable=r.trainable,dtype=r.dtype)
                    for r in base.parameters])


def _require_route_scope(campaign):
    from mora3.research.primary_campaign_construction62 import _ROUTE_KEYS,_TEXT_KEYS,MCSA,finite,legacy

    validation_keys={finite.INVERSE,legacy.FREE,MCSA}|set(_TEXT_KEYS.values())
    _require({r.task_key for r in campaign.train_routes}==_ROUTE_KEYS
             and {r.task_key for r in campaign.validation.tasks}==validation_keys,
             'Exact complete production 62-route and five-validation scope required')


def _reviewed_evidence(root, inputs, owner_state_sha, owner_layout_sha, heads):
    from . import full_owner
    from mora3.research import mup_campaign_worker
    from mora3.research.mup_sweep_selection import MAX_UPDATES

    campaign=inputs.campaign
    cohort=campaign.validation
    source=inputs.source_snapshot.sha256
    evidence={row.key:_metadata(root,row.reference) for row in inputs.evidence}
    common=dict(source_snapshot_sha256=source)
    schedule=dict(common,schema='Mora3.ReviewedPrimarySourceRoleSchedule/v1',
        full_horizon=MAX_UPDATES,task_cycle_sha256=digest(canonical(campaign.task_cycle)),
        train_route_plan_sha256=digest(canonical([asdict(r) for r in campaign.train_routes])),
        validation_cohort_sha256=cohort.sha256,validation_group_guard_sha256=cohort.same_snapshot_group_guard_sha256,
        validation_initial_reference_sha256=cohort.initial_reference_artifact_sha256,
        original_roles_preserved=True,global_group_review_accepted=True,matched_validation_reference_accepted=True)
    _same(evidence['source_role_schedule'],schedule,'Exact reviewed full source-role and validation evidence required')
    owner=dict(common,schema='Mora3.ReviewedCompletePrimaryOwnerEvidence/v1',
        owner_implementation_sha256=sha256(Path(full_owner.__file__).read_bytes()).hexdigest(),
        owner_extra_state_sha256=owner_state_sha,owner_parameter_layout_sha256=owner_layout_sha,
        width_family_sha256=campaign.width_family_sha256,active_owner_branches=list(heads),
        full_source_owner_complete=True,native_operators_qualified=True,strict_checkpoint_recovery_qualified=True)
    _same(evidence['complete_owner_evidence'],owner,'Exact complete reviewed owner/operator/recovery evidence required')
    runtime=evidence['runtime_profile']
    expected=dict(common,schema='Mora3.ReviewedPrimaryRuntimeProfile/v1',width_family_sha256=campaign.width_family_sha256,
        full_horizon=MAX_UPDATES,
        max_one_update_seconds=str(campaign.max_one_update_seconds),
        max_matched_validation_seconds=str(campaign.max_matched_validation_seconds),
        complete_operator_runtime_measured=True,full_horizon_capacity_verified=True)
    if 'native_cpu_prefetch' in runtime:
        options=runtime['native_cpu_prefetch']
        _require(type(options) is dict and set(options)=={'workers','task_keys'}
                 and type(options['workers']) is int and 1<=options['workers']<=4
                 and type(options['task_keys']) is list and options['task_keys']
                 and all(type(k) is str for k in options['task_keys'])
                 and len(set(options['task_keys']))==len(options['task_keys'])
                 and set(options['task_keys'])<={r.task_key for r in campaign.train_routes},
                 'Exact bounded native CPU prefetch profile required')
        expected['native_cpu_prefetch']=options
    _same(runtime,expected,'Exact measured full runtime and capacity evidence required')
    worker=dict(common,schema='Mora3.ReviewedExistingPrimaryWorkerEvidence/v1',
        worker_implementation_sha256=sha256(Path(mup_campaign_worker.__file__).read_bytes()).hexdigest(),
        exclusive_worker_control_reviewed=True,terminal_prefix_and_reservation_recovery_reviewed=True)
    _same(evidence['existing_worker'],worker,'Exact reviewed existing worker evidence required')


def load_reviewed_primary_release(root, *, path, expected_sha256):
    """Authenticate release authority and canonical metadata without an owner."""
    from mora3.research.primary_campaign_assembly import _reference,load_primary_campaign_request

    release_ref=_reference({'path':path,'sha256':expected_sha256})
    _require((release_ref.path,release_ref.sha256) in _REVIEWED_RELEASES,
             'No repository-reviewed full primary campaign training release matches this pin')
    root=Path(root).resolve()
    release=_metadata(root,release_ref)
    fields={'schema','scope','primary_request','campaign_sha256','trial_id','evidence',
            'verified_window_capacity_sha256','review_state'}
    if release.get('schema')==_LAUNCH_RELEASE_SCHEMA:
        fields.add('launch_request')
    _require(set(release)==fields and release['schema'] in (_RELEASE_SCHEMA,_LAUNCH_RELEASE_SCHEMA)
             and release['scope']=='full-primary-campaign-training'
             and release['review_state']=='accepted-full-campaign',
             'Exact repository-reviewed full primary release required')
    if release['schema']==_LAUNCH_RELEASE_SCHEMA:
        _reference(release['launch_request'])
    request=_reference(release['primary_request'])
    declaration=_metadata(root,request)
    # Authenticate path class and bytes before the existing canonical loader
    # follows its declared JSON metadata references.
    for key in ('source_snapshot','campaign','validation','width_family'):
        _metadata(root,_reference(declaration[key]))
    for key in ('configurations','evidence'):
        for value in declaration[key].values():
            _metadata(root,_reference(value))
    inputs=load_primary_campaign_request(root,request.path,request.sha256)
    campaign=inputs.campaign
    _require(campaign.sha256==release['campaign_sha256'] and inputs.trial.trial_id==release['trial_id']
             and inputs.source_view_policy=='effective-study-v2'
             and release['evidence']=={row.key:asdict(row.reference) for row in inputs.evidence},
             'Exact campaign, trial and complete reviewed evidence pins required')
    _require_route_scope(campaign)
    # Capacity binds the campaign hash, which already includes the runtime
    # profile digest. The external reviewed release avoids a digest cycle.
    window=_reference({'path':'capacity','sha256':release['verified_window_capacity_sha256']}).sha256
    owner_evidence=_metadata(root, next(row.reference for row in inputs.evidence
                                        if row.key=='complete_owner_evidence'))
    state_sha=_reference({'path':'owner-state','sha256':owner_evidence.get('owner_extra_state_sha256')}).sha256
    layout_sha=_reference({'path':'owner-layout','sha256':owner_evidence.get('owner_parameter_layout_sha256')}).sha256
    heads=_NATIVE29_HEADS if release['schema']==_LAUNCH_RELEASE_SCHEMA else _HEADS
    _reviewed_evidence(root,inputs,state_sha,layout_sha,heads)
    return release_ref, release, inputs, window


def authorize(owner, root, *, path, expected_sha256):
    """Bind one exact owner instance and one canonical campaign trial."""
    from .full_owner import SourceBoundOwner

    _require(os.getpid()==_PROCESS and type(owner) is SourceBoundOwner,'Exact local SourceBoundOwner required')
    release_ref,release,inputs,window=load_reviewed_primary_release(root,path=path,expected_sha256=expected_sha256)
    campaign=inputs.campaign
    heads=_NATIVE29_HEADS if release['schema']==_LAUNCH_RELEASE_SCHEMA else _HEADS
    state,numel,layout=_owner_identity(owner,heads)
    configurations=dict(inputs.configurations)
    _require(state['source_snapshot_sha256']==inputs.source_snapshot.sha256
             and numel==campaign.proxy_parameter_count and layout==_base_family_layout(inputs.family.base),'Live owner differs from the reviewed base family member')
    # Factory inputs and the stored owner metadata have deliberately different
    # layouts. Source identity is top-level; the owner adds heads/layers and a
    # vocabulary digest. Never require equality to asdict(PrimaryConfiguration).
    from mora3.research.campaign26_transfer import require_primary_target_configuration
    matched = require_primary_target_configuration(owner, configurations['base'])
    _require(canonical(matched)==canonical(state), 'Factory owner changed between native admission checks')
    state_sha=digest(canonical(state))
    _reviewed_evidence(Path(root).resolve(),inputs,state_sha,layout,heads)
    token=_Admission(release_ref.sha256,campaign.sha256,inputs.trial.trial_id,state_sha,layout,window,heads)
    with _LOCK:
        _require(owner not in _ACTIVE,'An owner instance can be admitted for one exact trial only')
        _ACTIVE[owner]=token


def require(owner, *, campaign_sha256=None, trial_id=None, window_capacity_sha256=None):
    from .full_owner import SourceBoundOwner

    _require(os.getpid()==_PROCESS and type(owner) is SourceBoundOwner,
             'Exact local SourceBoundOwner required')
    with _LOCK:
        token=_ACTIVE.get(owner)
    _require(token is not None,'construction is not full corpus/operator/recovery/resource campaign admission')
    _require((campaign_sha256 is None or token.campaign_sha256==campaign_sha256)
             and (trial_id is None or token.trial_id==trial_id),'Training release belongs to a different campaign or trial')
    _require(window_capacity_sha256 is None or token.window_capacity_sha256==window_capacity_sha256,
             'Live full-horizon capacity differs from the reviewed release')
    state,_,layout=_owner_identity(owner,token.heads)
    _require(digest(canonical(state))==token.state_sha256 and layout==token.layout_sha256,
             'Live owner configuration or parameter layout changed after reviewed admission')
