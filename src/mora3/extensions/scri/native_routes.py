"""Launch-inactive native SCRI route preparation on the existing finite reader.

Original row authentication, rights, split admission and complete-owner launch
remain with the coordinator. This module neither changes the owner gate nor
creates a competing training loop. Zero available targets never execute a model.
"""
from hashlib import sha256
from .contracts import StateRecord,TransitionPair,compile_transition,canonical,digest,number
from .curation import strict_json
from .model import SCRITargetAxis
from .spatial import build_neighborhood_query

SCHEMAS={'scri-transition':'Mora3.SCRITransitionEnvelope/v1',
         'scri-spatial':'Mora3.SCRISpatialEnvelope/v1'}


def decode_envelope(raw,*,task_key,axis):
    if type(raw)not in(bytes,str)or task_key not in SCHEMAS or type(axis)is not SCRITargetAxis:
        raise ValueError('explicit native SCRI envelope and source target axis required')
    d=strict_json(raw)
    common={'schema','source_snapshot_sha256'}
    wanted=common|({'baseline','outcome','pairing_kind','pairing_evidence_sha256'}if task_key=='scri-transition'
        else{'baseline_records','query_record_id','radius','target_axis','target_values','target_observed'})
    if type(d)is not dict or set(d)!=wanted or d['schema']!=SCHEMAS[task_key]:
        raise ValueError('exact versioned native SCRI envelope fields required')
    digest(d['source_snapshot_sha256'],'native source snapshot')
    if task_key=='scri-transition':
        pair=TransitionPair(StateRecord.from_mapping(d['baseline']),StateRecord.from_mapping(d['outcome']),
            d['pairing_kind'],d['pairing_evidence_sha256'])
        compiled=compile_transition(pair)
        if tuple(tuple(x)for x in compiled['target']['axis'])!=axis.features:
            raise ValueError('exact complete transition target axis required')
        value=pair;targets=compiled['target']['values'];observed=compiled['target']['observed']
        records=(pair.baseline,pair.outcome)
    else:
        if type(d['baseline_records'])is not list or type(d['target_axis'])is not list:
            raise ValueError('exact native spatial record and axis arrays required')
        if tuple(tuple(x)for x in d['target_axis'])!=axis.features:raise ValueError('spatial source target axis mismatch')
        records=tuple(StateRecord.from_mapping(x)for x in d['baseline_records'])
        value=build_neighborhood_query(records,query_record_id=d['query_record_id'],target_axis=axis.features,radius=d['radius'])
        targets=d['target_values'];observed=d['target_observed']
    if (type(targets)is not list or type(observed)is not list or len(targets)!=len(axis.features)
            or len(observed)!=len(targets)or any(type(m)is not bool or m!=(v is not None)for m,v in zip(observed,targets))):
        raise ValueError('complete native targets and exact missingness mask required')
    axis.__post_init__()
    for i,(target,present) in enumerate(zip(targets,observed)):
        if not present:continue
        if axis.likelihood=='categorical':
            if type(target)is not str or target not in axis.categories[i]:
                raise ValueError('unknown native categorical target')
        else:
            number(target,'native target')
            if axis.likelihood=='poisson' and (target<0 or int(target)!=target or target>2**53):
                raise ValueError('exact nonnegative native count target required')
    # This is a train route, never a mechanism for reclassifying official heldouts.
    if any(r.official_split not in(None,'train')for r in records):
        raise ValueError('official heldout cannot enter native train route')
    if len({r.source.evidence_kind for r in records})!=1:
        raise ValueError('one evidence stratum per native task required')
    return value,targets,observed,d['source_snapshot_sha256']


def build_native_scri_binding(owner,route):
    """Return an existing NativeRouteBinding; its engine still demands owner admission."""
    from mora3.model.full_owner import SourceBoundOwner
    from mora3.research.primary_native_routes import FiniteNativeTrainRoute,NativeRouteBinding,PreparedNativeTargets,NativeRouteWindow
    if type(owner)is not SourceBoundOwner or owner.scri is None or type(route)is not FiniteNativeTrainRoute:
        raise ValueError('exact registered SCRI primary owner and source-bound finite route required')
    snapshot=owner.get_extra_state()['source_snapshot_sha256']
    if route.task_key not in SCHEMAS or route.source_snapshot_sha256!=snapshot:
        raise ValueError('exact primary route/source snapshot required')
    axis=owner.scri.core.axis
    def prepare(window):
        if type(window)is not NativeRouteWindow:raise ValueError('native reader window required')
        route._view.require_native_rows(window.batch,split='train')
        values=[];targets=[];masks=[]
        for row in window.batch.native_rows.to_pylist():
            raw=row.get('native_payload')
            if type(raw)not in(bytes,str):raise ValueError('complete original native payload required')
            rawbytes=raw.encode()if type(raw)is str else raw
            if sha256(rawbytes).hexdigest()!=row.get('native_payload_sha256'):
                raise ValueError('original native SCRI payload hash mismatch')
            value,target,mask,binding=decode_envelope(raw,task_key=route.task_key,axis=axis)
            if binding!=snapshot:raise ValueError('task source snapshot mismatch')
            values.append(value);targets.append(target);masks.append(mask)
        count=sum(sum(mask)for mask in masks)
        if not count:return PreparedNativeTargets(0,'no observed source targets',None)
        def evaluate(model):
            if model is not owner:raise ValueError('prepared source route requires the exact registered owner')
            owner.get_extra_state()
            predictions=(owner.scri.forward_spatial(tuple(values))if route.task_key=='scri-spatial'else owner.scri(tuple(values)))
            result=owner.scri.objective(predictions,targets,masks)
            if result['observed_targets']!=count:raise ValueError('native target denominator changed')
            return result['loss']
        return PreparedNativeTargets(count,None,evaluate)
    return NativeRouteBinding(route,prepare,'native negative log likelihood','observed native coordinates')


def registration_proposal(source_snapshot_sha256):
    digest(source_snapshot_sha256,'source snapshot')
    return dict(schema='Mora3.SCRINativeRegistrationProposal/v1',source_snapshot_sha256=source_snapshot_sha256,
        routes=SCHEMAS,prepare='mora3.extensions.scri.native_routes.build_native_scri_binding',
        owner='mora3.model.full_owner.construct_primary_owner',
        role_builder='mora3.extensions.scri.native_owner.native_scri_roles',
        production_launch_enabled=False,newly_admitted_sources=0,
        required_before_activation=('original source/revision/rights/counts','curation and specimen/study/region/lineage groups',
            'complete native manifests and masks','same-snapshot coordinator owner attestations','operator/resource/recovery qualification'))
