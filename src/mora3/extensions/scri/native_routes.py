"""Launch-inactive native SCRI route preparation on the existing finite reader.

Original row authentication, rights, split admission and complete-owner launch
remain with the coordinator. This module neither changes the owner gate nor
creates a competing training loop. Zero available targets never execute a model.
"""
from hashlib import sha256
from dataclasses import dataclass, asdict, field
from decimal import Decimal, localcontext
from pathlib import Path
import torch
from .contracts import StateRecord,TransitionPair,compile_transition,canonical,digest,number
from .curation import strict_json
from .model import SCRITargetAxis
from .spatial import build_neighborhood_query

SCHEMAS={'scri-transition':'Mora3.SCRITransitionEnvelope/v1',
         'scri-spatial':'Mora3.SCRISpatialEnvelope/v1'}


def decode_envelope(raw,*,task_key,axis,split='train'):
    if split not in ('train','validation'):
        raise ValueError('Only explicit train or validation roles are supported')
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
    if task_key=='scri-spatial':
        # The envelope carries targets outside the predictive graph, but they
        # are not independent labels: each must be the exact original query
        # cell observation on that native axis. No missing-to-zero conversion,
        # relabeling, implicit alignment or fabricated supervised coordinates.
        original=next(r for r in records if r.record_id==d['query_record_id'])
        originals={(m.modality,m.feature_id):m for m in original.measurements}
        for native_axis,target,present in zip(axis.features,targets,observed):
            modality,feature,unit=native_axis
            measured=originals.get((modality,feature))
            if (measured is None or measured.unit!=unit
                    or canonical((measured.value,measured.observed))!=canonical((target,present))):
                raise ValueError('original spatial query measurement, unit and mask must exactly match supervised label')
    # This is a train route, never a mechanism for reclassifying official heldouts.
    if any(r.official_split not in(None,split)for r in records):
        raise ValueError('official heldout or original source split cannot be reassigned')
    if len({r.source.evidence_kind for r in records})!=1:
        raise ValueError('one evidence stratum per native task required')
    return value,targets,observed,d['source_snapshot_sha256']


@dataclass(frozen=True)
class SCRICampaignAdapter:
    """Inspectable source-bound train/validation adapter, independent of weights."""
    route: object
    axis: SCRITargetAxis
    validation_source_paths: tuple[str,...]=()
    max_payload_bytes: int=2_000_000
    _identity: bytes=field(init=False,repr=False)

    def __post_init__(self):
        from mora3.research.primary_native_routes import FiniteNativeTrainRoute
        if (type(self.route)is not FiniteNativeTrainRoute or type(self.axis)is not SCRITargetAxis
                or self.route.task_key not in SCHEMAS or type(self.max_payload_bytes)is not int
                or self.max_payload_bytes<1 or type(self.validation_source_paths)is not tuple
                or any(type(x)is not str or not x for x in self.validation_source_paths)
                or len(set(self.validation_source_paths))!=len(self.validation_source_paths)):
            raise ValueError('Exact native SCRI route, axis and explicit resource bounds required')
        self.axis.__post_init__()
        object.__setattr__(self,'_identity',canonical(self._binding()))

    def _binding(self):
        from mora3.research.primary_native_routes import finite_view_policy_sha256
        r=self.route
        if finite_view_policy_sha256(r._view)!=r.native_view_policy_sha256:
            raise ValueError('Native SCRI view binding changed')
        validation=[]
        for path in self.validation_source_paths:
            matches=[d for d in r._view.descriptors if d['source_shard']['path']==path]
            if len(matches)!=1:raise ValueError('Exact original validation shard required')
            descriptor=matches[0]
            if not any((c['source_family'],c['task_name'],c['native_schema'],c['source_revision'])in r._identity
                       and c['split']=='validation' and c['eligible_view_disposition']==1 and c['rows']>0
                       for c in descriptor['source_task_schema_counts']):
                raise ValueError('Validation shard lacks the original eligible SCRI task')
            validation.append(descriptor['source_shard'])
        dependencies={name:sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                      for name in ('contracts.py','curation.py','model.py','spatial.py','native_owner.py','native_routes.py')}
        return dict(schema='Mora3.SCRICampaignAdapter/v2',source_snapshot_sha256=r.source_snapshot_sha256,
            task_key=r.task_key,source_identities=sorted(r._identity),source_paths=r._paths,
            source_shards=[sha256(x).hexdigest()for x in r._shard_refs],batch_size=r._batch_size,
            initial_cursor_sha256=sha256(r.initial_cursor).hexdigest(),native_view_policy_sha256=r.native_view_policy_sha256,
            axis=asdict(self.axis),validation_shards=validation,max_payload_bytes=self.max_payload_bytes,
            dependencies=dependencies,objective='mean native NLL over explicitly observed coordinates',
            test_selection=False,admission_authority=False)

    def executable_binding(self):
        result=self._binding()
        if canonical(result)!=self._identity:raise ValueError('Native SCRI source/axis/route binding changed')
        return result

    def train_binding(self):
        from mora3.research.primary_native_routes import NativeRouteBinding
        self.executable_binding()
        return NativeRouteBinding(self.route,self.prepare,'native negative log likelihood','observed native coordinates')

    def _targets(self,batch,split):
        from mora3.data.finite_union_native_rows import SelectedNativeSourceBatch
        self.executable_binding();r=self.route
        if type(batch)is not SelectedNativeSourceBatch or batch.original_split!=split:
            raise ValueError('Exact original native SCRI source batch and split required')
        allowed=r._paths if split=='train'else self.validation_source_paths
        if batch.source_shard['path']not in allowed:
            raise ValueError('Batch source shard is outside the declared SCRI role')
        r._view.require_native_rows(batch,split=split)
        values=[];targets=[];masks=[]
        for row in batch.native_rows.to_pylist():
            identity=tuple(row[k]for k in ('source_family','task_name','native_schema','source_revision'))
            if identity not in r._identity or row['split']!=split:
                raise ValueError('Original SCRI source task or revision differs')
            raw=row.get('native_payload')
            if type(raw)not in(bytes,str):raise ValueError('Complete original native payload required')
            rawbytes=raw.encode()if type(raw)is str else raw
            if len(rawbytes)>self.max_payload_bytes:raise ValueError('Complete native SCRI payload exceeds byte budget')
            if sha256(rawbytes).hexdigest()!=row.get('native_payload_sha256'):
                raise ValueError('Original native SCRI payload hash mismatch')
            value,target,mask,snapshot=decode_envelope(raw,task_key=r.task_key,axis=self.axis,split=split)
            if snapshot!=r.source_snapshot_sha256:raise ValueError('Task source snapshot mismatch')
            values.append(value);targets.append(target);masks.append(mask)
        return tuple(values),targets,masks,sum(sum(mask)for mask in masks)

    def _owner(self,model):
        from mora3.model.full_owner import SourceBoundOwner
        from .native_owner import NativeSCRIModel
        if (type(model)is not SourceBoundOwner or type(model.scri)is not NativeSCRIModel
                or model.get_extra_state()['source_snapshot_sha256']!=self.route.source_snapshot_sha256
                or model.scri.core.axis!=self.axis):
            raise ValueError('Exact registered SCRI owner, native axis and source snapshot required')
        return model.scri

    def _loss(self,model,prepared):
        values,targets,masks,count=prepared;branch=self._owner(model)
        prediction=branch.forward_spatial(values)if self.route.task_key=='scri-spatial'else branch(values)
        result=branch.objective(prediction,targets,masks)
        if result['observed_targets']!=count:raise ValueError('Native target denominator changed')
        return result['loss']

    def prepare(self,window):
        from mora3.research.primary_native_routes import NativeRouteWindow,PreparedNativeTargets
        if type(window)is not NativeRouteWindow:raise ValueError('Exact native SCRI train window required')
        prepared=self._targets(window.batch,'train');count=prepared[3]
        if not count:return PreparedNativeTargets(0,'no observed source targets',None)
        def evaluate(model):
            self._owner(model)
            model.require_training_admission()
            return self._loss(model,prepared)
        return PreparedNativeTargets(count,None,evaluate)

    def validation_observed_targets(self,batch):
        return self._targets(batch,'validation')[3]

    def evaluate_validation(self,model,batch):
        from mora3.research.mup_campaign import ValidationBatchReduction
        if torch.is_grad_enabled():raise ValueError('Native SCRI validation must be evaluation-only')
        prepared=self._targets(batch,'validation');count=prepared[3]
        if not count:return ValidationBatchReduction(Decimal(0),0,'no observed source targets')
        loss=self._loss(model,prepared)
        with localcontext()as ctx:
            ctx.prec=34;numerator=Decimal(str(float(loss)))*count
        return ValidationBatchReduction(numerator,count,None)


def build_native_scri_binding(owner,route):
    """Compatibility entry point using the inspectable typed campaign adapter."""
    from mora3.model.full_owner import SourceBoundOwner
    from mora3.research.primary_native_routes import FiniteNativeTrainRoute
    if (type(owner)is not SourceBoundOwner or owner.scri is None or type(route)is not FiniteNativeTrainRoute
            or route.source_snapshot_sha256!=owner.get_extra_state()['source_snapshot_sha256']):
        raise ValueError('Exact registered SCRI owner and native source route required')
    return SCRICampaignAdapter(route,owner.scri.core.axis).train_binding()


def registration_proposal(source_snapshot_sha256):
    digest(source_snapshot_sha256,'source snapshot')
    return dict(schema='Mora3.SCRINativeRegistrationProposal/v1',source_snapshot_sha256=source_snapshot_sha256,
        routes=SCHEMAS,prepare='mora3.extensions.scri.native_routes.build_native_scri_binding',
        owner='mora3.model.full_owner.construct_primary_owner',
        role_builder='mora3.extensions.scri.native_owner.native_scri_roles',
        production_launch_enabled=False,newly_admitted_sources=0,
        required_before_activation=('original source/revision/rights/counts','curation and specimen/study/region/lineage groups',
            'complete native manifests and masks','same-snapshot coordinator owner attestations','operator/resource/recovery qualification'))
