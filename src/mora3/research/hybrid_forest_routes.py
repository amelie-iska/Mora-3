"""Counted native engine adapter for source-bound hybrid forest learning.

This is a route inside NativeMuUpdateEngine, not another optimizer, training
worker or admission authority. Source conditions and a separate fixed reward
checkpoint are immutable inputs. All proposed RNG/cursor changes are owned by
the existing engine; no wrapper retries an optimizer failure.
"""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import torch
from mora3.data.schema import GraphRecord
from mora3.data.tasks import ConditionalTask
from mora3.model.forest_flow import _wire, forest_context_sha256
from mora3.model.forest_training import ForestBatchIdentity
from mora3.model.hybrid_forest_teacher import (NativeHybridForestTeacher, FrozenForestReward,
    hybrid_forest_objective, _tensor_sha)
from mora3.model.full_owner import SourceBoundOwner
from .forest_training_routes import _clone
from .primary_training import NativeMuUpdateEngine, _rng_state, _restore_rng, _require_forest_cursor


@dataclass(frozen=True)
class HybridForestCondition:
    record: GraphRecord
    task: ConditionalTask
    root_ids: tuple
    root_states: torch.Tensor
    identity: ForestBatchIdentity


def _condition_wire(condition):
    if (type(condition)is not HybridForestCondition or type(condition.record)is not GraphRecord
            or type(condition.task)is not ConditionalTask or type(condition.identity)is not ForestBatchIdentity):
        raise ValueError('Exact native problem, known-root export and source identity required')
    condition.identity.__post_init__()
    return _wire(dict(identity=asdict(condition.identity),record=asdict(condition.record),
        task=condition.task.canonical_bytes().decode(),root_ids=condition.root_ids,
        roots=_tensor_sha(condition.root_states)))


class HybridForestTrainingRoute:
    """Finite original-condition segment; generated forests are not source rows."""
    def __init__(self,teacher,reward,conditions,*,task_key,trial_id,candidates,
                 balance_weight=1.,student_weight=1.,max_passes=1,max_resident_bytes=64*1024**2):
        if (type(teacher)is not NativeHybridForestTeacher or type(reward)is not FrozenForestReward
                or type(conditions)is not tuple or not 1<=len(conditions)<=1024
                or any(type(x)is not str or not 0<len(x)<=4096 for x in (task_key,trial_id))
                or type(candidates)is not int or candidates<1
                or type(max_passes)is not int or not 1<=max_passes<=1_000_000
                or type(max_resident_bytes)is not int or not 1<=max_resident_bytes<=512*1024**2
                or any(type(w)not in(int,float)or not math.isfinite(w)or w<=0
                       for w in (balance_weight,student_weight))):
            raise ValueError('Exact finite hybrid route, trial and positive fixed loss weights required')
        self.teacher=teacher;self.reward=reward;self.task_key=task_key;self.trial_id=trial_id
        self.candidates=candidates;self.balance_weight=balance_weight;self.student_weight=student_weight
        self.max_passes=max_passes;self.max_resident_bytes=max_resident_bytes
        self._teacher_owner=(teacher,);self._reward_owner=(reward,)
        self._heads_sha256=sha256(_wire(teacher.binding())).hexdigest()
        self._reward_sha256=reward.validate(teacher)
        ids=set();ordinals=set();size=0
        for condition in conditions:
            raw=_condition_wire(condition);identity=condition.identity
            if (identity.split!='train' or len(identity.record_ids)!=1
                    or identity.source_snapshot_sha256!=teacher.topology.field.spec.source_snapshot_sha256
                    or identity.coordinate_sha256!=teacher.topology.field.spec.coordinate_sha256
                    or identity.encoder_sha256!=teacher.spec.encoder_sha256
                    or identity.group_policy_sha256!=teacher.spec.group_policy_sha256
                    or identity.export_sha256!=teacher.spec.source_export_sha256
                    or identity.skeleton_policy_sha256!=teacher.topology.spec.construction_policy_sha256):
                raise ValueError('One original train-only condition on the exact native source/encoder/group/policy required')
            key=(identity.source_revision,identity.export_sha256,identity.record_ids[0])
            ordinal=(identity.source_revision,identity.export_sha256,identity.original_ordinals[0])
            if key in ids or ordinal in ordinals:
                raise ValueError('Duplicate original source condition/ordinal in hybrid route')
            ids.add(key);ordinals.add(ordinal)
            teacher.preflight(condition.record,condition.task,condition.root_ids,condition.root_states,candidates)
            size+=len(raw)+condition.root_states.numel()*condition.root_states.element_size()
            if size>max_resident_bytes:
                raise ValueError('Complete original-condition segment exceeds explicit byte bound; no truncation')
        self._conditions=tuple(_clone(c)for c in conditions)
        self._condition_pins=tuple(sha256(_condition_wire(c)).hexdigest()for c in self._conditions)
        self._objective_wire=_wire(self._objective())
        self.objective_sha256=sha256(self._objective_wire).hexdigest()
        self._declaration_wire=_wire(self._declaration())
        self.route_sha256=sha256(self._declaration_wire).hexdigest()
        self.resident_bytes=size

    @property
    def source_snapshot_sha256(self):
        return self.teacher.topology.field.spec.source_snapshot_sha256

    def _objective(self):
        return dict(schema='Mora3.HybridForestObjective/v1',heads_sha256=self._heads_sha256,
            reward_sha256=self._reward_sha256,candidates=self.candidates,
            balance_weight=self.balance_weight,student_weight=self.student_weight,
            source_snapshot_sha256=self.source_snapshot_sha256,
            sample_selection='all-unpruned-complete-forests',split='train')

    def _declaration(self):
        return dict(schema='Mora3.FiniteHybridForestRoute/v1',task_key=self.task_key,trial_id=self.trial_id,
            objective_sha256=self.objective_sha256,source_snapshot_sha256=self.source_snapshot_sha256,
            conditions=self._condition_pins,max_passes=self.max_passes,max_resident_bytes=self.max_resident_bytes)

    def _verify(self):
        if (self._teacher_owner!=(self.teacher,)or self._reward_owner!=(self.reward,)
                or sha256(_wire(self.teacher.binding())).hexdigest()!=self._heads_sha256
                or self.reward.validate(self.teacher)!=self._reward_sha256
                or _wire(self._objective())!=self._objective_wire
                or sha256(self._objective_wire).hexdigest()!=self.objective_sha256
                or _wire(self._declaration())!=self._declaration_wire
                or sha256(self._declaration_wire).hexdigest()!=self.route_sha256
                or tuple(sha256(_condition_wire(c)).hexdigest()for c in self._conditions)!=self._condition_pins):
            raise ValueError('Frozen hybrid source, reward, owner or objective changed')

    def binding(self):
        self._verify()
        return dict(schema='Mora3.NativeForestObjectiveBinding/v3',trial_id=self.trial_id,task_key=self.task_key,
            source_snapshot_sha256=self.source_snapshot_sha256,route_sha256=self.route_sha256,
            objective_sha256=self.objective_sha256,heads_sha256=self._heads_sha256,
            total_windows=len(self._conditions)*self.max_passes,
            sampling_method='engine-owned-hybrid-topology-Gaussian-flow-v1',
            skeleton_policy='generated-complete-forest-on-policy-hybrid-v1')

    def _cursor(self,position):
        return _wire(dict(schema='Mora3.ForestTrainingCursor/v1',route_sha256=self.route_sha256,position=position))

    def initial_cursor(self):
        self._verify();return self._cursor(0)

    def select(self,cursor):
        binding=self.binding();_require_forest_cursor(binding,cursor)
        position=json.loads(cursor)['position']
        if position==binding['total_windows']:
            return dict(condition=None,status='source-exhausted',cursor_before=cursor,cursor_after=cursor)
        index=position%len(self._conditions)
        return dict(condition=_clone(self._conditions[index]),condition_sha256=self._condition_pins[index],
            status='source-prepared',cursor_before=cursor,cursor_after=self._cursor(position+1))


def native_engine_hybrid_forest_loss(engine,route,*,trial_id):
    if type(engine)is not NativeMuUpdateEngine or type(engine.model)is not SourceBoundOwner:
        raise ValueError('Exact native engine/full owner required')
    engine.model.require_training_admission(campaign_sha256=engine.binding_sha256,trial_id=trial_id)
    if type(route)is not HybridForestTrainingRoute or trial_id!=route.trial_id:
        raise ValueError('Exact source-bound hybrid route and counted trial required')
    binding=route.binding();engine.require_forest_objective_binding(**binding)
    teacher=route.teacher;owner=engine.model
    if (teacher.topology is not owner.forest_topology or teacher.topology.field is not owner.forest_flow
            or not any(m is teacher.policy for m in owner.modules())
            or engine.next_task!=route.task_key
            or owner.get_extra_state()['source_snapshot_sha256']!=route.source_snapshot_sha256):
        raise ValueError('Exact registered native teacher and next source task required')
    sampler=engine._sampler_state()
    if sampler is None or torch.device(sampler['device'])!=teacher.topology.field.readout.weight.device:
        raise ValueError('Existing checkpoint-owned same-device generator required')
    before=engine.source_cursors[route.task_key];selected=route.select(before)
    condition=selected.pop('condition')
    if condition is None:
        return dict(selected,loss=None,optimizer_updates=0,training_admission=False)
    global_before=_rng_state();rng_before=engine.sampler_generator.get_state().clone()
    try:
        result=hybrid_forest_objective(teacher,condition.record,condition.task,route.reward,
            root_ids=condition.root_ids,root_states=condition.root_states,count=route.candidates,
            generator=engine.sampler_generator,split=condition.identity.split,
            balance_weight=route.balance_weight,student_weight=route.student_weight)
        engine.require_forest_objective_binding(**route.binding())
        if sha256(_condition_wire(condition)).hexdigest()!=selected['condition_sha256']:
            raise ValueError('Original native condition changed during objective evaluation')
        if not result['loss'].requires_grad:
            raise ValueError('Native hybrid objective must be differentiable')
    except BaseException:
        engine.sampler_generator.set_state(rng_before);_restore_rng(global_before);raise
    selected['status']='objective-prepared-not-updated'
    return dict(selected,**result,original_source_rows=1,
                original_record_ids=condition.identity.record_ids,
                original_ordinals=condition.identity.original_ordinals,
                source_evidence_kind=condition.identity.evidence_kind)


def run_native_hybrid_forest_update(engine,route,*,trial_id):
    prepared=native_engine_hybrid_forest_loss(engine,route,trial_id=trial_id)
    if prepared['loss']is None:return prepared
    # After entering update, only the counted native recovery protocol may resume.
    event=engine.update(route.task_key,prepared['loss'],cursor_before=prepared['cursor_before'],
                        cursor_after=prepared['cursor_after'])
    return dict(status='updated',event=event,condition_sha256=prepared['condition_sha256'],
        balance_loss=float(prepared['balance']['loss'].detach()),
        student_loss=float(prepared['student']['loss'].detach()),
        sample_sha256=prepared['batch'].sample_sha256,original_source_rows=1,
        cursor_before=prepared['cursor_before'],cursor_after=prepared['cursor_after'])
