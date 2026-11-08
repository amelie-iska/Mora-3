"""Five-stage sampler curriculum; neither DFT validation nor physical kinetics.

No optimizer, serialization loader, admission or automatic checkpoint reuse.
Caller must durably save model/optimizer/RNG plus this exact binding atomically.
"""
from dataclasses import dataclass,replace,asdict
from mora3.boltzmann_target import BoltzmannTarget
from mora3.teacher_protocol import TeacherRefusal,finite_float,hex_sha,canonical,sha

@dataclass(frozen=True,slots=True)
class AnnealingPlan:
    final_target: BoltzmannTarget
    updates_per_stage: tuple[int,...]
    gaps_kelvin: tuple[float,...]=(160.,80.,40.,20.)

    def __post_init__(self):
        if type(self.final_target)is not BoltzmannTarget or self.final_target.teacher.family not in ('UMA','ESEN'):
            raise TeacherRefusal('exact UMA/ESEN final target required')
        if (type(self.updates_per_stage)is not tuple or len(self.updates_per_stage)!=5
            or any(type(n)is not int or n<1 for n in self.updates_per_stage)):
            raise TeacherRefusal('five explicit positive update budgets required')
        if (type(self.gaps_kelvin)is not tuple or len(self.gaps_kelvin)!=4
            or any(not finite_float(x) or x<=0 for x in self.gaps_kelvin)
            or any(a<=b for a,b in zip(self.gaps_kelvin,self.gaps_kelvin[1:]))):
            raise TeacherRefusal('four strictly decreasing positive kelvin gaps required')
        ts=self.temperatures_kelvin
        drops=tuple(a-b for a,b in zip(ts,ts[1:]))
        if any(a<=b for a,b in zip(drops,drops[1:])):
            raise TeacherRefusal('represented adjacent temperature drops must strictly decrease')
        if any(not finite_float(t) for t in ts) or any(a<=b for a,b in zip(ts,ts[1:])):
            raise TeacherRefusal('five representable strictly decreasing temperatures required')

    @property
    def temperatures_kelvin(self):
        t=self.final_target.temperature_kelvin
        return tuple(t+sum(self.gaps_kelvin[i:]) for i in range(4))+(t,)

    def target(self,stage):
        if type(stage)is not int or not 0<=stage<5:raise TeacherRefusal('stage index 0..4 required')
        return replace(self.final_target,temperature_kelvin=self.temperatures_kelvin[stage])

    def binding(self):
        return dict(schema='Mora3.BoltzmannAnnealingPlan/v1',final_target=self.final_target.checkpoint_binding(),
            gaps_kelvin=list(self.gaps_kelvin),temperatures_kelvin=list(self.temperatures_kelvin),
            updates_per_stage=list(self.updates_per_stage),stage_targets=[self.target(i).checkpoint_binding() for i in range(5)],
            semantics='sampler-curriculum-not-physical-time-or-DFT-validation')

    @property
    def sha256(self):return sha(canonical(self.binding()))

@dataclass(frozen=True,slots=True)
class AnnealingState:
    plan: AnnealingPlan
    completed_updates: tuple[int,...]=(0,0,0,0,0)
    checkpoint_payload_sha256: str|None=None

    def __post_init__(self):
        if type(self.plan)is not AnnealingPlan or type(self.completed_updates)is not tuple or len(self.completed_updates)!=5:
            raise TeacherRefusal('exact plan and ordered progress required')
        open_stage=False
        for n,budget in zip(self.completed_updates,self.plan.updates_per_stage):
            if type(n)is not int or not 0<=n<=budget or (open_stage and n):raise TeacherRefusal('non-prefix stage progress')
            if n<budget:open_stage=True
        if (sum(self.completed_updates)>0 and not hex_sha(self.checkpoint_payload_sha256)) or (self.checkpoint_payload_sha256 is not None and not hex_sha(self.checkpoint_payload_sha256)):
            raise TeacherRefusal('exact model/optimizer/RNG payload digest required after progress')

    @property
    def active_stage(self):
        return next((i for i,(n,b) in enumerate(zip(self.completed_updates,self.plan.updates_per_stage)) if n<b),None)

    def evidence_binding(self,role):
        if role not in ('training','validation') or (self.active_stage is None and role=='training'):raise TeacherRefusal('active training stage or exact validation role required')
        i=4 if self.active_stage is None else self.active_stage
        return dict(schema='Mora3.BoltzmannAnnealingEvidence/v1',plan_sha256=self.plan.sha256,stage=i,
            target=self.plan.target(i).checkpoint_binding(),role=role,completed_updates=list(self.completed_updates))

    def require_evidence(self,binding,role):
        if canonical(binding)!=canonical(self.evidence_binding(role)):raise TeacherRefusal('stage/temperature/progress/role evidence mismatch')

    def record_update(self,evidence,*,checkpoint_payload_sha256):
        self.require_evidence(evidence,'training')
        if not hex_sha(checkpoint_payload_sha256) or checkpoint_payload_sha256==self.checkpoint_payload_sha256:
            raise TeacherRefusal('fresh durable optimizer/model/RNG payload identity required')
        counts=list(self.completed_updates);counts[self.active_stage]+=1
        return replace(self,completed_updates=tuple(counts),checkpoint_payload_sha256=checkpoint_payload_sha256)

    def checkpoint_binding(self):
        return dict(schema='Mora3.BoltzmannAnnealingCheckpoint/v1',plan=self.plan.binding(),
            completed_updates=list(self.completed_updates),active_stage=self.active_stage,
            checkpoint_payload_sha256=self.checkpoint_payload_sha256)

    def validate_checkpoint_binding(self,binding):
        if canonical(binding)!=canonical(self.checkpoint_binding()):raise TeacherRefusal('annealing checkpoint mismatch; no relabel or implicit resume')

def annealed_cartesian_teacher_trajectory_loss(policy,state,teacher,*,batch_size,generator,request_id):
    """Fresh stage-conditioned paths/rewards; completion acknowledged separately."""
    if type(state)is not AnnealingState or state.active_stage is None:raise TeacherRefusal('active exact annealing state required')
    from mora3.model.cartesian_teacher_trajectory import cartesian_teacher_trajectory_loss
    result=cartesian_teacher_trajectory_loss(policy,state.plan.target(state.active_stage),teacher,
        batch_size=batch_size,generator=generator,request_id=request_id)
    if result['target_sha256']!=state.plan.target(state.active_stage).sha256:raise TeacherRefusal('returned target mismatch')
    return dict(result,annealing_evidence=state.evidence_binding('training'))
