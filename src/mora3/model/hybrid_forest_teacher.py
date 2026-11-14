"""Native hybrid forest GFlowNet teacher and conditional flow-matching student.

Terminal objects are ordered forests AND every free Euclidean latent, not an
endpoint or a beam-selected answer. Counting measure over the finite canonical
DFS grammar is combined with a topology-specific product Lebesgue measure.
Topology is drawn first; each Gaussian increment sees only original conditions,
given roots and that node's ancestors. Construction reversal is deterministic.

The fixed reward is a normalized grammar/random-walk density times exp(-E),
where E is nonnegative frozen criterion energy. Thus its partition is in (0,1].
This is a Mora-3 adaptation of continuous/hybrid GFN and conditional FM theory,
not a claim of learned reward proportionality, calibrated validity or admission.
No parameters, optimizer, worker or training budget are created here.
"""
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import copy
import math
import torch
from torch.nn import functional as F

from mora3.data.schema import GraphRecord, Node, Edge, Measurement
from mora3.data.tasks import ConditionalTask
from mora3.data.provenance import Provenance
from mora3.reasoning.forest_topology import prefix_state, canonical_forest_actions, maximum_forest_nodes
from mora3.reasoning.flow_forest_refinement import ForestState, forest_fingerprint
from mora3.reasoning.continuous_gfn import diagonal_gaussian_log_density
from .forest_flow import _wire, _sha, _parameter_versions, forest_context_sha256, forest_flow_matching_loss
from .forest_topology import (NativeForestTopologyHead, sample_forest_topologies,
                              sampled_topology_forest, topology_log_probability)
from .forest_quality import NativeForestQualityHead
from .profile_gfn import ProfileGFNPolicy


def _tensor_sha(value):
    value=value.detach().contiguous()
    return sha256(_wire([list(value.shape),str(value.dtype)])+
                  value.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()


def _model_sha(model):
    """Bind actual parameter AND buffer values, not a configuration-only label."""
    before=_parameter_versions(model)
    h=sha256(_wire(model.get_extra_state()))
    for kind,items in (('parameter',model.named_parameters()),('buffer',model.named_buffers())):
        for name,value in items:
            h.update(_wire([kind,name,_tensor_sha(value)]))
    if before!=_parameter_versions(model):
        raise ValueError('Native model changed during immutable snapshot binding')
    return h.hexdigest()


def _storage(model):
    return {(str(p.device),p.untyped_storage().data_ptr()) for p in model.parameters() if p.numel()}


def _train(split):
    if split!='train':
        raise ValueError('Original train-only hybrid source conditions required')


@dataclass(frozen=True)
class HybridForestSpec:
    encoder_sha256: str
    group_policy_sha256: str
    source_export_sha256: str
    context_coefficient: float=.25
    max_prefix_bytes: int=2_000_000

    def __post_init__(self):
        for value in (self.encoder_sha256,self.group_policy_sha256,self.source_export_sha256):
            _sha(value)
        if (type(self.context_coefficient)not in(int,float)
                or not math.isfinite(self.context_coefficient)or self.context_coefficient<=0
                or type(self.max_prefix_bytes)is not int or not 1<=self.max_prefix_bytes<=8_000_000):
            raise ValueError('Explicit finite ancestor coefficient and prefix-byte capacity required')


class NativeHybridForestTeacher:
    """Parameter-free composition of two already-registered native heads.

    The endpoint policy is reused with a distinct graph-encoded objective tag and
    parent-relative increment semantics; its original endpoint route is unchanged.
    Eval mode makes sampling/rescoring deterministic without disabling gradients.
    """
    def __init__(self,topology,policy,spec):
        if (type(topology)is not NativeForestTopologyHead or type(policy)is not ProfileGFNPolicy
                or type(spec)is not HybridForestSpec):
            raise ValueError('Exact native topology, profile GFN and hybrid specification required')
        self.topology=topology;self.policy=policy;self.spec=spec
        self._owners=(topology,policy)
        self._configuration=_wire(self._binding())

    def _binding(self):
        self.spec.__post_init__();top=self.topology;policy=self.policy
        top.get_extra_state();policy.validate_parameterization_configuration()
        field=top.field;geometry=field.geometry
        if ((top,policy)!=self._owners or policy.dimension!=field.spec.dimension
                or policy.context_width!=field.width
                or geometry._source_backbone_reference[0]is not policy.profiles.shared.backbone
                or any(getattr(geometry.graph,n)is not getattr(policy.profiles.shared.backbone,n)
                       for n in ('feature','role','graph_token','layers','final_norm'))):
            raise ValueError('Exact shared native transformer, context and latent axis required')
        ref=field.readout.weight
        if any(p.device!=ref.device or p.dtype!=ref.dtype for p in policy.parameters()):
            raise ValueError('Exact same-device and precision native teacher required')
        return dict(schema='Mora3.NativeHybridForestTeacher/v1',spec=asdict(self.spec),
            topology=top.get_extra_state(),policy=policy.get_extra_state(),
            terminal_object='canonical-ordered-forest-and-all-nonroot-latents',
            measure='finite-topology-counting-times-topology-specific-Lebesgue',
            continuous_kernel='parent-relative-diagonal-Gaussian-with-ancestor-only-context',
            backward='deterministic-canonical-construction-deletion-mass-one',
            partition_condition='original-context-and-given-roots-only-not-sampled-topology',
            forest_selection='all-unedited-samples-no-rejection-or-beam',
            adds_parameters=False,training_admission=False)

    def binding(self):
        result=self._binding()
        if _wire(result)!=self._configuration:
            raise ValueError('Frozen hybrid source/method/operator binding changed')
        return copy.deepcopy(result)

    def snapshot_sha256(self):
        return sha256(_wire([self.binding(),_model_sha(self.topology),_model_sha(self.policy)])).hexdigest()

    def preflight(self,record,task,root_ids,root_states,count):
        self.binding();field=self.topology.field;ref=field.readout.weight
        if (type(root_ids)is not tuple or not root_ids or len(set(root_ids))!=len(root_ids)
                or any(type(x)is not str or not 0<len(x)<=4096 for x in root_ids)
                or any(x.startswith('mora-generated-node:')for x in root_ids)
                or type(root_states)is not torch.Tensor or root_states.requires_grad
                or root_states.shape!=(len(root_ids),field.spec.dimension)
                or root_states.dtype!=ref.dtype or root_states.device!=ref.device
                or not torch.isfinite(root_states).all()
                or type(count)is not int or not 1<=count<=self.topology.spec.max_batch_size
                or any(m.training for head in self._owners for m in head.modules())):
            raise ValueError('Exact detached given roots, bounded count and eval-mode native heads required')
        bounds=self.topology.spec.bounds
        reachable_depth=min(bounds.max_depth,bounds.max_nodes-len(root_ids)) if bounds.max_children else 0
        if reachable_depth>self.policy.horizon:
            raise ValueError('Whole reachable forest depth exceeds registered native GFN horizon')
        if count*(2*bounds.max_nodes-len(root_ids))>self.topology.spec.max_total_decisions:
            raise ValueError('Complete batch topology decision capacity exceeded')
        self.topology.preflight(record,task,root_count=len(root_ids),actions=())
        # Root-prefix bytes are checked before sampling or invoking any neural operator.
        self._prefix_graph(root_states,(),None,None)

    def _prefix_graph(self,roots,ancestors,forest,active_root):
        """Given roots plus this ancestry only; no sibling, future or target nodes."""
        values=[row for row in roots]
        values.extend(forest.states[i] for i in ancestors if i>=len(roots))
        nodes=[];edges=[];roles=[]
        for i,row in enumerate(values):
            fields=[Measurement('objective','hybrid-complete-forest-v1',None,'observed',True),
                    Measurement('given_root',i<len(roots),None,'observed',True),
                    Measurement('active_root',i==active_root,None,'observed',True)]
            fields.extend(Measurement('latent_coordinate:'+str(j),float(v),'latent-coordinate','predicted',True)
                          for j,v in enumerate(row))
            nodes.append(Node('given-reasoning-root'if i<len(roots)else'generated-ancestor',tuple(fields)))
            roles.extend((('node',i,m.name),'condition')for m in fields)
            if i>=len(roots):
                edges.append(Edge(active_root if i==len(roots)else i-1,i,'computational-ancestry',True))
        # A missing request is a target role, never a future latent value.
        nodes[-1]=Node(nodes[-1].kind,nodes[-1].measurements+
                       (Measurement('requested_next_state',None,None,'missing',False),))
        roles.append((('node',len(nodes)-1,'requested_next_state'),'target'))
        record=GraphRecord('hybrid-reasoning-prefix',tuple(nodes),tuple(edges),
            (Provenance('generated-latent-computation','v1','not-biological-causality',
             self.topology.field.spec.source_snapshot_sha256,'internal-derived-no-source-admission'),))
        task=ConditionalTask(tuple(roles),topology_known=True)
        if len(_wire(asdict(record)))+len(task.canonical_bytes())>self.spec.max_prefix_bytes:
            raise ValueError('Complete ancestor payload exceeds explicit capacity; no truncation')
        return record,task

    def _context(self,record,task,roots,ancestors=(),forest=None,active_root=None):
        prefix,prefix_task=self._prefix_graph(roots,ancestors,forest,active_root)
        field=self.topology.field
        base=field.encode_context(record,task)
        encoded=field.encode_context(prefix,prefix_task)
        context=base+self.spec.context_coefficient*encoded*(1+base)
        if not torch.isfinite(context).all():
            raise ValueError('Nonfinite native ancestor context')
        return context

    def log_partition(self,record,task,roots):
        return self.policy.log_partition(self._context(record,task,roots)).reshape(())

    def transition_parameters(self,record,task,forest,node):
        self.binding()
        if type(forest)is not ForestState:
            raise ValueError('Exact complete forest required')
        forest.__post_init__();roots=len(forest.clamped)
        if (forest.clamped!=tuple(range(roots)) or forest.parents[:roots]!=(-1,)*roots
                or type(node)is not int or not roots<=node<len(forest.parents)
                or forest.states.requires_grad or forest.coordinate_sha256!=self.topology.field.spec.coordinate_sha256
                or forest.context_sha256!=forest_context_sha256(record,task,
                     source_snapshot_sha256=self.topology.field.spec.source_snapshot_sha256)):
            raise ValueError('Detached native ancestor state and original condition binding required')
        chain=[];parent=forest.parents[node]
        while parent>=0:
            chain.append(parent);parent=forest.parents[parent]
        chain=tuple(reversed(chain));layer=len(chain)-1
        if layer>=self.policy.horizon:
            raise ValueError('Ancestor depth exceeds the frozen native policy horizon')
        context=self._context(record,task,forest.states[:roots],chain,forest,chain[0])
        mean,log_std,_=self.policy.parameters_at(forest.states[forest.parents[node]:forest.parents[node]+1],layer,context)
        scale=log_std.exp()
        if not torch.isfinite(scale).all()or(scale<=0).any():
            raise ValueError('Unrepresentable native Gaussian scale; no clipping or replacement')
        return mean[0],log_std[0]


def uniform_topology_log_mass(bounds,root_count,actions):
    """Normalized reference via uniform allowed actions, not uniform final trees."""
    final=prefix_state(bounds,root_count,actions)
    if not final.complete:
        raise ValueError('Complete canonical topology required for reference mass')
    return -sum(math.log(len(prefix_state(bounds,root_count,actions[:i]).allowed_actions))
                for i in range(len(actions)))


class FrozenForestReward:
    """Detached, fixed criterion-energy surrogate; not a biological validity oracle.

    Only explicitly supported criteria enter the energy; unknown criteria are not
    silently assigned negative labels. A source-trained frozen snapshot must be
    separately qualified before production admission. Digests alone do not do so.
    """
    def __init__(self,quality,teacher,*,training_evidence_sha256,supported_criteria,
                 scale=1.,node_weight=1.,split='train'):
        _train(split);_sha(training_evidence_sha256)
        if type(quality)is not NativeForestQualityHead or type(teacher)is not NativeHybridForestTeacher:
            raise ValueError('Exact frozen native quality head and teacher required')
        if (type(supported_criteria)is not tuple or not supported_criteria
                or len(set(supported_criteria))!=len(supported_criteria)
                or any(c not in quality.spec.criteria for c in supported_criteria)
                or type(scale)not in(int,float)or not math.isfinite(scale)or scale<=0
                or type(node_weight)not in(int,float)or not math.isfinite(node_weight)or node_weight<0):
            raise ValueError('Explicit supported criterion subset and finite reward scales required')
        self.quality=quality;self.supported_criteria=supported_criteria
        self.scale=float(scale);self.node_weight=float(node_weight)
        self.training_evidence_sha256=training_evidence_sha256
        self._teacher_configuration=_wire(teacher.binding())
        self._configuration=_wire(self._binding())
        self._snapshot=_model_sha(quality)
        self.validate(teacher)

    def _binding(self):
        return dict(schema='Mora3.FrozenHybridForestReward/v1',quality=self.quality.get_extra_state(),
            training_evidence_sha256=self.training_evidence_sha256,criteria=self.supported_criteria,
            scale=self.scale,node_weight=self.node_weight,split='train',
            topology_reference='uniform-allowed-DFS-actions-not-uniform-topologies',
            coordinate_reference='independent-zero-mean-parent-relative-Gaussian-increments',
            energy='mean-supported-criterion-softplus-negative-logit-plus-weighted-nonroot-node-mean',
            reward_partition_upper_bound=1,calibrated=False,training_admission=False)

    def validate(self,teacher):
        quality=self.quality;field=teacher.topology.field
        if (_wire(self._binding())!=self._configuration or _model_sha(quality)!=self._snapshot
                or _wire(teacher.binding())!=self._teacher_configuration
                or any(p.requires_grad for p in quality.parameters())
                or any(m.training for m in quality.modules())
                or _storage(quality)&(_storage(teacher.topology)|_storage(teacher.policy))
                or quality.field.spec.source_snapshot_sha256!=field.spec.source_snapshot_sha256
                or quality.field.spec.coordinate_sha256!=field.spec.coordinate_sha256
                or quality.field.spec.dimension!=field.spec.dimension
                or quality.field.spec.max_nodes<teacher.topology.spec.bounds.max_nodes
                or any(p.dtype!=field.readout.weight.dtype or p.device!=field.readout.weight.device
                       for p in quality.parameters())):
            raise ValueError('Separate immutable frozen same-source/coordinate reward snapshot required')
        return sha256(_wire([self._configuration.decode(),self._snapshot])).hexdigest()

    @torch.no_grad()
    def evaluate(self,teacher,record,task,forest):
        self.validate(teacher)
        roots=len(forest.clamped);canonical=canonical_forest_actions(teacher.topology.spec.bounds,
            forest.parents,root_count=roots)
        q=forest.states.new_tensor(uniform_topology_log_mass(teacher.topology.spec.bounds,roots,canonical.actions))
        if len(forest.parents)>roots:
            delta=torch.stack([forest.states[i]-forest.states[p] for i,p in enumerate(forest.parents)if p>=0])
            q=q+diagonal_gaussian_log_density(delta,torch.zeros_like(delta),
                torch.full_like(delta,math.log(self.scale))).sum()
        out=self.quality(record,task,forest.states[None],forest)
        index=[self.quality.spec.criteria.index(c)for c in self.supported_criteria]
        energy=F.softplus(-out['criterion_logits'][0,index]).mean()
        if len(forest.parents)>roots:
            energy=energy+self.node_weight*F.softplus(-out['node_logits'][0,roots:][:,index]).mean()
        if not torch.isfinite(energy)or energy<0 or not torch.isfinite(q-energy):
            raise ValueError('Finite nonnegative frozen criterion energy required')
        return dict(log_reference=q.detach(),energy=energy.detach(),log_reward=(q-energy).detach())


@dataclass(frozen=True)
class HybridForestBatch:
    forests: tuple
    topology_samples: tuple
    root_ids: tuple
    root_states: torch.Tensor
    teacher_sha256: str
    context_sha256: str
    sampler: str
    teacher_owner: tuple
    generator_before: torch.Tensor
    generator_after: torch.Tensor
    sample_sha256: str


def _batch_sha(batch):
    return sha256(_wire(dict(forests=[forest_fingerprint(f)for f in batch.forests],
        topologies=batch.topology_samples,root_ids=batch.root_ids,roots=_tensor_sha(batch.root_states),
        teacher=batch.teacher_sha256,context=batch.context_sha256,sampler=batch.sampler,
        generator_before=_tensor_sha(batch.generator_before),generator_after=_tensor_sha(batch.generator_after)))).hexdigest()


def _verify_batch(teacher,record,task,batch):
    if type(batch)is not HybridForestBatch or batch.teacher_owner!=(teacher,):
        raise ValueError('Exact native on-policy batch owner required')
    teacher.preflight(record,task,batch.root_ids,batch.root_states,len(batch.forests))
    context=forest_context_sha256(record,task,source_snapshot_sha256=teacher.topology.field.spec.source_snapshot_sha256)
    if (batch.sampler!='native-unpruned-topology-then-ancestor-Gaussians-v1'
            or batch.teacher_sha256!=teacher.snapshot_sha256()or batch.context_sha256!=context
            or len(batch.topology_samples)!=len(batch.forests)or _batch_sha(batch)!=batch.sample_sha256):
        raise ValueError('Edited, pruned, stale or wrong-condition samples are not native on-policy data')
    roots=len(batch.root_ids)
    for forest,sample in zip(batch.forests,batch.topology_samples):
        state=prefix_state(teacher.topology.spec.bounds,roots,tuple(sample['actions']))
        if (not state.complete or forest.parents!=state.parents or tuple(sample['parents'])!=state.parents
                or forest.skeleton_origin!='generated' or not forest.is_complete
                or forest.node_ids[:roots]!=batch.root_ids or forest.clamped!=tuple(range(roots))
                or forest.states.requires_grad or not torch.equal(forest.states[:roots],batch.root_states)
                or forest.context_sha256!=context or forest.coordinate_sha256!=teacher.topology.field.spec.coordinate_sha256):
            raise ValueError('Complete unmodified ordered forest and given roots required')


@torch.no_grad()
def sample_hybrid_forests(teacher,record,task,*,root_ids,root_states,count,generator):
    if type(teacher)is not NativeHybridForestTeacher:
        raise ValueError('Exact native hybrid teacher required')
    teacher.preflight(record,task,root_ids,root_states,count)
    if type(generator)is not torch.Generator or torch.device(generator.device)!=root_states.device:
        raise ValueError('Explicit same-device sampler generator required')
    before=generator.get_state().clone();pin=teacher.snapshot_sha256()
    context=forest_context_sha256(record,task,source_snapshot_sha256=teacher.topology.field.spec.source_snapshot_sha256)
    try:
        samples=sample_forest_topologies(teacher.topology,record,task,root_count=len(root_ids),count=count,generator=generator)
        forests=[]
        for sample in samples['samples']:
            forest=sampled_topology_forest(teacher.topology,record,task,sample,root_ids=root_ids,root_states=root_states)
            for i in range(len(root_ids),len(forest.parents)):
                mean,ls=teacher.transition_parameters(record,task,forest,i)
                value=forest.states[forest.parents[i]]+mean+ls.exp()*torch.randn(mean.shape,
                    dtype=mean.dtype,device=mean.device,generator=generator)
                if not torch.isfinite(value).all():
                    raise ValueError('Nonfinite complete-forest sample; no silent retry or clipping')
                forest.states[i]=value
            forests.append(forest)
        batch=HybridForestBatch(tuple(forests),samples['samples'],root_ids,root_states.detach().clone(),pin,context,
            'native-unpruned-topology-then-ancestor-Gaussians-v1',(teacher,),before,generator.get_state().clone(),'')
        batch=replace(batch,sample_sha256=_batch_sha(batch))
        _verify_batch(teacher,record,task,batch)
        return batch
    except BaseException:
        generator.set_state(before)
        raise


def hybrid_forest_balance(teacher,record,task,batch,reward,*,split):
    """Exact joint TB loss. Sampled values and frozen rewards carry no gradients."""
    _train(split);_verify_batch(teacher,record,task,batch)
    if type(reward)is not FrozenForestReward:
        raise ValueError('Exact immutable finite-partition forest reward required')
    reward.validate(teacher)
    log_z=teacher.log_partition(record,task,batch.root_states)
    logs=[];rewards=[]
    for forest in batch.forests:
        lp=topology_log_probability(teacher.topology,record,task,forest.parents,
                                   root_count=len(batch.root_ids))['log_probability']
        for i in range(len(batch.root_ids),len(forest.parents)):
            mean,ls=teacher.transition_parameters(record,task,forest,i)
            lp=lp+diagonal_gaussian_log_density(forest.states[i]-forest.states[forest.parents[i]],mean,ls)
        logs.append(lp);rewards.append(reward.evaluate(teacher,record,task,forest))
    log_forward=torch.stack(logs)
    fields={name:torch.stack([row[name]for row in rewards])for name in ('log_reference','energy','log_reward')}
    residual=log_z+log_forward-fields['log_reward'];loss=residual.square().mean()
    if not torch.isfinite(loss):
        raise ValueError('Nonfinite complete-forest trajectory balance')
    _verify_batch(teacher,record,task,batch);reward.validate(teacher)
    return dict(loss=loss,residual=residual,log_partition=log_z,log_forward=log_forward,**fields,
        backward_log_mass=0.,reward_proportionality_verified=False,training_admission=False)


def hybrid_forest_student_loss(teacher,record,task,batch,*,generator,split):
    """Conditional FM to all detached teacher forests, not top-k/reweighted edits.

    Each forest contributes one mean; root-only forests contribute zero rather
    than causing rejection or renormalization of the sampled topology law.
    """
    _train(split);_verify_batch(teacher,record,task,batch)
    if type(generator)is not torch.Generator or torch.device(generator.device)!=batch.root_states.device:
        raise ValueError('Explicit same-device student generator required')
    before=generator.get_state().clone();losses=[];count=0
    try:
        for forest in batch.forests:
            if len(forest.parents)==len(forest.clamped):
                losses.append(teacher.topology.field.readout.weight.sum()*0.)
                continue
            target=forest.states.detach()[None]
            noise=torch.randn(target.shape,dtype=target.dtype,device=target.device,generator=generator)
            time=torch.rand(1,dtype=target.dtype,device=target.device,generator=generator)
            available=torch.ones_like(target,dtype=torch.bool);available[:,list(forest.clamped)]=False
            out=forest_flow_matching_loss(teacher.topology.field,record,task,forest,target,noise,time,available)
            losses.append(out['loss']);count+=out['supervised_coordinates']
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise ValueError('Nonfinite teacher-to-student flow objective')
        _verify_batch(teacher,record,task,batch)
        return dict(loss=loss,supervised_coordinates=count,source_forests=len(losses),
                    teacher_convergence_verified=False,training_admission=False)
    except BaseException:
        generator.set_state(before)
        raise


def hybrid_forest_objective(teacher,record,task,reward,*,root_ids,root_states,count,generator,split,
                            balance_weight=1.,student_weight=1.):
    """One fully differentiable native teacher/student objective, zero updates."""
    _train(split)
    if (type(teacher)is not NativeHybridForestTeacher or type(reward)is not FrozenForestReward
            or any(type(w)not in(int,float)or not math.isfinite(w)or w<=0 for w in (balance_weight,student_weight))):
        raise ValueError('Exact hybrid teacher/reward and fixed positive objective coefficients required')
    teacher.preflight(record,task,root_ids,root_states,count);reward.validate(teacher)
    if type(generator)is not torch.Generator or torch.device(generator.device)!=root_states.device:
        raise ValueError('Explicit same-device engine-owned sampler required')
    before=generator.get_state().clone()
    try:
        batch=sample_hybrid_forests(teacher,record,task,root_ids=root_ids,root_states=root_states,count=count,generator=generator)
        balance=hybrid_forest_balance(teacher,record,task,batch,reward,split=split)
        student=hybrid_forest_student_loss(teacher,record,task,batch,generator=generator,split=split)
        loss=balance_weight*balance['loss']+student_weight*student['loss']
        if not torch.isfinite(loss):raise ValueError('Finite hybrid joint objective required')
        return dict(loss=loss,balance=balance,student=student,batch=batch,
                    optimizer_updates=0,training_admission=False)
    except BaseException:
        generator.set_state(before)
        raise


def native_hybrid_forest_objective(owner,teacher,*args,**kwargs):
    """Registered native owner entry; no source access precedes full admission."""
    from .full_owner import SourceBoundOwner
    if type(owner)is not SourceBoundOwner:
        raise ValueError('Exact native source owner required')
    owner.require_training_admission()
    if (type(teacher)is not NativeHybridForestTeacher or owner.forest_topology is not teacher.topology
            or owner.forest_flow is not teacher.topology.field
            or not any(m is teacher.policy for m in owner.modules())):
        raise ValueError('Exact factory-registered teacher heads required')
    return hybrid_forest_objective(teacher,*args,**kwargs)
