"""Continuous GFlowNet learning on complete ordered trajectory objects.

Explicit adaptation of continuous GFN theory (2301.12594v2): a state is the
entire prefix, not only its endpoint. Extending a prefix samples one Euclidean
increment; reversing the construction deletes the last entry deterministically.
The backward kernel is therefore a Dirac mass, NOT the native endpoint-space
learned Gaussian backward density. The original endpoint-GFN path is unchanged.

This version uses a fixed, forced terminal horizon. A positive finite trajectory
reward is a normalized Gaussian random-walk path density times exp(-E), E>=0.
This proves a finite partition <=1 (not convergence or biological validity).
A full-history-dependent penalty is now legitimate because the terminal object
IS the history. No beam-selected rollout is passed off as on-policy sampling.
The module creates no optimizer and no new learned parameters.
"""
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import re
import torch
from mora3.model.profile_gfn import ProfileGFNPolicy,conditional_trajectory_balance
from .gfn_sampling import sample_paths,score_paths
from .continuous_gfn import diagonal_gaussian_log_density
from .persistent_topology import topological_metric_loss
from .geometry_analogy import physical_distances


def _sha(x):
    if type(x)is not str or re.fullmatch('[0-9a-f]{64}',x)is None:
        raise ValueError('Exact immutable source/configuration evidence required')


def _wire(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def _tensor_hash(x):
    if not isinstance(x,torch.Tensor):raise ValueError('Native tensor required')
    value=x.detach().cpu().contiguous()
    return sha256(_wire(dict(shape=list(value.shape),dtype=str(value.dtype)))+value.view(torch.uint8).numpy().tobytes()).hexdigest()


def _input_hash(views):
    digest=sha256()
    for v in views:
        raw=v.input_bytes
        if type(raw)is not bytes:raise ValueError('Native exact input bytes required')
        digest.update(len(raw).to_bytes(8,'big'));digest.update(raw)
    return digest.hexdigest()


def _versions(policy):return tuple((id(p),p._version)for p in policy.parameters())


def _policy_hash(policy):return sha256(_wire(policy.get_extra_state())).hexdigest()


def _paths_hash(paths):
    return sha256(_wire({k:_tensor_hash(v)if isinstance(v,torch.Tensor)else v for k,v in sorted(paths.items())})).hexdigest()


def _validate_paths(paths,*,horizon,dimension,batch_size=None):
    if type(paths)is not dict or set(paths)!={'states','lengths','mask','optional_stop'}:
        raise ValueError('Exact native complete path batch required')
    x=paths['states'];lengths=paths['lengths'];mask=paths['mask']
    if (not isinstance(x,torch.Tensor)or not x.is_floating_point()or x.ndim!=3
            or x.shape[1:]!=(horizon,dimension)or not 1<=len(x)<=32 or x.requires_grad
            or not torch.isfinite(x).all()or batch_size is not None and len(x)!=batch_size):
        raise ValueError('Detached finite complete bounded trajectory states required')
    if (paths['optional_stop']is not False or not isinstance(lengths,torch.Tensor)
            or lengths.shape!=(len(x),)or lengths.dtype!=torch.long or lengths.device!=x.device
            or not torch.all(lengths==horizon)or not isinstance(mask,torch.Tensor)
            or mask.shape!=(len(x),horizon)or mask.dtype!=torch.bool or mask.device!=x.device or not mask.all()):
        raise ValueError('Every path must genuinely reach the fixed terminal horizon; no censored prefix admission')
    return x


class PathSpaceReward:
    """Frozen full-trajectory geometry guide with explicit non-experimental provenance.

    Reference coordinates must come from the SAME declared encoder/frame as the
    compared states, or an independently verified mapping recorded by alignment
    SHA. This reader does not discover that map. Reference data are train-only,
    group-disjoint from the query under the declared grouping policy.
    """
    def __init__(self,reference,*,source_sha256,encoder_sha256,alignment_sha256,
                 group_policy_sha256,guide_groups,query_groups,scale=1.,metric_weight=1.,
                 topology_weight=0.,evidence_kind='model_generated',guide_split='train'):
        for digest in(source_sha256,encoder_sha256,alignment_sha256,group_policy_sha256):_sha(digest)
        if (not isinstance(reference,torch.Tensor)or reference.ndim!=3 or not reference.is_floating_point()
                or not torch.isfinite(reference).all()or not 1<=len(reference)<=32
                or not 2<=reference.shape[1]<=64 or not 1<=reference.shape[2]<=64
                or not torch.all(reference[:,0]==0)):
            raise ValueError('Complete finite Euclidean guides including the zero root required')
        if type(scale)not in(int,float)or not math.isfinite(scale)or scale<=0:
            raise ValueError('Frozen finite positive path base-measure scale required')
        for w in(metric_weight,topology_weight):
            if type(w)not in(int,float)or not math.isfinite(w)or w<0:
                raise ValueError('Nonnegative finite full-trajectory penalty weights required')
        if guide_split!='train' or evidence_kind not in('model_generated','computed','synthetic_fixture'):
            raise ValueError('Train-only reference trajectories are not automatically observed biological outcomes')
        for groups in(guide_groups,query_groups):
            if (type(groups)is not tuple or len(groups)!=len(reference)
                    or any(type(g)is not tuple or not g or len(set(g))!=len(g)
                           or any(type(x)is not str or not x for x in g)for g in groups)):
                raise ValueError('Complete per-trajectory problem/source grouping required')
        if set(x for g in guide_groups for x in g)&set(x for g in query_groups for x in g):
            raise ValueError('Guide/query group leakage is forbidden')
        self.reference=reference.detach().clone();self.guide_groups=guide_groups;self.query_groups=query_groups
        self.scale=float(scale);self.metric_weight=float(metric_weight);self.topology_weight=float(topology_weight)
        self._metadata=dict(schema='Mora3.CompleteTrajectoryReward/v1',source_sha256=source_sha256,
            encoder_sha256=encoder_sha256,alignment_sha256=alignment_sha256,group_policy_sha256=group_policy_sha256,
            guide_groups=guide_groups,query_groups=query_groups,scale=self.scale,metric_weight=self.metric_weight,
            topology_weight=self.topology_weight,evidence_kind=evidence_kind,guide_split=guide_split,
            reference_sha256=_tensor_hash(self.reference),terminal_object='entire-ordered-reasoning-trajectory',
            base_measure='product-Lebesgue-over-full-Euclidean-prefix-increments;fixed-horizon',
            reward_partition_upper_bound=1.,learned_reasoning_improvement_proved=False,
            topology_tie_policy='all-edges-at-persistence-critical-levels',biological_causality_verified=False)
        self._binding=_wire(self._metadata)

    def binding(self):
        if (_wire(self._metadata)!=self._binding or _tensor_hash(self.reference)!=self._metadata['reference_sha256']
                or self.guide_groups!=self._metadata['guide_groups']or self.query_groups!=self._metadata['query_groups']
                or any(getattr(self,k)!=self._metadata[k]for k in('scale','metric_weight','topology_weight'))):
            raise ValueError('Frozen path reward/source/alignment configuration changed')
        return json.loads(self._binding)

    @torch.no_grad()
    def __call__(self,paths):
        self.binding();b,points,d=self.reference.shape
        x=_validate_paths(paths,horizon=points-1,dimension=d,batch_size=b)
        if x.device!=self.reference.device or x.dtype!=self.reference.dtype:
            raise ValueError('Guide/state precision and device must agree')
        delta=torch.cat((x[:,:1],x[:,1:]-x[:,:-1]),dim=1)
        q=diagonal_gaussian_log_density(delta,torch.zeros_like(delta),torch.full_like(delta,math.log(self.scale))).sum(-1)
        energies=[]
        for i in range(b):
            full=torch.cat((torch.zeros_like(x[i,:1]),x[i]),0)
            a=physical_distances(self.reference[i],scale=self.scale)
            z=physical_distances(full,scale=self.scale)
            row,col=torch.triu_indices(points,points,offset=1,device=x.device)
            metric=(a[row,col]-z[row,col]).square().mean()
            topo=(topological_metric_loss(a,z)['loss']if self.topology_weight else metric*0)
            energies.append(self.metric_weight*metric+self.topology_weight*topo)
        energy=torch.stack(energies).to(dtype=x.dtype)
        result=q-energy
        if not torch.isfinite(result).all()or not torch.all(energy>=0):
            raise ValueError('Nonfinite/nonpositive-base-measure trajectory reward refused')
        return result.detach()


@dataclass(frozen=True)
class NativePathBatch:
    paths: dict
    sampler: str
    input_sha256: str
    source_context_sha256: str
    policy_sha256: str
    paths_sha256: str
    context_sha256: str
    policy_versions: tuple
    policy_owner: tuple
    history_owner: tuple = ()
    history_sha256: str | None = None
    history_versions: tuple = ()


def sample_native_path_batch(policy,views,*,generator,source_context_sha256,history_context=None):
    """Use the original normalized native sampler, no top-k/beam rejection or clipping."""
    if type(policy)is not ProfileGFNPolicy or not isinstance(generator,torch.Generator):
        raise ValueError('Exact shared native profile GFN and explicit generator required')
    _sha(source_context_sha256);views=tuple(views)
    if not 1<=len(views)<=32 or not 1<=policy.horizon<=63 or not 1<=policy.dimension<=64:
        raise ValueError('Bounded complete path-space diagnostic/route capacity exceeded')
    if history_context is not None:
        from .trajectory_history import NativePrefixContext
        if (type(history_context)is not NativePrefixContext or history_context.policy is not policy
                or history_context.source_context_sha256!=source_context_sha256):
            raise ValueError('Exact native full-prefix source/policy conditioner required')
        history_context.binding()
    with torch.no_grad():
        context=policy.context(views)
        if history_context is None:
            paths=sample_paths(policy,len(views),generator=generator,context=context,optional_stop=False)
        else:
            gd=generator.device
            if gd.type=='cuda' and gd.index is None:gd=torch.device('cuda',torch.cuda.current_device())
            if gd!=context.device:raise ValueError('Explicit same-device isolated sampling generator required')
            x=context.new_zeros(len(views),policy.horizon,policy.dimension)
            for layer in range(policy.horizon):
                c=history_context.condition(x[:,:layer],context)
                previous=x[:,layer-1]if layer else x.new_zeros(len(x),policy.dimension)
                mean,ls,_=policy.parameters_at(previous,layer,c)
                scale=ls.exp()
                if not torch.isfinite(scale).all()or (scale<=0).any():raise ValueError('Invalid native Gaussian scale')
                x[:,layer]=mean+scale*torch.randn(mean.shape,dtype=mean.dtype,device=mean.device,generator=generator)
                if not torch.isfinite(x[:,layer]).all():raise ValueError('Nonfinite sampled prefix')
            paths=dict(states=x,lengths=torch.full((len(x),),policy.horizon,device=x.device,dtype=torch.long),
                mask=torch.ones((len(x),policy.horizon),device=x.device,dtype=torch.bool),optional_stop=False)
    _validate_paths(paths,horizon=policy.horizon,dimension=policy.dimension,batch_size=len(views))
    return NativePathBatch(paths,'native-unpruned-forced-horizon'if history_context is None
        else'native-unpruned-history-conditioned-forced-horizon',_input_hash(views),source_context_sha256,
        _policy_hash(policy),_paths_hash(paths),_tensor_hash(context),_versions(policy),(policy,),
        ()if history_context is None else(history_context,),
        None if history_context is None else sha256(_wire(history_context.binding())).hexdigest(),
        ()if history_context is None else history_context.parameter_versions())


def pathspace_balance(log_z,log_forward,log_reward):
    """TB on the prefix tree: backward deletion mass=1, forced final stop mass=1."""
    return conditional_trajectory_balance(log_z,log_forward,torch.zeros_like(log_forward),
        torch.zeros_like(log_z),log_reward,torch.ones_like(log_forward,dtype=torch.bool))


def pathspace_objective(policy,views,batch,reward,*,history_context=None):
    """Compute the complete native shared-policy loss, no optimizer or admission.

    A diagnostic loss callable is not authority to train production. The native
    owner entry below enforces the existing complete-dataset/model admission.
    Only same-snapshot on-policy data are accepted by this conservative adapter;
    a separately qualified off-policy replay path would need its own contract.
    """
    if type(policy)is not ProfileGFNPolicy or type(batch)is not NativePathBatch or type(reward)is not PathSpaceReward:
        raise ValueError('Exact native policy, sampled path receipt and immutable path reward required')
    views=tuple(views)
    expected_sampler='native-unpruned-forced-horizon'if history_context is None else'native-unpruned-history-conditioned-forced-horizon'
    if (batch.policy_owner!=(policy,)or batch.sampler!=expected_sampler
            or batch.policy_versions!=_versions(policy)or batch.policy_sha256!=_policy_hash(policy)
            or batch.input_sha256!=_input_hash(views)or batch.paths_sha256!=_paths_hash(batch.paths)):
        raise ValueError('Changed policy/input/trajectory or beam-selected sample cannot claim native on-policy status')
    _validate_paths(batch.paths,horizon=policy.horizon,dimension=policy.dimension,batch_size=len(views))
    context=policy.context(views)
    if batch.context_sha256!=_tensor_hash(context):raise ValueError('Sampling and scoring context differ')
    if history_context is None:
        if batch.history_owner or batch.history_sha256 is not None or batch.history_versions:
            raise ValueError('History-conditioned sampling cannot be rescored without its prefix adapter')
        scores=score_paths(policy,batch.paths,context=context)
        log_forward=scores['log_forward'];initial_context=context
    else:
        from .trajectory_history import NativePrefixContext
        if (type(history_context)is not NativePrefixContext or batch.history_owner!=(history_context,)
                or history_context.policy is not policy
                or history_context.source_context_sha256!=batch.source_context_sha256
                or batch.history_sha256!=sha256(_wire(history_context.binding())).hexdigest()
                or batch.history_versions!=history_context.parameter_versions()):
            raise ValueError('Full-prefix adapter/configuration/parameters changed between sample and score')
        x=batch.paths['states'];terms=[];initial_context=None
        for layer in range(policy.horizon):
            c=history_context.condition(x[:,:layer],context)
            if layer==0:initial_context=c
            previous=x[:,layer-1]if layer else x.new_zeros(len(x),policy.dimension)
            mean,ls,_=policy.parameters_at(previous,layer,c)
            terms.append(diagonal_gaussian_log_density(x[:,layer],mean,ls))
        log_forward=torch.stack(terms,1)
    log_reward=reward(batch.paths)
    result=pathspace_balance(policy.log_partition(initial_context),log_forward,log_reward)
    return dict(**result,log_reward=log_reward,observed_trajectories=len(views),
        observed_transitions=int(batch.paths['mask'].sum()),
        terminal_object='entire-ordered-reasoning-trajectory',
        backward_semantics='deterministic-prefix-deletion-Dirac',
        endpoint_backward_head_used=False,shared_native_forward_policy=True,
        forward_history_parameterization=('existing native last-latent plus biological context; not an added history encoder'
            if history_context is None else'complete-prefix-native-shared-TokenGT'),
        optimizer_updates=0,biological_causality_verified=False,training_admission=False)


def native_pathspace_objective(owner,policy_module_path,views,batch,reward,*,history_context=None,history_module_path=None):
    from mora3.model.full_owner import SourceBoundOwner
    if type(owner)is not SourceBoundOwner:raise ValueError('Exact existing full native owner required')
    owner.require_training_admission()
    if history_context is None:
        raise ValueError('Production path-space training requires full-prefix task-conditioned policy/normalizer context')
    if type(policy_module_path)is not str:raise ValueError('Registered native policy path required')
    policy=owner.get_submodule(policy_module_path)
    if type(policy)is not ProfileGFNPolicy:raise ValueError('Registered native conditional policy required')
    if batch.source_context_sha256!=owner.get_extra_state()['source_snapshot_sha256']:
        raise ValueError('Complete native source snapshot does not match the trajectory batch')
    if history_context is not None:
        if type(history_module_path)is not str or owner.get_submodule(history_module_path)is not history_context.geometry:
            raise ValueError('History encoder must be the exact already registered native geometry owner')
    elif history_module_path is not None:raise ValueError('History module path without a bound adapter')
    return pathspace_objective(policy,views,batch,reward,history_context=history_context)


def native_engine_pathspace_loss(engine, *, policy_module_path, history_module_path,
                                 history_context, views, reward, reward_binding_sha256,
                                 trial_id, task_key, cursor_before):
    """Bind an admitted full-prefix GFN objective to the native update engine RNG.

    This is an objective *preparation* boundary, not a source reader, an
    optimizer step, a trial reservation, or a release authority. The calling
    counted runner must still check its exact source route and cursor, perform
    the one native update, and durably pin checkpoint/terminal evidence.

    On-policy draws use only the generator owned and checkpointed by *engine*;
    a failed preparation restores its pre-draw state. A FoT beam-selected path
    cannot enter this boundary as a substitute for native GFN sampling.
    """
    from mora3.model.full_owner import SourceBoundOwner
    from mora3.research.primary_training import NativeMuUpdateEngine
    from .trajectory_history import NativePrefixContext

    if type(engine) is not NativeMuUpdateEngine or type(engine.model) is not SourceBoundOwner:
        raise ValueError('Exact native update engine and registered source-bound owner required')
    owner = engine.model
    # Never sample or even inspect a guide until the original, independent
    # source/construction/model gate has admitted this exact campaign and trial.
    owner.require_training_admission()
    if (type(trial_id) is not str or not trial_id or type(task_key) is not str
            or not task_key or type(cursor_before) is not bytes or not cursor_before
            or task_key != engine.next_task or engine.source_cursors.get(task_key) != cursor_before):
        raise ValueError('Exact current native source task and reader cursor required')
    owner.require_training_admission(campaign_sha256=engine.binding_sha256,trial_id=trial_id)
    if (type(policy_module_path) is not str or not policy_module_path
            or type(history_module_path) is not str or not history_module_path
            or type(history_context) is not NativePrefixContext):
        raise ValueError('Exact registered shared GFN and complete-prefix geometry paths required')
    if type(reward) is not PathSpaceReward:
        raise ValueError('Exact frozen complete-trajectory reward and guide required')
    _sha(reward_binding_sha256)
    if sha256(_wire(reward.binding())).hexdigest() != reward_binding_sha256:
        raise ValueError('Frozen trajectory reward/guide/source/alignment binding differs')
    source_snapshot = owner.get_extra_state()['source_snapshot_sha256']
    policy = owner.get_submodule(policy_module_path)
    geometry = owner.get_submodule(history_module_path)
    if (type(policy) is not ProfileGFNPolicy
            or history_context.policy is not policy or history_context.geometry is not geometry
            or history_context.source_context_sha256 != source_snapshot):
        raise ValueError('Exact source-snapshot GFN policy and shared complete-prefix owner required')
    prefix_binding=history_context.binding()
    # The independently checkpointed sampler is permitted only for the exact
    # trial/source/guide/prefix/method declared before any counted run.
    engine.require_trajectory_objective_binding(
        trial_id=trial_id,task_key=task_key,source_snapshot_sha256=source_snapshot,
        policy_module_path=policy_module_path,history_module_path=history_module_path,
        prefix_binding_sha256=sha256(_wire(prefix_binding)).hexdigest(),
        reward_binding_sha256=reward_binding_sha256,
        sampling_method='native-unpruned-history-conditioned-forced-horizon',
        terminal_object='entire-ordered-reasoning-trajectory',
        schema='Mora3.NativePathspaceBinding/v1')
    saved_sampler = engine._sampler_state()
    if saved_sampler is None:
        raise ValueError('Checkpoint-owned independent native GFN sampler required')
    if engine.sampler_generator is not engine._sampler_owner[0]:
        raise ValueError('Checkpointed GFN sampler owner changed')
    try:
        rows = tuple(views)
        batch = sample_native_path_batch(policy, rows, generator=engine.sampler_generator,
                                         source_context_sha256=source_snapshot,
                                         history_context=history_context)
        result = native_pathspace_objective(owner, policy_module_path, rows, batch,
                                           reward, history_context=history_context,
                                           history_module_path=history_module_path)
        engine._sampler_state()
        if task_key != engine.next_task or engine.source_cursors[task_key] != cursor_before:
            raise ValueError('Native source task or cursor changed during trajectory preparation')
        return batch, result
    except Exception:
        # No optimizer update has happened here, so a failed objective must not
        # advance the independently checkpointed on-policy sampling cursor.
        engine._sampler_owner[0].set_state(saved_sampler['state'])
        raise
