"""Preserve the full proxy ledger and select a transfer point at one frozen validation exposure."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from itertools import product
import re

MAX_UPDATES=5000
MAX_SECONDS=Decimal(1800)
MAX_GPU_SECONDS=Decimal(43200)
FAMILY_MACRO_POLICY='source-family-macro-v1'
SEEDS=(2026110201,2026110202)
LR=(Decimal('0.0001'),Decimal('0.0003'),Decimal('0.001'))
READOUT=(Decimal('0.5'),Decimal('1'))
ATTENTION=(Decimal('0.5'),Decimal('1'))
_FAIL={'refused','oom','diverged','interrupted','missing_metrics','insufficient_exposure'}
_ACCEPT={'completed','wall_time_censored'}
_DIGEST=re.compile('[0-9a-f]{64}')


def _digest(s):return type(s) is str and _DIGEST.fullmatch(s) is not None


def _n(x,lo=0,hi=None):return type(x) is int and x>=lo and (hi is None or x<=hi)


def _decimal(x):
    if (type(x) is not Decimal or not x.is_finite() or len(x.as_tuple().digits)>34
            or abs(x.adjusted())>300):
        raise ValueError('finite explicit decimal coordinate with bounded precision required')
    return x


@dataclass(frozen=True)
class ProxyTrial:
    trial_id: str
    hp_id: str
    learning_rate: Decimal
    readout_multiplier: Decimal
    attention_key_multiplier: Decimal
    seed: int


def planned_proxy_trials():
    """Immutable 3 × 2 × 2 grid, with the same two seeds at every point."""
    runs=[]
    for i,(lr,readout,attn) in enumerate(product(LR,READOUT,ATTENTION),1):
        hp=f'hp-{i:02d}'
        for seed in SEEDS:
            runs.append(ProxyTrial(f'{hp}-{seed}',hp,lr,readout,attn,seed))
    return tuple(runs)


@dataclass(frozen=True)
class FrozenValidationTask:
    task_key: str
    validation_source_sha256: str
    initial_reference_loss: Decimal
    minimum_normalization_scale: Decimal
    eligible_observed_targets: int

    def __post_init__(self):
        if (type(self.task_key) is not str or not self.task_key or not _digest(self.validation_source_sha256)
                or _decimal(self.minimum_normalization_scale)<=0 or not _n(self.eligible_observed_targets,1)):
            raise ValueError('complete validation task and strictly positive scale floor required')
        _decimal(self.initial_reference_loss)


@dataclass(frozen=True)
class FrozenProxySelection:
    campaign_manifest_sha256: str
    validation_exposure_sha256: str
    comparison_update: int
    tasks: tuple[FrozenValidationTask,...]
    task_families: tuple[tuple[str,str],...] = ()

    def __post_init__(self):
        if (not _digest(self.campaign_manifest_sha256) or not _digest(self.validation_exposure_sha256)
                or not _n(self.comparison_update,1,MAX_UPDATES) or type(self.tasks) is not tuple or not self.tasks
                or any(type(t) is not FrozenValidationTask for t in self.tasks)):
            raise ValueError('complete exact predeclared campaign and validation exposure required')
        keys=tuple(t.task_key for t in self.tasks)
        if keys!=tuple(sorted(set(keys))):raise ValueError('unique ordered validation task axis required')
        for t in self.tasks:t.__post_init__()
        if (
            type(self.task_families) is not tuple
            or any(type(row) is not tuple or len(row)!=2
                   or any(type(value) is not str or not value.strip() for value in row)
                   for row in self.task_families)
            or self.task_families and tuple(key for key,_ in self.task_families)!=keys
        ):
            raise ValueError('exact ordered validation task family membership required')


@dataclass(frozen=True)
class ValidationCoordinate:
    task_key: str
    source_sha256: str
    split: str
    observed_targets: int
    mean_loss: Decimal


@dataclass(frozen=True)
class ProxyRunOutcome:
    trial_id: str
    campaign_manifest_sha256: str
    run_artifact_sha256: str
    status: str
    completed_updates: int
    gpu_seconds: Decimal
    validation_exposure_sha256: str | None=None
    comparison_update: int | None=None
    validation: tuple[ValidationCoordinate,...]=()
    failure_reason: str | None=None


def _eligible_trial(contract,row):
    if not (
        type(row) is ProxyRunOutcome and row.campaign_manifest_sha256==contract.campaign_manifest_sha256
        and _digest(row.run_artifact_sha256) and _n(row.completed_updates,0,MAX_UPDATES)
        and 0<=_decimal(row.gpu_seconds)<=MAX_SECONDS
        and type(row.validation) is tuple and type(row.status) is str and row.status in _FAIL|_ACCEPT
    ):
        raise ValueError('invalid run identity, coverage, budget or terminal disposition')
    if row.status in _FAIL:
        if type(row.failure_reason) is not str or not row.failure_reason.strip():
            raise ValueError('failure/censoring reason must remain explicit')
        # Failed rows may retain forensic validation metrics in their raw artifact.
        # They must not carry any eligible comparison packet into the selector.
        if row.validation or row.comparison_update is not None or row.validation_exposure_sha256 is not None:
            raise ValueError('failed run cannot provide a selection packet')
        return None
    if row.failure_reason is not None:
        raise ValueError('eligible terminal comparison must not have a failure reason')
    if row.status=='completed' and row.completed_updates!=MAX_UPDATES:
        raise ValueError('completed planned run did not reach its terminal update')
    if row.status=='wall_time_censored' and row.completed_updates==MAX_UPDATES:
        raise ValueError('update-complete run cannot also be marked truncated')
    if (row.validation_exposure_sha256!=contract.validation_exposure_sha256
            or type(row.comparison_update) is not int or row.comparison_update!=contract.comparison_update
            or row.completed_updates<row.comparison_update or len(row.validation)!=len(contract.tasks)):
        raise ValueError('predeclared matched comparison exposure must be complete')
    scores=[]
    with localcontext() as ctx:
        ctx.prec=50
        for metric,task in zip(row.validation,contract.tasks,strict=True):
            if (type(metric) is not ValidationCoordinate or metric.task_key!=task.task_key
                    or metric.source_sha256!=task.validation_source_sha256 or metric.split!='validation'
                    or not _n(metric.observed_targets,1) or metric.observed_targets!=task.eligible_observed_targets):
                raise ValueError('test, wrong-source, duplicate, missing or incomplete validation cannot select transfer')
            ref=task.initial_reference_loss
            scale=max(abs(ref),task.minimum_normalization_scale)
            scores.append((_decimal(metric.mean_loss)-ref)/scale)
        if not contract.task_families:
            return sum(scores,Decimal(0))/len(scores)
        grouped={}
        for (_,family),score in zip(contract.task_families,scores,strict=True):
            grouped.setdefault(family,[]).append(score)
        return sum((sum(group,Decimal(0))/len(group) for group in grouped.values()),Decimal(0))/len(grouped)


def select_proxy_transfer(contract,outcomes):
    """Compute a seed-mean macro normalized validation score; never launch a run."""
    if type(contract) is not FrozenProxySelection:raise ValueError('exact frozen comparison contract required')
    contract.__post_init__()
    grid=planned_proxy_trials();outcomes=tuple(outcomes)
    if len(outcomes)!=len(grid) or any(type(r) is not ProxyRunOutcome or type(r.trial_id) is not str for r in outcomes):
        raise ValueError('all 24 explicit terminal trial outcomes are required')
    indexed={r.trial_id:r for r in outcomes}
    if set(indexed)!={t.trial_id for t in grid}:
        raise ValueError('no duplicate, unknown or omitted trial identities')
    if len({r.run_artifact_sha256 for r in outcomes if _digest(r.run_artifact_sha256)})!=len(outcomes):
        raise ValueError('one immutable distinct terminal artifact is required per trial')
    scored={t.trial_id:_eligible_trial(contract,indexed[t.trial_id]) for t in grid}
    ranked=[];ineligible=[]
    with localcontext() as ctx:
        ctx.prec=50
        if sum((r.gpu_seconds for r in outcomes),Decimal(0))>MAX_GPU_SECONDS:
            raise ValueError('aggregate frozen GPU budget exceeded')
        for a,b in zip(grid[::2],grid[1::2],strict=True):
            x,y=scored[a.trial_id],scored[b.trial_id]
            if x is None or y is None:
                ineligible.append(a.hp_id);continue
            spent=indexed[a.trial_id].gpu_seconds+indexed[b.trial_id].gpu_seconds
            ranked.append(((x+y)/2,spent,a.hp_id,a))
    ranked.sort(key=lambda row:row[:3])
    winner=ranked[0][3] if ranked else None
    result=dict(campaign_manifest_sha256=contract.campaign_manifest_sha256,comparison_update=contract.comparison_update,
        full_terminal_trial_count=len(grid),eligible_hyperparameter_points=len(ranked),
        retained_terminal_dispositions=tuple((t.trial_id,indexed[t.trial_id].status,indexed[t.trial_id].run_artifact_sha256) for t in grid),
        selection_order=tuple((key,str(score),str(spent)) for score,spent,key,_ in ranked),
        ineligible_hyperparameter_points=tuple(ineligible),
        frozen_transfer_hyperparameters=None if winner is None else dict(hp_id=winner.hp_id,
            learning_rate=str(winner.learning_rate),readout_multiplier=str(winner.readout_multiplier),
            attention_key_multiplier=str(winner.attention_key_multiplier)),
        target_validation_or_test_used=False,launch_authorized=False)
    if contract.task_families:
        result['validation_aggregation']=dict(policy=FAMILY_MACRO_POLICY,task_families=contract.task_families)
    return result
