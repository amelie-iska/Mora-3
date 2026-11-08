"""Read-only full-horizon capacity gate, usable before engine construction.

No campaign schema changes, pass increases, cycle/weight changes, or reservations.
A pass only proves available source windows, not nonempty targets or training admission.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
from hashlib import sha256
from mora3.data.admet import digest,canonical
from mora3.research.primary_campaign_assembly import _reference,_read
from mora3.research.mup_campaign import FrozenPrimaryProxyCampaign
from mora3.research.mup_sweep_selection import MAX_UPDATES
from .finite_window_schedule import exact_capacities
from .primary_native_routes import PreparedNativeTargets

@dataclass(frozen=True)
class RuntimeCapacityRequest:
    source_snapshot:dict
    campaign_reference:dict
    max_identity_metadata_bytes:int
    max_replay_windows:int
    def __post_init__(self):
        _reference(self.source_snapshot);_reference(self.campaign_reference)
        if any(type(n) is not int or n<1 for n in (self.max_identity_metadata_bytes,self.max_replay_windows)):
            raise ValueError('Explicit positive identity-metadata and target replay budgets required')

class WindowCapacityRefusal(ValueError):
    def __init__(self,deficits):
        self.deficits=tuple(deficits)
        super().__init__('Full campaign task cycle exceeds exact source or usable target capacity: '+str(len(self.deficits))+' routes')

@dataclass(frozen=True)
class VerifiedWindowCapacity:
    campaign_sha256:str
    source_snapshot_sha256:str
    horizon:int
    capacities:tuple
    required_calls:tuple
    observed_replays:tuple
    @property
    def sha256(self):return digest(canonical(dict(schema='Mora3.FullHorizonWindowCapacity/v1',campaign_sha256=self.campaign_sha256,source_snapshot_sha256=self.source_snapshot_sha256,horizon=self.horizon,capacities=[asdict(c) for c in self.capacities],required_calls=self.required_calls,observed_replays=self.observed_replays,training_authorized=False,optimizer_steps_guaranteed=False)))
    @property
    def training_authorized(self):return False

def scheduled_calls(task_cycle,total_updates):
    if type(task_cycle) is not tuple or not task_cycle or any(type(k) is not str or not k for k in task_cycle) or type(total_updates) is not int or total_updates<1:raise ValueError('Exact explicit nonempty task cycle and horizon required')
    from collections import Counter
    complete,remainder=divmod(total_updates,len(task_cycle));counts=Counter({k:n*complete for k,n in Counter(task_cycle).items()});counts.update(task_cycle[:remainder])
    return tuple(sorted(counts.items()))

def replay_observed_windows(binding, *, required_updates, per_update_scan_limit, remaining_budget):
    """Follow the live sequence without evaluating targets or committing a cursor."""
    iterator = iter(binding.route.iter_from(binding.route.initial_cursor))
    trace = sha256()
    consumed = observed = 0
    reason = None
    try:
        for _ in range(required_updates):
            for _ in range(per_update_scan_limit):
                if consumed >= remaining_budget:
                    raise ValueError('Exact native target replay window budget exceeded')
                window = next(iterator,None)
                if window is None:
                    reason = 'source-exhausted'
                    break
                consumed += 1
                prepared = binding.prepare(window)
                if type(prepared) is not PreparedNativeTargets:
                    raise ValueError('Exact native prepared target disposition required')
                prepared.__post_init__()
                trace.update(canonical((window.evidence_sha256,digest(window.cursor_after),
                    prepared.observed_positions,prepared.unavailable_reason)))
                if prepared.observed_positions:
                    observed += 1
                    break
            else:
                reason = 'per-update-scan-limit'
            if reason is not None:
                break
    finally:
        iterator.close()
    return dict(observed_updates=observed,consumed_windows=consumed,trace_sha256=trace.hexdigest(),reason=reason)


def require_runtime_window_capacity(root,*,campaign,bindings,request):
    """Call after metadata route assembly, before constructing an optimizer.

    Worker integration repeats this gate before any reservation or runner call.
    Unsupported route/batch metadata proofs fail closed. The gate never creates
    files and does not accept externally asserted capacity counts.
    """
    if type(campaign) is not FrozenPrimaryProxyCampaign or type(request) is not RuntimeCapacityRequest:raise ValueError('Exact frozen campaign and separate pinned runtime capacity request required')
    campaign.__post_init__();request.__post_init__();root=Path(root).resolve()
    cr=_reference(request.campaign_reference);sr=_reference(request.source_snapshot)
    craw=_read(root,cr);sraw=_read(root,sr)
    if cr.sha256!=campaign.sha256 or craw!=campaign.wire() or sr.sha256!=campaign.source_snapshot_sha256:raise ValueError('Exact immutable campaign and source snapshot pins differ')
    if type(bindings) is not dict or set(bindings)!={p.task_key for p in campaign.train_routes}:raise ValueError('Exact planned route key set required')
    ordered=tuple(sorted(bindings.items()))
    capacities=exact_capacities(root,bindings=ordered,plans=campaign.train_routes,source_snapshot=request.source_snapshot,max_identity_metadata_bytes=request.max_identity_metadata_bytes)
    counts=scheduled_calls(campaign.task_cycle,MAX_UPDATES);needed=dict(counts)
    if any(count == 0 for _,count in counts):
        raise ValueError('Every declared native task must occur inside the planned update horizon')
    deficits=tuple(dict(task_key=c.task_key,required_updates=needed[c.task_key],
        original_windows=c.original_windows,per_update_scan_limit=c.per_update_scan_limit,
        available_windows=c.source_window_upper_bound,initial_cursor_sha256=c.initial_cursor_sha256,
        adapter_sha256=c.adapter_sha256,evidence_sha256=c.evidence_sha256)
        for c in capacities if needed[c.task_key]>c.source_window_upper_bound)
    if deficits:
        raise WindowCapacityRefusal(deficits)
    remaining=request.max_replay_windows
    replays=[]
    for c in capacities:
        count=needed[c.task_key]
        replay=replay_observed_windows(bindings[c.task_key],required_updates=count,
            per_update_scan_limit=c.per_update_scan_limit,remaining_budget=remaining)
        remaining-=replay['consumed_windows']
        row=dict(task_key=c.task_key,required_updates=count,**replay)
        replays.append(row)
        if replay['observed_updates']!=count:
            raise WindowCapacityRefusal((row,))
    if _read(root,cr)!=craw or _read(root,sr)!=sraw:raise ValueError('Capacity request artifacts changed during verification')
    return VerifiedWindowCapacity(campaign.sha256,sr.sha256,MAX_UPDATES,capacities,counts,tuple(replays))
