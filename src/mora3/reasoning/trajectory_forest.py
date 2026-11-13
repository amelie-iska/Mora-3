"""Whole-history structural control on the actual native FoT forest.

Expansion/verifier receive complete root-to-candidate paths. Equal latent vectors
with different ancestries remain distinct states. Computational parenthood does
not establish biological causation. Pruned beam search is never labeled an
on-policy GFlowNet rollout; only genuinely terminal leaves are complete paths.
"""
from dataclasses import asdict,dataclass
from hashlib import sha256
import json
import re
import torch
from .fot_forest import grow_forest
from .analogy_control import state_fingerprint


def _sha(x):
    if type(x)is not str or re.fullmatch('[0-9a-f]{64}',x)is None:
        raise ValueError('Immutable path/context/evidence digest required')


def history_fingerprint(history):
    if type(history)is not tuple or not 1<=len(history)<=65:
        raise ValueError('Bounded complete immutable root-to-state history required')
    serialized=[];previous=None;tree=None
    for i,node in enumerate(history):
        if type(node)is not dict or not {'node_id','parent','depth','tree','state','terminal'}<=set(node):
            raise ValueError('Complete native ancestry node required')
        for k in('node_id','depth','tree'):
            if type(node[k])is not int or node[k]<0:raise ValueError('Exact ancestry indices required')
        if node['depth']!=i or node['parent']!=previous or type(node['terminal'])is not bool:
            raise ValueError('History is not a contiguous root-to-state reasoning path')
        if tree is None:tree=node['tree']
        if node['tree']!=tree:raise ValueError('A path cannot cross independent forest roots')
        serialized.append(dict(node_id=node['node_id'],parent=previous,depth=i,tree=tree,
            terminal=node['terminal'],state_sha256=state_fingerprint(node['state'])))
        previous=node['node_id']
    return sha256(json.dumps(serialized,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _clone(history):
    return tuple(dict(n,state=n['state'].detach().clone())for n in history)


@dataclass(frozen=True)
class TrajectoryDecision:
    status: str
    history_sha256: str
    context_sha256: str
    evidence_sha256: str

    def __post_init__(self):
        if self.status not in('admissible','refused','unknown'):
            raise ValueError('Explicit acceptance/refusal/abstention required')
        for value in(self.history_sha256,self.context_sha256,self.evidence_sha256):_sha(value)


def grow_trajectory_forest(roots,expand,certify,*,context_sha256,branches,beam,max_depth,max_candidates,min_distance=0.):
    """Preserve whole-history structural checks, candidate scores and all failures.

    The caller supplies a domain-specific verifier. It must not infer biological
    causality merely from neural scores or directions. No confidence calibration
    or reward density is fabricated from a structural acceptance decision.
    """
    _sha(context_sha256)
    if not callable(expand)or not callable(certify):raise ValueError('Explicit history proposal and verifier required')
    if max_depth>64:raise ValueError('Whole-trajectory proof depth budget exceeded')
    decisions=[];next_node_id=0
    def history_expand(history,depth,count):
        nonlocal next_node_id
        history_fingerprint(history)
        next_node_id=max(next_node_id,history[-1]['node_id']+1)
        states,scores,valid,terminal=expand(_clone(history),depth,count)
        if (not isinstance(states,torch.Tensor)or states.shape!=(count,roots.shape[1])
                or not isinstance(scores,torch.Tensor)or scores.shape!=(count,)
                or not isinstance(valid,torch.Tensor)or valid.shape!=(count,)or valid.dtype!=torch.bool
                or not isinstance(terminal,torch.Tensor)or terminal.shape!=(count,)or terminal.dtype!=torch.bool):
            raise ValueError('Complete native FoT proposal arrays required')
        mask=valid.clone()
        for i in range(count):
            child=dict(node_id=next_node_id+i,parent=history[-1]['node_id'],depth=depth,
                tree=history[-1]['tree'],state=states[i].detach().clone(),terminal=bool(terminal[i]))
            path=history+(child,);digest=history_fingerprint(path)
            verdict=certify(_clone(path))
            if type(verdict)is not TrajectoryDecision:raise ValueError('Exact whole-trajectory verdict required')
            verdict.__post_init__()
            if verdict.history_sha256!=digest or verdict.context_sha256!=context_sha256:
                raise ValueError('Certificate does not bind the complete current ancestry and context')
            mask[i]&=verdict.status=='admissible'
            decisions.append(dict(node_id=child['node_id'],depth=depth,**asdict(verdict)))
        next_node_id+=count
        return states,scores,mask,terminal
    def forbidden_endpoint_callback(*_):
        raise ValueError('Whole-history adapter unexpectedly reduced to endpoint callback')
    forest=grow_forest(roots,forbidden_endpoint_callback,history_expand=history_expand,
        branches=branches,beam=beam,max_depth=max_depth,max_candidates=max_candidates,min_distance=min_distance)
    complete=[];censored=[]
    for leaves in forest['leaves']:
        yes=[];no=[]
        for leaf in leaves:
            path=[];n=leaf
            while n is not None:path.append(n);n=forest['nodes'][n]['parent']
            path=tuple(reversed(path))
            (yes if forest['nodes'][leaf]['terminal']else no).append(path)
        complete.append(tuple(yes));censored.append(tuple(no))
    return dict(**forest,complete_paths=tuple(complete),censored_paths=tuple(censored),
        trajectory_certificates=tuple(decisions),context_sha256=context_sha256,
        gfn_on_policy=False,biological_causality_verified=False,training_admission=False)
