"""Bounded continuous inference adaptation of 2412.09078v1 §3.1/Eq3.

Per-tree layer-wise top-m, with caller-owned proposals, validity and terminal
semantics. This is not self-correction, consensus or an on-policy GFN kernel.
"""
import math
import torch
from .fot_selection import select_branches


@torch.no_grad()
def grow_forest(roots, expand, *, branches, beam, max_depth, max_candidates,
                min_distance=0., history_expand=None):
    """Grow independent trees from floating [trees,dimension] roots.

    expand(parent_state, arrival_depth, count) returns states[count,D],
    scores[count], valid[count], terminal[count] on the root dtype/device.
    Scores are absolute per-layer priorities, not accumulated log densities.
    At most max_candidates proposals per tree, roots excluded. Final partial
    layers use parent order, then child order; this is an explicit budget bias.
    All proposals are retained with parent IDs and selected flags for audit.
    Invalid outputs are explicit booleans; numerical corruption raises.
    Terminal selected leaves accumulate; only nonterminal selections expand.
    reasons describes why growth ended, even if earlier terminal leaves exist.
    States are cloned/detached, and neither inputs nor callback results mutate.
    Optional history_expand replaces expand with (cloned_ancestry, depth, count).
    It receives the full root-to-parent chain with stable node_id and parent IDs;
    mutating a callback copy cannot change the retained forest. Default behavior
    and its probability-free inference semantics are unchanged.
    Storage O(trees*max_candidates*dimension), excluding callback workspace.
    """
    if any(type(v) is not int or v <= 0 for v in (branches, beam, max_depth, max_candidates)):
        raise ValueError('Require positive integer budgets/depth')
    if (isinstance(min_distance,bool) or not isinstance(min_distance,(int,float))
        or not math.isfinite(min_distance) or min_distance < 0):
        raise ValueError('Require finite nonnegative distance')
    if (roots.ndim != 2 or min(roots.shape) <= 0 or not roots.is_floating_point()
        or not torch.isfinite(roots).all()):
        raise ValueError('Require finite nonempty floating roots [T,D]')
    if history_expand is not None and not callable(history_expand):
        raise ValueError('Explicit callable history-aware proposal adapter required')
    nodes=[]; leaves=[]; reasons=[]; counts=[]
    for tree, root in enumerate(roots):
        root_id=len(nodes)
        nodes.append(dict(tree=tree,parent=None,depth=0,state=root.detach().clone(),
                          score=None,valid=True,terminal=False,selected=True))
        active=[root_id]; done=[]; used=0
        for depth in range(1,max_depth+1):
            layer=[]
            for parent in active:
                count=min(branches,max_candidates-used)
                if count==0:break
                if history_expand is None:
                    x,s,v,t=expand(nodes[parent]['state'].clone(),depth,count)
                else:
                    # Provide complete cloned lineage, not an endpoint-keyed cache.
                    ancestry=[]; current=parent
                    while current is not None:
                        node=nodes[current]
                        ancestry.append(dict(node, node_id=current, state=node['state'].clone()))
                        current=node['parent']
                    x,s,v,t=history_expand(tuple(reversed(ancestry)),depth,count)
                if (x.shape!=(count,roots.shape[1]) or x.dtype!=roots.dtype or x.device!=roots.device
                    or s.shape!=(count,) or s.dtype!=roots.dtype or s.device!=roots.device
                    or v.shape!=(count,) or v.dtype!=torch.bool or v.device!=roots.device
                    or t.shape!=(count,) or t.dtype!=torch.bool or t.device!=roots.device
                    or not torch.isfinite(x).all() or not torch.isfinite(s).all()):
                    raise ValueError('Invalid expansion tensors')
                for j in range(count):
                    layer.append(len(nodes))
                    nodes.append(dict(tree=tree,parent=parent,depth=depth,
                        state=x[j].detach().clone(),score=s[j].item(),valid=v[j].item(),
                        terminal=t[j].item(),selected=False))
                used+=count
            x=torch.stack([nodes[i]['state'] for i in layer])
            s=roots.new_tensor([nodes[i]['score'] for i in layer])
            valid=torch.tensor([nodes[i]['valid'] for i in layer],dtype=torch.bool,device=roots.device)
            chosen=select_branches(x,s,beam,min_distance=min_distance,eligible=valid).tolist()
            selected=[layer[j] for j in chosen]
            active=[]
            for i in selected:
                nodes[i]['selected']=True
                (done if nodes[i]['terminal'] else active).append(i)
            if not selected:
                reason='invalid';break
            if not active:
                reason='terminal';break
            if depth==max_depth:
                reason='depth';done.extend(active);break
            if used==max_candidates:
                reason='budget';done.extend(active);break
        leaves.append(done);reasons.append(reason);counts.append(used)
    return dict(nodes=nodes,leaves=leaves,reasons=reasons,counts=counts)
