"""Exact all-pairs maximin square loss (Turaga et al., 0911.5372v1 Eq5).

Undirected connected graph; sigmoid logits are affinities. Labels >=0 are
supervised categories (including zero); -1 excludes endpoints, not transit nodes.
MST decisions run on detached CPU logits; selected edge values retain autograd.
Stable input-edge order breaks ties, a deterministic subgradient adaptation.
No sampling, graph cropping, neural default change or constrained-MALIS claim.
"""
import torch

CONVENTION='maximin-all-supervised-pairs-half-square-sigmoid-stable-edge-v1'


def maximin_objective(logits, edges, labels):
    """Mean half-square connectivity error over all supervised unordered pairs.

    Kruskal merges count newly connected same/different-label endpoint pairs.
    Every pair is counted once at its maximum-path bottleneck. Memory is linear
    in graph plus component label histograms, rather than a dense pair matrix.
    Disconnected graphs are refused rather than silently dropping pairs.
    """
    if (logits.ndim!=1 or logits.dtype not in (torch.float32,torch.float64) or not torch.isfinite(logits).all()
            or edges.shape!=(logits.numel(),2) or edges.dtype!=torch.long
            or labels.ndim!=1 or labels.dtype!=torch.long or labels.numel()<1
            or edges.device!=logits.device or labels.device!=logits.device
            or (labels < -1).any()):
        raise ValueError('Finite edge logits, int64 aligned edges and labels >=-1 required')
    n=labels.numel()
    ee=edges.detach().cpu().tolist()
    if any(i<0 or j<0 or i>=n or j>=n or i==j for i,j in ee):
        raise ValueError('Invalid graph endpoint')
    parent=list(range(n)); sizes=[1]*n
    hist=[{} if y<0 else {y:1} for y in labels.detach().cpu().tolist()]
    totals=[sum(h.values()) for h in hist]
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    order=torch.argsort(logits.detach().cpu(),descending=True,stable=True).tolist()
    chosen=[]; positives=[]; negatives=[]; merges=0
    for k in order:
        i,j=ee[k]; a,b=root(i),root(j)
        if a==b: continue
        if sizes[a]<sizes[b]: a,b=b,a
        positive=sum(count*hist[a].get(label,0) for label,count in hist[b].items())
        negative=totals[a]*totals[b]-positive
        chosen.append(k); positives.append(positive); negatives.append(negative)
        parent[b]=a; sizes[a]+=sizes[b]; totals[a]+=totals[b]; merges+=1
        for label,count in hist[b].items(): hist[a][label]=hist[a].get(label,0)+count
        hist[b]={}
    if merges!=n-1: raise ValueError('Connected graph required')
    count=sum(positives)+sum(negatives)
    if count==0: return logits.sum()*0
    values=logits[torch.tensor(chosen,device=logits.device)].sigmoid()
    pos=logits.new_tensor(positives); neg=logits.new_tensor(negatives)
    return (.5*(pos*(1-values).square()+neg*values.square())).sum()/count


def raster_maximin_objective(logits, instances):
    """Full four-neighbour image graph; background/overlap excluded as endpoints.

    Paths retain every pixel. Instance IDs are target-side categories only.
    Horizontal then vertical edge order fixes the documented tie convention.
    """
    from .instance_affinity import affinity_targets
    affinity_targets(instances)  # Reuse strict boolean instance validation.
    if logits.shape != (2, *instances.shape[1:]) or logits.device != instances.device:
        raise ValueError('Two aligned affinity logit planes required')
    _, h, w = instances.shape
    labels = torch.full((h, w), -1, dtype=torch.long, device=logits.device)
    for i, plane in enumerate(instances):
        labels[plane] = i
    labels[instances.sum(0) != 1] = -1
    ids = torch.arange(h*w, device=logits.device).reshape(h, w)
    edges = torch.cat((torch.stack((ids[:, :-1].flatten(), ids[:, 1:].flatten()), 1),
                       torch.stack((ids[:-1, :].flatten(), ids[1:, :].flatten()), 1)))
    values = torch.cat((logits[0, :, :-1].flatten(), logits[1, :-1, :].flatten()))
    return maximin_objective(values, edges, labels.flatten())
