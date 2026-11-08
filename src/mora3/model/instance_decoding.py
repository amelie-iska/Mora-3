"""Explicit CPU discrete affinity components and maximum-cardinality IoU matching.

Local evaluation protocol, not watershed/MALIS or a benchmark reproduction.
No area filter. Thresholds fixed by caller before evaluation. Split/merge counts
use ANY positive intersection, so tiny overlaps count; overlapping truth planes
are preserved and evaluated independently. Matching maximizes count, not total IoU.
"""
import math
import numpy as np
import torch


def decode_instances(logits, *, foreground_threshold=.5, affinity_threshold=.5):
    """Return row-major connected-component IDs [H,W] on CPU; 0 is background."""
    if (logits.ndim!=3 or logits.shape[0]!=3 or min(logits.shape[1:])<1
            or not logits.is_floating_point() or not torch.isfinite(logits).all()
            or not 0<foreground_threshold<1 or not 0<affinity_threshold<1):
        raise ValueError('Finite foreground/right/down logits and thresholds in (0,1) required')
    z=logits.detach().cpu()
    fg=z[0]>=math.log(foreground_threshold/(1-foreground_threshold))
    edges=z[1:]>=math.log(affinity_threshold/(1-affinity_threshold))
    h,w=fg.shape
    labels=torch.zeros((h,w),dtype=torch.long)
    count=0
    for r,c in fg.nonzero().tolist():
        if labels[r,c]: continue
        count+=1; labels[r,c]=count; stack=[(r,c)]
        while stack:
            y,x=stack.pop()
            for yy,xx,channel,ay,ax in ((y,x+1,0,y,x),(y,x-1,0,y,x-1),
                                       (y+1,x,1,y,x),(y-1,x,1,y-1,x)):
                if (0<=yy<h and 0<=xx<w and fg[yy,xx] and not labels[yy,xx]
                        and edges[channel,ay,ax]):
                    labels[yy,xx]=count; stack.append((yy,xx))
    return labels


def match_instances(predicted, truth, *, iou_threshold=.5, backend="contingency"):
    """One-to-one maximum-cardinality matching at IoU >= threshold.

    Iterative augmenting paths avoid recursion limits; empty truth planes rejected.
    The default exact contingency backend counts each truth plane separately;
    backend="dense" retains the pixelwise oracle for verification.
    Returned IoUs describe the deterministic matching, not max-sum-IoU assignment.
    """
    if (predicted.ndim!=2 or predicted.dtype!=torch.long or (predicted<0).any()
            or truth.ndim!=3 or truth.dtype!=torch.bool or truth.shape[1:]!=predicted.shape
            or not 0<iou_threshold<=1):
        raise ValueError('Nonnegative int64 labels, aligned boolean truth and threshold in (0,1] required')
    if backend not in ("dense", "contingency"):
        raise ValueError("Unknown object metric backend")
    p=predicted.detach().cpu(); t=truth.detach().cpu()
    ids=torch.unique(p); ids=ids[ids>0]
    areas=t.sum((1,2))
    if (areas==0).any(): raise ValueError('Empty truth instance plane')
    ious=torch.zeros((len(ids),len(t)),dtype=torch.float64)
    overlaps=torch.zeros_like(ious,dtype=torch.bool)
    if backend == "dense":
        # Retained pixelwise reference, including its FP32 division rounding.
        for i,pid in enumerate(ids):
            mask=p==pid
            intersection=(t & mask).sum((1,2))
            overlaps[i]=intersection>0
            ious[i]=intersection/(areas+mask.sum()-intersection)
    else:
        # Compact IDs avoid allocation proportional to arbitrary source label IDs.
        # Each truth plane is independent: overlaps are never flattened/relabelled.
        labels,inverse,counts=np.unique(p.numpy(),return_inverse=True,return_counts=True)
        positive=labels>0
        inverse=inverse.reshape(-1)
        intersections=np.zeros((int(positive.sum()),len(t)),dtype=np.int64)
        for j,plane in enumerate(t.numpy()):
            intersections[:,j]=np.bincount(inverse[plane.reshape(-1)],
                                           minlength=len(labels))[positive]
        intersection=torch.from_numpy(intersections)
        pred_areas=torch.from_numpy(counts[positive])
        overlaps=intersection>0
        # Use the oracle's integer true-division dtype before FP64 storage.
        ious[:]=intersection/(areas[None,:]+pred_areas[:,None]-intersection)
    adjacency=[torch.where(row>=iou_threshold)[0].tolist() for row in ious]
    owner={}; matched={}
    for root in range(len(ids)):
        queue=[root]; seen_p={root}; parent={}; terminal=None
        for node in queue:
            for target in adjacency[node]:
                if target in parent: continue
                parent[target]=node
                if target not in owner:
                    terminal=target; break
                other=owner[target]
                if other not in seen_p: seen_p.add(other); queue.append(other)
            if terminal is not None: break
        while terminal is not None:
            node=parent[terminal]; previous=matched.get(node)
            owner[terminal]=node; matched[node]=terminal; terminal=previous
    tp=len(matched)
    return dict(tp=tp,fp=len(ids)-tp,fn=len(t)-tp,
                matched_iou=[float(ious[i,j]) for i,j in sorted(matched.items())],
                split_objects=int((overlaps.sum(0)>1).sum()),
                merge_objects=int((overlaps.sum(1)>1).sum()),
                predicted_objects=len(ids),target_objects=len(t),iou_threshold=iou_threshold)
