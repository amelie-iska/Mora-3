"""All-slot/all-pixel affine Bernoulli set loss without dense full-slot logits.

Matching uses complete FP64 costs and detached exact Hungarian assignment. All
positive target incidences are retained, including overlaps and repeated planes.
The second pass checkpoints each affine BCE block with full feature/head/offset
gradients. CNN features, source masks and upstream activations are separately
owned resource costs; the explicit workspace bound does not claim total memory.
"""
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from scipy.optimize import linear_sum_assignment
from .instance_affinity import OverlappingInstanceHead


def streamed_affine_mask_set_objective(head,features,instances,*,offset=None,slot_block_size=32,
        pixel_block_size=8192,positive_incidence_block_size=65536,recompute=True,max_workspace_bytes=1024**3):
    if (type(head)is not OverlappingInstanceHead or head.slot_policy!='source-envelope-v3'
            or any(type(n)is not int or n<1 for n in(slot_block_size,pixel_block_size,positive_incidence_block_size,max_workspace_bytes))
            or type(recompute)is not bool):raise ValueError('explicit complete source head/stream workspace policy required')
    head.get_extra_state();env=head.source_envelope;w=head.projection.weight
    if (not isinstance(features,torch.Tensor)or features.ndim!=4 or features.shape[:2]!=(1,head.channels)
            or not 1<=features.shape[2]<=env.max_height or not 1<=features.shape[3]<=env.max_width
            or features.dtype!=w.dtype or features.device!=w.device or not torch.isfinite(features).all()
            or not isinstance(instances,torch.Tensor)or instances.ndim!=3 or instances.dtype!=torch.bool
            or instances.shape[1:]!=features.shape[2:]or instances.shape[0]>head.slots):
        raise ValueError('complete finite native features and every literal ordered Boolean source layer required')
    q=head.slots;n=len(instances);p=features.shape[2]*features.shape[3]
    qb=min(q,slot_block_size);pb=min(p,pixel_block_size);ib=min(max(n,1)*pb,positive_incidence_block_size)
    # Conservative explicit bound for detached cost copies, positive indices,
    # gather/scatter and affine/BCE blocks; original features/masks excluded.
    workspace=16*q*q+8*q+32*n*pb+8*qb*ib+40*qb*pb+16*head.channels*pb
    if workspace>max_workspace_bytes:raise ValueError('complete stream workspace bound exceeded; no crop')
    if offset is None:offset=w.new_zeros(q)
    if (not isinstance(offset,torch.Tensor)or offset.shape!=(q,)or offset.device!=w.device
            or offset.dtype!=w.dtype or not torch.isfinite(offset).all()):raise ValueError('matching finite source task offset required')
    factor=head.output_mult/(head.channels/head.base_channels)
    x=features[0].reshape(head.channels,p).T;target=instances.reshape(n,p)
    weight=w[:,:,0,0];bias=head.projection.bias
    with torch.no_grad():
        costs=torch.zeros(q,q,device=w.device,dtype=torch.float64);baseline=costs.new_zeros(q)
        for start in range(0,p,pixel_block_size):
            stop=min(p,start+pixel_block_size)
            # Sparse positive incidence retains every overlapping native plane.
            nz=target[:,start:stop].nonzero(as_tuple=False)
            for left in range(0,q,slot_block_size):
                right=min(q,left+slot_block_size)
                logits=F.linear(x[start:stop]*factor,weight[left:right],bias[left:right]).T+offset[left:right,None]
                xb=logits.double();baseline[left:right]+=F.softplus(xb).sum(1)
                for k in range(0,len(nz),positive_incidence_block_size):
                    part=nz[k:k+positive_incidence_block_size].to(w.device)
                    costs[left:right,:n].scatter_add_(1,part[:,0][None].expand(right-left,-1),-xb[:,part[:,1]])
        costs=(costs+baseline[:,None])/p
        if not torch.isfinite(costs).all():raise ValueError('finite complete source assignment costs required')
        rows,columns=linear_sum_assignment(costs.cpu().numpy())
        if tuple(rows)!=tuple(range(q)):raise ValueError('complete source slot assignment required')
    def segment(xb,ww,bb,oo,source_masks,*,columns,pstart,pstop):
        # Recreate only this target block during backward. The checkpoint holds
        # references to the one original source mask tensor, never a full set of
        # newly allocated assigned GPU targets or logits.
        truth=torch.zeros(len(columns),pstop-pstart,dtype=torch.bool,device=w.device)
        valid=[i for i,c in enumerate(columns)if c<n]
        if valid:
            positions=torch.tensor(valid,device=w.device)
            indices=torch.tensor([columns[i]for i in valid],device=source_masks.device)
            truth[positions]=source_masks[indices,pstart:pstop].to(w.device)
        logits=F.linear(xb*factor,ww,bb).T+oo[:,None]
        return F.binary_cross_entropy_with_logits(logits.double(),truth.double(),reduction='sum')
    total=x.sum()*0
    for left in range(0,q,slot_block_size):
        right=min(q,left+slot_block_size);col=tuple(int(c)for c in columns[left:right])
        for start in range(0,p,pixel_block_size):
            stop=min(p,start+pixel_block_size)
            args=(x[start:stop],weight[left:right],bias[left:right],offset[left:right],target)
            kwargs=dict(columns=col,pstart=start,pstop=stop)
            value=(checkpoint(segment,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
                if recompute and torch.is_grad_enabled()else segment(*args,**kwargs))
            total=total+value
    return total/(q*p)


def streamed_ordered_affine_mask_list_objective(head,features,instances,*,offset=None,slot_block_size=32,
        pixel_block_size=8192,recompute=True,max_workspace_bytes=1024**3):
    """Actual K supplied source planes only; no matching/padding-label inference."""
    if (type(head)is not OverlappingInstanceHead or head.slot_policy!='source-layer-list-v4'
            or any(type(v)is not int or v<1 for v in(slot_block_size,pixel_block_size,max_workspace_bytes))
            or type(recompute)is not bool):raise ValueError('explicit ordered complete-source stream policy required')
    head.get_extra_state();env=head.source_envelope;w=head.projection.weight
    if (not isinstance(features,torch.Tensor)or features.ndim!=4 or features.shape[:2]!=(1,head.channels)
            or not 1<=features.shape[2]<=env.max_height or not 1<=features.shape[3]<=env.max_width
            or features.dtype!=w.dtype or features.device!=w.device or not torch.isfinite(features).all()
            or not isinstance(instances,torch.Tensor)or instances.ndim!=3 or instances.dtype!=torch.bool
            or instances.shape[1:]!=features.shape[2:]or len(instances)>head.slots):
        raise ValueError('complete source-supplied ordered Boolean layer list required')
    n=len(instances);p=features.shape[2]*features.shape[3];qb=min(head.slots,slot_block_size);pb=min(p,pixel_block_size)
    if 48*qb*pb+16*head.channels*pb>max_workspace_bytes:raise ValueError('complete source ordered stream workspace exceeded')
    if offset is None:offset=w.new_zeros(head.slots)
    if offset.shape!=(head.slots,)or offset.dtype!=w.dtype or offset.device!=w.device or not torch.isfinite(offset).all():
        raise ValueError('complete ordered-source task offsets required')
    if n==0:return features.sum()*0+w.sum()*0+head.projection.bias.sum()*0+offset.sum()*0
    x=features[0].reshape(head.channels,p).T;target=instances.reshape(n,p)
    weight=w[:,:,0,0];bias=head.projection.bias;factor=head.output_mult/(head.channels/head.base_channels)
    def segment(xb,ww,bb,oo,source_masks,*,left,right,start,stop):
        truth=source_masks[left:right,start:stop].to(w.device)
        logits=F.linear(xb*factor,ww,bb).T+oo[:,None]
        return F.binary_cross_entropy_with_logits(logits.double(),truth.double(),reduction='sum')
    total=x.sum()*0
    for left in range(0,n,slot_block_size):
        right=min(n,left+slot_block_size)
        for start in range(0,p,pixel_block_size):
            stop=min(p,start+pixel_block_size);args=(x[start:stop],weight[left:right],bias[left:right],offset[left:right],target)
            opts=dict(left=left,right=right,start=start,stop=stop)
            loss=(checkpoint(segment,*args,use_reentrant=False,preserve_rng_state=True,**opts)
                if recompute and torch.is_grad_enabled()else segment(*args,**opts))
            total=total+loss
    return total/(n*p)
