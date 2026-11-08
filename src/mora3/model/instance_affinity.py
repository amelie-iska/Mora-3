"""Local 2D neighbor-affinity BCE adaptation, not MALIS/Rand optimization.

Channels: right, down, anchored at source pixel. Background-background is
negative; only a shared unique foreground instance is positive. Edges incident
to overlapping memberships are unsupervised, preserving source planes rather
than assigning arbitrary object precedence. Borders are unsupervised.
Targets must remain on the loss side, never in encoder conditioning.
"""
import torch
from torch.nn import functional as F

CONVENTION = 'right-down-unique-foreground-ignore-overlap-bce-v1'


def affinity_targets(instances):
    """Return boolean [2,H,W] targets and validity, independent of plane order."""
    if (instances.ndim != 3 or instances.dtype != torch.bool
            or min(instances.shape[1:]) < 1):
        raise ValueError('Boolean instance planes [N,H,W] required')
    n,h,w=instances.shape
    counts=instances.sum(0)
    # Reduction avoids an N*H*W integer workspace and preserves arbitrary IDs.
    labels=torch.zeros((h,w),dtype=torch.long,device=instances.device)
    for i in range(n):
        labels=torch.where(instances[i],i+1,labels)
    unique=counts<=1
    targets=torch.zeros((2,h,w),dtype=torch.bool,device=instances.device)
    valid=torch.zeros_like(targets)
    for channel,(a,b) in enumerate((( (slice(None),slice(None,-1)),(slice(None),slice(1,None))),
                                    ( (slice(None,-1),slice(None)),(slice(1,None),slice(None))))):
        v=unique[a]&unique[b]
        valid[channel][a]=v
        targets[channel][a]=v & (labels[a]>0) & (labels[a]==labels[b])
    return targets,valid


def affinity_objective(logits,instances):
    """Mean valid-edge Bernoulli NLL; no valid edges yields differentiable zero."""
    targets,valid=affinity_targets(instances)
    if (logits.shape != targets.shape or logits.device != instances.device
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError('Aligned finite floating affinity logits required')
    losses=F.binary_cross_entropy_with_logits(logits,targets.to(logits.dtype),reduction='none')
    return (losses*valid).sum()/valid.sum().clamp_min(1)


def balanced_affinity_objective(logits, instances):
    """Equal mean NLL over present interior, separation and background strata.

    Local boundary-weighting adaptation, not U-Net Eq2 or a topology loss.
    Separation is a valid negative edge with at least one foreground endpoint;
    background is a valid negative edge with neither endpoint foreground.
    Empty strata have no weight; entirely unsupervised images yield zero.
    Overlap and borders retain the reference ignore policy. Targets only.
    """
    targets, valid = affinity_targets(instances)
    if (logits.shape != targets.shape or logits.device != instances.device
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError('Aligned finite floating affinity logits required')
    foreground = instances.any(0)
    incident = torch.zeros_like(valid)
    incident[0, :, :-1] = foreground[:, :-1] | foreground[:, 1:]
    incident[1, :-1, :] = foreground[:-1, :] | foreground[1:, :]
    groups = torch.stack((targets & valid, valid & ~targets & incident,
                          valid & ~incident))
    counts = groups.sum((1, 2, 3))
    losses = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    means = (groups * losses).sum((1, 2, 3)) / counts.clamp_min(1)
    return means.sum() / (counts > 0).sum().clamp_min(1)


def overlapping_set_objective(logits, instances, *, pixel_block_size=None,
                              slot_policy='bounded-max64-v1'):
    """Mask-only bipartite Bernoulli adaptation, not the DETR objective.

    Minimize mean pixel BCE over one-to-one slot/target assignments, padding
    missing objects with empty masks. Overlapping target planes stay separate.
    Targets never enter a predictor. No objectness, box, Dice or focal loss is
    implied. Default capacity is 64 slots; explicit extended-max512-v2 accepts
    up to 512. Excess objects refuse rather than truncate. This loss-only policy
    does not widen source-bound tasks or admit training. A head remains separate.
    Assignment is detached; ties use one valid minimizing subgradient.
    Opt-in pixel blocking accumulates the same FP64 costs without full-image
    FP64 logit/target copies. Reduction order may alter roundoff and near-tie
    assignments. It does not block logits, final BCE/autograd or target storage,
    widen slot capacity, or establish a total-memory/speed improvement.
    """
    from scipy.optimize import linear_sum_assignment
    if slot_policy not in ('bounded-max64-v1', 'extended-max512-v2'):
        raise ValueError('unknown instance assignment slot policy')
    max_slots = 512 if slot_policy == 'extended-max512-v2' else 64
    if pixel_block_size is not None and (type(pixel_block_size) is not int
                                        or pixel_block_size < 1):
        raise ValueError('pixel block size must be a positive integer')
    if (logits.ndim != 3 or instances.ndim != 3
            or not logits.is_floating_point() or instances.dtype != torch.bool
            or logits.device != instances.device
            or logits.shape[1:] != instances.shape[1:]
            or min(logits.shape) < 1 or not 1 <= logits.shape[0] <= max_slots
            or instances.shape[0] > logits.shape[0]
            or not torch.isfinite(logits).all()
            or (instances.shape[0] and not instances.flatten(1).any(1).all())):
        raise ValueError(f'finite aligned logits, nonempty Boolean instances and 1..{max_slots} sufficient slots required')
    q = logits.shape[0]
    n = instances.shape[0]
    # Match in FP64 without constructing a Q*N*H*W pairwise workspace.
    with torch.no_grad():
        x = logits.detach().flatten(1)
        y = instances.flatten(1)
        if pixel_block_size is None:
            x = x.double()
            y = y.double()
            baseline = F.softplus(x).mean(1)
            costs = baseline[:, None].expand(q, q).clone()
            if n:
                costs[:, :n] -= x @ y.T / x.shape[1]
        else:
            # Each temporary pixel dimension is <= block size. Q*Q costs
            # remain resident; all pixels/ordered planes contribute exactly once.
            baseline = torch.zeros(q, dtype=torch.float64, device=logits.device)
            costs = torch.zeros((q, q), dtype=torch.float64, device=logits.device)
            for start in range(0, x.shape[1], pixel_block_size):
                xb = x[:, start:start + pixel_block_size].double()
                baseline += F.softplus(xb).sum(1)
                if n:
                    yb = y[:, start:start + pixel_block_size].double()
                    costs[:, :n] -= xb @ yb.T
            costs += baseline[:, None]
            costs /= x.shape[1]
        rows, columns = linear_sum_assignment(costs.cpu().numpy())
        assigned = torch.zeros_like(logits)
        for row, column in zip(rows, columns):
            if column < n:
                assigned[row] = instances[column]
    return F.binary_cross_entropy_with_logits(logits, assigned)


class OverlappingInstanceHead(torch.nn.Module):
    """Opt-in mask-slot affine head over shared native-resolution features.

    This mask-only adaptation predicts independent Bernoulli planes, without
    softmax, objectness, boxes or target inputs. It is not a DETR decoder.
    The explicit byte budget bounds the output tensor only, not total memory;
    inputs, autograd, matching and upstream encoders require separate accounting.
    Default capacity is 64 slots. Explicit extended-max512-v2 opts into up to
    512 slots with distinct serialized configuration; it does not widen the
    separate loss/task policy. No measured full-source memory/quality claim.
    No automatic attachment, checkpoint migration or training admission.
    """
    def __init__(self, channels, *, slots, max_output_bytes,
                 slot_policy='bounded-max64-v1',parameterization_policy='legacy',base_channels=None,output_mult=1.0,source_envelope=None):
        super().__init__()
        from .raster_source_envelope import RasterSourceEnvelope
        full=slot_policy in('source-envelope-v3','source-layer-list-v4')
        if (slot_policy not in ('bounded-max64-v1','extended-max512-v2','source-envelope-v3','source-layer-list-v4')
                or (full and(type(source_envelope)is not RasterSourceEnvelope or slots!=source_envelope.max_instances
                    or parameterization_policy!='mup-mask-slots-v1')) or(not full and source_envelope is not None)):
            raise ValueError('explicit complete source instance head slot policy required')
        if full:source_envelope.binding_for_slot_policy(slot_policy)
        self.source_envelope=source_envelope
        max_slots = source_envelope.max_instances if full else (512 if slot_policy=='extended-max512-v2'else 64)
        if (type(channels) is not int or channels < 1
                or type(slots) is not int or not 1 <= slots <= max_slots
                or type(max_output_bytes) is not int or max_output_bytes < 1):
            raise ValueError('positive channels/byte budget and integer slots within explicit policy required')
        import math
        if (parameterization_policy not in ('legacy','mup-mask-slots-v1')
                or (parameterization_policy=='legacy' and (base_channels is not None or output_mult!=1))
                or (parameterization_policy!='legacy' and (type(base_channels) is not int
                    or base_channels<1 or channels<base_channels or type(output_mult) not in (int,float)
                    or not math.isfinite(output_mult) or output_mult<=0))):
            raise ValueError('explicit compatible mask-slot readout parameterization required')
        self.parameterization_policy,self.base_channels,self.output_mult=parameterization_policy,base_channels,float(output_mult)
        self._mup_declaration=None
        self.slot_policy = slot_policy
        self.channels = channels
        self.slots = slots
        self.max_output_bytes = max_output_bytes
        self.projection = torch.nn.Conv2d(channels, slots, 1)
        if parameterization_policy!='legacy':
            with torch.no_grad():
                self.projection.weight.mul_(math.sqrt(channels/base_channels))
                self.projection.bias.mul_(math.sqrt(channels/base_channels))
            self._mup_declaration=self._current_declaration()

    def _current_declaration(self):
        value=(self.channels,self.slots,self.max_output_bytes,self.slot_policy,
               self.parameterization_policy,self.base_channels,self.output_mult)
        return value+((self.source_envelope.binding_for_slot_policy(self.slot_policy),)if self.slot_policy in('source-envelope-v3','source-layer-list-v4')else())

    def get_extra_state(self):
        if self._mup_declaration is not None:
            if (self._mup_declaration!=self._current_declaration()
                    or type(self.projection) is not torch.nn.Conv2d
                    or (self.projection.in_channels,self.projection.out_channels,self.projection.kernel_size,
                        self.projection.stride,self.projection.padding,self.projection.dilation,self.projection.groups,
                        self.projection.padding_mode,self.projection.transposed,self.projection.output_padding)
                    !=(self.channels,self.slots,(1,1),(1,1),(0,0),(1,1),1,'zeros',False,(0,0))):
                raise ValueError('mask-slot readout configuration drift')
            state=dict(version=3,architecture='independent-affine-mask-slots-mup-v1',channels=self.channels,
                slots=self.slots,max_output_bytes=self.max_output_bytes,max_extent=1000,slot_policy=self.slot_policy,
                max_slots=512 if self.slot_policy=='extended-max512-v2' else 64,training_admission=False,
                mup=dict(policy=self.parameterization_policy,base_channels=self.base_channels,output_mult=self.output_mult,
                    readout='native-times-sqrt-channels-over-base-input-divided-by-channels-over-base'))
            if self.slot_policy in('source-envelope-v3','source-layer-list-v4'):
                state.update(version=5 if self.slot_policy=='source-layer-list-v4'else 4,source_envelope=self.source_envelope.binding_for_slot_policy(self.slot_policy),max_slots=self.source_envelope.max_instances,
                    max_extent=max(self.source_envelope.max_height,self.source_envelope.max_width))
            return state
        if self.slot_policy == 'extended-max512-v2':
            return dict(version=2, architecture='independent-affine-mask-slots-v2',
                        channels=self.channels, slots=self.slots,
                        max_output_bytes=self.max_output_bytes, max_extent=1000,
                        slot_policy=self.slot_policy, max_slots=512,
                        training_admission=False)
        return dict(version=1, architecture='independent-affine-mask-slots-v1',
                    channels=self.channels, slots=self.slots,
                    max_output_bytes=self.max_output_bytes, max_extent=1000,
                    training_admission=False)

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError('overlapping instance head configuration mismatch')

    def forward(self, features):
        """Accept only [1,C,H,W] features; targets belong exclusively to loss."""
        weight = self.projection.weight
        if (not isinstance(features, torch.Tensor) or features.ndim != 4
                or features.shape[:2] != (1, self.channels)
                or not 1 <= min(features.shape[2:])
                or (features.shape[2]>self.source_envelope.max_height or features.shape[3]>self.source_envelope.max_width
                    if self.slot_policy in('source-envelope-v3','source-layer-list-v4')else max(features.shape[2:])>1000)
                or features.dtype != weight.dtype or features.device != weight.device
                or self.slots * features.shape[2] * features.shape[3] * features.element_size() > self.max_output_bytes
                or not torch.isfinite(features).all()):
            raise ValueError('finite aligned native features within extent/output-byte policy required')
        if self._mup_declaration is not None:
            self.get_extra_state()
            features=features*(self.output_mult/(self.channels/self.base_channels))
        return self.projection(features)[0]
