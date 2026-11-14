"""Isolated prefix-only forest context adapter for neural token continuation.

This adapter does not qualify source provenance, frozen encoder exports, muP
roles or training admission. The caller must supply separately bound prior
ancestor states; the full candidate/target forest states are never consulted.
No descendants, sibling topology, unrevealed node states or test labels are
introduced into the native graph computation.
"""
from dataclasses import dataclass
import re
import torch

from mora3.reasoning.flow_forest_refinement import ForestState
from .forest_token_decoder import NativeForestTokenDecoder, _tensor_sha


def _pin(value):
    if type(value) is not str or re.fullmatch(r'[0-9a-f]{64}',value) is None:
        raise ValueError('Exact lowercase SHA256 source/evidence pin required')


@dataclass(frozen=True)
class KnownForestCausalPrefix:
    """Unadmitted externally supplied prior source-identified ancestor states."""
    source_snapshot_sha256: str
    context_sha256: str
    coordinate_sha256: str
    evidence_sha256: str
    values_sha256: str
    node_ids: tuple[str,...]
    states: torch.Tensor

    def __post_init__(self):
        for pin in (self.source_snapshot_sha256,self.context_sha256,
                    self.coordinate_sha256,self.evidence_sha256,self.values_sha256):
            _pin(pin)
        ids=self.node_ids
        values=self.states
        if (type(ids) is not tuple or not 1<=len(ids)<=511
                or len(set(ids))!=len(ids)
                or any(type(x)is not str or not x or len(x)>4096 for x in ids)
                or type(values)is not torch.Tensor or values.ndim!=2
                or values.shape[0]!=len(ids) or not 1<=values.shape[1]<=256
                or values.dtype not in (torch.float32,torch.float64)
                or values.is_meta or values.requires_grad
                or not torch.isfinite(values).all()):
            raise ValueError('Complete finite frozen explicitly identified prior ancestor states required')
        if _tensor_sha(values)!=self.values_sha256:
            raise ValueError('Known causal prefix tensor hash mismatch')


def causal_node_view(decoder, template, node_index, known):
    """Construct a minimal exact ancestor chain with a zero unknown target.

    The template is only used to establish *the selected node's ancestry*.
    Siblings, descendants, all full-forest states and future dispositions do
    not enter the returned tensor or skeleton. This is a conditional-path
    mechanism, not proof that the chosen path was available at inference time.
    """
    if (type(decoder)is not NativeForestTokenDecoder
            or type(template)is not ForestState
            or type(known)is not KnownForestCausalPrefix):
        raise ValueError('Exact native decoder, forest and prior prefix required')
    decoder.get_extra_state();template.__post_init__();known.__post_init__()
    if (type(node_index)is not int or not 0<=node_index<len(template.parents)
            or node_index in template.clamped or not template.is_complete
            or template.skeleton_origin!='declared'):
        raise ValueError('An explicitly declared free target node is required')
    spec=decoder.field.spec
    if (known.source_snapshot_sha256!=spec.source_snapshot_sha256
            or known.context_sha256!=template.context_sha256
            or known.coordinate_sha256!=spec.coordinate_sha256
            or template.coordinate_sha256!=spec.coordinate_sha256):
        raise ValueError('Known causal prefix source/context/coordinate mismatch')
    path=[]
    cursor=node_index
    while cursor!=-1:
        path.append(cursor)
        cursor=template.parents[cursor]
    path.reverse()
    ancestors=path[:-1]
    if (known.node_ids!=tuple(template.node_ids[i] for i in ancestors)
            or len(path)>spec.max_nodes or len(path)>decoder.field.geometry.config['id_width']):
        raise ValueError('Prefix must contain exactly the ordered real ancestors')
    ref=decoder.codebook
    if (known.states.shape[1]!=spec.dimension or known.states.dtype!=ref.dtype
            or known.states.device!=ref.device or template.states.device!=ref.device
            or template.states.dtype!=ref.dtype):
        raise ValueError('Prefix must use the exact source-native axes, dtype and device')
    # Do not read template.states: it may encode the target and future.
    frozen=known.states.detach().clone()
    zeros=frozen.new_zeros((1,spec.dimension))
    values=torch.cat((frozen,zeros),dim=0)
    prefix_template=ForestState(
        tuple(template.node_ids[i] for i in path),
        tuple([-1]+list(range(len(path)-1))),
        tuple(['open']*len(ancestors)+['terminal']),
        values.clone(),tuple(range(len(ancestors))),
        tuple(template.directions[i] for i in path),
        known.context_sha256,known.coordinate_sha256,'declared')
    return values[None],prefix_template,len(path)-1
