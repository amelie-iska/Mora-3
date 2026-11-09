"""Source-bound spatial branch with an independent exact incidence coordinate axis.

The annotation payload and all transformer operators are shared by identity with
an existing owner. Only spatial incidence and the finite output head are new.
No existing rank, weights, defaults, source admission or campaign gate is changed.
Reference composition: mora3.model.cartesian_gfn_policy.SharedCartesianGFNPolicy.
"""
from dataclasses import asdict, dataclass
import copy
from hashlib import sha256
import torch
from torch import nn

from mora3.model.tokengt import TokenGT
from mora3.model.mixed_payload_projection import exact_wire
from mora3.model.morphology_mup import _native_roles
from mora3.model.parameterization import parameter_ledger
from mora3.model.mu_optimizer import _storage_signature
from .contracts import canonical, digest
from .model import SharedSCRIModel, SCRITargetAxis, masked_native_objective, scri_parameter_roles
from .spatial import SpatialQuery, validate_spatial_query


@dataclass(frozen=True)
class SpatialBudget:
    max_cells: int
    max_entities: int
    max_attention_elements: int
    max_condition_bytes: int
    max_batch_size: int

    def __post_init__(self):
        if any(type(x) is not int or x < 1 for x in asdict(self).values()) or self.max_cells > 128:
            raise ValueError('explicit positive integer spatial bounds, at most 128 cells required')
        if self.max_entities < self.max_cells + 1:
            raise ValueError('entity bound must include all nodes and the graph token')


class NativeSCRIModel(nn.Module):
    """Launch-inactive spatial/transition branch on the literal annotation owner.

    Spatial predictions read node zero's representation (token position one),
    unlike transition predictions which retain the original graph readout. A
    separate fixed incidence basis changes neither the primary rank-two path
    nor its input semantics. All aliases and the additional incidence weights
    must appear in any complete-owner optimizer/checkpoint inventory.
    """
    def __init__(self, shared, axis, budget, source_snapshot_sha256):
        super().__init__()
        if type(budget) is not SpatialBudget or type(axis) is not SCRITargetAxis:
            raise ValueError('typed native spatial budget and target axis required')
        budget.__post_init__();axis.__post_init__();digest(source_snapshot_sha256,'source snapshot')
        self.budget=budget
        self.source_snapshot_sha256=source_snapshot_sha256
        self.core=SharedSCRIModel(shared,axis,max_batch_size=budget.max_batch_size,
            max_condition_bytes=budget.max_condition_bytes,max_spatial_cells=budget.max_cells)
        cfg=copy.deepcopy(shared.backbone.configuration())
        cfg['id_width']=budget.max_cells
        # A new native graph object with fresh incidence; no protected owner mutation.
        self.graph=TokenGT.from_configuration(cfg)
        p=next(shared.parameters())
        self.graph.to(device=p.device,dtype=p.dtype)
        for name in ('feature','role','graph_token','layers','final_norm'):
            setattr(self.graph,name,getattr(shared.backbone,name))
        self._identity=(self.core,self.graph,self.graph.incidence)
        self._configuration=self._state()

    @property
    def training_admitted(self):
        return False

    @property
    def spatial_identifier_axis(self):
        return tuple('SCRI-spatial-orthogonal-coordinate:'+str(i) for i in range(self.budget.max_cells))

    def _state(self):
        return dict(schema='Mora3.NativeSCRISpatialOwner/v1',core=self.core.get_extra_state(),
            graph=self.graph.configuration(),budget=asdict(self.budget),
            source_snapshot_sha256=self.source_snapshot_sha256,
            spatial_readout='explicit-query-node-zero-at-token-one',
            identifiers='exact-identity-zero-padded;fixed-across-width-family',
            topology='complete-native-radius-reciprocal;no-causal-edge-claim',
            admission='external-coordinator-only;constructor-is-not-admission')

    def get_extra_state(self):
        if (self._parameters or self._buffers or set(self._modules)!={'core','graph'}
                or any(a is not b for a,b in zip((self.core,self.graph,self.graph.incidence),self._identity))
                or type(self.graph) is not TokenGT
                or self.graph.incidence is self.core.shared.backbone.incidence
                or any(getattr(self.graph,n) is not getattr(self.core.shared.backbone,n)
                    for n in ('feature','role','graph_token','layers','final_norm'))):
            raise ValueError('native SCRI shared identity or topology drift')
        if not exact_wire(self._state(),self._configuration):
            raise ValueError('native SCRI source/axis/resource/parameterization drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):
            raise ValueError('native SCRI checkpoint binding mismatch')

    def load_state_dict(self,state_dict,strict=True,assign=False):
        if strict is not True or assign is not False:
            raise ValueError('strict in-place checkpoint recovery preserves shared owners')
        return super().load_state_dict(state_dict,strict=True,assign=False)

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state()
        if local_metadata.get('assign_to_params_buffers',False):
            raise ValueError('checkpoint assignment cannot replace shared parameter owners')
        expected=self.state_dict()
        incoming={k[len(prefix):]:v for k,v in state_dict.items() if k.startswith(prefix)}
        if set(incoming)!=set(expected):raise ValueError('complete native SCRI checkpoint keys required')
        for name,ref in expected.items():
            value=incoming[name]
            if isinstance(ref,torch.Tensor):
                if (not isinstance(value,torch.Tensor) or value.is_meta or value.shape!=ref.shape
                        or value.dtype!=ref.dtype or value.layout!=ref.layout or not torch.isfinite(value).all()):
                    raise ValueError('native SCRI checkpoint tensor mismatch')
            elif not exact_wire(value,ref):
                raise ValueError('native SCRI checkpoint source/configuration mismatch')
        aliases={}
        for name,p in self.named_parameters(remove_duplicate=False):aliases.setdefault(id(p),[]).append(name)
        for names in aliases.values():
            if any(not torch.equal(incoming[names[0]],incoming[n]) for n in names[1:]):
                raise ValueError('native SCRI checkpoint alias values disagree')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def _queries(self,queries):
        self.get_extra_state()
        if type(queries) not in (tuple,list) or not 1<=len(queries)<=self.budget.max_batch_size:
            raise ValueError('complete bounded spatial query batch required')
        largest=0
        for q in queries:
            if type(q) is not SpatialQuery:raise ValueError('typed spatial query required')
            validate_spatial_query(q)
            if not exact_wire(q.target_axis,self.core.axis.features):
                raise ValueError('exact native target axis and units required')
            n=len(q.record.nodes);tokens=1+n+len(q.record.edges);largest=max(largest,tokens)
            if n>self.budget.max_cells or tokens>self.budget.max_entities:
                raise ValueError('complete spatial node/entity budget exceeded; never truncate')
            if len(canonical(asdict(q.record)))+len(q.task.canonical_bytes())>self.budget.max_condition_bytes:
                raise ValueError('complete spatial payload budget exceeded')
        cfg=self.graph.configuration()
        # Padded batch work, not the misleading sum of individual graph sizes.
        elements=len(queries)*cfg['heads']*cfg['layers']*largest*largest
        if elements>self.budget.max_attention_elements:
            raise ValueError('complete padded-batch attention budget exceeded')
        return queries

    def forward_spatial(self,queries):
        queries=self._queries(queries);p=self.graph.feature.weight;graphs=[]
        for q in queries:
            ids=torch.eye(len(q.record.nodes),self.budget.max_cells,device=p.device,dtype=p.dtype)
            graphs.append(self.core.shared.payload.forward_task(q.record,task=q.task,identifiers=ids))
        latent=self.graph(graphs)[0][:,1]
        return self.core.output(self.core.shared._readout_features(latent))

    def forward(self,pairs):
        self.get_extra_state()
        return self.core(pairs)

    def objective(self,parameters,values,observed):
        self.get_extra_state()
        return masked_native_objective(parameters,values,observed,self.core.axis)

    def require_training_admission(self):
        raise ValueError('native SCRI construction is not source/corpus/operator/launch admission')


def native_scri_roles(model,*,identifier_axis):
    if type(model) is not NativeSCRIModel:raise ValueError('exact native SCRI owner required')
    model.get_extra_state()
    roles={'core.'+n:r for n,r in scri_parameter_roles(model.core,identifier_axis=identifier_axis).items()}
    roles.update({'graph.'+n:r for n,r in _native_roles(model.graph,model.spatial_identifier_axis).items()})
    _storage_signature(model);parameter_ledger(model,roles)
    return roles
