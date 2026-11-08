"""Opt-in shared variable-N all-atom Cartesian GFN, no physical admission.

The fixed-dimensional conditional view is not a Module: binding a new source
system adds no parameters. Every transition reencodes the complete system with
shared actual TokenGT transforms. Six shared per-atom outputs specify a joint
normalized diagonal Gaussian on all free Cartesian logits. Conditional logZ uses
only static caller conditions. Teacher RPC, properness, source admission and
experimental validity remain the reward/campaign owner's responsibility.
"""
from dataclasses import asdict,dataclass
import copy
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from mora3.teacher_protocol import TeacherRefusal,hex_sha
from mora3.boltzmann_target import BoltzmannTarget,BoxSupport
from .tokengt import TokenGT
from .mixed_payload_projection import exact_wire

BOND_KINDS=('covalent','coordinate','reaction_partial','noncovalent_contact','caller_declared')
FEATURE_NAMES=(tuple('atomic-number-onehot:'+str(z) for z in range(1,119))+
    ('formal-charge','formal-charge-unknown','total-charge','spin-multiplicity','pbc-x','pbc-y','pbc-z',
     'bond-order','bond-order-unknown','topology-unknown')+
    tuple('cell-rowmajor:'+str(i) for i in range(9))+
    ('atom-fixed',)+tuple('fixed-xyz:'+x for x in 'xyz')+
    tuple('free-box-lower:'+x for x in 'xyz')+tuple('free-box-upper:'+x for x in 'xyz')+
    ('temperature-kelvin',)+tuple('bond-kind:'+x for x in BOND_KINDS)+
    tuple('periodic-image:'+x for x in 'xyz')+('periodic-image-unknown','energy-reference-eV','optional-stop')+
    tuple('current-xyz:'+x for x in 'xyz')+tuple('current-free-logit:'+x for x in 'xyz')+
    ('current-state-available','generative-layer-over-horizon'))


@dataclass(frozen=True,slots=True)
class CartesianBudget:
    max_atoms:int
    max_entities:int
    max_identifier_elements:int
    max_student_parameter_elements:int
    max_batch_size:int
    max_attention_elements:int

    def __post_init__(self):
        if any(type(v)is not int or v<=0 for v in asdict(self).values()):
            raise TeacherRefusal('explicit positive complete-system resource bounds required')


def box_coordinates_and_log_measure(logits,support):
    """Exact logit box map and normalized-uniform-box base Jacobian.

    This is the reviewed prototype equation: logJacobian-logVolume is
    sum(logsigmoid(z)+logsigmoid(-z)); Cartesian widths cancel in this correction
    only. Saturation refuses instead of clipping, cropping, or resampling.
    """
    if (type(support)is not BoxSupport or not isinstance(logits,torch.Tensor)
            or logits.ndim!=2 or logits.shape[1]!=support.dimension or not len(logits)
            or logits.dtype not in (torch.float32,torch.float64) or not torch.isfinite(logits).all()):
        raise TeacherRefusal('complete finite Cartesian-logit state required')
    physical=logits.to(dtype=torch.float64)
    lo=physical.new_tensor(support.lower_angstrom);hi=physical.new_tensor(support.upper_angstrom)
    free=lo+(hi-lo)*torch.sigmoid(physical)
    if not torch.isfinite(free).all() or not ((free>lo)&(free<hi)).all():
        raise TeacherRefusal('unrepresentable support-boundary sample; no clipping/resampling')
    correction=(F.logsigmoid(physical)+F.logsigmoid(-physical)).sum(-1)
    full=physical.new_empty(len(logits),len(support.fixed_xyz_angstrom),3)
    for i,row in enumerate(support.fixed_xyz_angstrom):
        if row is not None:full[:,i,:]=physical.new_tensor(row)
    for k,i in enumerate(support.free_atom_indices):full[:,i,:]=free[:,3*k:3*k+3]
    correction=correction.to(dtype=logits.dtype)
    if not torch.isfinite(correction).all():raise TeacherRefusal('unrepresentable box/base-measure correction')
    return full,correction


def normalized_terminal_log_reward(logits,energies_ev,target,*,horizon):
    """Pure equation, not a teacher-evaluation or admission authentication API.

    Caller MUST authenticate frozen evaluations and source/properness before
    passing their complete energies. No teacher or source lookup occurs here.
    """
    if (type(target)is not BoltzmannTarget or type(energies_ev)is not tuple
            or len(energies_ev)!=len(logits) or type(horizon)is not int or horizon<=0):
        raise TeacherRefusal('exact complete target/energy batch and horizon required')
    full,correction=box_coordinates_and_log_measure(logits,target.support)
    energy_terms=logits.new_tensor([target.energy_log_reward(x) for x in energies_ev])
    reward=(energy_terms+correction-(math.log(horizon) if target.optional_stop else 0.)).detach()
    if not torch.isfinite(reward).all():raise TeacherRefusal('nonfinite normalized terminal reward')
    return full,reward


class SharedCartesianGFNPolicy(nn.Module):
    def __init__(self,backbone,*,id_width,horizon,budget,teacher_sha256,
                 input_condition_contract_sha256,readout_mult=1.,backbone_identifier_axis=('context','observation')):
        super().__init__()
        if (type(backbone)is not TokenGT or type(budget)is not CartesianBudget
                or type(id_width)is not int or id_width<budget.max_atoms
                or type(horizon)is not int or horizon<=0 or not hex_sha(teacher_sha256)
                or not hex_sha(input_condition_contract_sha256)
                or type(readout_mult)not in (int,float) or not math.isfinite(readout_mult) or readout_mult<=0):
            raise TeacherRefusal('exact native backbone/complete-source rank/teacher/resource contract required')
        native=backbone.configuration();p=next(backbone.parameters())
        if (native['schema']!='tokengt-native-configuration-v3' or native['attention']!='vanilla'
                or native['feature_width']!=native['width'] or p.dtype not in (torch.float32,torch.float64)):
            raise TeacherRefusal('explicit applied ordinary native width family with complete hidden feature axis required')
        if (type(backbone_identifier_axis)is not tuple or len(backbone_identifier_axis)!=native['id_width']
                or any(type(x)is not str or not x for x in backbone_identifier_axis)
                or len(set(backbone_identifier_axis))!=len(backbone_identifier_axis)):
            raise TeacherRefusal('explicit complete ordered source backbone identifier axis required')
        self.backbone_identifier_axis=backbone_identifier_axis
        width=native['width'];base=native['parameterization']['base_width']
        # Exact unique total includes original input incidence, independent full
        # Cartesian incidence, finite-feature input and four finite readouts.
        expected=sum(x.numel() for x in backbone.parameters())+len(FEATURE_NAMES)*width+2*id_width*width+14*width+14
        if expected>budget.max_student_parameter_elements:
            raise TeacherRefusal('complete shared student parameter budget exceeded before allocation')
        self.backbone=backbone;self._source_backbone_reference=(backbone,)
        self.id_width=id_width;self.horizon=horizon;self.budget=budget
        self.teacher_sha256=teacher_sha256;self.input_condition_contract_sha256=input_condition_contract_sha256
        self.readout_mult=float(readout_mult);self.base_width=base;self.width=width
        cfg=copy.deepcopy(native);cfg['id_width']=id_width
        self.graph=TokenGT.from_configuration(cfg)
        for name in ('feature','role','graph_token','layers','final_norm'):
            setattr(self.graph,name,getattr(backbone,name))
        self.geometry_features=nn.Linear(len(FEATURE_NAMES),width,bias=False)
        self.forward_head=nn.Linear(width,6);self.backward_head=nn.Linear(width,6)
        self.stop_head=nn.Linear(width,1);self.normalizer=nn.Linear(width,1)
        with torch.no_grad():
            for head in (self.forward_head,self.backward_head,self.stop_head,self.normalizer):
                head.weight.mul_(math.sqrt(width/base));head.bias.mul_(math.sqrt(width/base))
        self.to(device=p.device,dtype=p.dtype)
        if sum(x.numel() for x in self.parameters())!=expected:
            raise TeacherRefusal('complete unique shared parameter accounting mismatch')
        self._configuration=self._current_configuration()

    def _current_configuration(self):
        return dict(schema='Mora3.SharedCartesianGFNPolicy/v1',policy='mup-cartesian-gfn-v1',
            native=self.backbone.configuration(),graph=self.graph.configuration(),id_width=self.id_width,
            backbone_identifier_axis=self.backbone_identifier_axis,
            horizon=self.horizon,budget=asdict(self.budget),teacher_sha256=self.teacher_sha256,
            input_condition_contract_sha256=self.input_condition_contract_sha256,
            base_width=self.base_width,readout_mult=self.readout_mult,features=FEATURE_NAMES,
            identifiers='exact-identity-zero-padded-source-atom-order-v1',
            state_space='Cartesian-open-box-logit;normalized-uniform-box-base',
            kernels='complete-state-TokenGT-shared-per-free-atom-xyz-mean-then-logstd-joint-product',
            normalizer='static-condition-only-shared-graph-token-linear',
            time='generative-DAG-not-physical-MD',teacher_parameters_inside_student=False,
            readout='fresh-sqrt-width-over-base;input-mult-over-width-ratio;bias-unscaled-in-forward')

    def get_extra_state(self):
        if (type(self.backbone)is not TokenGT or self.backbone is not self._source_backbone_reference[0]
                or type(self.graph)is not TokenGT or self.graph.incidence is self.backbone.incidence
                or any(getattr(self.graph,n)is not getattr(self.backbone,n)
                       for n in ('feature','role','graph_token','layers','final_norm'))):
            raise ValueError('Cartesian native shared operator alias drift')
        if set(self._modules)!={'backbone','graph','geometry_features','forward_head','backward_head','stop_head','normalizer'}:
            raise ValueError('Cartesian operator ownership drift')
        for name,out,inside,bias in (('geometry_features',self.width,len(FEATURE_NAMES),False),
                ('forward_head',6,self.width,True),('backward_head',6,self.width,True),
                ('stop_head',1,self.width,True),('normalizer',1,self.width,True)):
            h=getattr(self,name)
            if (type(h)is not nn.Linear or h._modules or h._buffers or (h.out_features,h.in_features)!=(out,inside)
                    or tuple(h.weight.shape)!=(out,inside) or (h.bias is not None)!=bias
                    or (bias and tuple(h.bias.shape)!=(out,))):
                raise ValueError('Cartesian exact finite feature/head configuration drift')
        if not exact_wire(self._current_configuration(),self._configuration):
            raise ValueError('Cartesian native/source/resource configuration drift')
        return copy.deepcopy(self._configuration)

    def set_extra_state(self,state):
        if not exact_wire(state,self.get_extra_state()):raise ValueError('Cartesian checkpoint configuration mismatch')

    def _load_from_state_dict(self,state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs):
        self.get_extra_state()
        for key,value in self.state_dict().items():
            if not isinstance(value,torch.Tensor) and not exact_wire(state_dict.get(prefix+key),value):
                raise ValueError('Cartesian checkpoint source/operator configuration mismatch')
        super()._load_from_state_dict(state_dict,prefix,local_metadata,strict,missing_keys,unexpected_keys,error_msgs)

    def bind(self,target):
        self.get_extra_state()
        if type(target)is not BoltzmannTarget or target.teacher.sha256!=self.teacher_sha256:
            raise TeacherRefusal('exact constructor-pinned frozen teacher target required')
        target.support.bind_system(target.system)
        return _BoundCartesianPolicy(self,target)

    def _readout(self,head,value):return head(value*(self.readout_mult/(self.width/self.base_width)))


class _BoundCartesianPolicy:
    """Ephemeral condition, never registered/serialized as a per-system head."""
    def __init__(self,owner,target):
        self.owner=owner;self.target=target;self.dimension=target.support.dimension;self.horizon=owner.horizon
        self._binding_reference=(owner,target,self.dimension,self.horizon)
        n=len(target.system.atom_ids);topology=json.loads(target.system.topology_json);bonds=topology.get('bonds')
        if bonds is not None and type(bonds)is not list:raise TeacherRefusal('explicit known bonds or unknownNone required')
        index={a:i for i,a in enumerate(target.system.atom_ids)}
        for bond in bonds or ():
            if (type(bond)is not dict or not {'first','second','kind','order','periodic_image','provenance'}<=set(bond)
                    or bond['first']not in index or bond['second']not in index
                    or bond['kind']not in BOND_KINDS or (bond['order']is not None
                    and (type(bond['order'])is not float or not math.isfinite(bond['order'])))):
                raise TeacherRefusal('exact complete source-declared bond map required')
            image=bond['periodic_image']
            if image is not None and (type(image)is not list or len(image)!=3
                    or any(type(v)is not int or (v!=0 and not target.system.pbc[j]) for j,v in enumerate(image))):
                raise TeacherRefusal('periodic image incompatible with source PBC')
            if bond['first']==bond['second'] and (image is None or not any(image)):
                raise TeacherRefusal('self bond requires an explicit nonzero periodic image')
        self._bonds=copy.deepcopy(bonds);self._index=index;self._n=n
        self._entities=n+2*len(bonds or ())
        self._budget(1)
        self._static,self._endpoints,self._roles=self._static_rows()

    def _budget(self,batch):
        if (self.owner is not self._binding_reference[0] or self.target is not self._binding_reference[1]
                or (self.dimension,self.horizon)!=self._binding_reference[2:]):
            raise TeacherRefusal('bound Cartesian source owner/target/dimension changed')
        m=self.owner;b=m.budget;cfg=m.graph.configuration();tokens=self._entities+1
        if (type(batch)is not int or not 1<=batch<=b.max_batch_size or self._n>b.max_atoms
                or self._n>m.id_width or tokens>b.max_entities or self._n*m.id_width>b.max_identifier_elements
                or batch*cfg['heads']*cfg['layers']*tokens*tokens>b.max_attention_elements):
            raise TeacherRefusal('complete atom/ID/entity/batch/attention budget exceeded; never crop')

    def _static_rows(self):
        t=self.target;s=t.system;support=t.support;free={a:k for k,a in enumerate(support.free_atom_indices)}
        cell=(0.,)*9 if s.cell_angstrom is None else tuple(x for row in s.cell_angstrom for x in row)
        def row(i,bond=None):
            z=[0.]*118
            if i is not None:z[s.atomic_numbers[i]-1]=1.
            q=None if i is None else s.formal_charges[i]
            fixed=i is not None and support.fixed_xyz_angstrom[i]is not None
            xyz=support.fixed_xyz_angstrom[i] if fixed else (0.,)*3
            k=free.get(i);lower=(0.,)*3 if k is None else support.lower_angstrom[3*k:3*k+3]
            upper=(0.,)*3 if k is None else support.upper_angstrom[3*k:3*k+3]
            order=None if bond is None else bond['order'];image=None if bond is None else bond['periodic_image']
            kind=[0.]*len(BOND_KINDS)
            if bond is not None:kind[BOND_KINDS.index(bond['kind'])]=1.
            result=[*z,0. if q is None else float(q),float(q is None),float(s.total_charge),float(s.spin_multiplicity),
                *(float(v) for v in s.pbc),0. if order is None else order,float(order is None),float(self._bonds is None),
                *cell,float(fixed),*xyz,*lower,*upper,t.temperature_kelvin,*kind,
                *((0.,)*3 if image is None else tuple(float(v) for v in image)),float(image is None),
                t.energy_reference_ev,float(t.optional_stop)]
            if len(result)!=len(FEATURE_NAMES)-8:raise TeacherRefusal('finite Cartesian feature-axis conservation failure')
            return result
        rows=[row(i) for i in range(self._n)];ends=[(i,i) for i in range(self._n)];roles=[0]*self._n
        for bond in self._bonds or ():
            i,j=self._index[bond['first']],self._index[bond['second']]
            reverse=copy.deepcopy(bond)
            if reverse['periodic_image']is not None:reverse['periodic_image']=[-v for v in reverse['periodic_image']]
            for endpoints,declaration in (((i,j),bond),((j,i),reverse)):
                rows.append(row(None,declaration));ends.append(endpoints);roles.append(1)
        return rows,ends,roles

    def _context(self,context):
        self.owner.get_extra_state();p=self.owner.normalizer.weight
        if (not isinstance(context,torch.Tensor) or context.ndim!=2 or context.shape[1]!=self.owner.width
                or context.dtype!=p.dtype or context.device!=p.device or not torch.isfinite(context).all()):
            raise TeacherRefusal('matching complete static condition context required')
        self._budget(len(context))

    def feature_rows(self,states,layer,*,static=False):
        self.owner.get_extra_state()
        p=self.owner.geometry_features.weight
        if (not isinstance(states,torch.Tensor) or states.ndim!=2 or states.shape[1]!=self.dimension
                or states.dtype!=p.dtype or states.device!=p.device or not torch.isfinite(states).all()
                or type(layer)is not int or not 0<=layer<=self.horizon or type(static)is not bool):
            raise TeacherRefusal('complete finite matching current state and valid layer required')
        self._budget(len(states))
        full=None if static else box_coordinates_and_log_measure(states,self.target.support)[0].to(dtype=p.dtype)
        free=states.new_zeros(len(states),self._n,3)
        if not static:free[:,self.target.support.free_atom_indices,:]=states.reshape(len(states),-1,3)
        result=[]
        for b in range(len(states)):
            dynamic=states.new_zeros(self._entities,8)
            if not static:
                dynamic[:self._n,:3]=full[b];dynamic[:self._n,3:6]=free[b]
                dynamic[:self._n,6]=1.;dynamic[:,-1]=layer/self.horizon
            rows=torch.cat((states.new_tensor(self._static),dynamic),-1)
            if not torch.isfinite(rows).all():raise TeacherRefusal('unrepresentable complete source/state feature values')
            result.append(rows)
        return result

    def graphs(self,states,layer,*,static=False):
        rows=self.feature_rows(states,layer,static=static);p=self.owner.geometry_features.weight
        ids=torch.eye(self._n,self.owner.id_width,dtype=p.dtype,device=p.device)
        ends=torch.tensor(self._endpoints,dtype=torch.long,device=p.device)
        roles=torch.tensor(self._roles,dtype=torch.long,device=p.device)
        return [dict(features=self.owner.geometry_features(r),endpoints=ends,roles=roles,identifiers=ids) for r in rows]

    def context(self,batch_size):
        self._budget(batch_size);p=self.owner.normalizer.weight
        state=p.new_zeros(batch_size,self.dimension)
        return self.owner.graph(self.graphs(state,0,static=True))[0][:,0]

    def parameters_at(self,states,layer,context,*,backward=False):
        self._context(context)
        if type(backward)is not bool or not isinstance(states,torch.Tensor) or states.ndim!=2 or len(states)!=len(context):raise TeacherRefusal('exact state/context batch and direction required')
        encoded=self.owner.graph(self.graphs(states,layer))[0]
        nodes=encoded[:,1:1+self._n][:,self.target.support.free_atom_indices,:]
        values=self.owner._readout(self.owner.backward_head if backward else self.owner.forward_head,nodes)
        mean=values[...,:3].flatten(1);ls=values[...,3:].flatten(1)
        stop=self.owner._readout(self.owner.stop_head,encoded[:,0]).squeeze(-1)
        if not all(torch.isfinite(v).all() for v in (mean,ls,stop)):raise TeacherRefusal('nonfinite normalized shared Cartesian kernel')
        return mean,ls,stop

    def log_partition(self,context):
        self._context(context);value=self.owner._readout(self.owner.normalizer,context).squeeze(-1)
        if not torch.isfinite(value).all():raise TeacherRefusal('nonfinite static conditional logZ')
        return value
