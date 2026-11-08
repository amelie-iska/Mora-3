"""Opt-in μAdamW for declared generic width families; no model qualification.

Reference: amelie-iska/mup 19814971934ef91dd546f88e913fc963e096d11c,
``mup/optim.py`` and ``mup/infshape.py`` (MIT). With two width axes, the
reference divides Adam LR by the *last* width-axis multiplier. By default it
multiplies decay by that multiplier, preserving lr*decay even for AdamW;
``decoupled_wd=True`` instead leaves decay unchanged. Other parameters keep LR.

This module neither initializes/scales a readout nor qualifies an operator.
Current native packed-QKV/incidence/tropical custom ledgers remain refused.
Finite axes, aliases, complete heads, initialization and checkpointed *model*
configuration remain separate prerequisites. A valid optimizer is not permission
to train, nor proof of a complete MuTransfer implementation.
"""
import copy
import hashlib
import json
import math
import weakref

import torch
from torch.optim import AdamW

from .parameterization import (
    Axis, ParameterLedger, ParameterEntry, WidthFamily, parameter_ledger, width_family,
)


_SCHEMA = 'mora3.mu-adamw.v1'
_REFERENCE = 'amelie-iska/mup@19814971934ef91dd546f88e913fc963e096d11c'


def _number(value, name, *, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f'{name} must be a finite {"positive" if positive else "nonnegative"} number')
    return float(value)


def _checked_family(family):
    """Recompute caller-supplied ledger counts; a dataclass is not a certificate."""
    if type(family) is not WidthFamily:
        raise ValueError('exact declared WidthFamily required')
    for ledger in (family.base, family.delta, family.target):
        if type(ledger) is not ParameterLedger or type(ledger.parameters) is not tuple:
            raise ValueError('exact immutable parameter ledgers required')
        names, sizes = set(), {}
        for entry in ledger.parameters:
            if (type(entry) is not ParameterEntry or type(entry.aliases) is not tuple
                    or not entry.aliases or any(type(n) is not str or not n for n in entry.aliases)
                    or tuple(sorted(set(entry.aliases))) != entry.aliases
                    or names.intersection(entry.aliases) or type(entry.shape) is not tuple
                    or type(entry.axes) is not tuple or len(entry.shape) != len(entry.axes)
                    or any(type(s) is not int or s < 1 for s in entry.shape)
                    or type(entry.numel) is not int or entry.numel != math.prod(entry.shape)
                    or type(entry.trainable) is not bool or type(entry.dtype) is not str
                    or not entry.dtype.startswith('torch.')
                    or entry.role not in ('input','hidden','output','embedding','embedding_readout',
                                          'vector','normalization','scalar')):
                raise ValueError('unsupported or inconsistent parameter entry (custom operators refuse)')
            names.update(entry.aliases)
            for axis, size in zip(entry.axes, entry.shape):
                if type(axis) is not Axis:
                    raise ValueError('exact axis declaration required')
                Axis(axis.kind, axis.identity)
                key=(axis.kind,axis.identity)
                if key in sizes and sizes[key] != size:
                    raise ValueError('same ordered axis has inconsistent size')
                sizes[key]=size
            if sum(a.kind == 'width' for a in entry.axes) > 2:
                raise ValueError('more than two width axes are unsupported by reference MuAdam')
        total=sum(e.numel for e in ledger.parameters)
        trainable=sum(e.numel for e in ledger.parameters if e.trainable)
        if (type(ledger.total_numel) is not int or ledger.total_numel != total
                or type(ledger.trainable_numel) is not int or ledger.trainable_numel != trainable):
            raise ValueError('ledger unique totals disagree with complete entries')
    if width_family(family.base,family.delta,family.target) != family:
        raise ValueError('declared width-family metadata drift')
    return family


def _family_digest(family):
    def ledger(value):
        return dict(schema=value.schema,total_numel=value.total_numel,
            trainable_numel=value.trainable_numel,parameters=[dict(
                aliases=e.aliases,shape=e.shape,axes=[(a.kind,a.identity) for a in e.axes],
                role=e.role,numel=e.numel,trainable=e.trainable,dtype=e.dtype)
                for e in value.parameters])
    wire=dict(schema=family.schema,base=ledger(family.base),delta=ledger(family.delta),
              target=ledger(family.target),widths=family.widths,total_numel=family.total_numel)
    return hashlib.sha256(json.dumps(wire,sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def _storage_signature(model):
    """Refuse distinct owners backed by one storage, even disjoint views.

    Identity-tied aliases are allowed. This conservative restriction prevents
    double Adam updates and implicit modification of a frozen storage alias.
    No tensor values are read and no storage is moved or allocated.
    """
    owners,storages,result=set(),set(),[]
    for name,p in model.named_parameters(remove_duplicate=False):
        if id(p) in owners:continue
        owners.add(id(p))
        if (p.device.type not in ('cpu','cuda') or p.layout != torch.strided
                or (p.requires_grad and not p.is_floating_point())):
            raise ValueError('only dense real floating CPU/CUDA trainable owners are qualified')
        storage=p.untyped_storage()
        key=(str(p.device),storage.data_ptr())
        if key in storages:
            raise ValueError('distinct parameter objects share storage')
        storages.add(key)
        result.append((name,id(p),key,storage.nbytes(),p.storage_offset(),tuple(p.stride())))
    return tuple(result)


def build_mu_adamw_groups(model, family, declarations, *, base_lr, weight_decay,
                         decay_aliases,decoupled_wd=False,member='target'):
    """Return one deterministic group per unique trainable parameter, without mutation.

    ``decay_aliases`` explicitly names every alias to decay. Either all aliases
    of a tied owner are named or none; partial ownership is rejected. Frozen
    owners are counted in the family but omitted from optimization. Caller axis
    declarations remain assertions, not source/model/initialization certificates.
    """
    base_lr=_number(base_lr,'base_lr',positive=True)
    weight_decay=_number(weight_decay,'weight_decay')
    if type(decoupled_wd) is not bool:
        raise ValueError('decoupled_wd requires an explicit bool')
    _checked_family(family)
    if type(member) is not str or member not in ('base','delta','target'):
        raise ValueError('explicit supported width-family member required')
    live=parameter_ledger(model,declarations)
    if live != getattr(family,member):
        raise ValueError('live aliases/shape/axis/dtype/trainability differ from selected member ledger')
    _storage_signature(model)
    if (type(decay_aliases) is not tuple
            or any(type(n) is not str for n in decay_aliases)
            or len(set(decay_aliases)) != len(decay_aliases)):
        raise ValueError('explicit unique immutable decay alias tuple required')
    named=dict(model.named_parameters(remove_duplicate=False))
    decay=set(decay_aliases)
    if not decay <= set(named):
        raise ValueError('unknown decay alias')
    fingerprint=_family_digest(family)
    base={e.aliases:e for e in family.base.parameters}
    result=[]
    for entry in live.parameters:
        chosen=decay.intersection(entry.aliases)
        if chosen and chosen != set(entry.aliases):
            raise ValueError('conflicting tied-alias decay declarations')
        if not entry.trainable:
            if chosen:raise ValueError('frozen owner cannot be selected for decay')
            continue
        infinite=[i for i,a in enumerate(entry.axes) if a.kind == 'width']
        multiplier=(entry.shape[infinite[-1]]/base[entry.aliases].shape[infinite[-1]]
                    if len(infinite)==2 else 1.)
        lr_scale=1./multiplier
        wd_scale=multiplier if not decoupled_wd else 1.
        actual_decay=weight_decay*wd_scale if chosen else 0.
        if not math.isfinite(actual_decay):
            raise ValueError('scaled decay overflow')
        result.append(dict(params=[named[entry.aliases[0]]],lr=base_lr*lr_scale,
            weight_decay=actual_decay,mup_aliases=entry.aliases,
            mup_lr_multiplier=lr_scale,mup_decay_multiplier=wd_scale if chosen else 0.,
            mup_family_sha256=fingerprint))
    if not result:
        raise ValueError('at least one trainable owner required')
    return result


class MuAdamW(AdamW):
    """Narrow native wrapper with ratio-preserving schedules and strict recovery.

    Standard common-factor schedulers (e.g. LambdaLR) are supported, including a
    zero final LR. Group-specific schedules, closures, load hooks and unversioned
    optimizer states refuse. Model weights/RNG and scheduler state must be saved
    and restored separately by the existing staged trainer; this class does not
    implement a whole-training checkpoint transaction or authorize execution.
    """
    def __init__(self, model, family, declarations, *, base_lr, weight_decay,
                 decay_aliases,decoupled_wd=False,betas=(.9,.999),eps=1e-8,member='target'):
        if (type(betas) is not tuple or len(betas) != 2
                or any(type(b) not in (int,float) or not math.isfinite(b) or not 0 <= b < 1
                       for b in betas)):
            raise ValueError('explicit finite Adam beta tuple required')
        eps=_number(eps,'eps',positive=True)
        groups=build_mu_adamw_groups(model,family,declarations,base_lr=base_lr,
            weight_decay=weight_decay,decay_aliases=decay_aliases,decoupled_wd=decoupled_wd,member=member)
        self._mup_model=weakref.ref(model)
        self._mup_family=family
        self._mup_member=member
        self._mup_declarations=dict(declarations)
        self._mup_bound=tuple(g['params'][0] for g in groups)
        self._mup_storage=_storage_signature(model)
        self._mup_metadata=dict(schema=_SCHEMA,reference=_REFERENCE,
            family_sha256=_family_digest(family),base_lr=float(base_lr),
            weight_decay=float(weight_decay),decay_aliases=tuple(decay_aliases),
            decoupled_wd=decoupled_wd,betas=tuple(float(b) for b in betas),eps=eps)
        if member!='target':
            self._mup_metadata.update(schema='mora3.mu-adamw.v2',member=member)
        super().__init__(groups,lr=float(base_lr),weight_decay=float(weight_decay),
                         betas=betas,eps=eps,foreach=False,fused=False)
        self._mup_groups=tuple({k:copy.deepcopy(v) for k,v in g.items() if k!='params'}
                               for g in self.param_groups)

    def __deepcopy__(self,memo):
        """Support existing staged ``deepcopy((model, optimizer))`` recovery only.

        Optimizer.__getstate__ omits subclass binding metadata. Reconstructing
        that metadata explicitly must use actual memo-copied model/Parameter
        identities; matching names or shapes alone is never sufficient.
        Standalone optimizer copying refuses rather than producing an unbound
        optimizer or implicitly creating another model.
        """
        self._validate_live()
        if type(self) is not MuAdamW:
            raise ValueError('exact native MuAdamW copy contract required')
        source=self._mup_model();model=memo.get(id(source))
        if model is None or model is source:
            raise ValueError('MuAdamW requires paired deepcopy((model, optimizer))')
        source_named=dict(source.named_parameters(remove_duplicate=False))
        copied_named=dict(model.named_parameters(remove_duplicate=False))
        if (type(model) is not type(source) or set(copied_named)!=set(source_named)
                or any(memo.get(id(p)) is not copied_named[name] for name,p in source_named.items())):
            raise ValueError('paired model/parameter memo identities required')
        metadata=self._mup_metadata
        result=MuAdamW(model,self._mup_family,self._mup_declarations,
            base_lr=metadata['base_lr'],weight_decay=metadata['weight_decay'],
            decay_aliases=metadata['decay_aliases'],decoupled_wd=metadata['decoupled_wd'],
            betas=metadata['betas'],eps=metadata['eps'],member=self._mup_member)
        memo[id(self)]=result
        result.load_state_dict(copy.deepcopy(self.state_dict(),memo))
        return result

    def _validate_groups(self, groups, *, serialized=False, master=None):
        if type(groups) not in (list,tuple) or len(groups) != len(self._mup_groups):
            raise ValueError('complete ordered optimizer groups required')
        common=None
        for index,(actual,expected) in enumerate(zip(groups,self._mup_groups)):
            if type(actual) is not dict or set(actual)-{'initial_lr'} != set(expected)|{'params'}:
                raise ValueError('optimizer group schema drift')
            params=actual['params']
            if type(params) is not list or len(params) != 1:
                raise ValueError('exact unique optimizer owner required')
            if serialized:
                if type(params[0]) is not int or params[0] != index:
                    raise ValueError('serialized optimizer owner/order drift')
            elif params[0] is not self._mup_bound[index]:
                raise ValueError('optimizer parameter identity drift')
            for key,value in expected.items():
                if key != 'lr' and (type(actual[key]) is not type(value) or actual[key] != value):
                    raise ValueError('optimizer scaling/decay/settings drift')
            rate=_number(actual['lr'],'scheduled group LR')/expected['mup_lr_multiplier']
            if common is None:common=rate
            if not math.isclose(common,rate,rel_tol=1e-12,abs_tol=0):
                raise ValueError('scheduler changed relative muP learning-rate multipliers')
            if 'initial_lr' in actual and actual['initial_lr'] != expected['lr']:
                raise ValueError('scheduler initial LR drift')
        if master is not None and not math.isclose(common,_number(master,'scheduled master LR'),
                                                   rel_tol=1e-12,abs_tol=0):
            raise ValueError('checkpoint scheduled master LR drift')
        return common

    def _validate_live(self):
        model=self._mup_model()
        if model is None:
            raise ValueError('live model owner no longer exists')
        # Parameter identity, not equal values or equal tensor shapes, owns Adam state.
        named=dict(model.named_parameters(remove_duplicate=False))
        selected=getattr(self._mup_family,self._mup_member)
        for entry,parameter in zip((e for e in selected.parameters if e.trainable),
                                   self._mup_bound):
            if any(named.get(alias) is not parameter for alias in entry.aliases):
                raise ValueError('live shared owner identity drift')
        if parameter_ledger(model,self._mup_declarations) != selected:
            raise ValueError('live family declaration drift')
        if _storage_signature(model) != self._mup_storage:
            raise ValueError('live parameter storage ownership changed')
        return self._validate_groups(self.param_groups)

    def step(self, closure=None):
        if closure is not None:
            raise ValueError('closures require a separately qualified execution contract')
        self._validate_live()
        return super().step()

    def state_dict(self):
        master=self._validate_live()
        result=super().state_dict()
        result['mora3_mu_adamw']={**copy.deepcopy(self._mup_metadata),
                                 'scheduled_master_lr':master}
        return result

    def load_state_dict(self, state_dict):
        """Prevalidate all owners, scaling and native Adam slots before any load.

        Metadata/slot refusals leave this optimizer unchanged. Existing model
        recovery remains an independently staged transaction, never a migration.
        """
        self._validate_live()
        if (type(state_dict) is not dict or set(state_dict) != {'state','param_groups','mora3_mu_adamw'}
                or type(state_dict['mora3_mu_adamw']) is not dict
                or set(state_dict['mora3_mu_adamw']) != set(self._mup_metadata)|{'scheduled_master_lr'}):
            raise ValueError('exact versioned mu-adamw checkpoint required')
        metadata=state_dict['mora3_mu_adamw']
        if any(type(metadata[k]) is not type(v) or metadata[k] != v
               for k,v in self._mup_metadata.items()):
            raise ValueError('checkpoint family/optimizer policy drift')
        self._validate_groups(state_dict['param_groups'],serialized=True,
                              master=metadata['scheduled_master_lr'])
        states=state_dict['state']
        if (type(states) is not dict or any(type(k) is not int or not 0 <= k < len(self._mup_bound)
                                          for k in states)):
            raise ValueError('unknown checkpoint moment owner')
        for index,slots in states.items():
            p=self._mup_bound[index]
            if type(slots) is not dict or set(slots) != {'step','exp_avg','exp_avg_sq'}:
                raise ValueError('exact native Adam slot schema required')
            step=slots['step']
            if (type(step) is not torch.Tensor or step.shape != torch.Size([])
                    or step.dtype != torch.float32 or step.device.type != 'cpu'
                    or not torch.isfinite(step) or step.item() < 0 or not step.item().is_integer()):
                raise ValueError('finite nonnegative integral native Adam step required')
            for key in ('exp_avg','exp_avg_sq'):
                moment=slots[key]
                if (type(moment) is not torch.Tensor or moment.layout != torch.strided
                        or moment.shape != p.shape
                        or moment.dtype != p.dtype or not torch.isfinite(moment).all()
                        or (key=='exp_avg_sq' and (moment<0).any())):
                    raise ValueError('checkpoint Adam moment shape/dtype/value drift')
        if self._optimizer_load_state_dict_pre_hooks or self._optimizer_load_state_dict_post_hooks:
            raise ValueError('optimizer load hooks require separate recovery qualification')
        incoming=copy.deepcopy({k:state_dict[k] for k in ('state','param_groups')})
        previous=copy.deepcopy(super().state_dict())
        try:
            result=super().load_state_dict(incoming)
            self._validate_live()
            return result
        except Exception:
            super().load_state_dict(previous)
            raise
